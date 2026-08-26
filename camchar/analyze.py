"""Post-acquisition photon-transfer analysis (temporal PTC / EMVA 1288).

Analyzes dark/ and flat/ subdirectories produced by `camchar get-dark-frames`
and `camchar get-flat-frames` and reports:

  K (system gain, e-/DN), read noise (e-), dark current (e-/s),
  saturation capacity (e-), PRNU (%), bias floor (DN), linearity.

Math (validated against Sony IMX174, see camera-noise-characterization skill):
  * temporal per-pixel variance:  Var = K*S + sigma_r^2   (DN units)
  * shot-noise identity:          Var/S = 1/K  in DN units  ->  K = S/Var
    (fitting V = K_fit*S + b gives K_fit = V/S ratio; the gain is 1/K_fit,
     NOT K_fit -- the classic trap. Anchor with the knee: K = Nsat/DN_clip.)
  * scaled data (12-bit << 4):    V16/S16 = 16 * (V12/S12), K12 = 16/(V16/S16)
  * reference document: docs/EMVA1288Linear_4.0Release.pdf -- all equation
    and section citations below follow EMVA 1288 R4 *Linear* numbering (the
    companion General PDF in docs/ numbers them differently). The PTC gain K
    is core Linear methodology (Sections 2.4 and 6.6). Two-frame estimators,
    dark-noise fit, quantization correction and highpass nonuniformities
    follow the equations cited inline.
"""

import json
from pathlib import Path

import numpy as np
import typer

from .io_utils import (
    DEFAULT_ROI_FRAC,
    DEFAULT_ROI_SPECIM,
    FRAME_MEAN_SPREAD_WARN_DN16,
    SAT_CLIP_FRAC,
    camera_dir_name,
    central_roi,
    stem_for,
)
from .plots import (
    save_dark_mean_plot,
    save_dark_variance_plot,
    save_linearity_plot,
    save_ptc_plot,
    save_snr_plot,
)

CLIP_DN = 65400  # anything >= this is clipped (IMX174: 4094<<4 = 65504)
DARK_CURRENT_FLAT_MAX_EXP = (
    0.01  # exposures <= 10 ms: dark-current shot noise negligible
)
SAT_MAX_DN = 65504  # sensor digital maximum (12-bit 4094 << 4)
PTC_FIT_SAT_FRAC = (
    0.7  # EMVA R4 Linear PTC regression range: minimum value to 70% saturation
)
QUANT_STEP_DN16 = 16.0  # DN16 per 12-bit LSB (DN12 << 4 storage)

_VENDOR_NAMES = {
    "playerone": "Player One",
    "basler": "Basler",
    "thorlabs": "Thorlabs",
}


def load_camera_meta(data_dir):
    """First metadata entry with a camera model, or None."""
    for seq_type in ("dark", "flat"):
        meta_path = Path(data_dir) / seq_type / "metadata.json"
        if not meta_path.exists():
            continue
        entries = json.loads(meta_path.read_text())
        for m in entries:
            if m.get("model"):
                return m
    return None


def _dataset_cfa_format(cam_dir):
    """Pixel format of the latest raw-Bayer acquisition, or '' if none.

    metadata.json is append-only and may mix sessions; scanning every entry
    (and keeping the last Bayer one) keeps the CFA decision and the phase
    labels consistent even when early entries predate pixel_format recording.
    """
    fmt = ""
    for seq_type in ("dark", "flat"):
        meta_path = Path(cam_dir) / seq_type / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            entries = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        for m in entries:
            candidate = str(m.get("pixel_format", ""))
            if candidate.startswith("Bayer"):
                fmt = candidate
    return fmt


def load_camera_info(data_dir):
    """Camera display label like 'Player One Apollo-M (IMX174)' from metadata."""
    m = load_camera_meta(data_dir)
    if not m:
        return None
    vendor = m.get("vendor", "")
    label = f"{_VENDOR_NAMES.get(vendor, vendor)} {m['model']}"
    if m.get("sensor"):
        label += f" ({m['sensor']})"
    return label


def resolve_data_dir(data_dir):
    """(camera_dir, layout) for the data under data_dir, or None.

    layout 'npy'    -- camchar acquisition layout: <dir>/dark|flat/metadata.json
    layout 'specim' -- SPECIM IQ export: <dir>/dark frames|flat-field frames

    data_dir itself is accepted when it already holds one of the layouts
    (legacy data/dark included). Otherwise subdirs are scanned: exactly one
    camera is used; several or none produce an error listing the choices.
    """
    from .specim import is_specim_dir  # local: keeps the spectral import lazy

    d = Path(data_dir)
    if not d.exists():
        typer.secho(f"  ! data dir not found: {d}", fg=typer.colors.YELLOW)
        return None
    if (d / "dark").exists() or (d / "flat").exists():
        return d, "npy"
    if is_specim_dir(d):
        return d, "specim"
    candidates = []
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "dark" / "metadata.json").exists() or (
            (sub / "flat" / "metadata.json").exists()
        ):
            candidates.append((sub, "npy"))
        elif is_specim_dir(sub):
            candidates.append((sub, "specim"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        typer.secho(
            f"  ! no camera data under {d} "
            f"(expected <root>/<vendor>_<model>_(<sensor>)/dark|flat "
            f"or a SPECIM IQ 'dark frames'/'flat-field frames' layout)",
            fg=typer.colors.YELLOW,
        )
        return None
    typer.secho(
        "  ! multiple camera dirs found -- pass one explicitly with --data:",
        fg=typer.colors.YELLOW,
    )
    for c, layout in candidates:
        typer.echo(f"      {c}  ({layout})")
    return None


def load_sequence(data_dir, seq_type):
    """Return [(exposure_s, stack), ...] sorted, skipping missing files.

    Metadata entries are deduped by (exposure_s, gain) -- append-only
    metadata.json accumulates entries across runs and can hold duplicates.
    """
    d = Path(data_dir) / seq_type
    meta_path = d / "metadata.json"
    if not meta_path.exists():
        typer.secho(
            f"  ! no metadata at {meta_path} -- nothing to analyze",
            fg=typer.colors.YELLOW,
        )
        return []
    entries = json.loads(meta_path.read_text())
    seen = set()
    out = []
    for m in entries:
        if m.get("sequence") != seq_type:
            continue
        key = (m["exposure_s"], m.get("gain"))
        if key in seen:
            continue
        seen.add(key)
        p = d / f"{stem_for(seq_type, m['exposure_s'], m['gain'])}.npy"
        if not p.exists():
            typer.secho(f"  ! missing {p.name}, skipping", fg=typer.colors.YELLOW)
            continue
        out.append((m["exposure_s"], np.load(p)))
    out.sort(key=lambda x: x[0])
    return out


def _box_filter(img, k):
    """k x k box filter (edge-padded, integral-image implementation)."""
    if k <= 1:
        return img.astype(np.float64)
    pad = k // 2
    p = np.pad(img.astype(np.float64), pad, mode="edge")
    ii = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    out = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    return out / (k * k)


def _binomial3(img):
    """Separable 3x3 binomial filter ([1,2,1]/4 per axis)."""
    p = np.pad(img.astype(np.float64), 1, mode="edge")
    h = (p[:-2, :] + 2 * p[1:-1, :] + p[2:, :]) / 4.0
    return (h[:, :-2] + 2 * h[:, 1:-1] + h[:, 2:]) / 4.0


def highpass(img):
    """EMVA 1288 R4 Section 8.1 highpass: img - binomial3(box11(box7(img)))."""
    return img - _binomial3(_box_filter(_box_filter(img, 7), 11))


_CFA_LABELS = ("R", "G1", "G2", "B")

# Pixel-format letters -> phase labels in _cfa_phases order (0,0),(0,1),(1,0),(1,1)
_CFA_LAYOUTS = {
    "RG": ("R", "G1", "G2", "B"),
    "GR": ("G1", "R", "B", "G2"),
    "GB": ("G1", "B", "R", "G2"),
    "BG": ("B", "G1", "G2", "R"),
}


def _cfa_labels(pixel_format=""):
    """Phase channel labels for a recorded Bayer pixel format (pure)."""
    fmt = str(pixel_format)
    if fmt.startswith("Bayer") and len(fmt) >= 7:
        return _CFA_LAYOUTS.get(fmt[5:7].upper(), _CFA_LABELS)
    return _CFA_LABELS


def _cfa_phases(img):
    """The four Bayer sub-lattices of an ROI image (pure).

    Order is (0,0), (0,1), (1,0), (1,1); label them with _cfa_labels() for
    the pixel format at hand.
    """
    return [img[i::2, j::2] for i in range(2) for j in range(2)]


def two_frame_stats(stack, roi):
    """EMVA 1288 R4 Linear two-frame estimates (Eqs. 18 and 32).

    Temporal variance from consecutive non-overlapping pairs (0,1), (2,3), ...:
        s2y.tf = mean((yA - yB)^2)/2 - (muA - muB)^2/2     (Eq. 18, incl. the
    Release-4 common-mode mean-difference correction).
    Spatial variance from the pair covariance (Eq. 32):
        s2y.tf = mean(yA*yB) - muA*muB
    which is exactly the fixed-pattern variance (temporal noise is uncorrelated
    between frames and cancels in the covariance).

    Returns (tf_tvar, tf_svar, tf_tvar_scatter, tf_svar_scatter): pair-averaged
    estimates plus the std across pairs (a direct uncertainty measure).
    """
    r = stack[:, roi[0], roi[1]].astype(np.float64)
    n_pairs = r.shape[0] // 2
    if n_pairs < 1:
        return 0.0, 0.0, 0.0, 0.0
    tvals = np.empty(n_pairs)
    svals = np.empty(n_pairs)
    for i in range(n_pairs):
        a = r[2 * i]
        b = r[2 * i + 1]
        mu_a, mu_b = a.mean(), b.mean()
        d = a - b
        tvals[i] = 0.5 * (d * d).mean() - 0.5 * (mu_a - mu_b) ** 2
        svals[i] = (a * b).mean() - mu_a * mu_b
    return (
        float(tvals.mean()),
        float(svals.mean()),
        float(tvals.std()),
        float(svals.std()),
    )


def roi_stats(stack, roi):
    """Per-pixel temporal mean/variance + spatial variance inside the ROI.

    N-frame estimates (mean/variance over all frames) plus the EMVA R4
    two-frame cross-check values (tf_*), see two_frame_stats().

    Temporal variance uses ddof=1 (EMVA R4 Linear Eq. 65, Section 8.2:
    1/(L-1)) -- the population (ddof=0) estimator is biased low by
    (N-1)/N = 5% at N=20, which the two-frame cross-check exposes (K,
    sigma_r, Nsat all shift). Spatial variance keeps ddof=0 (EMVA spatial
    sums use 1/(NM); bias < 0.01% at 10^4 pixels).
    """
    r = stack[:, roi[0], roi[1]].astype(np.float64)
    tf_t, tf_s, tf_ts, tf_ss = two_frame_stats(stack, roi)
    frame_means = r.mean(axis=(1, 2))
    return {
        "mean": r.mean(),
        "tvar": r.var(axis=0, ddof=1).mean(),  # temporal variance, avg over pixels
        "svar": r.var(axis=1).mean(),  # spatial variance, avg over frames
        "tf_tvar": tf_t,  # two-frame temporal variance (Eq. 18)
        "tf_svar": tf_s,  # two-frame spatial variance (Eq. 32)
        "tf_tvar_scatter": tf_ts,  # pair-to-pair std of tf_tvar
        "tf_svar_scatter": tf_ss,  # pair-to-pair std of tf_svar
        "frame_mean_spread": float(frame_means.max() - frame_means.min()),
        "sat_frac": float((r >= SAT_MAX_DN).mean()),  # EMVA 6.6 saturation test
    }


def usable_flat_points(rows):
    """Flat rows usable for the EMVA-extras point selection (pure).

    Primary rule: sat_frac <= SAT_CLIP_FRAC (EMVA 6.6 -- a majority-pinned
    point carries no variance). When no row passes, fall back to all rows
    below the clip mean (data that never saturates properly). Returns
    (subset, used_fallback).
    """
    ok = [rs for rs in rows if rs[1]["sat_frac"] <= SAT_CLIP_FRAC]
    if ok:
        return ok, False
    return [rs for rs in rows if rs[1]["mean"] < CLIP_DN], True


def fit_dark_stats(rows):
    """Eq. 29/30 dark fits from roi_stats rows [(exposure_s, stats)] (pure).

    bias is the measured mean of the shortest exposure (not the fit
    intercept); the Eq. 30 variance-fit slope is the second (variance-based)
    dark-current estimate. The two-frame temporal variance (Eq. 18) is the
    primary input -- it rejects between-acquisition drift -- so sigma_r and
    the variance-based dark current come from it; the N-frame values are the
    cross-check (sigma_r_nf_dn, dark_current_var_dn2_per_s_nf).
    """
    t = np.array([e for e, _ in rows])
    mu = np.array([s["mean"] for _, s in rows])
    tvars = np.array([s["tvar"] for _, s in rows])
    tf_tvars = np.array([s["tf_tvar"] for _, s in rows])

    slope = np.polyfit(t, mu, 1)[0]  # Eq. 29: mu_d = mu_d0 + mu_I.y t
    tf_var_slope, tf_var_offset = np.polyfit(t, tf_tvars, 1)  # Eq. 30 (two-frame)
    nf_var_slope, nf_var_offset = np.polyfit(t, tvars, 1)  # N-frame cross-check
    short = t <= DARK_CURRENT_FLAT_MAX_EXP
    if short.any():
        sigma_r_med = float(np.sqrt(np.median(tf_tvars[short])))
    else:
        sigma_r_med = float(np.sqrt(np.median(tf_tvars)))  # no short exposure available
    return {
        "bias_dn": float(mu[0]),
        "dark_current_dn_per_s": float(slope),
        "dark_current_var_dn2_per_s": float(tf_var_slope),
        "dark_current_var_dn2_per_s_nf": float(nf_var_slope),
        "sigma_r_dn": float(np.sqrt(max(tf_var_offset, 0.0))),
        "sigma_r_nf_dn": float(np.sqrt(max(nf_var_offset, 0.0))),
        "sigma_r_median_dn": sigma_r_med,
    }


def analyze_dark(seq, roi):
    """Bias floor, dark current, read noise from dark frames."""
    typer.secho(
        "=== DARK (tf = two-frame primary; nf = N-frame cross-check) ===",
        bold=True,
        fg=typer.colors.CYAN,
    )
    if not seq:
        return None
    rows = []
    for exp_s, st in seq:
        s = roi_stats(st, roi)
        rows.append((exp_s, s))
        typer.echo(
            f"  {exp_s * 1000:8.1f} ms  mean {s['mean']:8.2f}  "
            f"tf {s['tf_tvar']:9.2f}  nf {s['tvar']:9.2f}  "
            f"svar {s['svar']:8.2f}  tfs {s['tf_svar']:8.2f}"
        )
        if s["frame_mean_spread"] > FRAME_MEAN_SPREAD_WARN_DN16:
            typer.secho(
                f"  ! {exp_s * 1000:8.1f} ms: frame means spread "
                f"{s['frame_mean_spread']:.1f} DN16 across frames -- "
                "black-level step or source jump; bias/dark-current/read-noise "
                "from this stack are unreliable, exclude or re-acquire it",
                fg=typer.colors.RED,
                bold=True,
            )

    out = fit_dark_stats(rows)
    out["rows"] = rows
    return out


def fit_flat_stats(rows, dark):
    """PTC fit and derived quantities from roi_stats rows (pure).

    EMVA R4 Linear Eq. 50: sigma_y^2 = sigma_y.dark^2 + K*(mu_y - mu_y.dark).
    The two-frame temporal variance (Eq. 18) is the primary estimator -- it
    rejects between-acquisition drift, which inflates the N-frame variance
    (the SPECIM degenerate-PTC cause). For each usable flat point the
    measured two-frame dark temporal variance (matched to the flat's exposure
    by interpolation over the dark rows, linearly extrapolated outside the
    dark exposure grid) is subtracted: dV = tf_tvar - sigma_y.dark^2(t_exp),
    S = mean - bias_dn, and K_fit is the zero-intercept least-squares slope
    K_fit = sum(S*dV)/sum(S^2) (the standard's p. 23 regression of
    sigma_y^2 - sigma_y.dark^2 vs the photo-induced signal). Points with
    dV <= 0 (dark-subtraction noise at the lowest signals) are excluded from
    the fit when at least 3 positive points remain, keeping the fit selection
    identical to the PTC plot and the per-point gain check. The intercept of
    the free (offset) fit of dV vs S is the sanity check: it must come out
    near 0, and a large deviation flags PTC curvature (PRNU^2*S^2
    super-linearity or dark drift). R^2 uses the uncentered denominator
    sum(dV^2) (the model is through the origin). Derives K12 = 16/K_fit,
    electron-unit read noises (Eq. 53 quantization-corrected variants
    included) and the quick upper-bound PRNU estimates; the N-frame
    equivalents (dV_nf, K12_nf, sigma_r_nf_e, prnu_nf_pct) are the
    cross-check. Returns None when fewer than 3 unclipped points are
    available.

    Usable = mean < CLIP_DN *and* sat_frac <= SAT_CLIP_FRAC (EMVA 6.6):
    heavily-pinned points can sit below the mean threshold while their
    variance has collapsed (pinned pixels don't fluctuate), which would
    bend the PTC fit. Per EMVA R4 Linear the regression must use all data
    points between the minimum value and 70% saturation (mu_sat = the
    measured saturation point, i.e. the largest mean passing the 6.6 test),
    so points above PTC_FIT_SAT_FRAC * mu_sat are excluded; when the data
    never reaches saturation (fewer than 3 points under the cap) all points
    are kept. The cap itself stays in raw-DN units; only the fit abscissa
    is bias-subtracted.
    """
    ok = [s for _, s in rows if s["mean"] < CLIP_DN and s["sat_frac"] <= SAT_CLIP_FRAC]
    if len(ok) < 3:
        return None
    mu_sat = max(s["mean"] for s in ok)
    cap = PTC_FIT_SAT_FRAC * mu_sat
    ok_fit = [s for s in ok if s["mean"] <= cap]
    if len(ok_fit) < 3:
        ok_fit = ok  # never reached saturation: keep all linear points
    capped = len(ok_fit) < len(ok)

    def _dark_tvar(key):
        d_ts = np.array([e for e, _ in dark["rows"]])
        d_vals = np.array([s[key] for _, s in dark["rows"]])
        if len(d_ts) < 2:
            med = float(np.median(d_vals)) if len(d_vals) else 0.0
            return lambda t: med
        order = np.argsort(d_ts)
        d_ts, d_vals = d_ts[order], d_vals[order]

        def at(t):
            t = float(t)
            if t < d_ts[0]:
                slope = (d_vals[1] - d_vals[0]) / (d_ts[1] - d_ts[0])
                return float(d_vals[0] + slope * (t - d_ts[0]))
            if t > d_ts[-1]:
                slope = (d_vals[-1] - d_vals[-2]) / (d_ts[-1] - d_ts[-2])
                return float(d_vals[-1] + slope * (t - d_ts[-1]))
            return float(np.interp(t, d_ts, d_vals))

        return at

    dark_tvar_nf = _dark_tvar("tvar")
    dark_tvar_tf = _dark_tvar("tf_tvar")
    dV_all = np.array([s["tf_tvar"] - dark_tvar_tf(t) for t, s in rows])
    dV_nf_all = np.array([s["tvar"] - dark_tvar_nf(t) for t, s in rows])
    ok_ids = {id(s) for s in ok_fit}
    fit_idx = [i for i, (_, s) in enumerate(rows) if id(s) in ok_ids]
    pos_idx = [i for i in fit_idx if dV_all[i] > 0]
    if len(pos_idx) >= 3:
        fit_idx = pos_idx  # keep fit/plot/check selections identical (dV > 0)
    if len(fit_idx) < 3 or not any(dV_all[i] > 0 for i in fit_idx):
        # degenerate: no point carries positive dark-subtracted variance
        # (e.g. flats acquired with the lens cap on)
        return None
    S = np.array([rows[i][1]["mean"] - dark["bias_dn"] for i in fit_idx])
    V = dV_all[fit_idx]
    K_fit = float(np.sum(S * V) / np.sum(S * S))  # zero-intercept LSQ (Eq. 50)
    V_hat = K_fit * S
    # uncentered R^2 (TSS = sum V^2): the model is through the origin, so the
    # mean-centered denominator is not the right reference
    tss = float(np.sum(V * V))
    ptc_r2 = 1.0 - np.sum((V - V_hat) ** 2) / tss if tss > 0 else float("nan")
    _, b_free = np.polyfit(S, V, 1)  # free-fit intercept: sanity check, ~0
    K12 = 16.0 / K_fit  # DN16 -> DN12: K12 = 16/(V16/S16)
    sigma_r_e = dark["sigma_r_dn"] * K12 / 16.0
    sigma_r_nf_e = dark["sigma_r_nf_dn"] * K12 / 16.0

    # Eq. 53: subtract the quantization variance (step^2/12) for the
    # physical-unit dark noise -- 21.3 DN16^2 here, ~14% of the dark variance
    sig_q2 = QUANT_STEP_DN16**2 / 12.0
    sigma_r_e_q = np.sqrt(max(dark["sigma_r_dn"] ** 2 - sig_q2, 0.0)) * K12 / 16.0
    sigma_r_nf_e_q = np.sqrt(max(dark["sigma_r_nf_dn"] ** 2 - sig_q2, 0.0)) * K12 / 16.0

    # N-frame PTC cross-check: same dark-subtracted zero-intercept fit on the
    # N-frame temporal variance
    V_nf = dV_nf_all[fit_idx]
    K_fit_nf = float(np.sum(S * V_nf) / np.sum(S * S))
    K12_nf = 16.0 / K_fit_nf

    prnu = []
    prnu_nf = []
    dark_tf_svar = (
        np.median(
            [s["tf_svar"] for e, s in dark["rows"] if e <= DARK_CURRENT_FLAT_MAX_EXP]
        )
        if dark
        else 0.0
    )
    for _, s in rows:
        if 5000 < s["mean"] < 64000:
            prnu.append(np.sqrt(max(s["tf_svar"] - dark_tf_svar, 0)) / s["mean"])
            prnu_nf.append(np.sqrt(max(s["svar"] - s["tvar"], 0)) / s["mean"])

    return {
        "K12": K12,
        "K12_nf": K12_nf,
        "sigma_r_e": sigma_r_e,
        "sigma_r_e_q": sigma_r_e_q,
        "sigma_r_nf_e": sigma_r_nf_e,
        "sigma_r_nf_e_q": sigma_r_nf_e_q,
        "Nsat": K12 * 4094,
        "bias_dn": dark["bias_dn"],
        "ptc_slope": K_fit,
        "ptc_intercept": 0.0,  # Eq. 50 zero-intercept fit on the dark-subtracted variance
        "ptc_intercept_free": b_free,  # sanity check: ~0 for a valid PTC
        "ptc_slope_nf": K_fit_nf,
        "ptc_r2": ptc_r2,
        "mu_sat_dn": mu_sat,
        "ptc_sat_70_dn": cap if capped else mu_sat,
        "ptc_capped": capped,
        "dV": dV_all.tolist(),  # two-frame sig_y^2 - sig_y,dark^2 per row (aligned with rows)
        "dV_nf": dV_nf_all.tolist(),  # N-frame variant, aligned with rows
        "prnu_pct": float(np.mean(prnu) * 100) if prnu else None,
        "prnu_nf_pct": float(np.mean(prnu_nf) * 100) if prnu_nf else None,
    }


def analyze_flats(seq, dark, roi):
    """K, Nsat, PRNU from the flat sweep (temporal variance)."""
    typer.secho(
        "\n=== FLAT (tf = two-frame primary; nf = N-frame cross-check) ===",
        bold=True,
        fg=typer.colors.CYAN,
    )
    if not seq:
        return None
    rows = []
    for exp_s, st in seq:
        s = roi_stats(st, roi)
        rows.append((exp_s, s))
        typer.echo(
            f"  {exp_s * 1000:8.1f} ms  mean {s['mean']:10.2f}  "
            f"tf {s['tf_tvar']:12.2f}  nf {s['tvar']:12.2f}  "
            f"svar {s['svar']:12.2f}  tfs {s['tf_svar']:12.2f}"
        )

    flat = fit_flat_stats(rows, dark)
    if flat is None:
        typer.secho(
            "  ! flat fit failed: <3 usable points (unclipped, positive "
            "dark-subtracted variance) -- check illumination / lens cap",
            fg=typer.colors.YELLOW,
        )
        return None
    flat["rows"] = rows

    K12 = flat["K12"]
    K12_nf = flat["K12_nf"]
    sigma_r_e = flat["sigma_r_e"]
    sigma_r_nf_e = flat["sigma_r_nf_e"]
    sigma_r_e_q = flat["sigma_r_e_q"]
    sigma_r_nf_e_q = flat["sigma_r_nf_e_q"]
    typer.secho(
        "\n=== PTC FIT (two-frame temporal variance, Eq. 50: dV = K*S, "
        "S = mu_y - mu_y,dark, dV = sig_y^2 - sig_y,dark^2) ===",
        bold=True,
        fg=typer.colors.CYAN,
    )
    typer.echo(
        f"  K (gain, two-frame) = {K12:6.2f} e-/DN12   "
        f"(V/S slope {flat['ptc_slope']:.3f}, Eq. 18)"
    )
    typer.echo(
        f"  K (N-frame) = {K12_nf:6.2f} e-/DN12   "
        f"(slope {flat['ptc_slope_nf']:.3f}; "
        f"{100 * (K12_nf - K12) / K12:+.1f}% vs two-frame)"
    )
    typer.echo(
        f"  free intercept = {flat['ptc_intercept_free']:+.1f} DN16^2 "
        f"(sanity check: ~0 for a valid PTC)"
    )
    typer.echo(
        f"  sigma_r      = {sigma_r_e:6.2f} e-  ({dark['sigma_r_dn']:.2f} DN16; "
        f"Eq. 53 quant-corrected {sigma_r_e_q:.2f} e-)"
    )
    typer.echo(
        f"  sigma_r (nf) = {sigma_r_nf_e:6.2f} e-  ({dark['sigma_r_nf_dn']:.2f} DN16; "
        f"corrected {sigma_r_nf_e_q:.2f} e-)"
    )
    typer.echo(f"  Nsat         = {flat['Nsat']:7.0f} e- (at 12-bit clip 4094)")
    if flat["prnu_pct"] is not None:
        typer.echo(
            f"  PRNU c       = {flat['prnu_pct']:5.2f} %  (two-frame Eq. 32 "
            f"covariance, DSNU-subtracted)"
        )
    if flat["prnu_nf_pct"] is not None:
        typer.echo(
            f"  PRNU c (nf)  = {flat['prnu_nf_pct']:5.2f} %  (upper bound, "
            f"includes diffuser texture)"
        )
    typer.secho(
        "\n  per-point gain check (K12 = 16*S/dV, S = mu_y - mu_y,dark, "
        "dV = sig_y^2 - sig_y,dark^2, two-frame; rows with dV <= 0 omitted):",
        bold=True,
        fg=typer.colors.CYAN,
    )
    bias = dark["bias_dn"]
    for i, (exp_s, s) in enumerate(rows):
        if s["mean"] < CLIP_DN and flat["dV"][i] > 0:
            typer.echo(
                f"    {exp_s * 1000:7.2f} ms  S={s['mean'] - bias:9.1f}  "
                f"K12={16 * (s['mean'] - bias) / flat['dV'][i]:6.2f} e-/DN12"
            )
    # linearity vs exposure
    typer.secho(
        "\n=== LINEARITY (mean vs exposure, bias-subtracted) ===",
        bold=True,
        fg=typer.colors.CYAN,
    )
    bias = dark["bias_dn"]
    for i in range(1, len(rows)):
        e0, s0 = rows[i - 1][0], rows[i - 1][1]["mean"] - bias
        e1, s1 = rows[i][0], rows[i][1]["mean"] - bias
        if s0 > 0 and s1 > 0:
            typer.echo(
                f"  {e0 * 1000:7.2f}->{e1 * 1000:7.2f} ms  ratio {s1 / s0:.3f} "
                f"(ideal {e1 / e0:.3f})"
            )
    return flat


def dsnu1288_core(d_img, tvar, n, cfa=False):
    """Eq. 66 DSNU1288 (DN16): sqrt of highpass dark variance minus the Eq. 36 temporal residual (pure).

    d_img is the frame-averaged dark ROI image (Eq. 33), tvar its temporal
    variance, n the frame count; the highpass is the Section 8.1 filter.
    With cfa=True the four Bayer sub-lattices are processed separately (the
    CFA period survives the highpass and would inflate the estimate) and the
    phase variances are averaged before the sqrt.
    """
    parts = _cfa_phases(d_img) if cfa else [d_img]
    s2 = np.mean([max(float(highpass(p).var()) - tvar / n, 0.0) for p in parts])
    return float(np.sqrt(s2))


def emva_extras_core(
    dk,
    flat,
    d_img,
    f_img,
    dark_tvar_short,
    dark_L,
    flat_tvar_50,
    flat_L,
    cfa=False,
):
    """Saturation/DR/PRNU1288/DSNU1288 numbers (pure, no printing).

    d_img/f_img are the ROI frame-averaged images (Eq. 33) of the
    shortest-exposure dark and the ~50%-saturation flat; dark_tvar_short and
    flat_tvar_50 their temporal variances; dark_L/flat_L the frame counts
    for the Eq. 36 temporal-residual removal. Returns None when no point
    passes the saturation test.
    With cfa=True the Bayer sub-lattices are pooled in the variance domain
    (see dsnu1288_core) and per-phase values are returned alongside.
    """
    ok, fallback = usable_flat_points(flat["rows"])
    if not ok:
        return None
    mu_sat = flat["mu_sat_dn"]  # same saturation point that bounds the PTC fit
    k12 = flat["K12"]
    bias = dk["bias_dn"]
    nsat = (mu_sat - bias) * k12 / 16.0

    sig = flat.get("sigma_r_e_q")
    if sig is not None:
        # Eq. 27: sqrt(sigma_d^2 + sigma_q^2/K^2 + 1/4) + 1/2 -- the
        # quantization variance is added back on the Eq. 53-corrected read
        # noise (numerically equal to using the measured dark sigma)
        q2_e = (QUANT_STEP_DN16**2 / 12.0) * (flat["K12"] / 16.0) ** 2
        sig2 = sig**2 + q2_e
    else:
        sig2 = flat["sigma_r_e"] ** 2  # measured sigma already contains sigma_q^2
    mu_min = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * sig2))  # Eq. 27 (SNR = 1 point)
    dr = nsat / mu_min  # Eq. 28

    d_parts = _cfa_phases(d_img) if cfa else [d_img]
    f_parts = _cfa_phases(f_img) if cfa else [f_img]
    s2_dark_parts = [
        max(float(highpass(p).var()) - dark_tvar_short / dark_L, 0.0) for p in d_parts
    ]
    s2_50_parts = [
        max(
            float(highpass(f_p - d_p).var())
            - flat_tvar_50 / flat_L
            - dark_tvar_short / dark_L,
            0.0,
        )
        for f_p, d_p in zip(f_parts, d_parts)
    ]
    s2_dark = float(np.mean(s2_dark_parts))
    s2_50 = float(np.mean(s2_50_parts))
    prnu1288 = (
        100.0 * np.sqrt(max(s2_50 - s2_dark, 0.0)) / (f_img.mean() - d_img.mean())
    )
    out = {
        "mu_sat_dn": mu_sat,
        "nsat_e": nsat,
        "mu_min_e": mu_min,
        "dr": dr,
        "dr_db": float(20 * np.log10(dr)) if dr > 0 else float("nan"),
        "prnu1288_pct": prnu1288,
        "dsnu1288_dn": float(np.sqrt(s2_dark)),
        "sat_fallback": fallback,
    }
    if cfa:
        # per-channel view (each phase normalized by its own signal; the
        # combined headline number uses the whole-ROI signal)
        out["prnu_phase_pct"] = [
            100.0 * np.sqrt(max(s50 - sd, 0.0)) / (f_p.mean() - d_p.mean())
            for f_p, d_p, s50, sd in zip(f_parts, d_parts, s2_50_parts, s2_dark_parts)
        ]
        out["dsnu_phase_dn"] = [float(np.sqrt(v)) for v in s2_dark_parts]
    return out


def emva1288_extras(dark_seq, flat_seq, dk, flat, roi, cfa=False, cfa_labels=None):
    """EMVA R4 Linear extras: saturation (6.6), DR (Eq. 28), PRNU1288/DSNU1288.

    PRNU/DSNU use the Section 8.1 highpass (7x7 + 11x11 box then 3x3 binomial,
    subtracted) on frame-averaged ROI images (Eq. 33), with the Eq. 36 temporal
    residual (sigma_y^2/L) removed and the dark image subtracted from the
    50%-saturation PRNU image (Eq. 67). DSNU1288 is reported in DN16 (Eq. 66
    would give e- via K); the threshold/DR are in electrons via the PTC gain
    (Eq. 66 needs no calibrated photodiode, only photon units do).
    cfa=True pools the Bayer sub-lattices per emva_extras_core and prints the
    per-channel values; cfa_labels overrides the channel names.
    """
    typer.secho(
        "\n=== EMVA 1288 R4 Linear (saturation, nonuniformity, DR) ===",
        bold=True,
        fg=typer.colors.CYAN,
    )
    rows = flat["rows"]
    ok, fallback = usable_flat_points(rows)
    if fallback and ok:
        typer.secho(
            "  ! no point under the clip-fraction limit -- using max unclipped",
            fg=typer.colors.YELLOW,
        )
    if not ok:
        typer.secho("  ! no usable points", fg=typer.colors.YELLOW)
        return

    d_stack = min(dark_seq, key=lambda es: es[0])[1]
    dark_tvar = dk["rows"][0][1]["tf_tvar"]
    mu_sat_max = max(s["mean"] for _, s in ok)
    target = 0.5 * (mu_sat_max + dk["bias_dn"])
    idx = min(range(len(rows)), key=lambda i: abs(rows[i][1]["mean"] - target))
    f_exp, f_stack = flat_seq[idx]
    d_img = d_stack[:, roi[0], roi[1]].astype(np.float64).mean(axis=0)
    f_img = f_stack[:, roi[0], roi[1]].astype(np.float64).mean(axis=0)

    x = emva_extras_core(
        dk,
        flat,
        d_img=d_img,
        f_img=f_img,
        dark_tvar_short=dark_tvar,
        dark_L=d_stack.shape[0],
        flat_tvar_50=rows[idx][1]["tf_tvar"],
        flat_L=f_stack.shape[0],
        cfa=cfa,
    )
    if x is None:
        return
    typer.echo(
        f"  mu_y.sat    = {x['mu_sat_dn']:.0f} DN16  "
        f"(<= {SAT_CLIP_FRAC * 100:.1f}% of pixels at {SAT_MAX_DN:.0f})"
    )
    typer.echo(
        f"  Nsat (EMVA) = {x['nsat_e']:.0f} e-  (bias-subtracted; "
        f"vs K*4094 = {flat['Nsat']:.0f} e-)"
    )
    typer.echo(
        f"  mu_e.min    = {x['mu_min_e']:.1f} e- (SNR = 1);  "
        f"DR = {x['dr']:.0f} (Eq. 28)  ({x['dr_db']:.1f} dB)"
    )
    typer.echo(
        f"  PRNU1288    = {x['prnu1288_pct']:.2f} %  "
        f"(highpass, {f_exp * 1000:.1f} ms ~ 50% sat)"
    )
    typer.echo(f"  DSNU1288    = {x['dsnu1288_dn']:.2f} DN16 (highpass dark image)")
    if cfa:
        labels = cfa_labels or _CFA_LABELS
        typer.echo(
            "  CFA phases ("
            + ",".join(labels)
            + "): PRNU "
            + "/".join(f"{v:.2f}" for v in x["prnu_phase_pct"])
            + " %, DSNU "
            + "/".join(f"{v:.2f}" for v in x["dsnu_phase_dn"])
            + " DN16"
        )
    return x


def snr_temporal_model(signal_e, sigma_d_e, k12):
    """EMVA R4 Linear Eq. 21: temporal-SNR model curve (pure).

    sigma_d_e is the Eq. 53 quantization-corrected read noise; the
    quantization variance sigma_q^2/K^2 is added explicitly so the model
    matches the standard's equation form (numerically equal to using the
    measured dark sigma, which already contains quantization).
    """
    q2_e = (QUANT_STEP_DN16**2 / 12.0) * (k12 / 16.0) ** 2
    return signal_e / np.sqrt(sigma_d_e**2 + q2_e + signal_e)


def snr_total_model(extras, signal_e, sigma_d_e, k12):
    """EMVA R4 Linear Eq. 69: total-SNR model curve vs mean signal (e-, pure).

    Adds the spatial nonuniformities (DSNU1288 in e- = DN16*K12/16, Eq. 66,
    and PRNU1288 as a fraction, Eq. 67) and the quantization noise
    sigma_q^2/K^2 to the temporal terms. sigma_d_e is the dark temporal
    noise in e- (the Eq. 53 quant-corrected read noise, so quantization is
    not double-counted); signal_e is the mean photo signal in e-.
    """
    dsnu_e = extras["dsnu1288_dn"] * k12 / 16.0
    prnu = extras["prnu1288_pct"] / 100.0
    q2_e = (QUANT_STEP_DN16**2 / 12.0) * (k12 / 16.0) ** 2
    var = sigma_d_e**2 + dsnu_e**2 + q2_e + signal_e + (prnu * signal_e) ** 2
    return signal_e / np.sqrt(var)


def snr_total_measured(extras, means, bias_dn, tvars, k12):
    """Measured total SNR (e-, pure) per exposure step.

    Two-frame temporal variance (tf_tvar, the primary estimator) plus the
    spatial nonuniformity variance s_y^2 = DSNU1288^2 + PRNU1288^2*(mu -
    mu.dark)^2 (Eq. 68), converted to e- with the gain; tf_tvar already
    includes the quantization noise.
    """
    dsnu2 = extras["dsnu1288_dn"] ** 2
    prnu = extras["prnu1288_pct"] / 100.0
    s2 = dsnu2 + (prnu * (means - bias_dn)) ** 2
    signal_e = (means - bias_dn) * k12 / 16.0
    return signal_e / np.sqrt((tvars + s2) * (k12 / 16.0) ** 2)


def run(data_dir, roi=None, bands=5):
    resolved = resolve_data_dir(data_dir)
    if resolved is None:
        return
    cam_dir, layout = resolved

    if layout == "specim":
        from .band_analyze import run as run_bands  # local: avoid import cycle

        run_bands(cam_dir, roi or DEFAULT_ROI_SPECIM, bands)
        return

    dark_seq = load_sequence(cam_dir, "dark")
    roi_label = None
    if roi is None:
        # Default ROI: central fraction of the recorded frames. Derived from
        # the stacks themselves, not metadata.json -- append-only dirs can
        # mix geometries (acA1920 pre/post Aug-2026).
        ref = dark_seq or load_sequence(cam_dir, "flat")
        if ref:
            height, width = ref[0][1].shape[1:3]
            roi = central_roi(width, height)
            roi_label = f"central {DEFAULT_ROI_FRAC:.0%}"
    else:
        roi_label = f"rows {roi[0]}:{roi[1]}, cols {roi[2]}:{roi[3]}"
    roi_slices = (
        tuple(slice(a, b) for a, b in zip(roi[0::2], roi[1::2])) if roi else None
    )

    header = f"Analyzing {cam_dir.resolve()}"
    if roi_label:
        header += f"  (ROI {roi_label})"
    typer.secho(header, bold=True, fg=typer.colors.CYAN)
    dk = analyze_dark(dark_seq, roi_slices)
    meta = load_camera_meta(cam_dir)
    camera = load_camera_info(cam_dir)
    # Raw-Bayer datasets: spatial metrics pool the four CFA sub-lattices.
    # The format comes from a scan of all metadata entries so it stays
    # consistent even when early entries predate pixel_format recording.
    bayer_format = _dataset_cfa_format(cam_dir)
    cfa = bool(bayer_format)
    cfa_labels = _cfa_labels(bayer_format)
    if dk:
        out_dir = Path("outputs") / (camera_dir_name(meta) if meta else cam_dir.name)
        typer.echo(f"\n  bias floor (shortest exp): {dk['bias_dn']:.2f} DN16")
        typer.echo(
            f"  dark current (mean, Eq. 29): {dk['dark_current_dn_per_s']:.4f} DN16/s"
        )
        typer.echo(
            f"  dark current (var, Eq. 30, two-frame): "
            f"{dk['dark_current_var_dn2_per_s']:.2f} DN16^2/s "
            f"(slope inflated by DCNU/drift vs the mean)"
        )
        typer.echo(
            f"  read noise (Eq. 30 fit offset): {dk['sigma_r_dn']:.2f} DN16 "
            f"(short-exposure median {dk['sigma_r_median_dn']:.2f})"
        )
        save_dark_mean_plot(dk["rows"], out_dir, roi_label, camera)
        d_stack = min(dark_seq, key=lambda es: es[0])[1]
        d_img = d_stack[:, roi_slices[0], roi_slices[1]].astype(np.float64).mean(axis=0)
        dsnu = dsnu1288_core(
            d_img, dk["rows"][0][1]["tf_tvar"], d_stack.shape[0], cfa=cfa
        )
        save_dark_variance_plot(dk["rows"], out_dir, roi_label, camera, dsnu=dsnu)
        flat_seq = load_sequence(cam_dir, "flat")
        if flat_seq and dark_seq:
            fh, fw = flat_seq[0][1].shape[1:3]
            dh, dw = dark_seq[0][1].shape[1:3]
            if (fh, fw) != (dh, dw):
                typer.secho(
                    f"  ! dark ({dh}x{dw}) and flat ({fh}x{fw}) frame sizes "
                    "differ (append-only dir with mixed geometries?) -- the "
                    "ROI may cover different physical regions",
                    fg=typer.colors.YELLOW,
                )
        flat = analyze_flats(flat_seq, dk, roi_slices)
        if flat:
            flat["extras"] = emva1288_extras(
                dark_seq,
                flat_seq,
                dk,
                flat,
                roi_slices,
                cfa=cfa,
                cfa_labels=cfa_labels,
            )
            save_linearity_plot(
                flat["rows"], flat["bias_dn"], out_dir, CLIP_DN, roi_label, camera
            )
            save_ptc_plot(flat, dk, out_dir, CLIP_DN, roi_label, camera)
            save_snr_plot(flat, out_dir, CLIP_DN, roi_label, camera)

        if bayer_format:
            # additive per-channel pass: every Bayer sub-lattice through the
            # same fits, one curve per channel in the *_bands plot variants
            from .cfa_analyze import run as cfa_run  # local: avoid import cycle

            cfa_run(
                cam_dir,
                dark_seq,
                flat_seq,
                bayer_format,
                roi,
                camera=camera,
                roi_label=roi_label,
            )
