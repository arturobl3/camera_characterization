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

import argparse
import json
from pathlib import Path

import numpy as np

from .io_utils import camera_dir_name, stem_for
from .plots import (
    save_dark_plot,
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
SAT_CLIP_FRAC = 0.002  # EMVA R4 Linear 6.6: saturation = <= 0.2% pixels at max
QUANT_STEP_DN16 = 16.0  # DN16 per 12-bit LSB (DN12 << 4 storage)

DEFAULT_ROI = (600, 800, 850, 1050)  # playerone Apollo-M central 200x200
DEFAULT_ROI_SPECIM = (156, 356, 156, 356)  # SPECIM IQ 512x512 central 200x200

_VENDOR_NAMES = {"playerone": "Player One"}


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
        print(f"  ! data dir not found: {d}")
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
        print(
            f"  ! no camera data under {d} "
            f"(expected <root>/<vendor>_<model>_(<sensor>)/dark|flat "
            f"or a SPECIM IQ 'dark frames'/'flat-field frames' layout)"
        )
        return None
    print("  ! multiple camera dirs found -- pass one explicitly with --data:")
    for c, layout in candidates:
        print(f"      {c}  ({layout})")
    return None


def load_sequence(data_dir, seq_type):
    """Return [(exposure_s, stack), ...] sorted, skipping missing files.

    Metadata entries are deduped by (exposure_s, gain) -- append-only
    metadata.json accumulates entries across runs and can hold duplicates.
    """
    d = Path(data_dir) / seq_type
    meta_path = d / "metadata.json"
    if not meta_path.exists():
        print(f"  ! no metadata at {meta_path} -- nothing to analyze")
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
            print(f"  ! missing {p.name}, skipping")
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
    return {
        "mean": r.mean(),
        "tvar": r.var(axis=0, ddof=1).mean(),  # temporal variance, avg over pixels
        "svar": r.var(axis=1).mean(),  # spatial variance, avg over frames
        "tf_tvar": tf_t,  # two-frame temporal variance (Eq. 18)
        "tf_svar": tf_s,  # two-frame spatial variance (Eq. 32)
        "tf_tvar_scatter": tf_ts,  # pair-to-pair std of tf_tvar
        "tf_svar_scatter": tf_ss,  # pair-to-pair std of tf_svar
        "sat_frac": float((r >= SAT_MAX_DN).mean()),  # EMVA 6.6 saturation test
    }


def fit_dark_stats(rows):
    """Eq. 29/30 dark fits from roi_stats rows [(exposure_s, stats)] (pure).

    bias is the measured mean of the shortest exposure (not the fit
    intercept); the Eq. 30 variance-fit slope is the second (variance-based)
    dark-current estimate.
    """
    t = np.array([e for e, _ in rows])
    mu = np.array([s["mean"] for _, s in rows])
    tvars = np.array([s["tvar"] for _, s in rows])
    tf_tvars = np.array([s["tf_tvar"] for _, s in rows])

    slope, intercept = np.polyfit(t, mu, 1)  # Eq. 29: mu_d = mu_d0 + mu_I.y t
    var_slope, var_offset = np.polyfit(t, tvars, 1)  # Eq. 30
    _, tf_var_offset = np.polyfit(t, tf_tvars, 1)
    short = t <= DARK_CURRENT_FLAT_MAX_EXP
    if short.any():
        sigma_r_med = float(np.sqrt(np.median(tvars[short])))
    else:
        sigma_r_med = float(np.sqrt(np.median(tvars)))  # no short exposure available
    return {
        "bias_dn": float(mu[0]),
        "dark_current_dn_per_s": float(slope),
        "dark_current_var_dn2_per_s": float(var_slope),
        "sigma_r_dn": float(np.sqrt(max(var_offset, 0.0))),
        "sigma_r_tf_dn": float(np.sqrt(max(tf_var_offset, 0.0))),
        "sigma_r_median_dn": sigma_r_med,
        "bias_fit_dn": float(intercept),
    }


def analyze_dark(seq):
    """Bias floor, dark current, read noise from dark frames."""
    print("=== DARK (tvar/svar = N-frame; tf = two-frame cross-check) ===")
    if not seq:
        return None
    rows = []
    for exp_s, st in seq:
        s = roi_stats(st, ROI)
        rows.append((exp_s, s))
        print(
            f"  {exp_s * 1000:8.1f} ms  mean {s['mean']:8.2f}  "
            f"tvar {s['tvar']:9.2f}  tf {s['tf_tvar']:9.2f}  "
            f"svar {s['svar']:8.2f}  tfs {s['tf_svar']:8.2f}"
        )

    out = fit_dark_stats(rows)
    out["rows"] = rows
    return out


def fit_flat_stats(rows, dark):
    """PTC fit and derived quantities from roi_stats rows (pure).

    Fits V = K_fit*S + b on (mean, tvar) of the usable rows and derives
    K12 = 16/K_fit, electron-unit read noises (Eq. 53 quantization-corrected
    variants included) and the quick upper-bound PRNU estimates. Returns
    None when fewer than 3 unclipped points are available.

    Usable = mean < CLIP_DN *and* sat_frac <= SAT_CLIP_FRAC (EMVA 6.6):
    heavily-pinned points can sit below the mean threshold while their
    variance has collapsed (pinned pixels don't fluctuate), which would
    bend the PTC fit.
    """
    usable = [
        (s["mean"], s["tvar"])
        for _, s in rows
        if s["mean"] < CLIP_DN and s["sat_frac"] <= SAT_CLIP_FRAC
    ]
    if len(usable) < 3:
        return None
    S = np.array([u[0] for u in usable])
    V = np.array([u[1] for u in usable])
    K_fit, b = np.polyfit(S, V, 1)
    V_hat = np.polyval([K_fit, b], S)
    ptc_r2 = 1.0 - np.sum((V - V_hat) ** 2) / np.sum((V - V.mean()) ** 2)
    K12 = 16.0 / K_fit  # DN16 -> DN12: K12 = 16/(V16/S16)
    sigma_r_e = dark["sigma_r_dn"] * K12 / 16.0
    sigma_r_tf_e = dark["sigma_r_tf_dn"] * K12 / 16.0

    # Eq. 53: subtract the quantization variance (step^2/12) for the
    # physical-unit dark noise -- 21.3 DN16^2 here, ~14% of the dark variance
    sig_q2 = QUANT_STEP_DN16**2 / 12.0
    sigma_r_e_q = np.sqrt(max(dark["sigma_r_dn"] ** 2 - sig_q2, 0.0)) * K12 / 16.0
    sigma_r_tf_e_q = np.sqrt(max(dark["sigma_r_tf_dn"] ** 2 - sig_q2, 0.0)) * K12 / 16.0

    # two-frame PTC cross-check: same fit on the Eq. 18 temporal variance
    V_tf = np.array(
        [
            s["tf_tvar"]
            for _, s in rows
            if s["mean"] < CLIP_DN and s["sat_frac"] <= SAT_CLIP_FRAC
        ]
    )
    K_fit_tf, b_tf = np.polyfit(S, V_tf, 1)
    K12_tf = 16.0 / K_fit_tf

    prnu = []
    prnu_tf = []
    dark_tf_svar = (
        np.median(
            [s["tf_svar"] for e, s in dark["rows"] if e <= DARK_CURRENT_FLAT_MAX_EXP]
        )
        if dark
        else 0.0
    )
    for _, s in rows:
        if 5000 < s["mean"] < 64000:
            prnu.append(np.sqrt(max(s["svar"] - s["tvar"], 0)) / s["mean"])
            prnu_tf.append(np.sqrt(max(s["tf_svar"] - dark_tf_svar, 0)) / s["mean"])

    return {
        "K12": K12,
        "K12_tf": K12_tf,
        "sigma_r_e": sigma_r_e,
        "sigma_r_e_q": sigma_r_e_q,
        "sigma_r_tf_e": sigma_r_tf_e,
        "sigma_r_tf_e_q": sigma_r_tf_e_q,
        "Nsat": K12 * 4094,
        "bias_dn": dark["bias_dn"],
        "ptc_slope": K_fit,
        "ptc_intercept": b,
        "ptc_slope_tf": K_fit_tf,
        "ptc_intercept_tf": b_tf,
        "ptc_r2": ptc_r2,
        "prnu_pct": float(np.mean(prnu) * 100) if prnu else None,
        "prnu_tf_pct": float(np.mean(prnu_tf) * 100) if prnu_tf else None,
    }


def analyze_flats(seq, dark):
    """K, Nsat, PRNU from the flat sweep (temporal variance)."""
    print("\n=== FLAT (tvar/svar = N-frame; tf = two-frame cross-check) ===")
    if not seq:
        return None
    rows = []
    for exp_s, st in seq:
        s = roi_stats(st, ROI)
        rows.append((exp_s, s))
        print(
            f"  {exp_s * 1000:8.1f} ms  mean {s['mean']:10.2f}  "
            f"tvar {s['tvar']:12.2f}  tf {s['tf_tvar']:12.2f}  "
            f"svar {s['svar']:12.2f}  tfs {s['tf_svar']:12.2f}"
        )

    flat = fit_flat_stats(rows, dark)
    if flat is None:
        print("  ! too few unclipped points -- check illumination level")
        return None
    flat["rows"] = rows

    K12 = flat["K12"]
    K12_tf = flat["K12_tf"]
    sigma_r_e = flat["sigma_r_e"]
    sigma_r_tf_e = flat["sigma_r_tf_e"]
    sigma_r_e_q = flat["sigma_r_e_q"]
    sigma_r_tf_e_q = flat["sigma_r_tf_e_q"]
    print("\n=== PTC FIT (temporal variance) ===")
    print(f"  K (gain)     = {K12:6.2f} e-/DN12   (V/S slope {flat['ptc_slope']:.3f})")
    print(
        f"  K (two-frame)= {K12_tf:6.2f} e-/DN12   "
        f"(slope {flat['ptc_slope_tf']:.3f}, "
        f"Eq. 18; {100 * (K12_tf - K12) / K12:+.1f}% vs N-frame)"
    )
    print(
        f"  sigma_r      = {sigma_r_e:6.2f} e-  ({dark['sigma_r_dn']:.2f} DN16; "
        f"Eq. 53 quant-corrected {sigma_r_e_q:.2f} e-)"
    )
    print(
        f"  sigma_r (tf) = {sigma_r_tf_e:6.2f} e-  ({dark['sigma_r_tf_dn']:.2f} DN16; "
        f"corrected {sigma_r_tf_e_q:.2f} e-)"
    )
    print(f"  Nsat         = {flat['Nsat']:7.0f} e- (at 12-bit clip 4094)")
    if flat["prnu_pct"] is not None:
        print(
            f"  PRNU c       = {flat['prnu_pct']:5.2f} %  (upper bound, "
            f"includes diffuser texture)"
        )
    if flat["prnu_tf_pct"] is not None:
        print(
            f"  PRNU c (tf)  = {flat['prnu_tf_pct']:5.2f} %  (Eq. 32 "
            f"covariance, DSNU-subtracted)"
        )
    print("\n  per-point gain check (K12 = 16*S/tvar):")
    for exp_s, s in rows:
        if s["mean"] < CLIP_DN:
            print(
                f"    {exp_s * 1000:7.2f} ms  S={s['mean']:9.1f}  "
                f"K12={16 * s['mean'] / s['tvar']:6.2f} e-/DN12"
            )
    # linearity vs exposure
    print("\n=== LINEARITY (mean vs exposure, bias-subtracted) ===")
    bias = dark["bias_dn"]
    for i in range(1, len(rows)):
        e0, s0 = rows[i - 1][0], rows[i - 1][1]["mean"] - bias
        e1, s1 = rows[i][0], rows[i][1]["mean"] - bias
        if s0 > 0 and s1 > 0:
            print(
                f"  {e0 * 1000:7.2f}->{e1 * 1000:7.2f} ms  ratio {s1 / s0:.3f} "
                f"(ideal {e1 / e0:.3f})"
            )
    return flat


def emva_extras_core(
    dk, flat, d_img, f_img, dark_tvar_short, dark_L, flat_tvar_50, flat_L
):
    """Saturation/DR/PRNU1288/DSNU1288 numbers (pure, no printing).

    d_img/f_img are the ROI frame-averaged images (Eq. 33) of the
    shortest-exposure dark and the ~50%-saturation flat; dark_tvar_short and
    flat_tvar_50 their temporal variances; dark_L/flat_L the frame counts
    for the Eq. 36 temporal-residual removal. Returns None when no point
    passes the saturation test.
    """
    rows = flat["rows"]
    ok = [rs for rs in rows if rs[1]["sat_frac"] <= SAT_CLIP_FRAC]
    fallback = not ok
    if not ok:
        ok = [rs for rs in rows if rs[1]["mean"] < CLIP_DN]
    if not ok:
        return None
    mu_sat = max(s["mean"] for _, s in ok)
    k12 = flat["K12"]
    bias = dk["bias_dn"]
    nsat = (mu_sat - bias) * k12 / 16.0

    sig = flat.get("sigma_r_e_q", flat["sigma_r_e"])
    mu_min = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * sig**2))  # SNR = 1 point
    dr = nsat / mu_min  # Eq. 28

    s2_dark = max(float(highpass(d_img).var()) - dark_tvar_short / dark_L, 0.0)
    s2_50 = max(
        float(highpass(f_img - d_img).var())
        - flat_tvar_50 / flat_L
        - dark_tvar_short / dark_L,
        0.0,
    )
    prnu1288 = (
        100.0 * np.sqrt(max(s2_50 - s2_dark, 0.0)) / (f_img.mean() - d_img.mean())
    )
    return {
        "mu_sat_dn": mu_sat,
        "nsat_e": nsat,
        "mu_min_e": mu_min,
        "dr": dr,
        "dr_db": float(20 * np.log10(dr)) if dr > 0 else float("nan"),
        "prnu1288_pct": prnu1288,
        "dsnu1288_dn": float(np.sqrt(s2_dark)),
        "sat_fallback": fallback,
    }


def emva1288_extras(dark_seq, flat_seq, dk, flat):
    """EMVA R4 Linear extras: saturation (6.6), DR (Eq. 28), PRNU1288/DSNU1288.

    PRNU/DSNU use the Section 8.1 highpass (7x7 + 11x11 box then 3x3 binomial,
    subtracted) on frame-averaged ROI images (Eq. 33), with the Eq. 36 temporal
    residual (sigma_y^2/L) removed and the dark image subtracted from the
    50%-saturation PRNU image (Eq. 67). DSNU1288 is reported in DN16 (Eq. 66
    would give e- via K); the threshold/DR are in electrons via the PTC gain
    (Eq. 66 needs no calibrated photodiode, only photon units do).
    """
    print("\n=== EMVA 1288 R4 Linear (saturation, nonuniformity, DR) ===")
    rows = flat["rows"]
    ok = [rs for rs in rows if rs[1]["sat_frac"] <= SAT_CLIP_FRAC]
    if not ok:
        print("  ! no point under the clip-fraction limit -- using max unclipped")
    if not ok and not [rs for rs in rows if rs[1]["mean"] < CLIP_DN]:
        print("  ! no usable points")
        return

    d_stack = min(dark_seq, key=lambda es: es[0])[1]
    dark_tvar = dk["rows"][0][1]["tvar"]
    mu_sat_max = max(
        s["mean"] for _, s in (ok or [rs for rs in rows if rs[1]["mean"] < CLIP_DN])
    )
    target = 0.5 * (mu_sat_max + dk["bias_dn"])
    idx = min(range(len(rows)), key=lambda i: abs(rows[i][1]["mean"] - target))
    f_exp, f_stack = flat_seq[idx]
    d_img = d_stack[:, ROI[0], ROI[1]].astype(np.float64).mean(axis=0)
    f_img = f_stack[:, ROI[0], ROI[1]].astype(np.float64).mean(axis=0)

    x = emva_extras_core(
        dk,
        flat,
        d_img=d_img,
        f_img=f_img,
        dark_tvar_short=dark_tvar,
        dark_L=d_stack.shape[0],
        flat_tvar_50=rows[idx][1]["tvar"],
        flat_L=f_stack.shape[0],
    )
    if x is None:
        return
    print(
        f"  mu_y.sat    = {x['mu_sat_dn']:.0f} DN16  "
        f"(<= {SAT_CLIP_FRAC * 100:.1f}% of pixels at {SAT_MAX_DN:.0f})"
    )
    print(
        f"  Nsat (EMVA) = {x['nsat_e']:.0f} e-  (bias-subtracted; "
        f"vs K*4094 = {flat['Nsat']:.0f} e-)"
    )
    print(
        f"  mu_e.min    = {x['mu_min_e']:.1f} e- (SNR = 1);  "
        f"DR = {x['dr']:.0f} (Eq. 28)  ({x['dr_db']:.1f} dB)"
    )
    print(
        f"  PRNU1288    = {x['prnu1288_pct']:.2f} %  "
        f"(highpass, {f_exp * 1000:.1f} ms ~ 50% sat)"
    )
    print(f"  DSNU1288    = {x['dsnu1288_dn']:.2f} DN16 (highpass dark image)")


def run(data_dir, roi=None, bands=5):
    resolved = resolve_data_dir(data_dir)
    if resolved is None:
        return
    cam_dir, layout = resolved

    if layout == "specim":
        from .band_analyze import run as run_bands  # local: avoid import cycle

        run_bands(cam_dir, roi or DEFAULT_ROI_SPECIM, bands)
        return

    global ROI
    roi = roi or DEFAULT_ROI
    ROI = tuple(slice(a, b) for a, b in zip(roi[0::2], roi[1::2]))

    print(
        f"Analyzing {cam_dir.resolve()}  (ROI rows {roi[0]}:{roi[1]}, "
        f"cols {roi[2]}:{roi[3]})"
    )
    dark_seq = load_sequence(cam_dir, "dark")
    dk = analyze_dark(dark_seq)
    if dk:
        camera = load_camera_info(cam_dir)
        meta = load_camera_meta(cam_dir)
        out_dir = Path("outputs") / (camera_dir_name(meta) if meta else cam_dir.name)
        print(f"\n  bias floor (shortest exp): {dk['bias_dn']:.2f} DN16")
        print(
            f"  dark current (mean, Eq. 29): {dk['dark_current_dn_per_s']:.4f} DN16/s"
        )
        print(
            f"  dark current (var, Eq. 30): "
            f"{dk['dark_current_var_dn2_per_s']:.2f} DN16^2/s "
            f"(slope inflated by DCNU/drift vs the mean)"
        )
        print(
            f"  read noise (Eq. 30 fit offset): {dk['sigma_r_dn']:.2f} DN16 "
            f"(short-exposure median {dk['sigma_r_median_dn']:.2f})"
        )
        save_dark_plot(dk["rows"], out_dir, roi, camera)
        save_dark_variance_plot(dk["rows"], out_dir, roi, camera)
        flat_seq = load_sequence(cam_dir, "flat")
        flat = analyze_flats(flat_seq, dk)
        if flat:
            emva1288_extras(dark_seq, flat_seq, dk, flat)
            save_linearity_plot(
                flat["rows"], flat["bias_dn"], out_dir, CLIP_DN, roi, camera
            )
            save_ptc_plot(flat, dk, out_dir, CLIP_DN, roi, camera)
            save_snr_plot(flat, out_dir, CLIP_DN, roi, camera)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="camchar analyze", description="Temporal PTC analysis of dark/flat data"
    )
    p.add_argument(
        "--data",
        default="data",
        help="data root or camera dir <root>/<vendor>_<model>_(<sensor>); "
        "a single camera dir under a root is auto-discovered (default: data)",
    )
    p.add_argument(
        "--roi",
        default=None,
        help=f"ROI as r0:r1:c0:c1 (default {':'.join(map(str, DEFAULT_ROI))}; "
        f"SPECIM IQ {':'.join(map(str, DEFAULT_ROI_SPECIM))})",
    )
    p.add_argument(
        "--bands",
        type=int,
        default=5,
        help="number of equispaced bands in per-band (SPECIM IQ) plots; "
        "ignored for monochrome npy data (default 5)",
    )
    args = p.parse_args(argv)
    roi = None
    if args.roi is not None:
        roi = [int(x) for x in args.roi.split(":")]
        if len(roi) != 4:
            p.error("--roi must be r0:r1:c0:c1")
    run(args.data, roi, args.bands)


if __name__ == "__main__":
    main()
