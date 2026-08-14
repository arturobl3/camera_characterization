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
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .io_utils import camera_dir_name, stem_for
from .plots import save_dark_plot, save_linearity_plot, save_ptc_plot, save_snr_plot

CLIP_DN = 65400  # anything >= this is clipped (IMX174: 4094<<4 = 65504)
DARK_CURRENT_FLAT_MAX_EXP = (
    0.01  # exposures <= 10 ms: dark-current shot noise negligible
)

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
    """Camera dir (containing dark/ and/or flat/) under data_dir.

    data_dir itself is returned when it already contains dark/ or flat/ (legacy
    layout). Otherwise subdirs with metadata.json are scanned: exactly one is
    used; several or none produce an error listing the choices.
    """
    d = Path(data_dir)
    if not d.exists():
        print(f"  ! data dir not found: {d}")
        return None
    if (d / "dark").exists() or (d / "flat").exists():
        return d
    candidates = [
        sub
        for sub in sorted(d.iterdir())
        if sub.is_dir()
        and (
            (sub / "dark" / "metadata.json").exists()
            or (sub / "flat" / "metadata.json").exists()
        )
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        print(
            f"  ! no camera data under {d} "
            f"(expected <root>/<vendor>_<model>_(<sensor>)/dark|flat)"
        )
        return None
    print("  ! multiple camera dirs found -- pass one explicitly with --data:")
    for c in candidates:
        print(f"      {c}")
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


def two_frame_stats(stack, roi):
    """EMVA 1288 Release 4 two-frame estimates (Eq. 18 + Eq. 32).

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

    Temporal variance uses ddof=1 (EMVA Eq. 44: 1/(L-1)) -- the population
    (ddof=0) estimator is biased low by (N-1)/N = 5% at N=20, which the
    two-frame cross-check exposes (K, sigma_r, Nsat all shift). Spatial
    variance keeps ddof=0 (EMVA spatial sums use 1/(NM); bias < 0.01% at
    10^4 pixels).
    """
    r = stack[:, roi[0], roi[1]].astype(np.float64)
    tf_t, tf_s, tf_ts, tf_ss = two_frame_stats(stack, roi)
    return {
        "mean": r.mean(),
        "tvar": r.var(axis=0, ddof=1).mean(),  # temporal variance, avg over pixels
        "svar": r.var(axis=1).mean(),  # spatial variance, avg over frames
        "tf_tvar": tf_t,               # two-frame temporal variance (Eq. 18)
        "tf_svar": tf_s,               # two-frame spatial variance (Eq. 32)
        "tf_tvar_scatter": tf_ts,      # pair-to-pair std of tf_tvar
        "tf_svar_scatter": tf_ss,      # pair-to-pair std of tf_svar
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

    t = np.array([e for e, _ in seq])
    mu = np.array([roi_stats(st, ROI)["mean"] for _, st in seq])
    slope, intercept = np.polyfit(t, mu, 1)
    bias = mu[0]

    # read noise: temporal sigma where tvar is flat (read-noise dominated)
    tvars = np.array(
        [roi_stats(st, ROI)["tvar"] for e, st in seq if e <= DARK_CURRENT_FLAT_MAX_EXP]
    )
    tf_tvars = np.array(
        [roi_stats(st, ROI)["tf_tvar"] for e, st in seq if e <= DARK_CURRENT_FLAT_MAX_EXP]
    )
    sigma_r = float(np.sqrt(np.median(tvars)))
    sigma_r_tf = float(np.sqrt(np.median(tf_tvars)))
    return {
        "bias_dn": bias,
        "dark_current_dn_per_s": slope,
        "sigma_r_dn": sigma_r,
        "sigma_r_tf_dn": sigma_r_tf,
        "bias_fit_dn": intercept,
        "rows": rows,
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

    usable = [(s["mean"], s["tvar"]) for _, s in rows if s["mean"] < CLIP_DN]
    if len(usable) < 3:
        print("  ! too few unclipped points -- check illumination level")
        return None
    S = np.array([u[0] for u in usable])
    V = np.array([u[1] for u in usable])
    K_fit, b = np.polyfit(S, V, 1)
    V_hat = np.polyval([K_fit, b], S)
    ptc_r2 = 1.0 - np.sum((V - V_hat) ** 2) / np.sum((V - V.mean()) ** 2)
    K12 = 16.0 / K_fit  # DN16 -> DN12: K12 = 16/(V16/S16)
    sigma_r_e = dark["sigma_r_dn"] * K12 / 16.0
    sigma_r_tf_e = dark["sigma_r_tf_dn"] * K12 / 16.0

    # two-frame PTC cross-check: same fit on the Eq. 18 temporal variance
    V_tf = np.array([s["tf_tvar"] for _, s in rows if s["mean"] < CLIP_DN])
    K_fit_tf, b_tf = np.polyfit(S, V_tf, 1)
    K12_tf = 16.0 / K_fit_tf

    prnu = []
    prnu_tf = []
    dark_tf_svar = (
        np.median(
            [
                s["tf_svar"]
                for e, s in dark["rows"]
                if e <= DARK_CURRENT_FLAT_MAX_EXP
            ]
        )
        if dark
        else 0.0
    )
    for _, s in rows:
        if 5000 < s["mean"] < 64000:
            prnu.append(np.sqrt(max(s["svar"] - s["tvar"], 0)) / s["mean"])
            prnu_tf.append(
                np.sqrt(max(s["tf_svar"] - dark_tf_svar, 0)) / s["mean"]
            )

    print("\n=== PTC FIT (temporal variance) ===")
    print(f"  K (gain)     = {K12:6.2f} e-/DN12   (V/S slope {K_fit:.3f})")
    print(
        f"  K (two-frame)= {K12_tf:6.2f} e-/DN12   (slope {K_fit_tf:.3f}, "
        f"Eq. 18; {100 * (K12_tf - K12) / K12:+.1f}% vs N-frame)"
    )
    print(
        f"  sigma_r      = {sigma_r_e:6.2f} e-  ({dark['sigma_r_dn']:.2f} DN16, "
        f"{dark['sigma_r_dn'] / 16:.2f} DN12)"
    )
    print(
        f"  sigma_r (tf) = {sigma_r_tf_e:6.2f} e-  "
        f"({dark['sigma_r_tf_dn']:.2f} DN16)"
    )
    print(f"  Nsat         = {K12 * 4094:7.0f} e- (at 12-bit clip 4094)")
    if prnu:
        print(
            f"  PRNU c       = {np.mean(prnu) * 100:5.2f} %  (upper bound, "
            f"includes diffuser texture)"
        )
    if prnu_tf:
        print(
            f"  PRNU c (tf)  = {np.mean(prnu_tf) * 100:5.2f} %  (Eq. 32 "
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
    return {
        "K12": K12,
        "sigma_r_e": sigma_r_e,
        "Nsat": K12 * 4094,
        "rows": rows,
        "bias_dn": dark["bias_dn"],
        "ptc_slope": K_fit,
        "ptc_intercept": b,
        "ptc_r2": ptc_r2,
    }


def run(data_dir, roi=None):
    global ROI
    ROI = tuple(slice(a, b) for a, b in zip(roi[0::2], roi[1::2]))

    cam_dir = resolve_data_dir(data_dir)
    if cam_dir is None:
        return
    print(
        f"Analyzing {cam_dir.resolve()}  (ROI rows {roi[0]}:{roi[1]}, "
        f"cols {roi[2]}:{roi[3]})"
    )
    dk = analyze_dark(load_sequence(cam_dir, "dark"))
    if dk:
        camera = load_camera_info(cam_dir)
        meta = load_camera_meta(cam_dir)
        out_dir = Path("outputs") / (camera_dir_name(meta) if meta else cam_dir.name)
        print(f"\n  bias floor (shortest exp): {dk['bias_dn']:.2f} DN16")
        print(
            f"  dark current: {dk['dark_current_dn_per_s']:.4f} DN16/s  "
            f"(bias fit {dk['bias_fit_dn']:.2f})"
        )
        print(f"  read noise (temporal sigma, dark): {dk['sigma_r_dn']:.2f} DN16")
        save_dark_plot(dk["rows"], out_dir, roi, camera)
        flat = analyze_flats(load_sequence(cam_dir, "flat"), dk)
        if flat:
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
        default="600:800:850:1050",
        help="ROI as r0:r1:c0:c1 (default 600:800:850:1050)",
    )
    args = p.parse_args(argv)
    roi = [int(x) for x in args.roi.split(":")]
    if len(roi) != 4:
        p.error("--roi must be r0:r1:c0:c1")
    run(args.data, roi)


if __name__ == "__main__":
    main()
