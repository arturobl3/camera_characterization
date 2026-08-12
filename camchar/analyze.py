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

from .io_utils import stem_for
from .plots import save_linearity_plot

CLIP_DN = 65400  # anything >= this is clipped (IMX174: 4094<<4 = 65504)
DARK_CURRENT_FLAT_MAX = 0.02  # exposures <= 20 ms: dark-current shot noise negligible


def load_sequence(data_dir, seq_type):
    """Return [(exposure_s, stack), ...] sorted, skipping missing files."""
    d = Path(data_dir) / seq_type
    meta_path = d / "metadata.json"
    if not meta_path.exists():
        print(f"  ! no metadata at {meta_path} -- nothing to analyze")
        return []
    entries = json.loads(meta_path.read_text())
    out = []
    for m in entries:
        if m.get("sequence") != seq_type:
            continue
        p = d / f"{stem_for(seq_type, m['exposure_s'], m['gain'])}.npy"
        if not p.exists():
            print(f"  ! missing {p.name}, skipping")
            continue
        out.append((m["exposure_s"], np.load(p)))
    out.sort(key=lambda x: x[0])
    return out


def roi_stats(stack, roi):
    """Per-pixel temporal mean/variance + spatial variance inside the ROI."""
    r = stack[:, roi[0], roi[1]].astype(np.float64)
    return {
        "mean": r.mean(),
        "tvar": r.var(axis=0).mean(),  # temporal variance, avg over pixels
        "svar": r.var(axis=1).mean(),  # spatial variance, avg over frames
    }


def analyze_dark(seq):
    """Bias floor, dark current, read noise from dark frames."""
    print("=== DARK ===")
    if not seq:
        return None
    for exp_s, st in seq:
        s = roi_stats(st, ROI)
        print(
            f"  {exp_s * 1000:8.1f} ms  mean {s['mean']:8.2f}  "
            f"tvar {s['tvar']:9.2f}  svar {s['svar']:9.2f}"
        )

    t = np.array([e for e, _ in seq])
    mu = np.array([roi_stats(st, ROI)["mean"] for _, st in seq])
    slope, intercept = np.polyfit(t, mu, 1)
    bias = mu[0]

    # read noise: temporal sigma where tvar is flat (read-noise dominated)
    tvars = np.array(
        [roi_stats(st, ROI)["tvar"] for e, st in seq if e <= DARK_CURRENT_FLAT_MAX]
    )
    sigma_r = float(np.sqrt(np.median(tvars)))
    return {
        "bias_dn": bias,
        "dark_current_dn_per_s": slope,
        "sigma_r_dn": sigma_r,
        "bias_fit_dn": intercept,
    }


def analyze_flats(seq, dark):
    """K, Nsat, PRNU from the flat sweep (temporal variance)."""
    print("\n=== FLAT (temporal variance) ===")
    if not seq:
        return None
    rows = []
    for exp_s, st in seq:
        s = roi_stats(st, ROI)
        rows.append((exp_s, s))
        print(
            f"  {exp_s * 1000:8.1f} ms  mean {s['mean']:10.2f}  "
            f"tvar {s['tvar']:12.2f}  svar {s['svar']:12.2f}"
        )

    usable = [(s["mean"], s["tvar"]) for _, s in rows if s["mean"] < CLIP_DN]
    if len(usable) < 3:
        print("  ! too few unclipped points -- check illumination level")
        return None
    S = np.array([u[0] for u in usable])
    V = np.array([u[1] for u in usable])
    K_fit, b = np.polyfit(S, V, 1)
    K12 = 16.0 / K_fit  # DN16 -> DN12: K12 = 16/(V16/S16)
    sigma_r_e = dark["sigma_r_dn"] * K12 / 16.0
    prnu = []
    for _, s in rows:
        if 5000 < s["mean"] < 64000:
            prnu.append(np.sqrt(max(s["svar"] - s["tvar"], 0)) / s["mean"])

    print("\n=== PTC FIT (temporal variance) ===")
    print(f"  K (gain)     = {K12:6.2f} e-/DN12   (V/S slope {K_fit:.3f})")
    print(
        f"  sigma_r      = {sigma_r_e:6.2f} e-  ({dark['sigma_r_dn']:.2f} DN16, "
        f"{dark['sigma_r_dn'] / 16:.2f} DN12)"
    )
    print(f"  Nsat         = {K12 * 4094:7.0f} e- (at 12-bit clip 4094)")
    if prnu:
        print(
            f"  PRNU c       = {np.mean(prnu) * 100:5.2f} %  (upper bound, "
            f"includes diffuser texture)"
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
    }


def run(data_dir, roi=None):
    global ROI
    ROI = tuple(slice(a, b) for a, b in zip(roi[0::2], roi[1::2]))

    print(
        f"Analyzing {Path(data_dir).resolve()}  (ROI rows {roi[0]}:{roi[1]}, "
        f"cols {roi[2]}:{roi[3]})"
    )
    dk = analyze_dark(load_sequence(data_dir, "dark"))
    if dk:
        print(f"\n  bias floor (shortest exp): {dk['bias_dn']:.2f} DN16")
        print(
            f"  dark current: {dk['dark_current_dn_per_s']:.4f} DN16/s  "
            f"(bias fit {dk['bias_fit_dn']:.2f})"
        )
        print(f"  read noise (temporal sigma, dark): {dk['sigma_r_dn']:.2f} DN16")
        flat = analyze_flats(load_sequence(data_dir, "flat"), dk)
        if flat:
            save_linearity_plot(flat["rows"], flat["bias_dn"], "outputs", CLIP_DN, roi)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="camchar analyze", description="Temporal PTC analysis of dark/flat data"
    )
    p.add_argument(
        "--data",
        default="data",
        help="directory with dark/ and flat/ subdirs (default: data)",
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
