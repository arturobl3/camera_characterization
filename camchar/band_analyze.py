"""Per-band EMVA 1288 analysis for hyperspectral ENVI data (SPECIM IQ).

Every spectral band is treated as an independent monochrome camera: the same
roi_stats/two-frame estimators and EMVA R4 Linear fits as the playerone
pipeline (analyze.py) are applied per band. Cubes are read once, cropped to
the ROI and scaled to DN16 by specim.py; per-exposure stacks are reduced to
per-band roi_stats rows plus frame-averaged ROI images (for the Section 8.1
highpass nonuniformities) before moving to the next exposure.

Output: a compact one-line-per-band table, an EMVA detail block per plotted
band, a CSV with every per-band parameter, and the same plot set as the
playerone path with one curve per selected (equispaced) band plus a
parameters-vs-wavelength summary figure.
"""

import csv
from pathlib import Path

import numpy as np

from . import analyze
from . import specim
from .analyze import (
    CLIP_DN,
    emva_extras_core,
    fit_dark_stats,
    fit_flat_stats,
    roi_stats,
)
from .plots import (
    save_band_parameters_plot,
    save_dark_plot_bands,
    save_dark_variance_plot_bands,
    save_linearity_plot_bands,
    save_ptc_plot_bands,
    save_snr_plot_bands,
)

_FULL = (slice(None), slice(None))  # stacks arrive already ROI-cropped
_CAMERA_LABEL = "SPECIM IQ"


def _load_kind_stats(exps, kind, r0, r1, c0, c1):
    """One pass over all exposures of a kind.

    Returns {'stats': {exp_s: [roi_stats per band]}, 'means': {exp_s: (R,C,B)
    float32 frame-average}, 'counts': {exp_s: n}, 'wl': wavelengths}.
    """
    stats, means, counts, wavelengths = {}, {}, {}, None
    for exp_s, exp_dir in exps:
        stack, wl = specim.load_exposure_stack(exp_dir, exp_s, r0, r1, c0, c1)
        if stack is None:
            continue
        if wavelengths is None:
            wavelengths = wl
        elif len(wl) != len(wavelengths):
            print(f"  ! [{kind}] {exp_s * 1000:g} ms: band count changed, skipping")
            continue
        n, _, _, n_bands = stack.shape
        print(
            f"  [{kind}] {exp_s * 1000:7.1f} ms: {n} cubes x {n_bands} bands "
            f"(DN16, ROI-cropped)"
        )
        stats[exp_s] = [roi_stats(stack[:, :, :, b], _FULL) for b in range(n_bands)]
        means[exp_s] = stack.mean(axis=0, dtype=np.float32)
        counts[exp_s] = n
        del stack
    return {"stats": stats, "means": means, "counts": counts, "wl": wavelengths}


def _extras_for_band(b, dark, flat, dk, flat_fit):
    """emva_extras_core inputs picked per band (mirrors emva1288_extras)."""
    rows = flat_fit["rows"]
    ok = [rs for rs in rows if rs[1]["sat_frac"] <= analyze.SAT_CLIP_FRAC]
    if not ok:
        ok = [rs for rs in rows if rs[1]["mean"] < CLIP_DN]
    if not ok:
        return None
    mu_sat = max(s["mean"] for _, s in ok)
    target = 0.5 * (mu_sat + dk["bias_dn"])
    idx = min(range(len(rows)), key=lambda i: abs(rows[i][1]["mean"] - target))
    e_dark = min(dark["stats"])
    e_flat = rows[idx][0]
    return emva_extras_core(
        dk,
        flat_fit,
        d_img=dark["means"][e_dark][:, :, b].astype(np.float64),
        f_img=flat["means"][e_flat][:, :, b].astype(np.float64),
        dark_tvar_short=dk["rows"][0][1]["tvar"],
        dark_L=dark["counts"][e_dark],
        flat_tvar_50=rows[idx][1]["tvar"],
        flat_L=flat["counts"][e_flat],
    )


def _analyze_band(b, wl_nm, dark, flat):
    dk_rows = [(e, dark["stats"][e][b]) for e in sorted(dark["stats"])]
    dk = fit_dark_stats(dk_rows)
    dk["rows"] = dk_rows
    res = {
        "band": b,
        "wl_nm": wl_nm,
        "dark": dk,
        "dark_rows": dk_rows,
        "flat": None,
        "flat_rows": [],
        "extras": None,
        "status": "no flat data",
    }
    if not flat["stats"]:
        return res

    fl_rows = [(e, flat["stats"][e][b]) for e in sorted(flat["stats"])]
    res["flat_rows"] = fl_rows
    n_usable = sum(
        1
        for _, s in fl_rows
        if s["mean"] < CLIP_DN and s["sat_frac"] <= analyze.SAT_CLIP_FRAC
    )
    flat_fit = fit_flat_stats(fl_rows, dk)
    if flat_fit is None:
        res["status"] = f"skipped ({n_usable} usable flat pts)"
        return res
    flat_fit["rows"] = fl_rows
    res["flat"] = flat_fit
    res["extras"] = _extras_for_band(b, dark, flat, dk, flat_fit)
    if flat_fit["ptc_slope"] <= 0:
        # V vs S slope <= 0: the fit is physically meaningless (drift between
        # acquisitions and/or saturation-collapsed variance) -- the two-frame
        # columns are the ones to look at.
        res["status"] = "degenerate PTC (see K_tf)"
    else:
        res["status"] = "ok" if res["extras"] else "ok (no sat point)"
    return res


def _select_bands(n_bands, n_sel):
    n_sel = max(1, min(int(n_sel), n_bands))
    sel = np.linspace(0, n_bands - 1, n_sel).round().astype(int)
    return sorted(set(sel.tolist()))


def _print_table(results):
    print("\n=== PER-BAND PARAMETERS (K/sigma_r in e-; DN = DN16) ===")
    print(
        f"{'band':>4} {'lambda':>7} {'K12':>6} {'K12tf':>6} {'sig_r':>6} "
        f"{'sr_q':>6} {'d.cur':>8} {'DSNU':>6} {'PRNU':>6} {'Nsat':>7} "
        f"{'DR':>6}  status"
    )
    for r in results:
        f, x, dk = r["flat"], r["extras"], r["dark"]
        if not f:
            print(
                f"{r['band']:4d} {r['wl_nm']:7.1f}  {'-':>6} {'-':>6} {'-':>6} "
                f"{'-':>6} {'-':>8} {'-':>6} {'-':>6} {'-':>7} {'-':>6}  "
                f"{r['status']}"
            )
            continue
        ok = (
            np.isfinite(f["K12"]) and f["K12"] > 0
        )  # degenerate fits report '-' for K-derived columns
        ok_tf = np.isfinite(f["K12_tf"]) and f["K12_tf"] > 0
        tail = (
            "      -      -      -      -"
            if x is None or not ok
            else (
                f" {x['dsnu1288_dn']:6.2f} {x['prnu1288_pct']:6.2f} "
                f"{x['nsat_e']:7.0f} {x['dr']:6.0f}"
            )
        )
        print(
            f"{r['band']:4d} {r['wl_nm']:7.1f} "
            + (f"{f['K12']:6.2f} " if ok else f"{'-':>6} ")
            + (f"{f['K12_tf']:6.2f} " if ok_tf else f"{'-':>6} ")
            + (
                f"{f['sigma_r_e']:6.2f} {f['sigma_r_e_q']:6.2f} "
                if ok
                else f"{'-':>6} {'-':>6} "
            )
            + f"{dk['dark_current_dn_per_s']:8.2f}"
            + tail
            + f"  {r['status']}"
        )


def _print_band_detail(r):
    dk, f, x = r["dark"], r["flat"], r["extras"]
    print(f"\n-- band {r['band']} ({r['wl_nm']:.2f} nm) --")
    print(
        f"  bias {dk['bias_dn']:.2f} DN16; dark current "
        f"{dk['dark_current_dn_per_s']:.4f} DN16/s "
        f"(var {dk['dark_current_var_dn2_per_s']:.2f} DN16^2/s)"
    )
    print(
        f"  sigma_r (Eq. 30) {dk['sigma_r_dn']:.2f} DN16 "
        f"(short-exposure median {dk['sigma_r_median_dn']:.2f})"
    )
    if not f:
        return
    if "degenerate" in r["status"]:
        print(
            f"  ! degenerate PTC fit (slope {f['ptc_slope']:.3f} <= 0): "
            f"N-frame tvar bent by drift/saturation -- trust K_tf = "
            f"{f['K12_tf']:.2f} e-/DN12"
        )
    print(
        f"  K = {f['K12']:.2f} e-/DN12 (two-frame {f['K12_tf']:.2f}, "
        f"{100 * (f['K12_tf'] - f['K12']) / f['K12']:+.1f}%); "
        f"PTC R2 = {f['ptc_r2']:.6f}"
    )
    print(
        f"  sigma_r = {f['sigma_r_e']:.2f} e- (tf {f['sigma_r_tf_e']:.2f}; "
        f"Eq. 53 quant-corrected {f['sigma_r_e_q']:.2f} e-)"
    )
    print(f"  Nsat (PTC, 4094 clip) = {f['Nsat']:.0f} e-")
    if f["prnu_pct"] is not None:
        print(f"  PRNU quick upper bound = {f['prnu_pct']:.2f} %")
    if x:
        print(
            f"  mu_y.sat = {x['mu_sat_dn']:.0f} DN16; Nsat (EMVA) = "
            f"{x['nsat_e']:.0f} e-; DR = {x['dr']:.0f} ({x['dr_db']:.1f} dB)"
        )
        print(
            f"  PRNU1288 = {x['prnu1288_pct']:.2f} %; "
            f"DSNU1288 = {x['dsnu1288_dn']:.2f} DN16"
        )


_CSV_COLUMNS = [
    "band",
    "wavelength_nm",
    "status",
    "K12_e_per_DN12",
    "K12_tf_e_per_DN12",
    "ptc_r2",
    "sigma_r_e",
    "sigma_r_e_q",
    "sigma_r_tf_e",
    "sigma_r_tf_e_q",
    "bias_dn16",
    "sigma_r_dn16",
    "sigma_r_median_dn16",
    "dark_current_dn16_per_s",
    "dark_current_var_dn16_2_per_s",
    "prnu_quick_pct",
    "mu_sat_dn16",
    "nsat_emva_e",
    "nsat_ptc_e",
    "mu_min_e",
    "dynamic_range",
    "dynamic_range_db",
    "prnu1288_pct",
    "dsnu1288_dn16",
]


def _write_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_COLUMNS)
        for r in results:
            f, x, dk = r["flat"], r["extras"], r["dark"]
            row = {
                "band": r["band"],
                "wavelength_nm": f"{r['wl_nm']:.2f}",
                "status": r["status"],
                "bias_dn16": f"{dk['bias_dn']:.2f}",
                "sigma_r_dn16": f"{dk['sigma_r_dn']:.3f}",
                "sigma_r_median_dn16": f"{dk['sigma_r_median_dn']:.3f}",
                "dark_current_dn16_per_s": f"{dk['dark_current_dn_per_s']:.4f}",
                "dark_current_var_dn16_2_per_s": f"{dk['dark_current_var_dn2_per_s']:.3f}",
            }
            if f:
                row.update(
                    {
                        "K12_e_per_DN12": f"{f['K12']:.4f}",
                        "K12_tf_e_per_DN12": f"{f['K12_tf']:.4f}",
                        "ptc_r2": f"{f['ptc_r2']:.6f}",
                        "sigma_r_e": f"{f['sigma_r_e']:.4f}",
                        "sigma_r_e_q": f"{f['sigma_r_e_q']:.4f}",
                        "sigma_r_tf_e": f"{f['sigma_r_tf_e']:.4f}",
                        "sigma_r_tf_e_q": f"{f['sigma_r_tf_e_q']:.4f}",
                        "nsat_ptc_e": f"{f['Nsat']:.1f}",
                        "prnu_quick_pct": (
                            f"{f['prnu_pct']:.3f}" if f["prnu_pct"] is not None else ""
                        ),
                    }
                )
            if x:
                row.update(
                    {
                        "mu_sat_dn16": f"{x['mu_sat_dn']:.1f}",
                        "nsat_emva_e": f"{x['nsat_e']:.1f}",
                        "mu_min_e": f"{x['mu_min_e']:.3f}",
                        "dynamic_range": f"{x['dr']:.1f}",
                        "dynamic_range_db": f"{x['dr_db']:.2f}",
                        "prnu1288_pct": f"{x['prnu1288_pct']:.3f}",
                        "dsnu1288_dn16": f"{x['dsnu1288_dn']:.3f}",
                    }
                )
            w.writerow([row.get(c, "") for c in _CSV_COLUMNS])
    print(f"  [csv] saved {path}")


def run(cam_dir, roi, n_plot_bands=5):
    r0, r1, c0, c1 = roi
    print(
        f"Analyzing {cam_dir.resolve()}  (per-band; ROI rows {r0}:{r1}, cols {c0}:{c1})"
    )

    dark_exps = specim.list_exposures(specim.kind_dir(cam_dir, "dark"))
    if not dark_exps:
        print("  ! no dark frames found -- nothing to analyze")
        return
    flat_exps = specim.list_exposures(specim.kind_dir(cam_dir, "flat"))
    print(
        f"  dark: {len(dark_exps)} exposures ("
        + ", ".join(f"{e * 1000:g}" for e, _ in dark_exps)
        + " ms); flat: "
        + (
            f"{len(flat_exps)} exposures ("
            + ", ".join(f"{e * 1000:g}" for e, _ in flat_exps)
            + " ms)"
            if flat_exps
            else "none"
        )
    )

    dark = _load_kind_stats(dark_exps, "dark", r0, r1, c0, c1)
    if not dark["stats"]:
        print("  ! no dark cubes loaded -- nothing to analyze")
        return
    flat = _load_kind_stats(flat_exps, "flat", r0, r1, c0, c1)
    wl = dark["wl"]
    n_bands = len(wl)
    print(
        f"\n  {n_bands} bands, {wl[0]:.1f}-{wl[-1]:.1f} nm; "
        f"analyzing each band as an independent camera"
    )

    results = [_analyze_band(b, float(wl[b]), dark, flat) for b in range(n_bands)]
    _print_table(results)

    gaps = [
        abs(r["flat"]["K12_tf"] - r["flat"]["K12"]) / r["flat"]["K12"]
        for r in results
        if r["flat"]
        and np.isfinite(r["flat"]["K12"])
        and r["flat"]["K12"] > 0
        and np.isfinite(r["flat"]["K12_tf"])
        and r["flat"]["K12_tf"] > 0
    ]
    n_degen = sum(1 for r in results if "degenerate" in r["status"])
    if n_degen:
        print(
            f"  ! {n_degen} bands with non-positive PTC slope (drift and/or "
            f"saturation-collapsed variance) -- K_tf is the robust estimate"
        )
    if gaps:
        med = float(np.median(gaps))
        flag = "  ! " if med > 0.01 else "  "
        print(
            f"{flag}two-frame vs N-frame K: median |K_tf-K|/K = {med * 100:.2f}% "
            f"({'>1% -- estimator bias or non-stationarity' if med > 0.01 else 'ok'})"
        )

    selected = _select_bands(n_bands, n_plot_bands)
    print(f"\n=== EMVA DETAIL (plotted bands: {selected}) ===")
    for r in results:
        if r["band"] in selected:
            _print_band_detail(r)

    out_dir = Path("outputs") / cam_dir.name
    _write_csv(results, out_dir / "band_parameters.csv")
    sel_results = [r for r in results if r["band"] in selected]
    save_dark_plot_bands(sel_results, out_dir, roi, _CAMERA_LABEL)
    save_dark_variance_plot_bands(sel_results, out_dir, roi, _CAMERA_LABEL)
    save_linearity_plot_bands(sel_results, out_dir, CLIP_DN, roi, _CAMERA_LABEL)
    save_ptc_plot_bands(sel_results, out_dir, CLIP_DN, roi, _CAMERA_LABEL)
    save_snr_plot_bands(sel_results, out_dir, CLIP_DN, roi, _CAMERA_LABEL)
    save_band_parameters_plot(results, selected, out_dir, _CAMERA_LABEL)
