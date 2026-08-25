"""Per-CFA-channel analysis for raw-Bayer datasets.

Mirrors camchar/band_analyze.py: every Bayer sub-lattice (R, G1, G2, B in
_cfa_phases order) is analyzed as an independent monochrome camera through
the same pure fits -- fit_dark_stats / fit_flat_stats / emva_extras_core --
so the per-channel numbers and the pooled headline numbers share one math
implementation (no duplicated EMVA math, same rule as the SPECIM path).
Results feed one curve per channel in the *_bands.png plot variants and
outputs/<camera>/cfa_parameters.csv. The pooled section from analyze.run()
stays the primary EMVA-consistent set; this is purely additive.
"""

import csv
from pathlib import Path

import numpy as np
import typer

from . import analyze
from .analyze import (
    CLIP_DN,
    _cfa_labels,
    emva_extras_core,
    fit_dark_stats,
    fit_flat_stats,
    roi_stats,
)
from .plots import (
    save_dark_plot_bands,
    save_dark_variance_plot_bands,
    save_linearity_plot_bands,
    save_ptc_plot_bands,
    save_snr_plot_bands,
)

# physical colors for the channel curves; G1/G2 get distinct greens so the
# four curves stay tellable apart
_LABEL_COLORS = {
    "R": "tab:red",
    "G": "tab:green",
    "G1": "tab:green",
    "G2": "tab:olive",
    "B": "tab:blue",
}


def _slice_phase(seq, i, j):
    """[(exposure, sub-lattice stack)] for CFA phase (i, j)."""
    return [(e, stack[:, i::2, j::2]) for e, stack in seq]


def _extras_for_channel(dark_seq_c, flat_seq_c, dk, flat_fit, roi_slices):
    """emva_extras_core inputs picked per channel (mirrors band_analyze).

    Inside a single sub-lattice there is no CFA structure left, so the core
    runs with cfa=False.
    """
    rows = flat_fit["rows"]
    ok = [rs for rs in rows if rs[1]["sat_frac"] <= analyze.SAT_CLIP_FRAC]
    if not ok:
        ok = [rs for rs in rows if rs[1]["mean"] < CLIP_DN]
    if not ok or not dark_seq_c:
        return None
    mu_sat = max(s["mean"] for _, s in ok)
    target = 0.5 * (mu_sat + dk["bias_dn"])
    idx = min(range(len(rows)), key=lambda k: abs(rows[k][1]["mean"] - target))
    _, d_stack = min(dark_seq_c, key=lambda es: es[0])
    _, f_stack = flat_seq_c[idx]
    dark_tvar_short = dk["rows"][0][1]["tf_tvar"]
    return emva_extras_core(
        dk,
        flat_fit,
        d_img=d_stack[:, roi_slices[0], roi_slices[1]].astype(np.float64).mean(axis=0),
        f_img=f_stack[:, roi_slices[0], roi_slices[1]].astype(np.float64).mean(axis=0),
        dark_tvar_short=dark_tvar_short,
        dark_L=d_stack.shape[0],
        flat_tvar_50=rows[idx][1]["tf_tvar"],
        flat_L=f_stack.shape[0],
        cfa=False,
    )


def _analyze_channel(label, i, j, dark_seq, flat_seq, roi_slices):
    """Full monochrome pipeline on one CFA sub-lattice (no printing)."""
    res = {
        "band": -1,
        "label": label,
        "plot_color": _LABEL_COLORS.get(label),
        "dark": None,
        "dark_rows": [],
        "flat": None,
        "flat_rows": [],
        "extras": None,
        "status": "no dark data",
    }
    if not dark_seq:
        return res
    dark_seq_c = _slice_phase(dark_seq, i, j)
    dk_rows = [(e, roi_stats(st, roi_slices)) for e, st in dark_seq_c]
    dk = fit_dark_stats(dk_rows)
    dk["rows"] = dk_rows
    res["dark"] = dk
    res["dark_rows"] = dk_rows
    if not flat_seq:
        res["status"] = "no flat data"
        return res

    flat_seq_c = _slice_phase(flat_seq, i, j)
    fl_rows = [(e, roi_stats(st, roi_slices)) for e, st in flat_seq_c]
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
    res["extras"] = _extras_for_channel(
        dark_seq_c, flat_seq_c, dk, flat_fit, roi_slices
    )
    if flat_fit["ptc_slope"] <= 0:
        # V vs S slope <= 0: physically meaningless fit (saturation-collapsed
        # variance and/or per-acquisition shutter/lamp jumps) -- same rule as
        # the SPECIM bands.
        res["status"] = "degenerate PTC"
    else:
        res["status"] = "ok" if res["extras"] else "ok (no sat point)"
    return res


def _print_table(results):
    print("\n=== PER-CFA-CHANNEL PARAMETERS (K/sigma_r in e-; DN = DN16) ===")
    print(
        f"{'ch':>4} {'K12':>6} {'K12nf':>6} {'sig_r':>6} "
        f"{'sr_q':>6} {'d.cur':>8} {'DSNU':>6} {'PRNU':>6} {'Nsat':>7} "
        f"{'DR':>6}  status"
    )
    for r in results:
        f, x, dk = r["flat"], r["extras"], r["dark"]
        if not f or not dk:
            print(
                f"{r['label']:>4} {'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>8} "
                f"{'-':>6} {'-':>6} {'-':>7} {'-':>6}  {r['status']}"
            )
            continue
        ok = np.isfinite(f["K12"]) and f["K12"] > 0
        ok_nf = np.isfinite(f["K12_nf"]) and f["K12_nf"] > 0
        tail = (
            "      -      -      -      -"
            if x is None or not ok
            else (
                f" {x['dsnu1288_dn']:6.2f} {x['prnu1288_pct']:6.2f} "
                f"{x['nsat_e']:7.0f} {x['dr']:6.0f}"
            )
        )
        print(
            f"{r['label']:>4} "
            + (f"{f['K12']:6.2f} " if ok else f"{'-':>6} ")
            + (f"{f['K12_nf']:6.2f} " if ok_nf else f"{'-':>6} ")
            + (
                f"{f['sigma_r_e']:6.2f} {f['sigma_r_e_q']:6.2f} "
                if ok
                else f"{'-':>6} {'-':>6} "
            )
            + f"{dk['dark_current_dn_per_s']:8.2f}"
            + tail
            + f"  {r['status']}"
        )


def _print_channel_detail(r):
    dk, f, x = r["dark"], r["flat"], r["extras"]
    print(f"\n-- channel {r['label']} --")
    if not dk:
        print(f"  {r['status']}")
        return
    print(
        f"  bias {dk['bias_dn']:.2f} DN16; dark current "
        f"{dk['dark_current_dn_per_s']:.4f} DN16/s "
        f"(var {dk['dark_current_var_dn2_per_s']:.2f} DN16^2/s; "
        f"nf {dk['dark_current_var_dn2_per_s_nf']:.2f})"
    )
    print(
        f"  sigma_r (Eq. 30) {dk['sigma_r_dn']:.2f} DN16 "
        f"(short-exposure median {dk['sigma_r_median_dn']:.2f})"
    )
    if not f:
        return
    if "degenerate" in r["status"]:
        print(
            f"  ! degenerate PTC fit (two-frame slope {f['ptc_slope']:.3f} <= 0): "
            "even the adjacent-pair variance is bent -- check the data"
        )
    print(
        f"  K (two-frame) = {f['K12']:.2f} e-/DN12 (N-frame {f['K12_nf']:.2f}, "
        f"{100 * (f['K12_nf'] - f['K12']) / f['K12']:+.1f}%); "
        f"PTC R2 = {f['ptc_r2']:.6f}"
    )
    print(
        f"  sigma_r = {f['sigma_r_e']:.2f} e- (nf {f['sigma_r_nf_e']:.2f}; "
        f"Eq. 53 quant-corrected {f['sigma_r_e_q']:.2f} e-)"
    )
    print(f"  Nsat (PTC, 4094 clip) = {f['Nsat']:.0f} e-")
    if x:
        print(
            f"  mu_y.sat = {x['mu_sat_dn']:.0f} DN16; Nsat (EMVA) = "
            f"{x['nsat_e']:.0f} e-; DR = {x['dr']:.0f} ({x['dr_db']:.1f} dB)"
        )
        print(
            f"  PRNU1288 = {x['prnu1288_pct']:.2f} %; "
            f"DSNU1288 = {x['dsnu1288_dn']:.2f} DN16"
        )


_CSV_SOURCE_COLUMNS = [
    "status",
    "K12_e_per_DN12",
    "K12_nf_e_per_DN12",
    "ptc_r2",
    "sigma_r_e",
    "sigma_r_e_q",
    "sigma_r_nf_e",
    "sigma_r_nf_e_q",
    "bias_dn16",
    "sigma_r_dn16",
    "sigma_r_median_dn16",
    "dark_current_dn16_per_s",
    "dark_current_var_dn16_2_per_s",
    "dark_current_var_dn16_2_per_s_nf",
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
_COLUMNS = ["channel", *_CSV_SOURCE_COLUMNS]


def _write_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_COLUMNS)
        for r in results:
            f, x, dk = r.get("flat"), r.get("extras"), r.get("dark")
            row = {"channel": r["label"], "status": r["status"]}
            if dk:
                row.update(
                    {
                        "bias_dn16": f"{dk['bias_dn']:.2f}",
                        "sigma_r_dn16": f"{dk['sigma_r_dn']:.3f}",
                        "sigma_r_median_dn16": f"{dk['sigma_r_median_dn']:.3f}",
                        "dark_current_dn16_per_s": f"{dk['dark_current_dn_per_s']:.4f}",
                        "dark_current_var_dn16_2_per_s": f"{dk['dark_current_var_dn2_per_s']:.3f}",
                        "dark_current_var_dn16_2_per_s_nf": (
                            f"{dk['dark_current_var_dn2_per_s_nf']:.3f}"
                        ),
                    }
                )
            if f:
                row.update(
                    {
                        "K12_e_per_DN12": f"{f['K12']:.4f}",
                        "K12_nf_e_per_DN12": f"{f['K12_nf']:.4f}",
                        "ptc_r2": f"{f['ptc_r2']:.6f}",
                        "sigma_r_e": f"{f['sigma_r_e']:.4f}",
                        "sigma_r_e_q": f"{f['sigma_r_e_q']:.4f}",
                        "sigma_r_nf_e": f"{f['sigma_r_nf_e']:.4f}",
                        "sigma_r_nf_e_q": f"{f['sigma_r_nf_e_q']:.4f}",
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
            w.writerow([row.get(c, "") for c in _COLUMNS])
    typer.secho(f"  [csv] saved {path}", fg=typer.colors.GREEN)


def run(cam_dir, dark_seq, flat_seq, bayer_format, roi, camera=None):
    labels = _cfa_labels(bayer_format)
    phases = list(zip(labels, (0, 0, 1, 1), (0, 1, 0, 1)))
    roi_slices = tuple(slice(a, b) for a, b in zip(roi[0::2], roi[1::2]))
    typer.secho(
        "\n=== PER-CFA-CHANNEL ANALYSIS (" + ",".join(labels) + ") ===",
        bold=True,
        fg=typer.colors.CYAN,
    )
    results = [
        _analyze_channel(label, i, j, dark_seq, flat_seq, roi_slices)
        for label, i, j in phases
    ]
    _print_table(results)

    gaps = [
        abs(r["flat"]["K12_nf"] - r["flat"]["K12"]) / r["flat"]["K12"]
        for r in results
        if r["flat"]
        and np.isfinite(r["flat"]["K12"])
        and r["flat"]["K12"] > 0
        and np.isfinite(r["flat"]["K12_nf"])
        and r["flat"]["K12_nf"] > 0
    ]
    n_degen = sum(1 for r in results if "degenerate" in r["status"])
    if n_degen:
        print(
            f"  ! {n_degen} channels with non-positive two-frame PTC slope "
            f"(saturation-collapsed variance) -- check the data"
        )
    if gaps:
        med = float(np.median(gaps))
        flag = "  ! " if med > 0.01 else "  "
        print(
            f"{flag}N-frame vs two-frame K across channels: median "
            f"|K_nf-K|/K = {med * 100:.2f}% "
            f"({'>1% -- drift between acquisitions' if med > 0.01 else 'ok'})"
        )

    print("\n=== EMVA DETAIL (per CFA channel) ===")
    for r in results:
        _print_channel_detail(r)

    out_dir = Path("outputs") / Path(cam_dir).name
    _write_csv(results, out_dir / "cfa_parameters.csv")
    save_dark_plot_bands(results, out_dir, camera=camera)
    save_dark_variance_plot_bands(results, out_dir, camera=camera)
    save_linearity_plot_bands(results, out_dir, CLIP_DN, camera=camera)
    save_ptc_plot_bands(results, out_dir, CLIP_DN, camera=camera)
    save_snr_plot_bands(results, out_dir, CLIP_DN, camera=camera)
