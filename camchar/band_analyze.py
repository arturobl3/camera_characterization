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
import typer

from . import analyze
from . import report
from . import specim
from .analyze import (
    CLIP_DN,
    SAT_MAX_DN,
    emva_extras_core,
    fit_dark_stats,
    fit_flat_stats,
)
from .plots import (
    save_band_mean_roi_plot,
    save_band_parameters_plot,
    save_dark_plot_bands,
    save_dark_variance_plot_bands,
    save_flat_uniformity_plot,
    save_linearity_plot_bands,
    save_ptc_plot_bands,
    save_snr_plot_bands,
)

_CAMERA_LABEL = "SPECIM IQ"


def _band_roi_stats(stack, n_bands):
    """roi_stats dicts for every band at once (one float64 conversion).

    Same estimators as analyze.roi_stats -- N-frame temporal variance with
    ddof=1, spatial variance with ddof=0, EMVA R4 Linear two-frame Eqs. 18/32
    with their pair-to-pair scatter, and the 6.6 saturation fraction --
    batched over the band axis so the stack is converted to float64 once
    instead of 2*B times (roi_stats and two_frame_stats each re-convert the
    stack per band). Stacks arrive already ROI-cropped.
    """
    f = stack.astype(np.float64)
    means = f.mean(axis=(0, 1, 2))
    tvars = f.var(axis=0, ddof=1).mean(axis=(0, 1))
    svars = f.var(axis=1).mean(axis=(0, 1))
    sat = (f >= SAT_MAX_DN).mean(axis=(0, 1, 2))
    n_pairs = f.shape[0] // 2
    if n_pairs >= 1:
        a = f[0 : 2 * n_pairs : 2]
        b = f[1 : 2 * n_pairs : 2]
        mu_a = a.mean(axis=(1, 2))
        mu_b = b.mean(axis=(1, 2))
        d = a - b
        pair_t = 0.5 * (d * d).mean(axis=(1, 2)) - 0.5 * (mu_a - mu_b) ** 2
        pair_s = (a * b).mean(axis=(1, 2)) - mu_a * mu_b
        tf_t, tf_s = pair_t.mean(axis=0), pair_s.mean(axis=0)
        tf_ts, tf_ss = pair_t.std(axis=0), pair_s.std(axis=0)
    else:
        tf_t = tf_s = tf_ts = tf_ss = np.zeros(n_bands)
    return [
        {
            "mean": float(means[b]),
            "tvar": float(tvars[b]),
            "svar": float(svars[b]),
            "tf_tvar": float(tf_t[b]),
            "tf_svar": float(tf_s[b]),
            "tf_tvar_scatter": float(tf_ts[b]),
            "tf_svar_scatter": float(tf_ss[b]),
            "sat_frac": float(sat[b]),
        }
        for b in range(n_bands)
    ]


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
            typer.secho(
                f"  ! [{kind}] {exp_s * 1000:g} ms: band count changed, skipping",
                fg=typer.colors.YELLOW,
            )
            continue
        n, _, _, n_bands = stack.shape
        typer.echo(
            f"  [{kind}] {exp_s * 1000:7.1f} ms: {n} cubes x {n_bands} bands "
            f"(DN16, ROI-cropped)"
        )
        stats[exp_s] = _band_roi_stats(stack, n_bands)
        means[exp_s] = stack.mean(axis=0, dtype=np.float32)
        counts[exp_s] = n
        del stack
    return {"stats": stats, "means": means, "counts": counts, "wl": wavelengths}


def _load_shortest_flat_frames(flat_exps, sel_bands):
    """Full-frame per-band means of the shortest flat exposure.

    Returns (imgs, wl_sel, exp_s, n_cubes): imgs is a (R, C, n_sel) float32
    frame-average over all cubes of the shortest flat exposure (DN16),
    wl_sel the wavelengths of the selected bands. One extra pass over a
    single exposure; feeds the illumination-uniformity 3x3 grid.
    """
    exp_s, exp_dir = flat_exps[0]
    hdrs = specim.iter_cubes(exp_dir)
    if not hdrs:
        return None, None, exp_s, 0
    img = specim.open_cube(hdrs[0])
    nl, ns = img.shape[0], img.shape[1]
    img.fid.close()
    stack, wl = specim.load_exposure_stack(exp_dir, exp_s, 0, nl, 0, ns)
    if stack is None:
        return None, None, exp_s, 0
    means = stack.mean(axis=0, dtype=np.float32)
    return means[:, :, sel_bands], wl[sel_bands], exp_s, stack.shape[0]


def _extras_for_band(b, dark, flat, dk, flat_fit):
    """emva_extras_core inputs picked per band (mirrors emva1288_extras)."""
    rows = flat_fit["rows"]
    ok, _ = analyze.usable_flat_points(rows)
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
        dark_tvar_short=dk["rows"][0][1]["tf_tvar"],
        dark_L=dark["counts"][e_dark],
        flat_tvar_50=rows[idx][1]["tf_tvar"],
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
        # V vs S slope <= 0: the fit is physically meaningless (saturation-
        # collapsed variance and/or per-acquisition shutter/lamp jumps) --
        # even the drift-robust two-frame variance is bent.
        res["status"] = "degenerate PTC"
    else:
        res["status"] = "ok" if res["extras"] else "ok (no sat point)"
    return res


def _select_bands(n_bands, n_sel):
    n_sel = max(1, min(int(n_sel), n_bands))
    sel = np.linspace(0, n_bands - 1, n_sel).round().astype(int)
    return sorted(set(sel.tolist()))


def _print_table(results):
    report.print_param_table(
        results,
        "\n=== PER-BAND PARAMETERS (K/sigma_r in e-; DN = DN16) ===",
        f"{'band':>4} {'lambda':>7} ",
        lambda r: f"{r['band']:4d} {r['wl_nm']:7.1f} ",
    )


def _print_band_detail(r):
    report.print_curve_detail(
        r["dark"],
        r["flat"],
        r["extras"],
        r["status"],
        f"\n-- band {r['band']} ({r['wl_nm']:.2f} nm) --",
        quick_prnu=True,
        degen_cause=True,
    )


_CSV_COLUMNS = ["band", "wavelength_nm", "status", *report.PARAM_COLUMNS]


def _write_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_COLUMNS)
        for r in results:
            row = {
                "band": r["band"],
                "wavelength_nm": f"{r['wl_nm']:.2f}",
                "status": r["status"],
            }
            if r["dark"]:
                row.update(report.csv_dark_cells(r["dark"]))
            row.update(report.csv_flat_cells(r.get("flat")))
            row.update(report.csv_extras_cells(r.get("extras")))
            w.writerow([row.get(c, "") for c in _CSV_COLUMNS])
    typer.secho(f"  [csv] saved {path}", fg=typer.colors.GREEN)


def run(cam_dir, roi, n_plot_bands=5):
    r0, r1, c0, c1 = roi
    typer.secho(
        f"Analyzing {cam_dir.resolve()}  (per-band; ROI rows {r0}:{r1}, cols {c0}:{c1})",
        bold=True,
        fg=typer.colors.CYAN,
    )

    dark_exps = specim.list_exposures(specim.kind_dir(cam_dir, "dark"))
    if not dark_exps:
        typer.secho(
            "  ! no dark frames found -- nothing to analyze", fg=typer.colors.YELLOW
        )
        return
    flat_exps = specim.list_exposures(specim.kind_dir(cam_dir, "flat"))
    typer.echo(
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
        typer.secho(
            "  ! no dark cubes loaded -- nothing to analyze", fg=typer.colors.YELLOW
        )
        return
    flat = _load_kind_stats(flat_exps, "flat", r0, r1, c0, c1)
    wl = dark["wl"]
    n_bands = len(wl)
    typer.echo(
        f"\n  {n_bands} bands, {wl[0]:.1f}-{wl[-1]:.1f} nm; "
        f"analyzing each band as an independent camera"
    )

    results = [_analyze_band(b, float(wl[b]), dark, flat) for b in range(n_bands)]
    _print_table(results)
    report.check_drift_and_degenerate(results, "bands")

    selected = _select_bands(n_bands, n_plot_bands)
    typer.secho(
        f"\n=== EMVA DETAIL (plotted bands: {selected}) ===",
        bold=True,
        fg=typer.colors.CYAN,
    )
    for r in results:
        if r["band"] in selected:
            _print_band_detail(r)

    out_dir = Path("outputs") / cam_dir.name
    _write_csv(results, out_dir / "band_parameters.csv")
    save_band_mean_roi_plot(
        dark["stats"], wl, out_dir, "dark", CLIP_DN, roi, _CAMERA_LABEL
    )
    if flat["stats"]:
        save_band_mean_roi_plot(
            flat["stats"], wl, out_dir, "flat", CLIP_DN, roi, _CAMERA_LABEL
        )
    if flat_exps:
        sel9 = _select_bands(n_bands, 9)
        f_imgs, wl_sel, exp_s, n_cubes = _load_shortest_flat_frames(flat_exps, sel9)
        if f_imgs is not None:
            save_flat_uniformity_plot(
                f_imgs, wl_sel, exp_s, n_cubes, out_dir, roi, _CAMERA_LABEL
            )
    sel_results = [r for r in results if r["band"] in selected]
    save_dark_plot_bands(sel_results, out_dir, roi, _CAMERA_LABEL)
    save_dark_variance_plot_bands(sel_results, out_dir, roi, _CAMERA_LABEL)
    save_linearity_plot_bands(sel_results, out_dir, CLIP_DN, roi, _CAMERA_LABEL)
    save_ptc_plot_bands(sel_results, out_dir, CLIP_DN, roi, _CAMERA_LABEL)
    save_snr_plot_bands(sel_results, out_dir, CLIP_DN, roi, _CAMERA_LABEL)
    save_band_parameters_plot(results, selected, out_dir, _CAMERA_LABEL)
