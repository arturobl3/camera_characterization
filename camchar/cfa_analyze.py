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
from . import report
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
    ok, _ = analyze.usable_flat_points(rows)
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
    report.print_param_table(
        results,
        "\n=== PER-CFA-CHANNEL PARAMETERS (K/sigma_r in e-; DN = DN16) ===",
        f"{'ch':>4} ",
        lambda r: f"{r['label']:>4} ",
    )


def _print_channel_detail(r):
    report.print_curve_detail(
        r.get("dark"),
        r.get("flat"),
        r.get("extras"),
        r["status"],
        f"\n-- channel {r['label']} --",
    )


_COLUMNS = ["channel", "status", *report.PARAM_COLUMNS]


def _write_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_COLUMNS)
        for r in results:
            row = {"channel": r["label"], "status": r["status"]}
            if r.get("dark"):
                row.update(report.csv_dark_cells(r["dark"]))
            row.update(report.csv_flat_cells(r.get("flat")))
            row.update(report.csv_extras_cells(r.get("extras")))
            w.writerow([row.get(c, "") for c in _COLUMNS])
    typer.secho(f"  [csv] saved {path}", fg=typer.colors.GREEN)


def run(cam_dir, dark_seq, flat_seq, bayer_format, roi, camera=None, roi_label=None):
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
    report.check_drift_and_degenerate(results, "channels")

    typer.secho(
        "\n=== EMVA DETAIL (per CFA channel) ===", bold=True, fg=typer.colors.CYAN
    )
    for r in results:
        _print_channel_detail(r)

    out_dir = Path("outputs") / Path(cam_dir).name
    _write_csv(results, out_dir / "cfa_parameters.csv")
    save_dark_plot_bands(results, out_dir, roi_label, camera=camera)
    save_dark_variance_plot_bands(results, out_dir, roi_label, camera=camera)
    save_linearity_plot_bands(results, out_dir, CLIP_DN, roi_label, camera=camera)
    save_ptc_plot_bands(results, out_dir, CLIP_DN, roi_label, camera=camera)
    save_snr_plot_bands(results, out_dir, CLIP_DN, roi_label, camera=camera)
