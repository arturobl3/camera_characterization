"""Shared presentation layer for multi-curve analyses (SPECIM bands and CFA
channels).

band_analyze.py and cfa_analyze.py render the same parameter table, EMVA
detail block, CSV and drift/degeneracy checks on top of the same pure fits
(analyze.fit_dark_stats / fit_flat_stats / emva_extras_core); the helpers
here are that shared rendering. Each caller supplies only its curve identity
columns ('name') -- every number-formatting decision lives here once.
"""

import numpy as np
import typer

_NUM_HEADER = (
    f"{'K12':>6} {'K12nf':>6} {'sig_r':>6} "
    f"{'sr_q':>6} {'d.cur':>8} {'DSNU':>6} {'PRNU':>6} {'Nsat':>7} "
    f"{'DR':>6}  status"
)

_DASH_CELLS = (
    f"{'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>8} {'-':>6} {'-':>6} {'-':>7} {'-':>6}"
)


def num_cells(f, x, dk):
    """The shared formatted numeric columns of one table row.

    Degenerate fits (non-positive K) report '-' for the K-derived columns;
    a missing EMVA-extras point dashes DSNU/PRNU/Nsat/DR.
    """
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
    return (
        (f"{f['K12']:6.2f} " if ok else f"{'-':>6} ")
        + (f"{f['K12_nf']:6.2f} " if ok_nf else f"{'-':>6} ")
        + (
            f"{f['sigma_r_e']:6.2f} {f['sigma_r_e_q']:6.2f} "
            if ok
            else f"{'-':>6} {'-':>6} "
        )
        + f"{dk['dark_current_dn_per_s']:8.2f}"
        + tail
    )


def print_param_table(results, title, lead_header, lead_cells):
    """Per-curve parameter table: caller-supplied identity columns plus the
    shared numeric columns; one line per result, status last."""
    typer.secho(title, bold=True, fg=typer.colors.CYAN)
    typer.echo(f"{lead_header}{_NUM_HEADER}")
    for r in results:
        f, x, dk = r.get("flat"), r.get("extras"), r.get("dark")
        if not f or not dk:
            typer.echo(f"{lead_cells(r)}{_DASH_CELLS}  {r['status']}")
            continue
        typer.echo(f"{lead_cells(r)}{num_cells(f, x, dk)}  {r['status']}")


def print_curve_detail(dk, f, x, status, header, quick_prnu=False, degen_cause=False):
    """Shared EMVA detail block for one curve (dark fit, PTC fit, extras).

    header is the curve's title line (e.g. '-- band 12 (550.00 nm) --');
    quick_prnu adds the quick upper-bound PRNU line (SPECIM bands carry a
    diffuser, CFA channels share the pooled optics); degen_cause extends the
    degenerate-fit warning with the saturation-collapse/shutter-jump causes.
    """
    typer.echo(header)
    if not dk:
        typer.echo(f"  {status}")
        return
    typer.echo(
        f"  bias {dk['bias_dn']:.2f} DN16; dark current "
        f"{dk['dark_current_dn_per_s']:.4f} DN16/s "
        f"(var {dk['dark_current_var_dn2_per_s']:.2f} DN16^2/s; "
        f"nf {dk['dark_current_var_dn2_per_s_nf']:.2f})"
    )
    typer.echo(
        f"  sigma_r (Eq. 30) {dk['sigma_r_dn']:.2f} DN16 "
        f"(short-exposure median {dk['sigma_r_median_dn']:.2f})"
    )
    if not f:
        return
    if "degenerate" in status:
        cause = (
            " (saturation collapse and/or per-acquisition shutter/lamp jumps)"
            if degen_cause
            else ""
        )
        typer.echo(
            f"  ! degenerate PTC fit (two-frame slope {f['ptc_slope']:.3f} <= 0): "
            f"even the adjacent-pair variance is bent{cause} -- check the data"
        )
    typer.echo(
        f"  K (two-frame) = {f['K12']:.2f} e-/DN12 (N-frame {f['K12_nf']:.2f}, "
        f"{100 * (f['K12_nf'] - f['K12']) / f['K12']:+.1f}%); "
        f"PTC R2 = {f['ptc_r2']:.6f}"
    )
    typer.echo(
        f"  sigma_r = {f['sigma_r_e']:.2f} e- (nf {f['sigma_r_nf_e']:.2f}; "
        f"Eq. 53 quant-corrected {f['sigma_r_e_q']:.2f} e-)"
    )
    typer.echo(f"  Nsat (PTC, 4094 clip) = {f['Nsat']:.0f} e-")
    if quick_prnu and f["prnu_pct"] is not None:
        typer.echo(f"  PRNU quick upper bound = {f['prnu_pct']:.2f} %")
    if x:
        typer.echo(
            f"  mu_y.sat = {x['mu_sat_dn']:.0f} DN16; Nsat (EMVA) = "
            f"{x['nsat_e']:.0f} e-; DR = {x['dr']:.0f} ({x['dr_db']:.1f} dB)"
        )
        typer.echo(
            f"  PRNU1288 = {x['prnu1288_pct']:.2f} %; "
            f"DSNU1288 = {x['dsnu1288_dn']:.2f} DN16"
        )


# CSV columns produced by the three cell builders below (identity/status
# columns are prepended by each caller).
PARAM_COLUMNS = [
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


def csv_dark_cells(dk):
    """Dark-fit CSV cells (bias/read-noise/dark-current block)."""
    return {
        "bias_dn16": f"{dk['bias_dn']:.2f}",
        "sigma_r_dn16": f"{dk['sigma_r_dn']:.3f}",
        "sigma_r_median_dn16": f"{dk['sigma_r_median_dn']:.3f}",
        "dark_current_dn16_per_s": f"{dk['dark_current_dn_per_s']:.4f}",
        "dark_current_var_dn16_2_per_s": f"{dk['dark_current_var_dn2_per_s']:.3f}",
        "dark_current_var_dn16_2_per_s_nf": (
            f"{dk['dark_current_var_dn2_per_s_nf']:.3f}"
        ),
    }


def csv_flat_cells(f):
    """PTC-fit CSV cells, or {} when the curve has no flat fit."""
    if not f:
        return {}
    return {
        "K12_e_per_DN12": f"{f['K12']:.4f}",
        "K12_nf_e_per_DN12": f"{f['K12_nf']:.4f}",
        "ptc_r2": f"{f['ptc_r2']:.6f}",
        "sigma_r_e": f"{f['sigma_r_e']:.4f}",
        "sigma_r_e_q": f"{f['sigma_r_e_q']:.4f}",
        "sigma_r_nf_e": f"{f['sigma_r_nf_e']:.4f}",
        "sigma_r_nf_e_q": f"{f['sigma_r_nf_e_q']:.4f}",
        "nsat_ptc_e": f"{f['Nsat']:.1f}",
        "prnu_quick_pct": (f"{f['prnu_pct']:.3f}" if f["prnu_pct"] is not None else ""),
    }


def csv_extras_cells(x):
    """EMVA-extras CSV cells (saturation/DR/nonuniformity), or {} when absent."""
    if not x:
        return {}
    return {
        "mu_sat_dn16": f"{x['mu_sat_dn']:.1f}",
        "nsat_emva_e": f"{x['nsat_e']:.1f}",
        "mu_min_e": f"{x['mu_min_e']:.3f}",
        "dynamic_range": f"{x['dr']:.1f}",
        "dynamic_range_db": f"{x['dr_db']:.2f}",
        "prnu1288_pct": f"{x['prnu1288_pct']:.3f}",
        "dsnu1288_dn16": f"{x['dsnu1288_dn']:.3f}",
    }


def check_drift_and_degenerate(results, noun="bands"):
    """Print the two data-quality indicators shared by both multi-curve runs.

    Counts curves whose PTC slope went non-positive (saturation-collapsed
    variance) and reports the median |K_nf-K|/K gap -- the between-acquisition
    drift indicator (>1% flags non-stationarity).
    """
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
        typer.secho(
            f"  ! {n_degen} {noun} with non-positive two-frame PTC slope "
            f"(saturation-collapsed variance) -- check the data",
            fg=typer.colors.YELLOW,
        )
    if gaps:
        med = float(np.median(gaps))
        flag = "  ! " if med > 0.01 else "  "
        typer.echo(
            f"{flag}N-frame vs two-frame K across {noun}: median "
            f"|K_nf-K|/K = {med * 100:.2f}% "
            f"({'>1% -- drift between acquisitions' if med > 0.01 else 'ok'})"
        )
