"""Plot generation for camchar analyze output (saved into outputs/)."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

_DEFAULT_ROI = (500, 900, 750, 1150)
_DEFAULT_ROI_SPECIM = (156, 356, 156, 356)


def _title_with_camera(base, camera, roi):
    title = base
    if camera:
        title += "\n" + camera
    if roi and tuple(roi) == _DEFAULT_ROI:
        title += " (central 400x400 px ROI)"
    elif roi and tuple(roi) == _DEFAULT_ROI_SPECIM:
        title += " (central 200x200 px ROI)"
    elif roi:
        title += f" (ROI rows {roi[0]}:{roi[1]}, cols {roi[2]}:{roi[3]})"
    return title


def _band_colors(n):
    """Distinct colors for per-band curves (viridis ramp)."""
    return plt.cm.viridis(np.linspace(0.0, 0.9, max(n, 1)))


def save_dark_mean_plot(rows, out_dir, roi=None, camera=None):
    """Mean dark DN vs exposure with the dark-current linear fit.

    rows is the 'rows' list from analyze_dark: (exposure_s, stats) with
    stats['mean']. Saves dark_mean_vs_exposure.png.
    """
    x = np.array([e for e, _ in rows]) * 1000
    y = np.array([s["mean"] for _, s in rows])
    if len(x) < 2:
        typer.secho(
            "  ! too few dark points for dark mean plot", fg=typer.colors.YELLOW
        )
        return None
    slope, intercept = np.polyfit(x, y, 1)
    yhat = np.polyval([slope, intercept], x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(x, y, "o", ms=5, label="dark (measured)")
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        x_line,
        np.polyval([slope, intercept], x_line),
        "-",
        lw=1.5,
        label="linear fit",
    )
    ax.set_xlabel(r"$t_{\mathrm{exp}}$ (ms)")
    ax.set_ylabel(r"$\mu_{y,\mathrm{dark}}$ (DN16)")
    ax.set_title(
        _title_with_camera(
            r"Dark signal $\mu_{y,\mathrm{dark}}$ vs $t_{\mathrm{exp}}$", camera, roi
        )
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.text(
        0.03,
        0.97,
        rf"$\mu_{{y,\mathrm{{dark}}}} = {slope:.4g}\,t {intercept:+.2f}$ (DN16)"
        "\n"
        f"bias offset = {intercept:.2f} DN16"
        "\n"
        f"R² = {r2:.6f}"
        "\n"
        rf"$\mu_{{I,y}} = {slope * 1000:.4f}$ DN16/s",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "dark_mean_vs_exposure.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return {"slope": slope, "intercept": intercept, "r2": r2}


def save_dark_variance_plot(rows, out_dir, roi=None, camera=None, dsnu=None):
    """Dark temporal variance vs exposure with the Eq. 30 fit.

    rows is the 'rows' list from analyze_dark: (exposure_s, stats) with
    stats['tvar']/stats['tf_tvar']. The N-frame fit's slope is the
    variance-based dark-current estimate and its offset is sigma_r^2
    (EMVA R4 Linear Eq. 30); the two-frame values are the cross-check.
    dsnu is DSNU1288 (Eq. 66, DN16) from the shortest dark stack, shown in
    the text box when provided. Saves dark_variance_vs_exposure.png.
    """
    x = np.array([e for e, _ in rows]) * 1000
    y = np.array([s["tvar"] for _, s in rows])
    y_tf = np.array([s["tf_tvar"] for _, s in rows])
    if len(x) < 2:
        typer.secho(
            "  ! too few dark points for dark variance plot", fg=typer.colors.YELLOW
        )
        return None
    slope, offset = np.polyfit(x, y, 1)
    slope_tf, offset_tf = np.polyfit(x, y_tf, 1)
    yhat = np.polyval([slope, offset], x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "o", ms=5, label=r"$\sigma_y^2$ (N-frame)")
    ax.plot(x, y_tf, "s", ms=4, alpha=0.7, label=r"$\sigma_y^2$ (two-frame, Eq. 18)")
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        x_line,
        np.polyval([slope, offset], x_line),
        "-",
        lw=1.5,
        label="Eq. 30 fit (N-frame)",
    )
    ax.plot(
        x_line,
        np.polyval([slope_tf, offset_tf], x_line),
        "--",
        lw=1.5,
        label="Eq. 30 fit (two-frame)",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_{\mathrm{exp}}$ (ms)")
    ax.set_ylabel(r"$\sigma_{y,\mathrm{dark}}^2$ (DN16²)")
    ax.set_title(
        _title_with_camera(
            r"Dark variance $\sigma_{y,\mathrm{dark}}^2$ vs $t_{\mathrm{exp}}$",
            camera,
            roi,
        )
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    text = (
        rf"$\sigma_y^2 = {slope:.4g}\,t {offset:+.2f}$ (DN16²)"
        "\n"
        f"R² = {r2:.6f}"
        "\n"
        rf"$\mu_{{I,y}} = {slope * 1000:.4f}$ DN16²/s (Eq. 30 slope)"
        "\n"
        rf"$\sigma_r = \sqrt{{\max(\mathrm{{offset}}, 0)}} = "
        rf"{np.sqrt(max(offset, 0.0)):.2f}$ DN16"
    )
    if dsnu is not None:
        text += "\n" + f"DSNU1288 = {dsnu:.2f} DN16"
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "dark_variance_vs_exposure.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return {"slope": slope, "offset": offset, "r2": r2}


def save_linearity_plot(rows, bias_dn, out_dir, clip_dn, roi=None, camera=None):
    """Mean DN (bias-subtracted) vs exposure with a linear fit.

    rows is a list of (exposure_s, stats) from analyze_flats, stats['mean'] in
    DN16. The fit uses unclipped points only (mean < clip_dn); saturated points
    are drawn but excluded. Saves linearity_mean_vs_exposure.png.
    """
    x = np.array([e for e, _ in rows]) * 1000
    y = np.array([s["mean"] for _, s in rows])
    fit_mask = y < clip_dn
    if fit_mask.sum() < 2:
        typer.secho(
            "  ! too few unclipped points for linearity plot", fg=typer.colors.YELLOW
        )
        return None

    xf, yf = x[fit_mask], y[fit_mask] - bias_dn
    slope, intercept = np.polyfit(xf, yf, 1)
    yhat = np.polyval([slope, intercept], xf)
    ss_res = float(np.sum((yf - yhat) ** 2))
    ss_tot = float(np.sum((yf - yf.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xf, yf, "o", ms=5, label="flat (unclipped)")
    ax.plot(
        x[~fit_mask],
        y[~fit_mask] - bias_dn,
        "rx",
        ms=7,
        label="saturated (excluded)",
    )
    x_line = np.linspace(0, float(xf.max()), 200)
    ax.plot(x_line, np.polyval([slope, intercept], x_line), "-", lw=1.5)
    ax.set_xlabel(r"$t_{\mathrm{exp}}$ (ms)")
    ax.set_ylabel(r"$\mu_y - \mu_{y,\mathrm{dark}}$ (DN16)")
    ax.set_title(
        _title_with_camera(
            r"Linearity: $\mu_y - \mu_{y,\mathrm{dark}}$ vs $t_{\mathrm{exp}}$",
            camera,
            roi,
        )
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.text(
        0.97,
        0.5,
        rf"$\mu_y - \mu_{{y,\mathrm{{dark}}}} = {slope:.2f}\,t {intercept:+.2f}$ (DN16)"
        "\n"
        f"R² = {r2:.6f}",
        transform=ax.transAxes,
        va="center",
        ha="right",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "linearity_mean_vs_exposure.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return {"slope": slope, "intercept": intercept, "r2": r2}


def save_ptc_plot(flat, dark, out_dir, clip_dn, roi=None, camera=None):
    """Log-log photon transfer curve (temporal variance vs mean signal).

    flat/dark are the return dicts from analyze_flats/analyze_dark. Unclipped
    flat points carry the shot-noise fit line V = K_fit*S + b; dark points show
    the read-noise region with a dashed floor at sigma_r^2. Saturated points
    are excluded (their variance is degenerate). Saves
    ptc_variance_vs_mean.png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    dx = np.array([s["mean"] for _, s in dark["rows"]])
    dv = np.array([s["tvar"] for _, s in dark["rows"]])
    m = dv > 0
    if m.any():
        ax.loglog(
            dx[m],
            dv[m],
            "^",
            color="tab:green",
            ms=6,
            label="dark (read-noise region)",
        )

    fx = np.array([s["mean"] for _, s in flat["rows"]])
    fv = np.array([s["tvar"] for _, s in flat["rows"]])
    cap = flat["ptc_sat_70_dn"]
    fit_mask = (fx <= cap) & (fx < clip_dn) & (fv > 0)
    if fit_mask.sum() < 2:
        typer.secho("  ! too few points for PTC plot", fg=typer.colors.YELLOW)
        plt.close(fig)
        return None
    ax.loglog(fx[fit_mask], fv[fit_mask], "o", ms=5, label="flat (fit, 0-70% sat)")
    above = (fx > cap) & (fx < clip_dn) & (fv > 0)
    if above.any():
        ax.loglog(
            fx[above],
            fv[above],
            "x",
            ms=6,
            color="0.5",
            alpha=0.8,
            label="excluded (>70% sat)",
        )

    x_line = np.linspace(fx[fit_mask].min(), fx[fit_mask].max(), 200)
    ax.loglog(
        x_line,
        np.polyval([flat["ptc_slope"], flat["ptc_intercept"]], x_line),
        "-",
        lw=1.5,
        label="linear fit",
    )

    sigma_r2 = dark["sigma_r_dn"] ** 2
    ax.axhline(
        sigma_r2,
        ls="--",
        color="0.4",
        lw=1.2,
        label=rf"$\sigma_r^2$ read-noise floor = {sigma_r2:.1f} DN16²",
    )
    ax.axvline(
        2**16,
        ls="--",
        color="tab:red",
        lw=1.2,
        label="2¹⁶ max DN16",
    )
    if flat["ptc_capped"]:
        ax.axvline(
            cap,
            ls="--",
            color="tab:orange",
            lw=1.2,
            label=rf"0.7·$\mu_{{s}}$ = {cap:.0f} DN16",
        )

    ax.set_xlabel(r"$\mu_y$ (DN16)")
    ax.set_ylabel(r"$\sigma_y^2$ (DN16²)")
    ax.set_title(
        _title_with_camera(
            r"Photon transfer curve: $\sigma_y^2$ vs $\mu_y$", camera, roi
        )
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.grid(which="minor", ls="--", alpha=0.3)
    ax.text(
        0.03,
        0.97,
        rf"$\sigma_y^2 = {flat['ptc_slope']:.4g}\,\mu_y {flat['ptc_intercept']:+.1f}$"
        " (DN16²)"
        "\n"
        f"R² = {flat['ptc_r2']:.6f}"
        "\n"
        f"K = {flat['K12']:.2f} e⁻/DN12"
        "\n"
        rf"$\sigma_r$ = {dark['sigma_r_dn']:.2f} DN16",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "ptc_variance_vs_mean.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None


def save_snr_plot(flat, out_dir, clip_dn, roi=None, camera=None):
    """SNR vs signal (e⁻) with the ideal-camera shot-noise reference.

    flat is the return dict from analyze_flats. Signal in electrons is
    bias-subtracted via K12; measured SNR comes from each point's temporal
    variance (EMVA 1288 R4 Linear Eq. 20). Overlays the fitted noise model
    (shot noise sigma_e^2 = mu_e, Eq. 13, + read noise) and the ideal-camera
    limit SNR = sqrt(S_e) (Eq. 23). Saves snr_vs_signal.png.
    """
    means = np.array([s["mean"] for _, s in flat["rows"]])
    tvars = np.array([s["tvar"] for _, s in flat["rows"]])
    m = (means < clip_dn) & (tvars > 0)
    if m.sum() < 2:
        typer.secho("  ! too few points for SNR plot", fg=typer.colors.YELLOW)
        return None
    k12 = flat["K12"]
    signal = (means[m] - flat["bias_dn"]) * k12 / 16.0
    snr = (means[m] - flat["bias_dn"]) / np.sqrt(tvars[m])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(signal, snr, "o", ms=5, label="measured (flat)")
    s_curve = np.linspace(signal.min(), signal.max(), 200)
    ax.loglog(
        s_curve,
        s_curve / np.sqrt(s_curve + flat["sigma_r_e"] ** 2),
        "-",
        lw=1.5,
        label=r"model: $\mu_e/\sqrt{\mu_e + \sigma_r^2}$",
    )
    ax.loglog(
        s_curve,
        np.sqrt(s_curve),
        "--",
        lw=1.5,
        color="0.4",
        label=r"ideal camera: $\sqrt{\mu_e}$",
    )
    ax.set_xlabel(r"$\mu_e$ (e⁻)")
    ax.set_ylabel("SNR")
    ax.set_title(_title_with_camera(r"SNR vs $\mu_e$", camera, roi))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.grid(which="minor", ls="--", alpha=0.3)
    ax.text(
        0.03,
        0.97,
        f"K = {k12:.2f} e⁻/DN12"
        "\n"
        rf"$\sigma_r$ = {flat['sigma_r_e']:.2f} e⁻"
        "\n"
        rf"$\mu_{{e,\mathrm{{sat}}}}$ = {flat['Nsat']:.0f} e⁻",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "snr_vs_signal.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None


# ---------------------------------------------------------------------------
# Per-band (SPECIM IQ hyperspectral) plots: one curve per selected band.
# sel_results is the 'results' list from band_analyze filtered to the plotted
# bands; each entry carries 'wl_nm', 'dark_rows', 'flat_rows', 'dark' (fit
# dict) and 'flat' (fit dict or None).
# ---------------------------------------------------------------------------


def save_dark_plot_bands(sel_results, out_dir, roi=None, camera=None):
    """Dark mean DN16 vs exposure, one curve per band.

    Saves dark_mean_vs_exposure_bands.png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    for color, r in zip(_band_colors(len(sel_results)), sel_results):
        rows = r["dark_rows"]
        if len(rows) < 2:
            continue
        x = np.array([e for e, _ in rows]) * 1000
        y = np.array([s["mean"] for _, s in rows])
        ax.plot(
            x,
            y,
            "o-",
            ms=4,
            lw=1.2,
            color=color,
            label=rf"$\lambda$ {r['wl_nm']:.0f} nm",
        )
        drew = True
    if not drew:
        typer.secho(
            "  ! too few dark points for per-band dark plot", fg=typer.colors.YELLOW
        )
        plt.close(fig)
        return None
    ax.set_xlabel(r"$t_{\mathrm{exp}}$ (ms)")
    ax.set_ylabel(r"$\mu_{y,\mathrm{dark}}$ (DN16)")
    ax.set_title(
        _title_with_camera(
            r"Dark signal $\mu_{y,\mathrm{dark}}$ vs $t_{\mathrm{exp}}$ (per band)",
            camera,
            roi,
        )
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "dark_mean_vs_exposure_bands.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None


def save_dark_variance_plot_bands(sel_results, out_dir, roi=None, camera=None):
    """Dark temporal variance vs exposure, one curve per band (Eq. 30).

    N-frame variance with its linear fit per band; the fit slope is the
    variance-based dark-current estimate and the offset is sigma_r^2.
    Saves dark_variance_vs_exposure_bands.png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    for color, r in zip(_band_colors(len(sel_results)), sel_results):
        rows = r["dark_rows"]
        if len(rows) < 2:
            continue
        x = np.array([e for e, _ in rows]) * 1000
        y = np.array([s["tvar"] for _, s in rows])
        ax.plot(
            x,
            y,
            "o-",
            ms=4,
            lw=1.2,
            color=color,
            label=rf"$\lambda$ {r['wl_nm']:.0f} nm",
        )
        slope, offset = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            xl, np.polyval([slope, offset], xl), "-", lw=1.2, color=color, alpha=0.6
        )
        drew = True
    if not drew:
        typer.secho(
            "  ! too few dark points for per-band dark variance plot",
            fg=typer.colors.YELLOW,
        )
        plt.close(fig)
        return None
    ax.set_xlabel(r"$t_{\mathrm{exp}}$ (ms)")
    ax.set_ylabel(r"$\sigma_{y,\mathrm{dark}}^2$ (DN16²)")
    ax.set_title(
        _title_with_camera(
            r"Dark variance $\sigma_{y,\mathrm{dark}}^2$ vs $t_{\mathrm{exp}}$"
            " (per band)",
            camera,
            roi,
        )
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "dark_variance_vs_exposure_bands.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None


def save_linearity_plot_bands(sel_results, out_dir, clip_dn, roi=None, camera=None):
    """Bias-subtracted flat mean vs exposure, one curve per band (log-log).

    Log-log because the signal level spans orders of magnitude across the
    spectrum (halogen lamp x sensor QE). Saturated points are marked x and
    excluded from the curve. Saves linearity_mean_vs_exposure_bands.png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    for color, r in zip(_band_colors(len(sel_results)), sel_results):
        if not r["flat_rows"] or not r["dark"]:
            continue
        rows = r["flat_rows"]
        bias = r["dark"]["bias_dn"]
        x = np.array([e for e, _ in rows]) * 1000
        y = np.array([s["mean"] for _, s in rows]) - bias
        m = (y > 0) & (np.array([s["mean"] for _, s in rows]) < clip_dn)
        if m.sum() < 1:
            continue
        ax.loglog(
            x[m],
            y[m],
            "o-",
            ms=4,
            lw=1.2,
            color=color,
            label=rf"$\lambda$ {r['wl_nm']:.0f} nm",
        )
        if (~m).any():
            ax.loglog(x[~m], np.maximum(y[~m], 1.0), "x", ms=7, color=color, alpha=0.7)
        drew = True
    if not drew:
        typer.secho(
            "  ! too few unclipped points for per-band linearity plot",
            fg=typer.colors.YELLOW,
        )
        plt.close(fig)
        return None
    ax.set_xlabel(r"$t_{\mathrm{exp}}$ (ms)")
    ax.set_ylabel(r"$\mu_y - \mu_{y,\mathrm{dark}}$ (DN16)")
    ax.set_title(
        _title_with_camera(
            r"Linearity: $\mu_y - \mu_{y,\mathrm{dark}}$ vs $t_{\mathrm{exp}}$"
            " (per band)",
            camera,
            roi,
        )
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "linearity_mean_vs_exposure_bands.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None


def save_ptc_plot_bands(sel_results, out_dir, clip_dn, roi=None, camera=None):
    """Log-log photon transfer curve with one curve + fit per band.

    Saves ptc_variance_vs_mean_bands.png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    for color, r in zip(_band_colors(len(sel_results)), sel_results):
        f = r["flat"]
        if not f:
            continue
        x = np.array([s["mean"] for _, s in f["rows"]])
        v = np.array([s["tvar"] for _, s in f["rows"]])
        cap = f["ptc_sat_70_dn"]
        m = (x <= cap) & (x < clip_dn) & (v > 0)
        if m.sum() < 2:
            continue
        ax.loglog(
            x[m],
            v[m],
            "o",
            ms=4,
            color=color,
            label=rf"$\lambda$ {r['wl_nm']:.0f} nm: K={f['K12']:.2f}",
        )
        ab = (x > cap) & (x < clip_dn) & (v > 0)
        if ab.any():
            ax.loglog(x[ab], v[ab], "x", ms=4, color="0.5", alpha=0.7)
        xl = np.linspace(x[m].min(), x[m].max(), 100)
        ax.loglog(
            xl,
            np.polyval([f["ptc_slope"], f["ptc_intercept"]], xl),
            "-",
            lw=1.2,
            color=color,
        )
        drew = True
    if not drew:
        typer.secho("  ! too few points for per-band PTC plot", fg=typer.colors.YELLOW)
        plt.close(fig)
        return None
    ax.set_xlabel(r"$\mu_y$ (DN16)")
    ax.set_ylabel(r"$\sigma_y^2$ (DN16²)")
    ax.set_title(
        _title_with_camera(
            r"Photon transfer curve: $\sigma_y^2$ vs $\mu_y$ (per band)",
            camera,
            roi,
        )
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.grid(which="minor", ls="--", alpha=0.3)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "ptc_variance_vs_mean_bands.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None


def save_snr_plot_bands(sel_results, out_dir, clip_dn, roi=None, camera=None):
    """SNR vs signal (e-) with one measured curve + model per band.

    Also draws the ideal-camera limit SNR = sqrt(mu_e) (dashed) across the
    combined signal range of all bands. Saves snr_vs_signal_bands.png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    sig_lo, sig_hi = [], []
    for color, r in zip(_band_colors(len(sel_results)), sel_results):
        f = r["flat"]
        if not f or not r["dark_rows"]:
            continue
        means = np.array([s["mean"] for _, s in f["rows"]])
        tvars = np.array([s["tvar"] for _, s in f["rows"]])
        m = (means < clip_dn) & (tvars > 0)
        if m.sum() < 2:
            continue
        signal = (means[m] - f["bias_dn"]) * f["K12"] / 16.0
        snr = (means[m] - f["bias_dn"]) / np.sqrt(tvars[m])
        ax.loglog(
            signal, snr, "o", ms=4, color=color, label=rf"$\lambda$ {r['wl_nm']:.0f} nm"
        )
        sc = np.linspace(signal.min(), signal.max(), 100)
        ax.loglog(sc, sc / np.sqrt(sc + f["sigma_r_e"] ** 2), "-", lw=1.2, color=color)
        sig_lo.append(signal.min())
        sig_hi.append(signal.max())
        drew = True
    if not drew:
        typer.secho("  ! too few points for per-band SNR plot", fg=typer.colors.YELLOW)
        plt.close(fig)
        return None
    si = np.linspace(min(sig_lo), max(sig_hi), 200)
    ax.loglog(
        si,
        np.sqrt(si),
        "--",
        lw=1.5,
        color="0.4",
        label=r"ideal camera: $\sqrt{\mu_e}$",
    )
    ax.set_xlabel(r"$\mu_e$ (e⁻)")
    ax.set_ylabel("SNR")
    ax.set_title(_title_with_camera(r"SNR vs $\mu_e$ (per band)", camera, roi))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.grid(which="minor", ls="--", alpha=0.3)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "snr_vs_signal_bands.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None


def save_band_parameters_plot(results, selected, out_dir, camera=None):
    """EMVA parameters vs wavelength: K, sigma_r, dark current, PRNU/DSNU.

    All bands are plotted; the bands chosen for the per-band plots are marked
    with vertical dotted lines. Saves band_parameters_vs_wavelength.png.
    """
    wl = np.array([r["wl_nm"] for r in results])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sel_wl = [results[b]["wl_nm"] for b in selected]

    def _pos(a):
        """Mask non-positive values (degenerate fits) so axes stay readable."""
        return np.where(np.asarray(a, dtype=float) > 0, a, np.nan)

    def _mark(ax):
        for w in sel_wl:
            ax.axvline(w, color="0.5", ls=":", lw=1, alpha=0.7)

    ax = axes[0][0]
    k = _pos([r["flat"]["K12"] if r["flat"] else np.nan for r in results])
    ktf = _pos([r["flat"]["K12_tf"] if r["flat"] else np.nan for r in results])
    ax.plot(wl, k, "o", ms=3, label="K (N-frame)")
    ax.plot(wl, ktf, ".", ms=3, alpha=0.6, label="K (two-frame)")
    ax.set_ylabel(r"$K$ (e⁻/DN12)")
    ax.set_title("System gain $K$")
    ax.legend(fontsize=9)
    _mark(ax)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    sr = _pos([r["flat"]["sigma_r_e"] if r["flat"] else np.nan for r in results])
    srq = _pos([r["flat"]["sigma_r_e_q"] if r["flat"] else np.nan for r in results])
    ax.plot(wl, sr, "o", ms=3, label=r"$\sigma_r$")
    ax.plot(wl, srq, ".", ms=3, alpha=0.6, label=r"$\sigma_r$ (Eq. 53 quant-corrected)")
    ax.set_ylabel(r"$\sigma_r$ (e⁻)")
    ax.set_title(r"Read noise $\sigma_r$")
    ax.legend(fontsize=9)
    _mark(ax)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    dc = np.array([r["dark"]["dark_current_dn_per_s"] for r in results])
    pos = dc > 0
    ax.semilogy(wl[pos], dc[pos], "o", ms=3)
    ax.set_ylabel(r"$\mu_{I,y}$ (DN16/s)")
    ax.set_xlabel("wavelength (nm)")
    ax.set_title("Dark current (Eq. 29 mean fit)")
    _mark(ax)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1][1]
    prnu = np.array(
        [r["extras"]["prnu1288_pct"] if r["extras"] else np.nan for r in results]
    )
    ax.plot(wl, prnu, "o", ms=3, color="tab:blue", label="PRNU1288 (%)")
    ax.set_ylabel("PRNU1288 (%)", color="tab:blue")
    ax.set_xlabel("wavelength (nm)")
    ax.set_title("Nonuniformity (highpass, EMVA §8.1)")
    ax2 = ax.twinx()
    dsnu = np.array(
        [r["extras"]["dsnu1288_dn"] if r["extras"] else np.nan for r in results]
    )
    ax2.plot(wl, dsnu, ".", ms=3, color="tab:red", label="DSNU1288 (DN16)")
    ax2.set_ylabel("DSNU1288 (DN16)", color="tab:red")
    _mark(ax)
    ax.grid(alpha=0.3)

    if camera:
        fig.suptitle(f"{camera} -- per-band EMVA 1288 R4 (Linear) parameters")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "band_parameters_vs_wavelength.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None
