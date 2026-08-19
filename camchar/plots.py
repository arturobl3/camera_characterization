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
    """Distinct colors for per-band curves (default C-series palette)."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [colors[i % len(colors)] for i in range(max(n, 1))]


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
        rf"$I_d = {slope * 1000:.4f}$ DN16/s",
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
    stats['tf_tvar']/stats['tvar']. The two-frame (Eq. 18) fit's slope is
    the primary variance-based dark-current estimate and its offset is
    sigma_r^2 (EMVA R4 Linear Eq. 30); the N-frame values are the
    cross-check. dsnu is DSNU1288 (Eq. 66, DN16) from the shortest dark
    stack, shown in the text box when provided. Saves
    dark_variance_vs_exposure.png.
    """
    x = np.array([e for e, _ in rows]) * 1000
    y = np.array([s["tf_tvar"] for _, s in rows])
    y_nf = np.array([s["tvar"] for _, s in rows])
    if len(x) < 2:
        typer.secho(
            "  ! too few dark points for dark variance plot", fg=typer.colors.YELLOW
        )
        return None
    slope, offset = np.polyfit(x, y, 1)
    slope_nf, offset_nf = np.polyfit(x, y_nf, 1)
    yhat = np.polyval([slope, offset], x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "o", ms=5, label=r"$\sigma_y^2$ (two-frame, Eq. 18)")
    ax.plot(x, y_nf, "s", ms=4, alpha=0.7, label=r"$\sigma_y^2$ (N-frame)")
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        x_line,
        np.polyval([slope, offset], x_line),
        "-",
        lw=1.5,
        label="Eq. 30 fit (two-frame)",
    )
    ax.plot(
        x_line,
        np.polyval([slope_nf, offset_nf], x_line),
        "--",
        lw=1.5,
        label="Eq. 30 fit (N-frame)",
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
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)
    text = (
        rf"$\sigma_y^2 = {slope:.4g}\,t {offset:+.2f}$ (DN16²)"
        "\n"
        f"R² = {r2:.6f}"
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
    """Log-log photon transfer curve (Eq. 50: two-frame dark-subtracted variance).

    flat/dark are the return dicts from analyze_flats/analyze_dark. The
    abscissa is the photo-induced signal mu_y - mu_y,dark and the ordinate
    dV = sigma_y^2 - sigma_y,dark^2 on the two-frame temporal variance
    (Eq. 18, primary estimator; dark variance matched per exposure), so the
    fit is the zero-intercept line dV = K*S and the read-noise floor sits
    at y = 0. The free-fit intercept (~0) is the sanity check.
    Saturated points are excluded (their variance is degenerate). Saves
    ptc_variance_vs_mean.png.
    """
    bias = flat["bias_dn"]
    fx = np.array([s["mean"] for _, s in flat["rows"]])
    fv = np.array(flat["dV"])
    cap = flat["ptc_sat_70_dn"]
    fit_mask = (fx <= cap) & (fx < clip_dn) & (fv > 0)
    if fit_mask.sum() < 2:
        typer.secho("  ! too few points for PTC plot", fg=typer.colors.YELLOW)
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(
        fx[fit_mask] - bias, fv[fit_mask], "o", ms=5, label="flat (fit, 0-70% sat)"
    )
    above = (fx > cap) & (fx < clip_dn) & (fv > 0)
    if above.any():
        ax.loglog(
            fx[above] - bias,
            fv[above],
            "x",
            ms=6,
            color="0.5",
            alpha=0.8,
            label="excluded (>70% sat)",
        )

    x_line = np.linspace(fx[fit_mask].min(), fx[fit_mask].max(), 200) - bias
    ax.loglog(
        x_line,
        flat["ptc_slope"] * x_line,
        "-",
        lw=1.5,
        label="linear fit (zero intercept)",
    )

    ax.axvline(
        2**16 - bias,
        ls="--",
        color="tab:red",
        lw=1.2,
        label="2¹⁶ max DN16",
    )
    if flat["ptc_capped"]:
        ax.axvline(
            cap - bias,
            ls="--",
            color="tab:orange",
            lw=1.2,
            label=rf"0.7·$\mu_{{s}}$ = {cap:.0f} DN16",
        )

    ax.set_xlabel(r"$\mu_y - \mu_{y,\mathrm{dark}}$ (DN16)")
    ax.set_ylabel(r"$\sigma_y^2 - \sigma_{y,\mathrm{dark}}^2$ (DN16²)")
    ax.set_title(
        _title_with_camera(
            r"Photon transfer curve: $\sigma_y^2 - \sigma_{y,\mathrm{dark}}^2$ "
            r"vs $\mu_y - \mu_{y,\mathrm{dark}}$",
            camera,
            roi,
        )
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.grid(which="minor", ls="--", alpha=0.3)
    ax.text(
        0.03,
        0.97,
        rf"$\Delta\sigma_y^2 = {flat['ptc_slope']:.4g}\,(\mu_y - "
        rf"\mu_{{y,\mathrm{{dark}}}})$ (DN16²)"
        "\n"
        f"R² = {flat['ptc_r2']:.6f}"
        "\n"
        f"K = {flat['K12']:.2f} e⁻/DN12"
        "\n"
        rf"free intercept = {flat['ptc_intercept_free']:+.1f} DN16² (~0 check)",
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
    bias-subtracted via K12; measured SNR comes from each point's two-frame
    temporal variance (EMVA 1288 R4 Linear Eq. 18, primary estimator).
    Overlays the fitted noise model
    (shot noise sigma_e^2 = mu_e, Eq. 13, + read noise) and the ideal-camera
    limit SNR = sqrt(S_e) (Eq. 23). When flat carries the EMVA extras
    (flat['extras']), also draws the Eq. 69 total SNR (DSNU1288 + PRNU1288 +
    quantization included): the model curve plus the measured total SNR at
    each exposure step. Saves snr_vs_signal.png.
    """
    from .analyze import snr_temporal_model, snr_total_measured, snr_total_model  # local: import cycle

    extras = flat.get("extras")
    means = np.array([s["mean"] for _, s in flat["rows"]])
    tvars = np.array([s["tf_tvar"] for _, s in flat["rows"]])
    m = (means < clip_dn) & (tvars > 0)
    if m.sum() < 2:
        typer.secho("  ! too few points for SNR plot", fg=typer.colors.YELLOW)
        return None
    k12 = flat["K12"]
    signal = (means[m] - flat["bias_dn"]) * k12 / 16.0
    snr = (means[m] - flat["bias_dn"]) / np.sqrt(tvars[m])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(signal, snr, "+", ms=7, label="measured (flat)")
    s_curve = np.linspace(signal.min(), signal.max(), 200)
    ax.loglog(
        s_curve,
        snr_temporal_model(s_curve, flat["sigma_r_e_q"], k12),
        "-",
        lw=1.5,
        label=r"model (Eq. 21): $\mu_e/\sqrt{\sigma_d^2 + \sigma_q^2/K^2 + \mu_e}$",
    )
    ax.loglog(
        s_curve,
        np.sqrt(s_curve),
        "--",
        lw=1.5,
        color="0.4",
        label=r"ideal camera: $\sqrt{\mu_e}$",
    )
    if extras:
        ax.loglog(
            signal,
            snr_total_measured(extras, means[m], flat["bias_dn"], tvars[m], k12),
            "x",
            ms=6,
            color="tab:purple",
            label="measured total SNR (Eq. 69)",
        )
        ax.loglog(
            s_curve,
            snr_total_model(extras, s_curve, flat["sigma_r_e_q"], k12),
            "-.",
            lw=1.5,
            color="tab:purple",
            label=r"total SNR model: $\mu_e/\sqrt{\sigma_d^2+\mathrm{DSNU}^2+"
            r"\sigma_q^2/K^2+\mu_e+\mathrm{PRNU}^2\mu_e^2}$",
        )
    else:
        typer.secho(
            "  ! no EMVA extras -- Eq. 69 total SNR omitted",
            fg=typer.colors.YELLOW,
        )
    ax.set_xlabel(r"$\mu_e$ (e⁻)")
    ax.set_ylabel("SNR")
    ax.set_title(_title_with_camera(r"SNR vs $\mu_e$", camera, roi))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.grid(which="minor", ls="--", alpha=0.3)
    text = (
        f"K = {k12:.2f} e⁻/DN12"
        "\n"
        rf"$\sigma_r$ = {flat['sigma_r_e']:.2f} e⁻"
        "\n"
        rf"$\sigma_d$ (Eq. 53) = {flat['sigma_r_e_q']:.2f} e⁻"
        "\n"
        rf"$\mu_{{e,\mathrm{{sat}}}}$ = {flat['Nsat']:.0f} e⁻"
        "\n"
        rf"$\mathrm{{SNR}}_{{\max}} = \sqrt{{\mu_{{e,\mathrm{{sat}}}}}}"
        rf" = {np.sqrt(flat['Nsat']):.1f}$"
    )
    if extras:
        text += (
            "\n"
            rf"$\mu_{{e,\mathrm{{min}}}}$ (Eq. 27) = {extras['mu_min_e']:.2f} e⁻"
            "\n"
            rf"DR (Eq. 28) = {extras['dr']:.0f} ({extras['dr_db']:.1f} dB)"
            "\n"
            rf"DSNU1288 = {extras['dsnu1288_dn'] * k12 / 16.0:.2f} e⁻"
            "\n"
            f"PRNU1288 = {extras['prnu1288_pct']:.2f} %"
        )
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
    """Dark mean DN16 vs exposure, one curve per band (semilog x).

    Matches the monochrome dark_mean_vs_exposure.png style: log exposure
    axis, per-band data markers (no connecting line) with the fitted linear
    curve overlaid, and a text box per band reporting the fit equation,
    R^2, bias offset (fit intercept) and dark current (fit slope per
    second). Saves dark_mean_vs_exposure_bands.png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    n_box = 0
    for color, r in zip(_band_colors(len(sel_results)), sel_results):
        rows = r["dark_rows"]
        if len(rows) < 2:
            continue
        x = np.array([e for e, _ in rows]) * 1000
        y = np.array([s["mean"] for _, s in rows])
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        xl = np.linspace(x.min(), x.max(), 100)
        ax.semilogx(
            x,
            y,
            "o",
            ms=5,
            color=color,
            label=rf"$\lambda$ {r['wl_nm']:.0f} nm",
        )
        ax.plot(xl, slope * xl + intercept, "-", lw=1.5, color=color)
        r2_txt = rf"$R^2 = {r2:.4f}$" if np.isfinite(r2) else r"$R^2 = \text{--}$"
        if n_box < 6:  # boxes stack every 0.16 of axes height; keep them on-axes
            ax.text(
                0.03,
                0.97 - 0.16 * n_box,
                rf"$\lambda$ {r['wl_nm']:.0f} nm:  "
                rf"$\mu_{{y,\mathrm{{dark}}}} = {slope:.4g}\,t {intercept:+.2f}$ DN16"
                "\n"
                rf"{r2_txt}  $\cdot$  bias = {intercept:.2f} DN16  $\cdot$  "
                rf"$I_d = {slope * 1000:.3f}$ DN16/s",
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            )
        n_box += 1
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
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "dark_mean_vs_exposure_bands.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None


def save_dark_variance_plot_bands(sel_results, out_dir, roi=None, camera=None):
    """Dark temporal variance vs exposure, one curve per band (semilog x).

    Matches the monochrome dark_variance_vs_exposure.png style: log exposure
    axis, per-band data markers (no connecting line) for the two-frame
    (Eq. 18, primary) and N-frame variances, the fitted Eq. 30 lines
    overlaid (solid two-frame, dashed N-frame cross-check), and a text box
    per band reporting the fit equation, R^2, variance-based dark current
    (fit slope per second) and read noise (sqrt of the fit offset), plus the
    per-band DSNU1288 (Eq. 66, dark-only) when the band has a flat fit.
    Saves dark_variance_vs_exposure_bands.png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    n_box = 0
    for color, r in zip(_band_colors(len(sel_results)), sel_results):
        rows = r["dark_rows"]
        if len(rows) < 2:
            continue
        x = np.array([e for e, _ in rows]) * 1000
        y = np.array([s["tf_tvar"] for _, s in rows])
        y_nf = np.array([s["tvar"] for _, s in rows])
        ax.semilogx(
            x,
            y,
            "o",
            ms=5,
            color=color,
            label=rf"$\lambda$ {r['wl_nm']:.0f} nm",
        )
        ax.semilogx(x, y_nf, "s", ms=4, alpha=0.7, color=color)
        slope, offset = np.polyfit(x, y, 1)
        slope_nf, offset_nf = np.polyfit(x, y_nf, 1)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, np.polyval([slope, offset], xl), "-", lw=1.5, color=color)
        ax.plot(
            xl,
            np.polyval([slope_nf, offset_nf], xl),
            "--",
            lw=1.5,
            color=color,
        )
        yhat = slope * x + offset
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r2_txt = rf"$R^2 = {r2:.4f}$" if np.isfinite(r2) else r"$R^2 = \text{--}$"
        extras = r.get("extras")
        dsnu = extras["dsnu1288_dn"] if extras else None
        box_text = (
            rf"$\lambda$ {r['wl_nm']:.0f} nm:  "
            rf"$\sigma_y^2 = {slope:.4g}\,t {offset:+.2f}$ DN16$^2$"
            "\n"
            rf"{r2_txt}  $\cdot$  "
            rf"$\sigma_r = {np.sqrt(max(offset, 0.0)):.2f}$ DN16"
        )
        if dsnu is not None:
            box_text += f"\nDSNU1288 = {dsnu:.2f} DN16"
        if n_box < 6:  # boxes stack every 0.16 of axes height; keep them on-axes
            ax.text(
                0.03,
                0.97 - 0.16 * n_box,
                box_text,
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85),
            )
        n_box += 1
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
    """Bias-subtracted flat mean vs exposure, one curve per band (linear).

    Linear scales: within one band the signal spans only the exposure grid
    (the log-log span argument applies across the spectrum, not per band).
    Saturated points are marked x and excluded from the curve and the fit.
    Each band's legend entry carries its linear fit mu = a*t + b (on the
    unclipped points) and the (centered) R^2. Saves
    linearity_mean_vs_exposure_bands.png.
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
        if m.sum() < 2:
            continue
        ax.plot(
            x[m],
            y[m],
            "o",
            ms=4,
            lw=1.2,
            color=color,
        )
        if (~m).any():
            ax.plot(x[~m], y[~m], "x", ms=7, color=color, alpha=0.7)
        a, b = np.polyfit(x[m], y[m], 1)  # mu = a*t + b on unclipped points
        y_hat = a * x[m] + b
        ss_res = float(np.sum((y[m] - y_hat) ** 2))
        ss_tot = float(np.sum((y[m] - y[m].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        xl = np.linspace(x[m].min(), x[m].max(), 100)
        r2_txt = rf"$R^2={r2:.4f}$" if np.isfinite(r2) else ""
        ax.plot(
            xl,
            a * xl + b,
            "-",
            lw=1.2,
            color=color,
            label=(
                rf"$\lambda$ {r['wl_nm']:.0f} nm: "
                rf"$\mu = {a:.6g}\,t {b:+.2f}$" + (f", {r2_txt}" if r2_txt else "")
            ),
        )
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
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "linearity_mean_vs_exposure_bands.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None


def save_ptc_plot_bands(sel_results, out_dir, clip_dn, roi=None, camera=None):
    """Log-log photon transfer curve with one curve + fit per band.

    Matches the monochrome ptc_variance_vs_mean.png style: abscissa is the
    photo-induced signal mu_y - mu_y,dark and the ordinate
    dV = sigma_y^2 - sigma_y,dark^2 on the two-frame temporal variance
    (Eq. 18, primary; per-band dark variance matched per exposure, from the
    shared fit), so the fit line is dV = K*S through the origin. Per band:
    fit markers with the excluded (>70% sat) points as x, the zero-intercept
    fit line, and the 0.7*mu_s cap lines; the 2^16 clip line and a single
    text box listing the per-band R^2 and K values (no equations) complete
    the monochrome style. Saves ptc_variance_vs_mean_bands.png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    labeled_excluded = False
    capped_any = False
    first_bias = None
    r2_entries, k_entries = [], []
    for color, r in zip(_band_colors(len(sel_results)), sel_results):
        f = r["flat"]
        if not f:
            continue
        bias = f["bias_dn"]
        if first_bias is None:
            first_bias = bias
        x = np.array([s["mean"] for _, s in f["rows"]])
        v = np.array(f["dV"])
        cap = f["ptc_sat_70_dn"]
        m = (x <= cap) & (x < clip_dn) & (v > 0)
        if m.sum() < 2:
            continue
        wl = rf"{r['wl_nm']:.0f}\mathrm{{nm}}"
        r2_entries.append(rf"$R^2_{{{wl}}} = {f['ptc_r2']:.4f}$")
        k_entries.append(rf"$K_{{{wl}}} = {f['K12']:.2f}$")
        ax.loglog(
            x[m] - bias,
            v[m],
            "o",
            ms=5,
            color=color,
            label=rf"$\lambda$ {r['wl_nm']:.0f} nm",
        )
        ab = (x > cap) & (x < clip_dn) & (v > 0)
        if ab.any():
            ax.loglog(
                x[ab] - bias,
                v[ab],
                "x",
                ms=6,
                color="0.5",
                alpha=0.8,
                label="excluded (>70% sat)" if not labeled_excluded else None,
            )
            labeled_excluded = True
        xl = np.linspace(x[m].min(), x[m].max(), 100) - bias
        ax.loglog(
            xl,
            f["ptc_slope"] * xl,
            "-",
            lw=1.5,
            color=color,
        )
        if f["ptc_capped"]:
            ax.axvline(
                cap - bias,
                ls="--",
                color="tab:orange",
                lw=1.2,
            )
            capped_any = True
        drew = True
    if not drew:
        typer.secho("  ! too few points for per-band PTC plot", fg=typer.colors.YELLOW)
        plt.close(fig)
        return None
    ax.axvline(
        2**16 - first_bias,
        ls="--",
        color="tab:red",
        lw=1.2,
        label="2¹⁶ max DN16",
    )
    handles, labels = ax.get_legend_handles_labels()
    if capped_any:
        from matplotlib.lines import Line2D

        handles.insert(
            0,
            Line2D(
                [0],
                [0],
                ls="--",
                color="tab:orange",
                lw=1.2,
                label="0.7·µ_s (per band)",
            ),
        )
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    per_line = 3
    r2_lines = [
        "  ".join(r2_entries[i : i + per_line])
        for i in range(0, len(r2_entries), per_line)
    ]
    k_lines = [
        "  ".join(k_entries[i : i + per_line]) + r"  e$^-$/DN12"
        for i in range(0, len(k_entries), per_line)
    ]
    ax.text(
        0.03,
        0.97,
        "\n".join(r2_lines + k_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )
    ax.set_xlabel(r"$\mu_y - \mu_{y,\mathrm{dark}}$ (DN16)")
    ax.set_ylabel(r"$\sigma_y^2 - \sigma_{y,\mathrm{dark}}^2$ (DN16²)")
    ax.set_title(
        _title_with_camera(
            r"Photon transfer curve: $\sigma_y^2 - \sigma_{y,\mathrm{dark}}^2$ "
            r"vs $\mu_y - \mu_{y,\mathrm{dark}}$ (per band)",
            camera,
            roi,
        )
    )
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
    combined signal range of all bands, and per band the Eq. 69 total SNR
    (model curve + measured points, from the band's EMVA extras). Saves
    snr_vs_signal_bands.png.
    """
    from .analyze import (
        snr_temporal_model,
        snr_total_measured,
        snr_total_model,
    )  # local: import cycle

    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    drew_total = False
    sig_lo, sig_hi = [], []
    snr_max_entries = []
    mu_min_entries = []
    dr_entries = []
    for color, r in zip(_band_colors(len(sel_results)), sel_results):
        f = r["flat"]
        if not f or not r["dark_rows"]:
            continue
        extras = r.get("extras")
        means = np.array([s["mean"] for _, s in f["rows"]])
        tvars = np.array([s["tf_tvar"] for _, s in f["rows"]])
        m = (means < clip_dn) & (tvars > 0)
        if m.sum() < 2:
            continue
        signal = (means[m] - f["bias_dn"]) * f["K12"] / 16.0
        snr = (means[m] - f["bias_dn"]) / np.sqrt(tvars[m])
        ax.loglog(
            signal, snr, "+", ms=6, color=color, label=rf"$\lambda$ {r['wl_nm']:.0f} nm"
        )
        sc = np.linspace(signal.min(), signal.max(), 100)
        ax.loglog(
            sc,
            snr_temporal_model(sc, f["sigma_r_e_q"], f["K12"]),
            "-",
            lw=1.2,
            color=color,
        )
        if extras:
            ax.loglog(
                signal,
                snr_total_measured(extras, means[m], f["bias_dn"], tvars[m], f["K12"]),
                "x",
                ms=5,
                color=color,
                alpha=0.7,
            )
            ax.loglog(
                sc,
                snr_total_model(extras, sc, f["sigma_r_e_q"], f["K12"]),
                "-.",
                lw=1.2,
                color=color,
                alpha=0.9,
            )
            drew_total = True
        sig_lo.append(signal.min())
        sig_hi.append(signal.max())
        snr_max_entries.append(
            rf"$\mathrm{{SNR}}_{{\max,\,{r['wl_nm']:.0f}\mathrm{{nm}}}}"
            rf" = {np.sqrt(f['Nsat']):.1f}$"
        )
        if extras:
            mu_min_entries.append(
                rf"$\mu_{{e,\mathrm{{min}},\,{r['wl_nm']:.0f}\mathrm{{nm}}}}"
                rf" = {extras['mu_min_e']:.2f}$"
            )
            dr_entries.append(
                rf"$\mathrm{{DR}}_{{{r['wl_nm']:.0f}\mathrm{{nm}}}}"
                rf" = {extras['dr']:.0f}$"
            )
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
    if drew_total:
        ax.plot(
            [],
            [],
            "x",
            ms=6,
            color="0.4",
            label="measured total SNR (Eq. 69)",
        )
        ax.plot(
            [],
            [],
            "-.",
            lw=1.5,
            color="0.4",
            label="total SNR model (Eq. 69)",
        )
    ax.set_xlabel(r"$\mu_e$ (e⁻)")
    ax.set_ylabel("SNR")
    ax.set_title(_title_with_camera(r"SNR vs $\mu_e$ (per band)", camera, roi))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.grid(which="minor", ls="--", alpha=0.3)
    per_line = 3
    lines = []
    for entries in (snr_max_entries, mu_min_entries, dr_entries):
        lines += [
            "  ".join(entries[i : i + per_line])
            for i in range(0, len(entries), per_line)
        ]
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )
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
    knf = _pos([r["flat"]["K12_nf"] if r["flat"] else np.nan for r in results])
    ax.plot(wl, k, "o", ms=3, label="K (two-frame)")
    ax.plot(wl, knf, ".", ms=3, alpha=0.6, label="K (N-frame)")
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
    ax.set_ylabel(r"$I_d$ (DN16/s)")
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


def save_flat_uniformity_plot(
    imgs, wl_sel, exp_ms, n_cubes, out_dir, roi=None, camera=None
):
    """3x3 grid of the shortest flat exposure's bands with the ROI overlaid.

    imgs is the (R, C, 9) float32 frame-average per band (DN16) from
    band_analyze._load_shortest_flat_frames. Each subplot stretches its own
    1-99 percentile range (per-band contrast, so dark-edge and bright bands
    are both readable); the dashed red rectangle is the analysis ROI, drawn
    on pixel edges so it aligns with imshow. Saves
    flat_uniformity_{exp_ms:g}ms.png.
    """
    from matplotlib.patches import Rectangle  # local: matches Line2D pattern

    n_sel = imgs.shape[2]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), squeeze=False)
    r0, r1, c0, c1 = roi
    for i in range(9):
        ax = axes[i // 3][i % 3]
        if i >= n_sel:
            ax.axis("off")
            continue
        b = imgs[:, :, i]
        vmin, vmax = np.percentile(b, (1, 99))
        ax.imshow(b, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
        ax.add_patch(
            Rectangle(
                (c0 - 0.5, r0 - 0.5),
                c1 - c0,
                r1 - r0,
                fill=False,
                ec="tab:red",
                ls="--",
                lw=1.2,
            )
        )
        ax.set_title(
            rf"$\lambda$ = {wl_sel[i]:.0f} nm · mean {b.mean():.0f} DN16",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        _title_with_camera(
            f"Flat-field uniformity ({exp_ms * 1000:g} ms, avg of {n_cubes} cubes)",
            camera,
            roi,
        ),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / f"flat_uniformity_{exp_ms * 1000:g}ms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None


def save_band_mean_roi_plot(stats, wl, out_dir, kind, clip_dn, roi=None, camera=None):
    """Bar chart of the per-band ROI mean, one subplot per exposure (SPECIM).

    stats is the 'stats' dict from band_analyze._load_kind_stats:
    {exp_s: [roi_stats per band]}, each band's 'mean' being the mean pixel
    value over all frames x the ROI in DN16. Every subplot shows one
    exposure's 204 bars (x = wavelength in nm, y = mean in DN16); bands whose
    mean reached the clip level are drawn red, with the 2^16 max-DN16 line
    as reference. Saves band_mean_roi_{kind}.png.
    """
    exps = sorted(stats)
    if not exps:
        typer.secho(
            f"  ! no {kind} exposures for per-band ROI mean plot",
            fg=typer.colors.YELLOW,
        )
        return None
    n_cols = min(len(exps), 4)
    n_rows = int(np.ceil(len(exps) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.4 * n_cols, 3.6 * n_rows), squeeze=False
    )
    wl = np.asarray(wl)
    bar_w = max(0.8 * (wl[-1] - wl[0]) / max(len(wl) - 1, 1), 1.0)
    for i, e in enumerate(exps):
        ax = axes[i // n_cols][i % n_cols]
        y = np.array([s["mean"] for s in stats[e]])
        if len(y) != len(wl):
            print(
                f"  ! [{kind}] {e * 1000:g} ms: {len(y)} bands != {len(wl)} "
                "wavelengths, skipping subplot"
            )
            continue
        sat = y >= clip_dn
        ax.bar(wl, y, width=bar_w, color=np.where(sat, "tab:red", "tab:blue"))
        ax.set_title(f"{e * 1000:g} ms", fontsize=10)
        ax.set_xlim(wl[0], wl[-1])
        ax.grid(alpha=0.3, axis="y")
        if sat.any():
            ax.axhline(2**16, ls="--", color="0.4", lw=1)
            ax.plot(
                [],
                [],
                "s",
                color="tab:red",
                label=f"mean \u2265 {clip_dn} DN16",
            )
            ax.legend(fontsize=8, loc="upper left")
    for i in range(len(exps), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")
    fig.suptitle(
        _title_with_camera(f"ROI mean per band ({kind})", camera, roi),
        fontsize=11,
    )
    fig.supxlabel("wavelength (nm)")
    fig.supylabel(r"$\mu_y$ (DN16)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / f"band_mean_roi_{kind}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.secho(f"  [plot] saved {path}", fg=typer.colors.GREEN)
    return None
