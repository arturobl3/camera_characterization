"""Plot generation for camchar analyze output (saved into outputs/)."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_ROI = (600, 800, 850, 1050)


def _title_with_camera(base, camera, roi):
    title = base
    if camera:
        title += "\n" + camera
    if roi and tuple(roi) == _DEFAULT_ROI:
        title += " (central 200x200 px ROI)"
    elif roi:
        title += f" (ROI rows {roi[0]}:{roi[1]}, cols {roi[2]}:{roi[3]})"
    return title


def save_dark_plot(rows, out_dir, roi=None, camera=None):
    """Mean dark DN vs exposure with the dark-current linear fit.

    rows is the 'rows' list from analyze_dark: (exposure_s, stats) with
    stats['mean']. Saves dark_mean_vs_exposure.png.
    """
    x = np.array([e for e, _ in rows]) * 1000
    y = np.array([s["mean"] for _, s in rows])
    if len(x) < 2:
        print("  ! too few dark points for dark plot")
        return None
    slope, intercept = np.polyfit(x, y, 1)
    yhat = np.polyval([slope, intercept], x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "o", ms=5, label="dark (measured)")
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
    print(f"  [plot] saved {path}")
    return {"slope": slope, "intercept": intercept, "r2": r2}


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
        print("  ! too few unclipped points for linearity plot")
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
        0.03,
        0.97,
        rf"$\mu_y - \mu_{{y,\mathrm{{dark}}}} = {slope:.2f}\,t {intercept:+.2f}$ (DN16)"
        "\n"
        f"R² = {r2:.6f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "linearity_mean_vs_exposure.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved {path}")
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
    fit_mask = (fx < clip_dn) & (fv > 0)
    if fit_mask.sum() < 2:
        print("  ! too few points for PTC plot")
        plt.close(fig)
        return None
    ax.loglog(fx[fit_mask], fv[fit_mask], "o", ms=5, label="flat (unclipped)")

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
    print(f"  [plot] saved {path}")
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
        print("  ! too few points for SNR plot")
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
    print(f"  [plot] saved {path}")
    return None
