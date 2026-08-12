"""Plot generation for camchar analyze output (saved into outputs/)."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_linearity_plot(rows, bias_dn, out_dir, clip_dn, roi=None):
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
    ax.set_xlabel("Exposure (ms)")
    ax.set_ylabel("Mean DN16 (bias-subtracted)")
    title = "Linearity: flat mean DN vs exposure"
    if roi:
        title += f" (ROI rows {roi[0]}:{roi[1]}, cols {roi[2]}:{roi[3]})"
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.text(
        0.03,
        0.97,
        f"y = {slope:.2f}·t {intercept:+.2f} DN16\nR² = {r2:.6f}",
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
