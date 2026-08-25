"""Saving acquired stacks + metadata, plus analysis constants shared by the
CLI, the fits and the plots (kept here because every consumer already
imports this module -- importing them from analyze.py would create cycles).

DN convention: frames are 12-bit data stored as DN12 << 4 ("DN16").
"""

import json
import time
from pathlib import Path

import numpy as np
import typer

DEFAULT_ROI_FRAC = 0.5  # default analysis ROI: central fraction of each dimension
DEFAULT_ROI_SPECIM = (140, 290, 156, 306)  # SPECIM IQ ROI, see uniformity plot
SAT_CLIP_FRAC = 0.002  # EMVA R4 Linear 6.6: saturation = <= 0.2% pixels at max


def central_roi(width, height, frac=DEFAULT_ROI_FRAC):
    """(r0, r1, c0, c1) covering the central ``frac`` of a width x height frame (pure).

    The side length is rounded down to an even pixel count so the four Bayer
    sub-lattices of a raw-Bayer dataset stay balanced, with a minimum of 2 px
    per side; the window is centered on the frame ((n - size) // 2).
    """

    def span(n):
        size = int(n * frac)
        size -= size % 2
        size = max(size, 2)
        start = (n - size) // 2
        return start, start + size

    r0, r1 = span(height)
    c0, c1 = span(width)
    return (r0, r1, c0, c1)


def stem_for(seq_type, exposure_s, gain):
    """Filename stem for an exposure: 'dark_00100ms_g0' or 'flat_004500us_g0'.

    Integer ms >= 1 -> ms names (backward compatible with existing data);
    everything else -> microsecond names (sub-ms / fractional-ms are unique).
    """
    exp_ms = exposure_s * 1000
    is_integer_ms = abs(exp_ms - round(exp_ms)) < 1e-9
    if exp_ms >= 1 and is_integer_ms:
        return f"{seq_type}_{int(round(exp_ms)):05d}ms_g{gain:g}"
    return f"{seq_type}_{int(round(exp_ms * 1000)):05d}us_g{gain:g}"


def camera_dir_name(info):
    """Folder name for a camera: 'playerone_Apollo-M_(IMX174)'.

    info is the camera info dict from backend.open() or a metadata entry
    (needs 'vendor', 'model', 'sensor' keys). Spaces become '-'.
    """
    name = "_".join(
        str(info.get(k, "")).strip()
        for k in ("vendor", "model")
        if str(info.get(k, "")).strip()
    )
    sensor = str(info.get("sensor", "")).strip()
    if sensor:
        name += f"_({sensor})"
    return name.replace(" ", "-")


def save_sequence(
    out_dir,
    seq_type,
    exposure_s,
    gain,
    stack,
    info,
    notes="",
    temp_start_c=None,
    temp_c=None,
):
    """Save one exposure's frames + a metadata JSON.

    out_dir : Path        output directory (created if missing)
    seq_type: str         'dark' or 'flat'
    stack   : np.ndarray  (n_frames, h, w) uint16
    info    : dict        camera info dict from backend.open()
    temp_start_c, temp_c: sensor temp before/after the exposure (optional)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = stem_for(seq_type, exposure_s, gain)
    np.save(out_dir / f"{stem}.npy", stack)

    meta = {
        "sequence": seq_type,
        "exposure_s": exposure_s,
        "gain": gain,
        "n_frames": int(stack.shape[0]),
        "width": int(stack.shape[2]),
        "height": int(stack.shape[1]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "temp_start_c": temp_start_c,
        "temp_c": temp_c,
        "notes": notes,
    }
    meta.update(info)
    meta_path = out_dir / "metadata.json"
    entries = []
    if meta_path.exists():
        entries = json.loads(meta_path.read_text())
    entries.append(meta)
    meta_path.write_text(json.dumps(entries, indent=2))

    # quick quality report
    means = stack.mean(axis=(1, 2))
    stds = stack.std(axis=(1, 2))
    typer.echo(
        f"  saved {stem}.npy: {stack.shape}  frame mean {means.mean():.2f} "
        f"(±{means.std():.2f} across frames), spatial std {stds.mean():.2f}"
    )
