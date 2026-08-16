"""Saving acquired stacks + metadata."""

import json
import time
from pathlib import Path

import numpy as np


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
    print(
        f"  saved {stem}.npy: {stack.shape}  frame mean {means.mean():.2f} "
        f"(±{means.std():.2f} across frames), spatial std {stds.mean():.2f}"
    )
