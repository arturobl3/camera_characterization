"""SPECIM IQ hyperspectral ENVI data discovery/loading for per-band analysis.

Layout (as exported by the camera):

    <root>/dark frames/<N> ms>/<ID>/capture/<ID>.hdr + <ID>.raw
    <root>/flat-field frames/<N> ms>/<ID>/capture/<ID>.hdr + <ID>.raw

Cubes are 512x512x204 uint16 BIL little-endian, 12-bit DN, with the band
wavelengths in the .hdr 'wavelength' entry (~397-1004 nm). Exposure comes
from the '<N> ms' folder name and is cross-checked against the .hdr 'tint'
entry (they matched in all acquisitions on disk).

The per-capture DARKREF_*/WHITEDARKREF_*/WHITEREF_* files are 512x1x204
slabs (the camera's stored internal calibrations, not per-frame data) and
are deliberately ignored: temporal statistics need the raw cubes. Cube
headers are the .hdr files whose stem is all digits.

DN scaling: raw 12-bit DN is shifted << 4 into DN16 (the same DN12<<4
storage the playerone pipeline uses), so every downstream constant
(CLIP_DN, SAT_MAX_DN, quantization step, K12 = 16/K_fit) is reused
unchanged.
"""

import re
from pathlib import Path

import numpy as np
import spectral.io.envi as envi

DARK_DIR = "dark frames"
FLAT_DIR = "flat-field frames"
RAW_MAX_DN = 4095  # 12-bit full scale; values above this cannot be <<4 safely
DN16_SHIFT = 4  # 12-bit DN -> DN16 (DN12 << 4)

_EXPOSURE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*ms\s*$", re.IGNORECASE)


def is_specim_dir(d):
    """True when d holds a SPECIM IQ 'dark frames'/'flat-field frames' layout."""
    d = Path(d)
    return (d / DARK_DIR).is_dir() or (d / FLAT_DIR).is_dir()


def kind_dir(root, seq_type):
    return Path(root) / (DARK_DIR if seq_type == "dark" else FLAT_DIR)


def list_exposures(kdir):
    """Sorted [(exposure_s, exp_dir)] -- one entry per '<N> ms' subfolder."""
    out = []
    kdir = Path(kdir)
    if not kdir.is_dir():
        return out
    for sub in sorted(kdir.iterdir()):
        if not sub.is_dir():
            continue
        m = _EXPOSURE_RE.match(sub.name)
        if not m:
            print(f"  ! skipping unrecognized folder: {sub}")
            continue
        out.append((float(m.group(1)) / 1000.0, sub))
    out.sort(key=lambda x: x[0])
    return out


def iter_cubes(exp_dir):
    """Sorted capture cube .hdr paths under <exp_dir>/<ID>/capture/.

    Only all-digit stems are cubes; DARKREF_*/WHITEREF_*/WHITEDARKREF_* are
    the camera's internal calibration slabs and are skipped.
    """
    return [h for h in sorted(exp_dir.glob("*/capture/*.hdr")) if h.stem.isdigit()]


def open_cube(hdr_path):
    """envi image for a capture header (data file is the sibling .raw)."""
    return envi.open(str(hdr_path), str(hdr_path.with_suffix(".raw")))


def wavelengths_of(img):
    wl = img.metadata.get("wavelength") or []
    return np.array([float(w) for w in wl])


def load_exposure_stack(exp_dir, exp_s, r0, r1, c0, c1):
    """All cubes of one exposure as a DN16 uint16 stack (n, R, C, bands).

    Reads each cube once, cropped to the ROI. Returns (stack, wavelengths)
    or (None, None) when the folder holds no capture cubes.
    """
    hdrs = iter_cubes(exp_dir)
    if not hdrs:
        print(f"  ! no capture cubes in {exp_dir}")
        return None, None

    imgs, n_bands = [], None
    for h in hdrs:
        img = open_cube(h)
        tint = img.metadata.get("tint")
        if tint is not None:
            try:
                tint_ms = float(tint)
            except ValueError:
                tint_ms = None
            if tint_ms is not None and abs(tint_ms - exp_s * 1000.0) > 1e-6:
                print(
                    f"  ! {h.stem}: hdr tint {tint_ms:g} ms != folder "
                    f"{exp_s * 1000:g} ms"
                )
        if n_bands is None:
            n_bands = int(img.shape[2])
        elif int(img.shape[2]) != n_bands:
            print(f"  ! {h.stem}: {img.shape[2]} bands != {n_bands}, skipping cube")
            continue
        imgs.append(img)
    if not imgs:
        return None, None

    wl = wavelengths_of(imgs[0])
    if len(wl) != n_bands:
        print(f"  ! wavelength list has {len(wl)} entries != {n_bands} bands")
        return None, None

    rows, cols = list(range(r0, r1)), list(range(c0, c1))
    out = np.empty((len(imgs), r1 - r0, c1 - c0, n_bands), dtype=np.uint16)
    for i, img in enumerate(imgs):
        cube = np.array(img.read_subimage(rows, cols))  # copy: view is read-only
        if int(cube.max()) > RAW_MAX_DN:
            print(
                f"  ! cube {i}: max DN {int(cube.max())} > {RAW_MAX_DN} "
                "(not 12-bit?) -- values may wrap on the <<4 shift"
            )
        np.left_shift(cube, DN16_SHIFT, out=cube)  # in place; safe for <= 4095
        out[i] = cube
    return out, wl
