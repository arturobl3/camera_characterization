# camchar data formats

This document describes every file format camchar reads and writes, how
raw-Bayer (color) acquisitions differ from monochrome ones, and why all
12-bit-sensor data is stored left-shifted by 4 ("DN16") rather than as native
ADC codes.

## Directory layout

```
data/                                  # acquisition root (--out, default 'data')
├── playerone_Apollo-M_(IMX174)/
│   ├── dark/
│   │   ├── dark_00010ms_g0.npy        # one uint16 stack per exposure
│   │   └── metadata.json              # append-only array of entries
│   └── flat/
│       ├── flat_004500us_g0.npy
│       └── metadata.json
└── basler_a2A3536-31ucBAS_(IMX676)/   # same structure for any vendor

outputs/                               # analysis artifacts (per camera)
└── playerone_Apollo-M_(IMX174)/*.png  # PTC, SNR, linearity, dark plots
```

- The camera folder name comes from `camchar/io_utils.py:camera_dir_name()`:
  `<vendor>_<model>_(<sensor>)`, spaces replaced by `-`. The legacy layout
  (`data/dark`, no camera folder) is still accepted by `analyze`.
- Plot filenames are stable (`ptc_variance_vs_mean.png`,
  `dark_mean_vs_exposure.png`, ...); rerunning `analyze` overwrites them.
- SPECIM IQ hyperspectral input is read-only (ENVI/BIL export); its outputs
  are `outputs/SPECIM_IQ/band_parameters.csv` plus per-band plot variants.
  camchar never writes hyperspectral cubes.

## Frame files (.npy)

One file per (sequence type, exposure): `{dark|flat}_{NNNNN}{unit}_g{gain}.npy`
containing a **uint16 stack of shape `(n_frames, height, width)`**, written
with `np.save`. Stacks (not per-frame files) because every statistic the
toolchain computes is temporal: two-frame Eq. 18 variances, N-frame ddof=1
variances, frame-mean drift checks.

Filename rules (`camchar/io_utils.py:stem_for()`):

| exposure | name |
|---|---|
| integer ms ≥ 1 | `dark_00010ms_g0` |
| sub-ms / fractional | `flat_004500us_g0` |

The microsecond form keeps fractional exposures unique. Gain is formatted
with `%g` (`g0`, `g6`, `g0.5`). Backends may clamp requested exposures to the
camera range; the *effective* exposure is what lands in the filename and
metadata (`last_exposure_s`). Note that clamped exposures can collide into
the same filename (silent overwrite) — avoid sweeping below the camera
minimum.

`analyze` reconstructs filenames from `metadata.json` through `stem_for()`;
if you rename or move `.npy` files by hand, keep the metadata consistent.

## metadata.json

Each sequence directory holds an **append-only JSON array** with one entry
per saved stack:

```json
{
  "sequence": "dark",
  "exposure_s": 0.01,
  "gain": 0.0,
  "n_frames": 10,
  "width": 1920,
  "height": 1200,
  "timestamp": "2026-08-23T22:05:43+0300",
  "temp_start_c": 45.2,
  "temp_c": 45.3,
  "notes": "lens cap on",
  "vendor": "basler",
  "model": "a2A3536-31ucBAS",
  "serial": "41952064",
  "sensor": "IMX676",
  "pixel_size_um": 2.0,
  "bit_depth": 12,
  "usb3": true,
  "pixel_format": "BayerRG12",
  "color": true,
  "black_level_node": 20.0,
  "black_level_dn12": null
}
```

Common keys (written by `save_sequence`) plus everything in the backend's
camera-info dict merged on top. Because the file only grows, it accumulates
duplicate entries across runs; `load_sequence()` treats the `.npy` files as
ground truth and deduplicates entries by `(exposure_s, gain)`.

Vendor-specific extras you may see:

| key | meaning | vendors |
|---|---|---|
| `pixel_format` | GenICam-style format string recorded at configure time | basler |
| `color` | `true` when the dataset is raw Bayer | basler |
| `black_level_node` | applied BlackLevel register value (units are model-specific) | basler ace 2 |
| `black_level_dn12` | estimated digital offset in 12-bit LSB (null when not applicable/measurable) | all |
| `gain_db` | dB equivalent of an integer gain index | thorlabs |

`width`/`height` describe the **acquisition AOI**. On Basler cameras the
Default user set's geometry is kept rather than forcing full-sensor size,
which can include optical-black rows (e.g. the a2A3536 sensor is 3536×3552
but acquires 3536×3536).

## Color cameras: raw Bayer stacks

Color sensors are acquired in their **native raw Bayer format** (e.g.
`BayerRG12`) and stored exactly like monochrome stacks — there is no
demosaicing anywhere in the toolchain. Each pixel keeps its CFA position,
so:

- **Temporal statistics need no special handling**: every pixel is compared
  only with itself across frames, regardless of which filter it sits under.
  K, σr, dark current, linearity and saturation are all computed unchanged.
- **Spatial statistics must not mix channels**: the 2×2 Bayer period survives
  the EMVA §8.1 highpass and would inflate DSNU/PRNU several-fold. When the
  dataset is Bayer (any `metadata.json` entry whose `pixel_format` starts
  with `"Bayer"`), the DSNU/PRNU cores pool the four CFA sub-lattices in the
  variance domain and print per-channel values (`R,G1,G2,B` labels follow the
  recorded layout).
- Never switch a color model to its `Mono*` pixel formats for measurements:
  those outputs are interpolated (demosaiced / green-upsampled) and destroy
  both temporal and spatial noise statistics.

## DN16: storing 12-bit ADCs left-shifted by 4

All three supported sensor families digitize to **12 bits** (Player One RAW16,
Basler `Mono12`/`BayerRG12`, Thorlabs DN12 frames). camchar stores every
frame as

```
stored_value (uint16) = native_DN12 << 4      # == DN12 * 16, called "DN16"
```

so the uint16 container is filled MSB-first and every vendor lands in the
same numeric scale at the backend boundary (`snap()`), before anything
downstream sees the data.

Why this convention instead of keeping native codes:

- **One scale for everything.** EMVA thresholds, clip levels, plot axes and
  fits are defined once in `analyze.py` and stay valid across vendors.
- **Quantization stays explicit.** One 12-bit LSB equals
  `QUANT_STEP_DN16 = 16` DN16; the quantization variance is
  σq² = 16²/12 ≈ 21.3 DN16² — the exact term subtracted in the
  Eq. 53 quantization-corrected read noise.
- **Saturation arithmetic is unambiguous.** Native 4094 maps to
  65 504 DN16 (`SAT_MAX_DN`, 4095 reserved as a flag); anything ≥ 65 400 DN16
  (`CLIP_DN`) counts as clipped when filtering PTC points.

Scale conversions to keep in mind:

| quantity | factor going DN12 → DN16 |
|---|---|
| signal / mean / readout value | ×16 |
| variance / σ² | ×256 |
| standard deviation σ | ×16 |
| quantization step (1 LSB) | ×16 |

The system gain K is reported in **e⁻/DN12** even though the variance-vs-
signal regression runs on DN16 data (`K12 = 16 / K_fit`): electrons per
*native* ADC count is what datasheets quote, so mixing scales when comparing
against a datasheet is the classic mistake. Worked example from a real dark
frame (a2A3536, BlackLevel register 20): stored mean 3716 DN16 = 232 DN12 of
black-level offset — a real pedestal, not zeros.

## Loading data outside the CLI

```python
import json
from pathlib import Path

import numpy as np

cam = Path("data/basler_a2A3536-31ucBAS_(IMX676)")
stack = np.load(cam / "flat" / "flat_00010ms_g0.npy")   # (n, h, w) uint16, DN16
entries = json.loads((cam / "flat" / "metadata.json").read_text())

dn12 = stack.astype(np.float64) / 16.0                   # back to native ADC units
```

Keep in mind: values ≥ 65 504 DN16 are saturated, ~65 400 is the practical
clip level, and a black-level pedestal (typically 80–370 DN16 depending on
camera) sits under every flat measurement — subtract the bias before
interpreting signal levels.
