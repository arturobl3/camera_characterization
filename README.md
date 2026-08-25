# camchar — camera PTC characterization toolkit

Acquisition CLI for photon-transfer-curve (PTC) camera characterization, following
EMVA 1288 / Janesick methodology. Vendor-agnostic backend design; currently
implements the **Player One Astronomy** backend (tested with Apollo-M / IMX174 on
macOS, SDK 3.10.1) and the **Basler** backend via pypylon (tested with
acA1920-155um / IMX174 over USB3 on Windows, pypylon 26.7).

> **New here?** Start with the step-by-step guide at
> [`docs/emva-noise-characterization-guide.md`](docs/emva-noise-characterization-guide.md)
> — the full EMVA 1288 pipeline (dark/flat acquisition, gain K, read noise,
> saturation, DSNU/PRNU, SNR/DR), with the pitfalls we hit and the camchar
> commands for each step.

## Quick start (macOS)

```bash
# prerequisites
brew install libusb
uv sync                      # creates .venv, installs camchar + numpy (uv.lock)

# warm up to steady-state operating temperature (auto-stops once stable)
uv run camchar warmup-sensor --vendor playerone

# dark frames: lens cap ON, dark room
uv run camchar get-dark-frames --vendor playerone \
    --exposures 1,10,100,500,2000 --frames 20 --gain 0
# flat frames: uniform broadband illumination (halogen + diffuser, or LED flat panel)
uv run camchar get-flat-frames --vendor playerone \
    --exposures 10,50,100,500 --frames 20 --gain 0 \
    --notes "green LED ~530nm, diffuser, 30 cm"

# analysis: temporal PTC -> K, read noise, Nsat, dark current, PRNU, linearity
uv run camchar analyze --data data
```

`python -m camchar ...` also works (same package). The console script and
module are interchangeable; on Windows lab machines run the same commands from
a `uv sync`'d checkout. `--vendor` (playerone | basler) is **required** on the
four acquisition commands (`analyze` is offline and takes none); Basler
`--gain` is in dB (float, 0–24), Player One gain is an integer.

Acquired sequences are organized per camera:
`<data root>/<vendor>_<model>_(<sensor>)/{dark,flat}/` (default root `data`,
override with `--out`), e.g. `data/playerone_Apollo-M_(IMX174)/dark/`.
Each exposure writes `{seq}_{exp}ms|us_g{gain}.npy` (uint16 stack, n×h×w) plus an
append-only `metadata.json` with camera info, exposure, gain, sensor temp
(`temp_start_c`/`temp_c`, before/after each exposure), notes.
Sub-ms / fractional-ms exposures use microsecond names (e.g. `flat_004500us`).

## CLI

```
camchar get-dark-frames --vendor playerone [--out ROOT] [--exposures MS,MS,...] [--frames N] [--gain G] [--notes TXT]
camchar get-flat-frames --vendor playerone [--out ROOT] [--exposures MS,MS,...] [--frames N] [--gain G] [--notes TXT]
camchar analyze --data DIR [--roi r0:r1:c0:c1] [--bands N]
camchar warmup-sensor --vendor basler
camchar source-stability-check --vendor basler [--exposures MS,MS,...] [--frames N] [--gain G]
camchar check-saturation --vendor basler [--exposures MS,MS,...] [--gain G]
```

Exposure times are given in milliseconds (fractional values allowed). Defaults: dark
and flat share 50 log-spaced exposures from 0.1–1000 ms (np.logspace), 10 frames each,
gain 0.
`--out` defaults to `data`; frames are written to
`<out>/<vendor>_<model>_(<sensor>)/dark|flat/`.
`warmup-sensor` runs continuous 0.1 s exposures and prints the timestamped sensor
temperature; it declares the temperature stable once it stays within 0.3 °C over the
last 2 min and auto-stops 1 min after that (Ctrl+C to stop earlier). Run it before
acquiring so darks/flats are taken at thermal equilibrium.
`source-stability-check` captures 4 consecutive frames at each exposure (default
0.01/0.1/1/10/100/1000 ms, spanning the camera's 10 µs–1 s default sweep) and prints the percent deviation of each frame's spatial mean versus
the first frame (the reference); any deviation above 0.1% triggers a per-exposure
warning and exit code 1. Run it on the flat-field source before acquisition —
different exposures probe different fluctuation timescales (short exposures see
fast ripple, long exposures catch slow drift).
`check-saturation` records 1 frame per default-sweep exposure and stops at the first
one whose clipped-pixel fraction trips the EMVA criterion (≥ 0.2 % of pixels at
65504 DN16 — the same bar `analyze` applies to PTC points). Ideal flat setup is
saturation beginning exactly at the 3rd-to-last sweep exposure (~687 ms on the
default grid): that verdict exits 0; anything else exits 1 and prints the exact
intensity multiplier to apply (extrapolated from a through-origin mean-vs-exposure
fit when nothing saturates during the sweep). Run it while tuning the light source,
before flat acquisition.
`analyze` accepts the data root (`data`) or a camera dir directly
(`data/playerone_Apollo-M_(IMX174)`); a single camera dir under the root is
auto-discovered. It reads `<camera>/dark/` + `<camera>/flat/` from metadata.json,
reports K (e⁻/DN12), read noise (e⁻, from darks, with an EMVA Eq. 53
quantization-corrected value), dark current (DN/s, from mean and variance),
N_sat (e⁻), PRNU (%), bias floor, per-point gain check, exposure linearity,
EMVA-style saturation / absolute sensitivity threshold / dynamic range, and
highpass-filtered PRNU1288/DSNU1288 (EMVA 1288 §8.1), and writes plots
(dark mean + dark variance vs exposure, linearity, PTC, SNR; the SNR plot adds the
EMVA Eq. 69 total-SNR curve and measured points including DSNU1288/PRNU1288) to
`outputs/<vendor>_<model>_(<sensor>)/`. ROI defaults
to a central 400×400 patch (500:900:750:1150); only the ROI needs uniform
illumination.
The **EMVA 1288 Release 4 two-frame method is the primary estimator** (Eq. 18
temporal variance with common-mode correction, Eq. 32 spatial covariance,
averaged over consecutive pairs): adjacent-pair differences reject
between-acquisition drift, so K/σr/dV/PRNU all come from it. The N-frame
values are the cross-check (`nf` columns and lines) and must agree to <~1%
for a stationary source; a systematic gap signals drift between
acquisitions. Temporal variance uses ddof=1 (EMVA R4 Linear Eq. 65): the ddof=0 population estimator is biased
low by (N−1)/N (5% at N=20) — this exact bug was caught by the two-frame
cross-check in the Aug 2026 Apollo-M dataset (K 8.91 → 8.46 e⁻/DN12).

## SPECIM IQ hyperspectral (per-band analysis)

`analyze` also handles SPECIM IQ exports. Point `--data` at the folder
holding the camera's `dark frames/` and `flat-field frames/` directories
(`data/SPECIM_IQ` here); with several cameras under one root, pass the
camera dir explicitly:

```bash
uv run camchar analyze --data data/SPECIM_IQ            # all 204 bands
uv run camchar analyze --data data/SPECIM_IQ --bands 7  # 7 curves per plot
```

- **Layout**: `<root>/<kind>/<N> ms>/<ID>/capture/<ID>.hdr|.raw` (ENVI, BIL,
  uint16, 512×512×204, 397–1004 nm). Exposure time comes from the `<N> ms`
  folder name, cross-checked against the header `tint`. The per-capture
  `DARKREF_*`/`WHITEREF_*`/`WHITEDARKREF_*` files are the camera's stored
  512×1×204 calibration slabs and are deliberately ignored — temporal
  statistics need the raw cubes.
- **DN scaling**: raw 12-bit DN is shifted `<<4` into DN16, so every
  constant and formula of the monochrome pipeline (clip 65400, saturation
  65504, quantization step 16, K12 = 16/K_fit) applies unchanged.
- **Per-band analysis**: every band is analyzed as an independent
  monochrome camera with the same EMVA R4 Linear math (same estimators,
  ddof conventions, two-frame cross-checks). PTC-fit points must pass the
  EMVA §6.6 saturation test (≤0.2% clipped pixels) — not just the mean
  clip threshold, since heavily pinned pixels lose their variance and bend
  the fit — and lie within the R4 Linear regression range (minimum value to
  70% of the measured saturation). Bands with <3 usable flat points (dim UV end, saturated bands)
  are `skipped`; bands whose V-vs-S slope comes out non-positive
  (saturation-collapsed variance and/or per-acquisition shutter/lamp jumps —
  with the two-frame primary the between-acquisition drift that used to bend
  the N-frame variance no longer affects the fit) are flagged `degenerate PTC`.
  A systematic two-frame-vs-N-frame K gap is the drift indicator.
- **Output**: one-line-per-band table, an EMVA detail block per plotted
  band, `outputs/SPECIM_IQ/band_parameters.csv` (every parameter × 204
  bands), the same plot set as the monochrome path with one curve per
  selected band (`--bands`, default 5, equispaced), plus a
  parameters-vs-wavelength summary figure (K, σr, dark current,
  PRNU/DSNU1288).
- ROI defaults to the central 200×200 (156:356:156:356) of the 512×512
  frame; `--roi` overrides.

## Project layout

```
camera_characterization/
├── camchar/
│   ├── cli.py                 # argparse CLI (get-dark-frames | get-flat-frames | analyze)
│   ├── analyze.py             # temporal PTC analysis (K, σr, Nsat, PRNU, linearity)
│   ├── band_analyze.py        # per-band EMVA analysis for hyperspectral data
│   ├── specim.py              # SPECIM IQ ENVI discovery/loading (12-bit -> DN16)
│   ├── plots.py               # matplotlib figures (Agg), monochrome + per-band
│   ├── io_utils.py            # npy + metadata.json saving, stem_for() naming
│   └── backends/
│       ├── base.py            # CameraBackend ABC
│       ├── playerone.py       # Player One backend (pitfalls encoded)
│       ├── basler.py          # Basler backend (pypylon, software trigger)
│       ├── thorlabs.py        # Thorlabs backend (TLCamera SDK, per-frame trigger)
│       └── __init__.py        # backend registry
├── vendor/
│   ├── playerone/
│   │   ├── lib/               # SDK binary for YOUR OS (see below)
│   │   └── py/                # vendored pyPOACamera.py (platform-aware loading)
│   └── thorlabs/
│       ├── lib/               # TLCamera SDK native DLLs (Windows x64 subset)
│       └── py/thorlabs_tsi_sdk/  # vendored official Python wrapper (0.0.8)
├── pyproject.toml             # uv-managed: uv sync, uv run camchar ...
├── data/                      # acquired sequences (gitignored)
│   ├── playerone_Apollo-M_(IMX174)/
│   │   ├── dark/              # .npy stacks + metadata.json
│   │   └── flat/
│   └── SPECIM_IQ/             # hyperspectral export (gitignored, ~48 GB)
│       ├── dark frames/<N> ms/<ID>/capture/
│       └── flat-field frames/<N> ms/<ID>/capture/
└── outputs/                   # plots per camera (gitignored)
    ├── playerone_Apollo-M_(IMX174)/
    └── SPECIM_IQ/             # + band_parameters.csv
```

Add a vendor: implement `CameraBackend` in `camchar/backends/<vendor>.py`, decorate
with `@register("<vendor>")`, done.

Data formats — `.npy` stacks, `metadata.json` fields, raw-Bayer handling for
color cameras, and the DN16 (12-bit << 4) storage convention — are documented
in [docs/file_formats.md](docs/file_formats.md).

## Player One SDK setup per OS

Put the SDK binary for your OS in `vendor/playerone/lib/`:

- **macOS** — from `PlayerOne_Camera_SDK_MacOS_V3.10.1.tar.gz`:
  copy `lib/libPlayerOneCamera.3.10.1.dylib` plus the three symlinks
  (`libPlayerOneCamera.dylib`, `.3.dylib`, `.3.10.dylib`). Then patch libusb:
  ```bash
  install_name_tool -change @rpath/libusb-1.0.0.dylib \
      /usr/local/opt/libusb/lib/libusb-1.0.0.dylib \
      vendor/playerone/lib/libPlayerOneCamera.3.10.1.dylib
  ```
- **Windows** — from `PlayerOne_Camera_SDK_Windows_V3.10.1.zip`:
  copy `lib/x64/PlayerOneCamera.dll` into `vendor/playerone/lib/`.
  Install the Player One camera driver from their software page (WinUSB).
- **Linux** — from the Linux SDK tarball: copy the four `.so` files for your arch.

## Basler (pypylon) setup

`uv add pypylon` — the wheel bundles the pylon runtime (Windows/macOS/Linux).
If the camera does not enumerate, install the pylon Software Suite to provide
the USB3 driver. The backend grabs `Mono12` (mono models) or **raw unpacked
BayerRG12** (color models) and shifts `<<4` to DN16, so the stored data and
every downstream constant match the Player One path.

Color models (tested: a2A3536-31ucBAS / IMX676): their `Mono*` pixel formats
are interpolated (demosaiced/green-upsampled) and never used; raw Bayer data
keeps temporal metrics exact per photosite, and `analyze` automatically pools
the four CFA sub-lattices for the spatial DSNU/PRNU so the Bayer pattern
cannot inflate them (per-channel values are printed).

Per-frame software triggering is used (no stale free-run buffers when
the exposure changes between snaps); the camera's stored user set is reset to
Default on `configure()` so pylon Viewer leftovers cannot leak in, and the
loaded geometry is kept (full-sensor size can include optical-black rows).
Exposures outside the camera range (21 µs on the acA1920-155um, 12 µs on the
a2A3536) are clamped with a warning, and the effective exposure is what lands
in metadata and filenames.

## Thorlabs (TLCamera SDK) setup

The runtime subset of Thorlabs' *Scientific Camera Interfaces* SDK is vendored:
native DLLs in `vendor/thorlabs/lib/` (camera sdk + zelux USB device + hotplug
monitor + logger, ~700 KB; the 180 MB color/polarization processors a mono
camera never loads are omitted) and the official Python wrapper in
`vendor/thorlabs/py/thorlabs_tsi_sdk/`. Windows x64 binaries only right now.
Install ThorCam once for the USB driver (WinUSB); after that no Thorlabs
software needs to run. Verified on the Kiralux LP126MU:

- frames arrive **right-aligned DN12** in uint16 and are shifted `<<4` to DN16
  at acquisition (same convention as basler/playerone);
- `--gain` is an integer gain index (0..480 on the LP126MU; the dB equivalent
  is recorded as `gain_db` in metadata);
- exposures outside 28 µs .. 14.7 s are clamped with a warning;
- **no sensor-temperature API exists in this SDK** — `warmup-sensor` refuses
  to run against thorlabs cameras (thermal stability cannot be verified);
  warm the camera up for a fixed time manually instead;
- hot-pixel correction is disabled by `configure()` (it substitutes neighbor
  values and would corrupt variance statistics).

Color models (tested: LP126CU) work through the same path: the SDK delivers
**raw right-aligned Bayer** (no demosaicing — its color processor is opt-in
and unused), `metadata.json` records `pixel_format` (e.g. `"BayerBG12"` for
the LP126CU, whose origin pixel is blue), and `analyze` automatically pools
the CFA sub-lattices for DSNU/PRNU like on Basler color models. On Bayer
datasets `analyze` additionally runs a **per-channel pass** (SPECIM-style):
every CFA sub-lattice goes through the same fits independently, giving one
curve per channel (R/G1/G2/B) in the `*_bands.png` plot variants plus
`cfa_parameters.csv` — additive to the pooled headline numbers.

Full SDK distribution (docs, installers, examples) lives untracked under
`Scientific Camera Interfaces/`.

## Known pitfalls (empirical, macOS SDK 3.10.1)

1. **Vendor Python wrapper ABI bug.** `POASetConfig`/`POAGetConfig` in the
   vendored `pyPOACamera.py` pass `c_int` where the C API takes an 8-byte
   `POAConfigValue` union. Config writes are silently corrupted (exposure stuck at
   its 10 µs minimum, readback garbage). The backend bypasses the wrapper and
   calls the DLL directly with a proper ctypes `Union` — do not "fix" the wrapper
   to route through it.
2. **`CloseCamera()` wedges the device on macOS.** After a clean close, the next
   process gets `POA_ERROR_INVALID_ID` until the camera is physically replugged.
   The backend never calls `CloseCamera`; process exit does the kernel cleanup.
   (On Windows this may not apply — verify before relying on it.)
3. **USB enumeration is flaky.** The device periodically drops and re-enumerates;
   a hub makes it worse (intermittent `INVALID_INDEX`/`INVALID_ID`). The backend
   retries enumeration at `open()`. Prefer a direct USB connection.
4. **Exposure config:** use `POA_EXP` (config 31, seconds, float) with fallback to
   `POA_EXPOSURE` (config 0, µs, int). Verified readback on Apollo-M: 0.1–2.0 s.

## Measurement protocol notes

- Dark frames: lens cap ON, complete darkness (fluorescent lights flicker at
  100 Hz and leak — turn them off or shield).
- Flat frames: uniform illumination only needs to cover the analysis ROI;
  validate spatial std < 1–2%. Halogen must be warm (30 min). Avoid colored
  diffusers (spectrally non-neutral).
- Frame counts: with a large ROI (10⁴+ pixels) 10–20 frames per exposure give
  ~1% temporal-variance precision; more frames are cheap on fast cameras.
- For QE (η) extraction you additionally need absolute irradiance at the sensor
  (calibrated photodiode) and the source spectrum — outside this CLI for now.
