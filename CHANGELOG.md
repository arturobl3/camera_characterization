# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-01

### Changed

- **Sensor-relative default ROI**: `camchar analyze` without `--roi` now uses
  the central 50% of each recorded frame's dimensions (`io_utils.central_roi`,
  even-sized window, minimum 2 px side) instead of the fixed IMX174-era patch
  (`500:900:750:1150`). This centers the analysis window on every sensor
  (Basler a2A3536, Thorlabs LP126MU/CU, ...) and scales with the data.
  Results are not directly comparable across this change; pass
  `--roi 500:900:750:1150` to reproduce legacy windows. Plot titles show a
  compact "central 50%" tag for the default window and exact coordinates for
  explicit ROIs; per-channel (CFA) plot titles now carry the same tag.
  SPECIM IQ keeps its dedicated default ROI.

## [1.0.0] - 2026-08-25

First stable release. All prior commits were unreleased development toward
this version; the notes below summarize the state at 1.0.0 and the changes
made immediately before the release.

### Added

- **Acquisition CLI** (`typer` app, `camchar` console script) with five
  hardware commands — `get-dark-frames`, `get-flat-frames`,
  `warmup-sensor`, `source-stability-check`, `check-saturation` — each with a
  required, enum-validated `--vendor` option (invalid values exit 2 before
  any backend is touched). Exit-code contract: warmup 130 on Ctrl+C / 1 on
  abort / 1 when the backend has no temperature API; stability check 1 when
  unstable; check-saturation 0 only when saturation begins exactly at the
  third-to-last sweep exposure, else 1 with an intensity multiplier.
- **Camera backends** behind one `CameraBackend` interface
  (`camchar/backends/base.py`), registered at import:
  - *Player One*: direct DLL config calls with a ctypes union, bypassing the
    vendored wrapper's ABI bug; enumeration retry at `open()`; deliberate
    no-op `close()` (macOS wedge workaround).
  - *Basler* (pypylon): software-triggered raw Mono12 / unpacked BayerRG12,
    Default user-set load so Viewer settings never leak, model-aware digital
    black-level setup (IMX174 DN12 target, ace 2 measured node values),
    exposure clamping with effective-exposure metadata.
  - *Thorlabs* (official `thorlabs_tsi_sdk` wrapper, vendored): right-aligned
    raw frames shifted to DN16, inclusive-ROI convention handled, hot-pixel
    correction forced off, real teardown `close()`, color models recorded as
    raw Bayer (`pixel_format` drives CFA-aware analysis downstream).
- **Offline analysis** (`camchar analyze`, works without any SDK installed):
  EMVA 1288 R4 *Linear* photon-transfer analysis — gain K via the Eq. 50
  zero-intercept PTC fit on dark-subtracted variance over the min-to-70%-sat
  regression range, read noise with the Eq. 53 quantization correction, dark
  current (Eq. 29 mean / Eq. 30 variance slopes), saturation per §6.6,
  threshold/DR (Eq. 27/28), PRNU1288/DSNU1288 via the §8.1 highpass with the
  Eq. 36 temporal-residual correction, linearity check, and SNR plots
  (Eq. 21 temporal and Eq. 69 total-SNR models vs measured points).
- **Two-frame estimators as primary** (Eq. 18 temporal variance, Eq. 32
  spatial covariance) with N-frame cross-check values everywhere
  (`*_nf`); a systematic two-frame-vs-N-frame gap is the documented drift
  indicator. Temporal variance uses ddof=1 (Eq. 65), spatial ddof=0.
- **SPECIM IQ hyperspectral path**: ENVI BIL export discovery/loading via an
  mmap ROI reader (`specim.py`; avoids spectral's per-pixel seek loop), every
  band analyzed as an independent monochrome camera, per-band table +
  `band_parameters.csv` + per-band plot variants + parameters-vs-wavelength
  summary with all-band and 500–800 nm means.
- **Per-CFA-channel analysis** for raw-Bayer datasets
  (`cfa_analyze.py`): R/G1/G2/B sub-lattices through the same pure fits,
  CFA-aware pooled spatial metrics in the headline numbers, per-channel
  table/detail/plots and `cfa_parameters.csv`.
- **Plot suite** (`plots.py`, Agg backend): dark mean/variance, linearity,
  log-log PTC, SNR — monochrome plus per-curve variants, each annotated with
  the fitted quantities and R².
- **Data conventions** (`io_utils.py`): `<root>/<vendor>_<model>_(<sensor>)/
  {dark,flat}/` layout, `{dark|flat}_{NNNNN}ms|us_g{gain}.npy` uint16 stacks
  (DN12 << 4 storage), append-only `metadata.json` deduped on load.
- **Test suite**: CLI exit-code tests against mocked backends and
  pure-function tests locking in the EMVA fit semantics,
  filename/metadata round-trips, backend helpers, and the SPECIM loader.

### Changed

- Refactor ahead of the release: shared constants (`DEFAULT_ROI_FRAC`,
  `DEFAULT_ROI_SPECIM`, `SAT_CLIP_FRAC`) live in `io_utils.py`;
  multi-curve presentation (parameter table, EMVA detail block, CSV cells,
  drift/degeneracy check) lives once in `report.py` and serves both the
  SPECIM and CFA paths; the EMVA 50%-saturation point selection rule lives
  once in `analyze.usable_flat_points()`; ROI slices are passed explicitly
  through the analysis functions (no module-global ROI); all user-facing
  output goes through typer.
- SPECIM IQ default ROI is rows 140:290 / cols 156:306 of the 512x512 frame
  (`--roi` overrides).

### Removed

- Dead code: legacy argparse entry point for `analyze`, unused fit-dict keys
  (`bias_fit_dn`, `ptc_intercept_nf`) and write-only backend state.

### Fixed

- Basler dark frames no longer underflow on the IMX174 (explicit black-level
  offset; previously ~97% of dark pixels pinned at zero, censoring the dark
  histogram and every variance-based quantity).
- Temporal-variance estimator bias (ddof=0 population formula under-reported
  K by ~(N-1)/N; caught by the two-frame cross-check on the Aug 2026 Apollo-M
  dataset).

[1.0.0]: https://github.com/arturobl3/camera_characterization/releases/tag/v1.0.0
