# camchar — camera PTC characterization toolkit

Acquisition CLI for photon-transfer-curve (PTC) camera characterization, following
EMVA 1288 / Janesick methodology. Vendor-agnostic backend design; currently
implements the **Player One Astronomy** backend (tested with Apollo-M / IMX174 on
macOS, SDK 3.10.1).

## Quick start (macOS)

```bash
# prerequisites
brew install libusb
uv sync                      # creates .venv, installs camchar + numpy (uv.lock)

# warm up to steady-state operating temperature (auto-stops once stable)
uv run camchar warmup-sensor

# dark frames: lens cap ON, dark room
uv run camchar get-dark-frames --out data/dark \
    --exposures 0.001,0.01,0.1,0.5,2.0 --frames 20 --gain 0
# flat frames: uniform broadband illumination (halogen + diffuser, or LED)
uv run camchar get-flat-frames --out data/flat \
    --exposures 0.01,0.05,0.1,0.5 --frames 20 --gain 0 \
    --notes "green LED ~530nm, diffuser, 30 cm"

# analysis: temporal PTC -> K, read noise, Nsat, dark current, PRNU, linearity
uv run camchar analyze --data data
```

`python -m camchar ...` also works (same package). The console script and
module are interchangeable; on Windows lab machines run the same commands from
a `uv sync`'d checkout.

Each exposure writes `{seq}_{exp}ms|us_g{gain}.npy` (uint16 stack, n×h×w) plus an
append-only `metadata.json` with camera info, exposure, gain, sensor temp
(`temp_start_c`/`temp_c`, before/after each exposure), notes.
Sub-ms / fractional-ms exposures use microsecond names (e.g. `flat_004500us`).

## CLI

```
camchar get-dark-frames --out DIR [--exposures S,S,...] [--frames N] [--gain G] [--notes TXT]
camchar get-flat-frames --out DIR [--exposures S,S,...] [--frames N] [--gain G] [--notes TXT]
camchar analyze --data DIR [--roi r0:r1:c0:c1]
camchar warmup-sensor
```

Defaults: dark 0.001–2.0 s (9 points), flat 0.01–1.0 s (5 points), 20 frames each, gain 0.
`warmup-sensor` runs continuous 0.5 s exposures and prints the timestamped sensor
temperature; it declares the temperature stable once it stays within 0.3 °C over the
last 2 min and auto-stops 1 min after that (Ctrl+C to stop earlier). Run it before
acquiring so darks/flats are taken at thermal equilibrium.
`analyze` reads `<data>/dark/` + `<data>/flat/` from metadata.json, reports K
(e⁻/DN12), read noise (e⁻, from darks), dark current (DN/s), N_sat (e⁻),
PRNU (%), bias floor, per-point gain check and exposure linearity. ROI defaults
to a central 200×200 patch (600:800:850:1050); only the ROI needs uniform
illumination.

## Project layout

```
camera_characterization/
├── camchar/
│   ├── cli.py                 # argparse CLI (get-dark-frames | get-flat-frames | analyze)
│   ├── analyze.py             # temporal PTC analysis (K, σr, Nsat, PRNU, linearity)
│   ├── io_utils.py            # npy + metadata.json saving, stem_for() naming
│   └── backends/
│       ├── base.py            # CameraBackend ABC
│       ├── playerone.py       # Player One backend (pitfalls encoded)
│       └── __init__.py        # backend registry
├── vendor/
│   └── playerone/
│       ├── lib/               # SDK binary for YOUR OS (see below)
│       └── py/                # vendored pyPOACamera.py (platform-aware loading)
├── pyproject.toml             # uv-managed: uv sync, uv run camchar ...
└── data/                      # acquired sequences (gitignored)
```

Add a vendor: implement `CameraBackend` in `camchar/backends/<vendor>.py`, decorate
with `@register("<vendor>")`, done.

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
