# AGENTS.md

## Commands

- Setup: `uv sync` (Python >=3.11; creates `.venv`)
- Run: `uv run camchar ...` or `uv run python -m camchar ...` (equivalent)
- Verify without hardware: `uv run camchar --help`; `uv run camchar analyze --data data`
- Lint: `uv run ruff check .` — Format: `uv run ruff format .` (ruff is a dev dependency; `vendor/` is excluded — never lint/format the vendored SDK wrapper)
- No test suite or typecheck is configured. Don't guess one into existence.

## Hardware dependence

- `get-dark-frames` / `get-flat-frames` require a physically connected Player One camera (direct USB, no hub). Don't run acquisition commands as a code check.
- `analyze` is offline: reads `<data>/dark/` + `<data>/flat/` via their `metadata.json`; writes plots to `outputs/` (CWD, gitignored) via `camchar/plots.py` (matplotlib, Agg backend).
- The backend retries camera enumeration for up to 60 s at `open()` by design (USB enumeration is flaky); don't shorten or remove this.

## Player One SDK — deliberate workarounds, do not "fix"

- The vendored wrapper `vendor/playerone/py/pyPOACamera.py` has an ABI bug: `POASetConfig`/`POAGetConfig` pass `c_int` where the C API expects an 8-byte `POAConfigValue` union, silently corrupting config writes. `camchar/backends/playerone.py` bypasses the wrapper and calls the DLL directly with a ctypes `Union`. Never route config reads/writes back through the wrapper.
- `PlayerOneBackend.close()` is intentionally a no-op: `CloseCamera()` wedges the device on macOS until physically replugged. Process exit is the cleanup.
- Exposure uses config 31 (`POA_EXP`, seconds, float) with fallback to config 0 (`POA_EXPOSURE`, µs, int).
- SDK binaries are OS-specific and must be placed in `vendor/playerone/lib/` (loading is platform-aware in the wrapper). Windows: `PlayerOneCamera.dll` + WinUSB driver. macOS: dylib(s) + Homebrew libusb patched in via `install_name_tool` (see README).

## Architecture and conventions

- Backend registry: new vendor = implement `CameraBackend` (`camchar/backends/base.py`) in `camchar/backends/<vendor>.py` with `@register("<vendor>")`; registration happens at import (bottom of `camchar/backends/__init__.py`).
- Data layout (gitignored): `{dark|flat}_{NNNNN}ms|us_g{gain}.npy` — uint16 stack `(n, h, w)` — plus an append-only `metadata.json` per directory. Naming rule lives in `camchar/io_utils.py:stem_for()` (integer ms >= 1 -> `ms`, else `us`); `analyze.py` reconstructs filenames from metadata via `stem_for`, so the two must stay in sync.
- Frames are 12-bit data stored as DN12 << 4 (DN16). Gain math in `camchar/analyze.py`: fit `V = K_fit*S + b` on temporal variance, then `K12 = 16/K_fit` — the gain is the inverse of the slope, not the slope. Clip threshold 65400 DN; default ROI `600:800:850:1050` (only the ROI must be uniformly illuminated).
- `uv.lock` is intentionally gitignored; don't commit it. `data/` holds large acquisition sequences — never commit it.
- README.md is the source of truth for the measurement protocol (dark/flat acquisition practice).
