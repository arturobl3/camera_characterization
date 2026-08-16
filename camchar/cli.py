"""CLI entry point (Typer).

Usage examples:
  python -m camchar get-dark-frames --vendor playerone --exposures 100,500,1000,2000 --frames 20 --gain 0
  python -m camchar get-flat-frames --vendor playerone --exposures 10,50,100 --frames 20 \\
      --gain 0 --notes "green LED ~530nm, 30cm"
  python -m camchar warmup-sensor --vendor basler
  python -m camchar source-stability-check --vendor basler

--vendor (playerone | basler) is required on every acquisition command;
analyze is offline and needs no vendor. Camera must be on and connected.
Lens cap ON for dark frames; uniform illumination (diffuser) for flat
frames. Frames are saved under <out>/<vendor>_<model>_(<sensor>)/dark|flat
(--out defaults to 'data').

analyze also handles SPECIM IQ hyperspectral exports: pass --data pointing
at the folder holding 'dark frames'/'flat-field frames'; every band is
analyzed independently (see camchar/band_analyze.py).
"""

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from .analyze import run as analyze_run
from .backends import get_backend
from .io_utils import camera_dir_name, save_sequence

# Exposures in milliseconds (converted to seconds only at the backend/
# metadata boundary)
EXPOSURE_MIN_MS = 0.1
EXPOSURE_MAX_MS = 1000.0
NUM_EXPOSURES = 50
DEFAULT_EXPOSURES_MS = np.logspace(
    np.log10(EXPOSURE_MIN_MS), np.log10(EXPOSURE_MAX_MS), NUM_EXPOSURES
)
DEFAULT_FRAMES = 10
WARMUP_EXPOSURE_S = 0.1
WARMUP_STABLE_WINDOW_S = 120.0
WARMUP_STABLE_TOL_C = 0.3
WARMUP_STOP_DELAY_S = 60.0
WARMUP_MAX_CONSECUTIVE_FAILS = 5
WARMUP_PRINT_INTERVAL_S = 5.0
SOURCE_STABILITY_EXPOSURES_MS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
SOURCE_STABILITY_FRAMES = 4
SOURCE_STABILITY_TOL_PCT = 0.1

app = typer.Typer(
    help="Camera PTC characterization toolkit (EMVA 1288 / photon transfer)",
    no_args_is_help=True,
    add_completion=False,
)


def _parse_exposures_ms(text):
    try:
        vals = [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise typer.BadParameter(f"invalid exposure list '{text}'")
    if not vals or any(v <= 0 for v in vals):
        raise typer.BadParameter("exposures must be positive milliseconds")
    return vals


def _ms_list(exposures_ms):
    return ", ".join(f"{e:g} ms" for e in exposures_ms)


def _vendor_option():
    return typer.Option(..., help="camera vendor backend (playerone | basler)")


def _run_sequence(vendor, out, exposures_ms, frames, gain, notes, seq_type):
    backend = get_backend(vendor)
    typer.secho(
        f"[{seq_type}] vendor={vendor}  opening camera...",
        bold=True,
        fg=typer.colors.CYAN,
    )
    info = backend.open()
    pixel = info["pixel_size_um"]
    typer.echo(
        f"[{seq_type}] camera: {info['model']} SN={info['serial']} "
        f"sensor={info['sensor']}" + (f" pixel={pixel:g}um" if pixel else "")
    )
    backend.configure(gain=gain)
    temp = backend.sensor_temp_c()
    out_dir = Path(out) / camera_dir_name(info) / seq_type
    typer.echo(f"[{seq_type}] sensor temp: {temp:.1f} C, gain: {gain:g}")
    typer.echo(f"[{seq_type}] saving to {out_dir}")

    total_frames = 0
    t0 = time.time()
    for exp_ms in exposures_ms:
        exp_s = exp_ms / 1000.0
        typer.echo(f"[{seq_type}] exposure {exp_ms:g} ms x {frames} frames ...")
        temp_before = backend.sensor_temp_c()
        stack = backend.snap(exp_s, frames)
        temp_after = backend.sensor_temp_c()
        # backends may clamp to the camera's exposure range; record the
        # effective exposure so metadata/filenames match what was captured
        eff_s = getattr(backend, "last_exposure_s", exp_s)
        save_sequence(
            out_dir,
            seq_type,
            eff_s,
            gain,
            stack,
            info,
            notes=notes,
            temp_start_c=temp_before,
            temp_c=temp_after,
        )
        total_frames += stack.shape[0]
        typer.echo(f"[{seq_type}] temp {temp_before:.1f} -> {temp_after:.1f} C")

    elapsed = time.time() - t0
    typer.secho(
        f"[{seq_type}] done: {len(exposures_ms)} exposures, {total_frames} frames, "
        f"{elapsed / 60:.1f} min -> {out_dir}",
        fg=typer.colors.GREEN,
        bold=True,
    )
    backend.close()


def _warmup_sensor(vendor):
    backend = get_backend(vendor)
    typer.secho(
        f"[warmup] vendor={vendor}  opening camera...", bold=True, fg=typer.colors.CYAN
    )
    info = backend.open()
    pixel = info["pixel_size_um"]
    typer.echo(
        f"[warmup] camera: {info['model']} SN={info['serial']} "
        f"sensor={info['sensor']}" + (f" pixel={pixel:g}um" if pixel else "")
    )
    backend.configure(gain=0)

    t_start = time.time()
    temp = temp_start = backend.sensor_temp_c()
    typer.echo(
        f"[warmup] starting at {temp_start:.1f} C; running continuous "
        f"{WARMUP_EXPOSURE_S:g} s exposures until stable "
        f"(spread <= {WARMUP_STABLE_TOL_C:g} C over {WARMUP_STABLE_WINDOW_S:.0f} s), "
        f"then auto-stopping after {WARMUP_STOP_DELAY_S:.0f} s; Ctrl+C to stop early"
    )
    samples = [] if math.isnan(temp_start) else [(t_start, temp_start)]
    stable_since = None
    fails = 0
    last_print = 0.0
    try:
        while True:
            try:
                backend.snap(WARMUP_EXPOSURE_S, 1)
                fails = 0
            except RuntimeError as exc:
                fails += 1
                if fails >= WARMUP_MAX_CONSECUTIVE_FAILS:
                    raise
                typer.secho(
                    f"[{time.strftime('%H:%M:%S')}] snap failed ({exc}), retrying",
                    fg=typer.colors.YELLOW,
                )
                time.sleep(2)
                continue
            now = time.time()
            temp = backend.sensor_temp_c()
            stamp = time.strftime("%H:%M:%S")
            due = now - last_print >= WARMUP_PRINT_INTERVAL_S
            if math.isnan(temp):
                if due:
                    typer.echo(f"[{stamp}] temperature read failed, skipping")
                    last_print = now
                continue
            samples = [(t, v) for t, v in samples if now - t <= WARMUP_STABLE_WINDOW_S]
            samples.append((now, temp))
            span = now - samples[0][0]
            spread = max(v for _, v in samples) - min(v for _, v in samples)
            line = f"[{stamp}] {temp:6.2f} C"
            if (
                now - t_start >= WARMUP_STABLE_WINDOW_S
                and spread <= WARMUP_STABLE_TOL_C
            ):
                if stable_since is None:
                    stable_since = now
                remaining = WARMUP_STOP_DELAY_S - (now - stable_since)
                if remaining <= 0:
                    typer.secho(
                        f"{line}   stable for {WARMUP_STOP_DELAY_S:.0f} s "
                        f"-- warmup complete",
                        fg=typer.colors.GREEN,
                        bold=True,
                    )
                    break
                if due:
                    typer.echo(
                        f"{line}   stable ({spread:.2f} C over {span:.0f} s), "
                        f"auto-stop in {remaining:.0f} s"
                    )
                    last_print = now
            else:
                stable_since = None
                if due:
                    if span >= 15:
                        rate = (temp - samples[0][1]) / span * 60
                        line += f"   {rate:+.1f} C/min"
                    typer.echo(line)
                    last_print = now
    except KeyboardInterrupt:
        typer.secho(
            f"\n[warmup] stopped by user at {temp:.1f} C", fg=typer.colors.YELLOW
        )
        backend.close()
        return 130
    except RuntimeError as exc:
        typer.secho(
            f"[warmup] aborted after {fails} consecutive failures: {exc}",
            fg=typer.colors.RED,
        )
        backend.close()
        return 1

    typer.secho(
        f"[warmup] done: {temp_start:.1f} C -> {temp:.1f} C in "
        f"{(time.time() - t_start) / 60:.1f} min",
        fg=typer.colors.GREEN,
        bold=True,
    )
    backend.close()
    return 0


def _source_stability_check(vendor, exposures_ms, frames, gain):
    backend = get_backend(vendor)
    typer.secho(
        f"[stability] vendor={vendor}  opening camera...",
        bold=True,
        fg=typer.colors.CYAN,
    )
    info = backend.open()
    typer.echo(
        f"[stability] camera: {info['model']} SN={info['serial']} "
        f"sensor={info['sensor']} pixel={info['pixel_size_um']}um"
    )
    backend.configure(gain=gain)
    temp = backend.sensor_temp_c()
    typer.echo(f"[stability] sensor temp: {temp:.1f} C, gain: {gain:g}")
    typer.echo(
        f"[stability] deviation of each frame mean vs frame 0 (reference); "
        f"warning threshold {SOURCE_STABILITY_TOL_PCT:g}%"
    )
    if frames < 2:
        typer.secho(
            "[stability] need at least 2 frames per exposure", fg=typer.colors.RED
        )
        backend.close()
        return 1

    unstable = []
    skipped = 0
    for exp_ms in exposures_ms:
        stack = backend.snap(exp_ms / 1000.0, frames)
        means = stack.mean(axis=(1, 2))
        ref = float(means[0])
        if ref <= 0:
            skipped += 1
            typer.echo(
                f"[stability] exposure {exp_ms:g} ms: reference mean is 0 -- skipping"
            )
            continue
        devs = 100.0 * (means - ref) / ref
        typer.echo(
            f"[stability] exposure {exp_ms:g} ms (reference mean {ref:.2f} DN16):"
        )
        for i in range(1, len(devs)):
            if abs(devs[i]) > SOURCE_STABILITY_TOL_PCT:
                typer.secho(
                    f"    frame {i}: {devs[i]:+.4f} %  <-- EXCEEDS THRESHOLD",
                    fg=typer.colors.RED,
                )
            else:
                typer.echo(f"    frame {i}: {devs[i]:+.4f} %")
        max_dev = float(np.abs(devs[1:]).max())
        if max_dev <= SOURCE_STABILITY_TOL_PCT:
            typer.secho(
                f"    max |dev| = {max_dev:.4f} %  -> stable",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(
                f"    max |dev| = {max_dev:.4f} %  -> WARNING: "
                f"source may not be stable",
                fg=typer.colors.RED,
            )
            unstable.append(exp_ms)

    if skipped == len(exposures_ms):
        typer.secho(
            "[stability] verdict: no usable frames (all reference means are 0) "
            "-- check the source/lens",
            fg=typer.colors.RED,
            bold=True,
        )
        backend.close()
        return 1
    if unstable:
        typer.secho(
            "[stability] verdict: WARNING -- source may not be stable at "
            + ", ".join(f"{e:g} ms" for e in unstable),
            fg=typer.colors.RED,
            bold=True,
        )
        backend.close()
        return 1
    typer.secho(
        "[stability] verdict: source stable at all exposures",
        fg=typer.colors.GREEN,
        bold=True,
    )
    backend.close()
    return 0


@app.command()
def get_dark_frames(
    vendor: str = _vendor_option(),
    out: Path = typer.Option(
        "data",
        help="root output directory; frames go to "
        "<out>/<vendor>_<model>_(<sensor>)/dark (default: data)",
    ),
    exposures: str = typer.Option(
        ",".join(map(str, DEFAULT_EXPOSURES_MS)),
        parser=_parse_exposures_ms,
        metavar="MS,MS,...",
        show_default=False,
        help="comma-separated exposure times in milliseconds "
        f"(default: {_ms_list(DEFAULT_EXPOSURES_MS)})",
    ),
    frames: int = typer.Option(DEFAULT_FRAMES, help="frames per exposure"),
    gain: float = typer.Option(
        0.0, help="fixed gain (dB for basler, unitless for playerone)"
    ),
    notes: str = typer.Option("dark", help="metadata notes, e.g. 'lens cap on'"),
):
    """Acquire dark frames (lens cap ON, dark room)."""
    _run_sequence(vendor, out, exposures, frames, gain, notes, "dark")


@app.command()
def get_flat_frames(
    vendor: str = _vendor_option(),
    out: Path = typer.Option(
        "data",
        help="root output directory; frames go to "
        "<out>/<vendor>_<model>_(<sensor>)/flat (default: data)",
    ),
    exposures: str = typer.Option(
        ",".join(map(str, DEFAULT_EXPOSURES_MS)),
        parser=_parse_exposures_ms,
        metavar="MS,MS,...",
        show_default=False,
        help="comma-separated exposure times in milliseconds "
        f"(default: {_ms_list(DEFAULT_EXPOSURES_MS)})",
    ),
    frames: int = typer.Option(DEFAULT_FRAMES, help="frames per exposure"),
    gain: float = typer.Option(
        0.0, help="fixed gain (dB for basler, unitless for playerone)"
    ),
    notes: str = typer.Option("flat", help="metadata notes, e.g. 'green LED ~530nm'"),
):
    """Acquire flat frames (uniform broadband illumination)."""
    _run_sequence(vendor, out, exposures, frames, gain, notes, "flat")


@app.command()
def analyze(
    data: Path = typer.Option(
        "data",
        help="data root or camera dir <root>/<vendor>_<model>_(<sensor>); "
        "a single camera dir under a root is auto-discovered (default: data)",
    ),
    roi: Optional[str] = typer.Option(
        None,
        help="ROI as r0:r1:c0:c1 (default 500:900:750:1150; SPECIM IQ "
        "hyperspectral data defaults to a central 156:356:156:356)",
    ),
    bands: int = typer.Option(
        5,
        help="number of equispaced bands in per-band (SPECIM IQ) plots; "
        "ignored for monochrome npy data",
    ),
):
    """Temporal PTC analysis of dark/flat data."""
    roi_list = None
    if roi is not None:
        roi_list = [int(x) for x in roi.split(":")]
        if len(roi_list) != 4:
            raise typer.BadParameter("--roi must be r0:r1:c0:c1")
    analyze_run(data, roi_list, bands)


@app.command()
def warmup_sensor(vendor: str = _vendor_option()):
    """Run the camera to warm it up to its steady-state operating temperature."""
    rc = _warmup_sensor(vendor)
    if rc:
        raise typer.Exit(rc)


@app.command()
def source_stability_check(
    vendor: str = _vendor_option(),
    exposures: str = typer.Option(
        ",".join(map(str, SOURCE_STABILITY_EXPOSURES_MS)),
        parser=_parse_exposures_ms,
        metavar="MS,MS,...",
        show_default=False,
        help="comma-separated exposure times in milliseconds "
        f"(default: {_ms_list(SOURCE_STABILITY_EXPOSURES_MS)})",
    ),
    frames: int = typer.Option(
        SOURCE_STABILITY_FRAMES, help="consecutive frames per exposure"
    ),
    gain: float = typer.Option(
        0.0, help="fixed gain (dB for basler, unitless for playerone)"
    ),
):
    """Check the light-source stability for flat-field measurements."""
    rc = _source_stability_check(vendor, exposures, frames, gain)
    if rc:
        raise typer.Exit(rc)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
