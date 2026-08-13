"""CLI entry point.

Usage examples:
  python -m camchar get-dark-frames --exposures 0.1,0.5,1.0,2.0 --frames 20 --gain 0
  python -m camchar get-flat-frames --exposures 0.01,0.05,0.1 --frames 20 \\
      --gain 0 --notes "green LED ~530nm, 30cm"
  python -m camchar warmup-sensor

Camera must be on and connected. Lens cap ON for dark frames; uniform
illumination (diffuser) for flat frames. Frames are saved under
<out>/<vendor>_<model>_(<sensor>)/dark|flat (--out defaults to 'data').
"""

import argparse
import math
import sys
import time
from pathlib import Path

from .analyze import run as analyze_run
from .backends import get_backend
from .io_utils import camera_dir_name, save_sequence

DEFAULT_DARK_EXPOSURES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 2.0]
DEFAULT_FLAT_EXPOSURES = [0.01, 0.05, 0.1, 0.5, 1.0]
WARMUP_EXPOSURE_S = 0.1
WARMUP_STABLE_WINDOW_S = 120.0
WARMUP_STABLE_TOL_C = 0.3
WARMUP_STOP_DELAY_S = 60.0
WARMUP_MAX_CONSECUTIVE_FAILS = 5
WARMUP_PRINT_INTERVAL_S = 5.0


def _parse_exposures(text):
    try:
        vals = [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid exposure list '{text}'")
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("exposures must be positive seconds")
    return vals


def _run_sequence(args, seq_type):
    backend = get_backend(args.vendor)
    print(f"[{seq_type}] vendor={args.vendor}  opening camera...")
    info = backend.open()
    print(
        f"[{seq_type}] camera: {info['model']} SN={info['serial']} "
        f"sensor={info['sensor']} pixel={info['pixel_size_um']}um"
    )
    backend.configure(gain=args.gain)
    temp = backend.sensor_temp_c()
    out_dir = Path(args.out) / camera_dir_name(info) / seq_type
    print(f"[{seq_type}] sensor temp: {temp:.1f} C, gain: {args.gain}")
    print(f"[{seq_type}] saving to {out_dir}")

    total_frames = 0
    t0 = __import__("time").time()
    for exp in args.exposures:
        print(f"[{seq_type}] exposure {exp:g}s x {args.frames} frames ...")
        temp_before = backend.sensor_temp_c()
        stack = backend.snap(exp, args.frames)
        temp_after = backend.sensor_temp_c()
        save_sequence(
            out_dir,
            seq_type,
            exp,
            args.gain,
            stack,
            info,
            notes=args.notes,
            temp_start_c=temp_before,
            temp_c=temp_after,
        )
        total_frames += stack.shape[0]
        print(f"[{seq_type}] temp {temp_before:.1f} -> {temp_after:.1f} C")

    elapsed = __import__("time").time() - t0
    print(
        f"[{seq_type}] done: {len(args.exposures)} exposures, {total_frames} frames, "
        f"{elapsed / 60:.1f} min -> {out_dir}"
    )
    backend.close()


def _warmup_sensor(args):
    backend = get_backend(args.vendor)
    print(f"[warmup] vendor={args.vendor}  opening camera...")
    info = backend.open()
    print(
        f"[warmup] camera: {info['model']} SN={info['serial']} "
        f"sensor={info['sensor']} pixel={info['pixel_size_um']}um"
    )
    backend.configure(gain=0)

    t_start = time.time()
    temp = temp_start = backend.sensor_temp_c()
    print(
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
                print(f"[{time.strftime('%H:%M:%S')}] snap failed ({exc}), retrying")
                time.sleep(2)
                continue
            now = time.time()
            temp = backend.sensor_temp_c()
            stamp = time.strftime("%H:%M:%S")
            due = now - last_print >= WARMUP_PRINT_INTERVAL_S
            if math.isnan(temp):
                if due:
                    print(f"[{stamp}] temperature read failed, skipping")
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
                    print(
                        f"{line}   stable for {WARMUP_STOP_DELAY_S:.0f} s "
                        f"-- warmup complete"
                    )
                    break
                if due:
                    print(
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
                    print(line)
                    last_print = now
    except KeyboardInterrupt:
        print(f"\n[warmup] stopped by user at {temp:.1f} C")
        backend.close()
        return 130
    except RuntimeError as exc:
        print(f"[warmup] aborted after {fails} consecutive failures: {exc}")
        backend.close()
        return 1

    print(
        f"[warmup] done: {temp_start:.1f} C -> {temp:.1f} C in "
        f"{(time.time() - t_start) / 60:.1f} min"
    )
    backend.close()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="camchar", description="Camera PTC acquisition CLI"
    )
    p.add_argument(
        "--vendor",
        default="playerone",
        help="camera vendor backend (default: playerone)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    for cmd, defaults, cap in (
        ("get-dark-frames", DEFAULT_DARK_EXPOSURES, "dark"),
        ("get-flat-frames", DEFAULT_FLAT_EXPOSURES, "flat"),
    ):
        sp = sub.add_parser(cmd)
        sp.add_argument(
            "--out",
            default="data",
            help="root output directory; frames go to "
            "<out>/<vendor>_<model>_(<sensor>)/dark|flat (default: data)",
        )
        sp.add_argument(
            "--exposures",
            type=_parse_exposures,
            default=defaults,
            help=f"comma-separated exposure times in seconds "
            f"(default: {','.join(map(str, defaults))})",
        )
        sp.add_argument("--frames", type=int, default=20, help="frames per exposure")
        sp.add_argument("--gain", type=int, default=0, help="fixed gain (default 0)")
        sp.add_argument(
            "--notes", default=cap, help="metadata notes, e.g. 'lens cap on'"
        )
        sp.add_argument(
            "--retry",
            type=int,
            default=60,
            help="seconds to retry camera enumeration (default 60)",
        )
        sp.set_defaults(_seq=cap, _retry_override=True)

    sp = sub.add_parser("analyze", help="temporal PTC analysis of dark/flat data")
    sp.add_argument(
        "--data",
        default="data",
        help="data root or camera dir <root>/<vendor>_<model>_(<sensor>); "
        "a single camera dir under a root is auto-discovered (default: data)",
    )
    sp.add_argument(
        "--roi",
        default="600:800:850:1050",
        help="ROI as r0:r1:c0:c1 (default 600:800:850:1050)",
    )
    sp.set_defaults(_analyze=True)

    sp = sub.add_parser(
        "warmup-sensor",
        help="run the camera to warm it up to its steady-state operating "
        "temperature (auto-stops 1 min after the temperature stabilizes)",
    )
    sp.set_defaults(_warmup=True)

    args = p.parse_args(argv)
    if getattr(args, "_analyze", False):
        roi = [int(x) for x in args.roi.split(":")]
        if len(roi) != 4:
            p.error("--roi must be r0:r1:c0:c1")
        return analyze_run(args.data, roi)
    if getattr(args, "_warmup", False):
        return _warmup_sensor(args)
    if args.command in ("get-dark-frames", "get-flat-frames"):
        seq = "dark" if args.command == "get-dark-frames" else "flat"
        if args._retry_override:
            # backend retry is handled inside the backend; CLI-level retry arg kept
            # for future backends that may need it
            pass
        _run_sequence(args, seq)
    else:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
