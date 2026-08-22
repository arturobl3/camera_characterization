"""Thorlabs scientific camera backend (TLCamera SDK via the official
``thorlabs_tsi_sdk`` Python wrapper, vendored under vendor/thorlabs/py).

Tested with a Kiralux LP126MU (4096x3000 mono CMOS, 12-bit) on Windows.
Pitfalls this backend encodes (all verified empirically on the hardware):

  1. Frame data is right-aligned at the camera bit depth in uint16 (the SDK
     docstring says bit_depth counts the *lower* bits; verified on the
     12-bit LP126MU: dark mean == black level, ~5.3 DN12). Frames are
     shifted left by 16 - bit_depth to DN16 in snap() so every downstream
     constant and fit is shared with the playerone/basler path.
  2. The ROI lower-right coordinate is inclusive on this camera family
     (full frame reads back as (0, 0, w-1, h-1)), contradicting the SDK
     docstring example ((100,100,600,600) -> "500x500"). configure() never
     rewrites a full-frame readback and verifies dimensions instead.
  3. The TLCamera C API has no temperature readout at all ->
     sensor_temp_c() returns NaN and the class declares has_temperature =
     False so the CLI refuses to run warmup-sensor against it.
  4. close() must be a real teardown (dispose camera, then SDK): Thorlabs
     docs warn of crashes on exit otherwise. Opposite of the Player One
     macOS wedge where close() must stay a no-op.
  5. DLL discovery: tl_camera.py loads 'thorlabs_tsi_camera_sdk.dll' by
     bare name at TLCameraSDK() construction time, so vendor/thorlabs/lib
     is put on the DLL search path (PATH prepend + os.add_dll_directory)
     at import, before any TLCameraSDK() exists. Only the mono runtime
     subset of the Native Toolkit is vendored (~700 KB: camera sdk +
     zelux USB device + hotplug monitor + logger); the color/polarization
     processor DLLs (~180 MB) are not needed for mono acquisition.
  6. Exposure is integer microseconds; LP126MU range is 28 us .. ~14.7 s.
     Out-of-range exposures are clamped with a warning like the basler
     backend and last_exposure_s records the effective value (sub-minimum
     exposures collide into one stem_for filename - known convention).
  7. Gain is an integer index (0..480 on the LP126MU); --gain is coerced
     to int (playerone convention) and the dB equivalent from
     convert_gain_to_decibels is recorded as gain_db in metadata.
  8. Hot-pixel correction substitutes neighbor values into flagged pixels,
     which destroys temporal-variance statistics; configure() turns it off
     when the camera supports it.
  9. Black level is left at the factory default (5 DN12 on the LP126MU,
     dark mean sits safely above zero) and recorded as black_level_dn12.
"""

import os
import sys
import time

import numpy as np

from . import register
from .base import CameraBackend

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PY_DIR = os.path.join(_ROOT, "vendor", "thorlabs", "py")
_LIB_DIR = os.path.join(_ROOT, "vendor", "thorlabs", "lib")

if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)
if sys.platform == "win32" and os.path.isdir(_LIB_DIR):
    # Must be in place before TLCameraSDK() loads the DLL by bare name.
    os.environ["PATH"] = _LIB_DIR + os.pathsep + os.environ["PATH"]
    try:
        os.add_dll_directory(_LIB_DIR)
    except OSError:
        pass

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK  # noqa: E402
from thorlabs_tsi_sdk.tl_camera_enums import OPERATION_MODE  # noqa: E402

_POLL_TIMEOUT_MS = 1000  # per-poll block; snap() enforces its own deadline
_GRAB_TIMEOUT_MARGIN_S = 10.0  # same spirit as basler's +5000 ms margin

# model -> sensor name; the TLCamera API exposes no sensor-model string,
# so entries go here only once confirmed (LP126MU dir name stays clean).
_KNOWN_MODELS = {}


@register("thorlabs")
class ThorlabsBackend(CameraBackend):
    vendor = "thorlabs"
    has_temperature = False  # no temperature API in the TLCamera SDK

    def __init__(self, retry_seconds=60, serial=None):
        self._retry = retry_seconds
        self._serial = serial
        self._sdk = None
        self._cam = None
        self._info = None
        self._w = self._h = 0
        self._shift = 4  # DN12 -> DN16 until configure() reads the real bit depth
        self.last_exposure_s = None

    # ---------- lifecycle ----------
    def open(self):
        if not os.path.isdir(_PY_DIR) or not os.path.isdir(_LIB_DIR):
            raise RuntimeError(
                f"Thorlabs SDK runtime missing: need {_PY_DIR} (python "
                f"package) and {_LIB_DIR} (native libraries)"
            )
        t0 = time.time()
        serials = []
        last_err = None
        while time.time() - t0 < self._retry:
            try:
                if self._sdk is None:
                    self._sdk = TLCameraSDK()
                serials = self._sdk.discover_available_cameras()
                if serials:
                    break
                last_err = RuntimeError("no cameras discovered yet")
            except Exception as exc:  # half-dead device / dll hiccup
                last_err = exc
                if self._sdk is not None:
                    try:
                        self._sdk.dispose()
                    except Exception:
                        # dispose() failed -> the wrapper's _is_sdk_open class
                        # flag stays True, so every further TLCameraSDK()
                        # would die with "already in use" and mask the real
                        # error. Stop retrying; last_err keeps the cause.
                        self._sdk = None
                        break
                    self._sdk = None
            time.sleep(2)
        if not serials:
            raise RuntimeError(
                "Thorlabs camera not found after %ds (%s). Check USB "
                "connection (prefer direct, no hub)." % (self._retry, last_err)
            )
        serial = self._serial or serials[0]
        if serial not in serials:
            self._close_sdk_only()
            raise RuntimeError(
                f"Thorlabs camera SN {serial} not found (found: {serials})"
            )
        self._cam = self._sdk.open_camera(serial)
        cam = self._cam
        model = cam.model
        self._info = {
            "vendor": "thorlabs",
            "model": model,
            "serial": cam.serial_number,
            "sensor": _KNOWN_MODELS.get(model, ""),
            "pixel_size_um": float(cam.sensor_pixel_width_um),
            "bit_depth": int(cam.bit_depth),
            "width": int(cam.sensor_width_pixels),
            "height": int(cam.sensor_height_pixels),
            "usb3": int(getattr(cam.usb_port_type, "value", 0)) == 2,
        }
        return dict(self._info)

    def configure(self, gain=0):
        if float(gain) != int(gain):
            raise RuntimeError(
                f"Thorlabs gain must be an integer index, got {gain!r} (basler uses dB)"
            )
        gain_idx = int(gain)
        lo, hi = self._cam.gain_range.min, self._cam.gain_range.max
        if not lo <= gain_idx <= hi:
            raise RuntimeError(f"gain {gain_idx} outside camera range {lo}..{hi}")
        cam = self._cam
        try:
            cam.gain = gain_idx
        except Exception as exc:
            raise RuntimeError(f"SetGain failed: {exc}")
        try:
            cam.is_hot_pixel_correction_enabled = False
        except Exception:
            pass  # unsupported -> nothing to disable
        cam.binx = 1
        cam.biny = 1
        w, h = cam.image_width_pixels, cam.image_height_pixels
        sw, sh = cam.sensor_width_pixels, cam.sensor_height_pixels
        if (w, h) != (sw, sh):
            # lower-right is inclusive on this family (see module docstring)
            cam.roi = (0, 0, sw - 1, sh - 1)
            w, h = cam.image_width_pixels, cam.image_height_pixels
        self._w, self._h = w, h
        # raw frames are right-aligned at the camera's bit depth; DN16 needs
        # 16 - bit_depth leading zero bits (4 on the 12-bit LP126MU)
        self._shift = max(0, 16 - int(cam.bit_depth))
        cam.frames_per_trigger_zero_for_unlimited = 1
        cam.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
        cam.image_poll_timeout_ms = _POLL_TIMEOUT_MS
        self._info["gain"] = gain_idx
        try:
            self._info["gain_db"] = float(cam.convert_gain_to_decibels(gain_idx))
        except Exception:
            self._info["gain_db"] = None
        try:
            self._info["black_level_dn12"] = int(cam.black_level)
        except Exception:
            self._info["black_level_dn12"] = None
        elo, ehi = cam.exposure_time_range_us
        print(
            f"  [thorlabs] exposure range: {elo} us .. {ehi / 1e6:g} s, "
            f"gain index {gain_idx}"
            + (
                f" ({self._info['gain_db']:.2f} dB)"
                if self._info["gain_db"] is not None
                else ""
            )
        )
        return self._info

    def sensor_temp_c(self):
        # No temperature API exists in the TLCamera C API (verified in
        # tl_camera_sdk.h); NaN makes every CLI temperature printout skip.
        return float("nan")

    def snap(self, exposure_s, n_frames):
        cam = self._cam
        want_us = exposure_s * 1e6
        elo, ehi = cam.exposure_time_range_us
        exp_us = min(max(int(round(want_us)), elo), ehi)
        if exp_us != int(round(want_us)):
            print(
                f"  [thorlabs] exposure {exposure_s * 1e3:g} ms clamped to "
                f"{exp_us / 1e3:g} ms (camera range {elo} us .. "
                f"{ehi / 1e6:g} s)"
            )
        cam.exposure_time_us = exp_us
        self.last_exposure_s = exp_us / 1e6
        deadline = time.monotonic() + (exp_us / 1e6 + _GRAB_TIMEOUT_MARGIN_S) * max(
            n_frames, 1
        )
        stack = np.empty((n_frames, self._h, self._w), dtype=np.uint16)
        cam.arm(n_frames)
        try:
            for i in range(n_frames):
                cam.issue_software_trigger()
                frame = cam.get_pending_frame_or_null()
                while frame is None:
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            f"no frame {i} within timeout (exposure {exp_us / 1e6:g} s)"
                        )
                    frame = cam.get_pending_frame_or_null()
                buf = frame.image_buffer
                if buf.shape != (self._h, self._w):
                    raise RuntimeError(
                        f"unexpected frame shape {buf.shape} "
                        f"(expected {self._h, self._w})"
                    )
                # right-aligned raw DN -> DN16 (module docstring, pitfall 1)
                stack[i] = np.copy(buf) << self._shift
        finally:
            try:
                cam.disarm()
            except Exception:
                pass
        return stack

    def close(self):
        """Real teardown: the SDK requires explicit disposal (crashes on
        exit otherwise). See module docstring, pitfall 4."""
        cam, sdk = self._cam, self._sdk
        self._cam = None
        self._sdk = None
        if cam is not None:
            try:  # dispose() disarms first, then closes the camera
                cam.dispose()
            except Exception:
                pass
        self._close_sdk_only(sdk)

    def _close_sdk_only(self, sdk=None):
        sdk = sdk if sdk is not None else self._sdk
        self._sdk = None
        if sdk is not None:
            try:
                sdk.dispose()
            except Exception:
                pass
