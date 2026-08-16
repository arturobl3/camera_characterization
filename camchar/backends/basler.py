"""Basler camera backend (pypylon).

Acquires raw 12-bit frames from Basler ace-class cameras (tested with
acA1920-155um / IMX174 over USB3). Conventions follow the Player One path:

  - PixelFormat Mono12 (unpacked) -> pypylon's grab_result.Array is a
    uint16 array holding DN12 values in the low bits; frames are stored
    left-shifted by 4 as DN16 so every downstream constant and fit in
    analyze.py is shared with the playerone path.
  - Software trigger per frame (TriggerMode On / FrameStart / Software):
    free-running acquisition would keep stale buffers from the previous
    exposure in the queue, and RetrieveResult does not tell you which
    exposure a frame belongs to. Triggering gives snap()-style, exact
    per-frame semantics and no flushing needed when exposure changes.
  - Gain is set via the 'Gain' node in dB (float, e.g. 0..24 dB on the
    acA1920-155um).
  - The camera's stored user set is reset to Default on configure() so
    leftover pylon Viewer settings (test patterns, ROIs, LUTs) never leak
    into measurements.
"""

import time

import numpy as np

try:
    from pypylon import pylon
except ImportError:  # pragma: no cover - only on misinstalled environments
    pylon = None

from . import register
from .base import CameraBackend

# model-name prefix -> (sensor, pixel size um); add entries as needed
_KNOWN_MODELS = {
    "acA1920-155": ("IMX174", 5.86),
}

_GRAB_TIMEOUT_MARGIN_MS = 5000  # same margin as the playerone GetImageData call


@register("basler")
class BaslerBackend(CameraBackend):
    vendor = "basler"

    def __init__(self, retry_seconds=60, serial=None):
        self._retry = retry_seconds
        self._serial = serial
        self._cam = None
        self._info = None
        self._w = self._h = 0
        self._grabbing = False

    # ---------- helpers ----------
    @staticmethod
    def _node(cam, name):
        """Return the GenApi node or None when unavailable (wrapped access)."""
        try:
            return getattr(cam, name)
        except Exception:
            return None

    @staticmethod
    def _node_get(node, default=None):
        if node is None:
            return default
        try:
            return node.GetValue()
        except Exception:
            return default

    def _snap_one(self, exposure_s):
        timeout_ms = max(int(exposure_s * 1000) + _GRAB_TIMEOUT_MARGIN_MS, 5000)
        self._cam.ExecuteSoftwareTrigger()
        with self._cam.RetrieveResult(timeout_ms) as res:
            if not res.GrabSucceeded():
                raise RuntimeError(
                    f"grab failed: {res.ErrorDescription} (exposure {exposure_s:g} s)"
                )
            return np.array(res.Array, dtype=np.uint16, copy=True)

    # ---------- lifecycle ----------
    def open(self):
        if pylon is None:
            raise RuntimeError("pypylon is not installed (uv add pypylon)")
        t0 = time.time()
        cam = None
        last_err = None
        while time.time() - t0 < self._retry:
            try:
                tl = pylon.TlFactory.GetInstance()
                devices = tl.EnumerateDevices()
                dev = None
                if self._serial:
                    for d in devices:
                        if d.GetSerialNumber() == self._serial:
                            dev = d
                            break
                    if dev is None:
                        raise RuntimeError(f"Basler camera SN {self._serial} not found")
                elif devices:
                    dev = devices[0]
                if dev is not None:
                    cam = pylon.InstantCamera(tl.CreateDevice(dev))
                    cam.Open()
                    break
            except Exception as exc:  # enumeration hiccup; keep retrying
                last_err = exc
                if cam is not None:
                    try:
                        cam.Close()
                    except Exception:
                        pass
                    cam = None
                time.sleep(2)
        if cam is None:
            raise RuntimeError(
                "Basler camera not found after %ds (%s). Check USB connection "
                "and pylon USB3 driver." % (self._retry, last_err)
            )

        self._cam = cam
        info = cam.DeviceInfo
        model = str(info.GetModelName())
        # SensorName is not exposed on some ace models; the model table is
        # authoritative for known cameras, the node is only a fallback.
        key = model[:-2] if model.endswith(("um", "gm")) else model
        if key in _KNOWN_MODELS:
            sensor, pixel = _KNOWN_MODELS[key]
        else:
            sensor = self._node_get(self._node(cam, "SensorName")) or model
            pixel = None
        try:
            w = int(cam.SensorWidth.GetValue())
            h = int(cam.SensorHeight.GetValue())
        except Exception:
            w = h = 0
        self._w, self._h = w, h
        self._info = {
            "vendor": "basler",
            "model": model,
            "serial": str(info.GetSerialNumber()),
            "sensor": sensor,
            "pixel_size_um": pixel,
            "bit_depth": 12,
            "width": w,
            "height": h,
            "usb3": "Usb" in str(info.GetDeviceClass()),
        }
        return dict(self._info)

    def configure(self, gain=0.0):
        cam = self._cam
        try:  # reset any leftover user settings (test pattern, ROI, LUT, ...)
            cam.UserSetSelector.SetValue("Default")
            cam.UserSetLoad.Execute()
        except Exception:
            pass
        # disable auto features so nothing drifts during a sequence
        for node_name in ("GainAuto", "ExposureAuto"):
            node = self._node(cam, node_name)
            if node is not None:
                try:
                    node.SetValue("Off")
                except Exception:
                    pass
        cam.OffsetX.SetValue(0)
        cam.OffsetY.SetValue(0)
        cam.Width.SetValue(cam.SensorWidth.GetValue())
        cam.Height.SetValue(cam.SensorHeight.GetValue())
        cam.PixelFormat.SetValue("Mono12")
        cam.TriggerMode.SetValue("On")
        cam.TriggerSource.SetValue("Software")
        try:
            cam.AcquisitionFrameRateEnable.SetValue(False)
        except Exception:
            pass
        try:
            cam.Gain.SetValue(float(gain))
        except Exception as exc:
            raise RuntimeError(f"SetGain failed: {exc}")
        try:  # cache the camera's exposure range for snap()-time clamping
            self._exp_min_us = float(cam.ExposureTime.GetMin())
            self._exp_max_us = float(cam.ExposureTime.GetMax())
            print(
                f"  [basler] exposure range: "
                f"{self._exp_min_us:g} us .. {self._exp_max_us / 1e6:g} s"
            )
        except Exception:
            self._exp_min_us = self._exp_max_us = None
        self._info["gain"] = float(gain)
        return self._info

    def sensor_temp_c(self):
        try:
            return float(self._cam.DeviceTemperature.GetValue())
        except Exception:
            return float("nan")

    def snap(self, exposure_s, n_frames):
        cam = self._cam
        exp_us = exposure_s * 1e6  # seconds -> microseconds
        if self._exp_min_us is not None:
            exp_us = min(max(exp_us, self._exp_min_us), self._exp_max_us)
            if exp_us != exposure_s * 1e6:
                print(
                    f"  [basler] exposure {exposure_s * 1e3:g} ms clamped to "
                    f"{exp_us / 1e3:g} ms (camera range "
                    f"{self._exp_min_us:g} us .. {self._exp_max_us / 1e6:g} s)"
                )
        cam.ExposureTime.SetValue(exp_us)
        self.last_exposure_s = exp_us / 1e6  # effective exposure for metadata
        if not self._grabbing:
            cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self._grabbing = True
        stack = np.empty((n_frames, self._h, self._w), dtype=np.uint16)
        for i in range(n_frames):
            frame = self._snap_one(exposure_s)
            if frame.shape != (self._h, self._w):
                raise RuntimeError(
                    f"unexpected frame shape {frame.shape} (expected {self._h, self._w})"
                )
            stack[i] = frame << 4  # DN12 -> DN16, same convention as playerone
        return stack

    def close(self):
        if self._cam is not None:
            try:
                if self._grabbing:
                    self._cam.StopGrabbing()
            except Exception:
                pass
            try:
                self._cam.Close()
            except Exception:
                pass
            self._cam = None
            self._grabbing = False
