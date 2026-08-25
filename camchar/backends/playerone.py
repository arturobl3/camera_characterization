"""Player One Astronomy camera backend.

Pitfalls this backend encodes (all found empirically on macOS, SDK 3.10.1):
  1. Vendor Python wrapper ABI bug: POASetConfig/POAGetConfig pass c_int where the
     C API takes an 8-byte POAConfigValue union (long intValue / double floatValue).
     Every config write is silently corrupted (e.g. exposure clamped to its 10 us
     minimum). Fix: call the DLL directly with a proper ctypes Union.
  2. CloseCamera() wedges the device on macOS: after a clean close, subsequent
     processes get POA_ERROR_INVALID_ID until the camera is physically replugged.
     Fix: never call CloseCamera; let process exit do the kernel cleanup.
  3. USB enumeration is flaky (device drops and re-enumerates; a hub makes it
     worse). Fix: retry enumeration at open(), and prefer direct USB connection.
  4. libusb dependency: the SDK dylib loads @rpath/libusb-1.0.0.dylib. On macOS,
     install libusb via Homebrew and patch the vendored dylib:
       install_name_tool -change @rpath/libusb-1.0.0.dylib \
         /usr/local/opt/libusb/lib/libusb-1.0.0.dylib libPlayerOneCamera.3.10.1.dylib
"""

import os
import sys
import time
from ctypes import POINTER, Union, byref, c_double, c_int, c_long

import numpy as np

from . import register
from .base import CameraBackend

_VENDOR_PY = os.path.join(
    os.path.dirname(__file__), "..", "..", "vendor", "playerone", "py"
)
sys.path.insert(0, _VENDOR_PY)
import pyPOACamera as poa  # noqa: E402


class _POAConfigValue(Union):
    _fields_ = [("intValue", c_long), ("floatValue", c_double), ("boolValue", c_int)]


dll = poa.dll
dll.POASetConfig.argtypes = [c_int, c_int, _POAConfigValue, c_int]
dll.POASetConfig.restype = c_int


def _set_cfg(cid, conf_id, value, is_auto=0):
    return poa.POAErrors(dll.POASetConfig(cid, conf_id, value, is_auto))


def _get_cfg_int(cid, conf_id):
    dll.POAGetConfig.argtypes = [c_int, c_int, POINTER(_POAConfigValue), POINTER(c_int)]
    dll.POAGetConfig.restype = c_int
    v, auto = _POAConfigValue(), c_int(0)
    e = dll.POAGetConfig(cid, conf_id, byref(v), byref(auto))
    return poa.POAErrors(e), v.intValue, bool(auto.value)


def _get_cfg_float(cid, conf_id):
    dll.POAGetConfig.argtypes = [c_int, c_int, POINTER(_POAConfigValue), POINTER(c_int)]
    dll.POAGetConfig.restype = c_int
    v, auto = _POAConfigValue(), c_int(0)
    e = dll.POAGetConfig(cid, conf_id, byref(v), byref(auto))
    return poa.POAErrors(e), v.floatValue, bool(auto.value)


def _errstr(e):
    """GetErrorString wants a POAErrors enum; direct-DLL calls return raw ints."""
    try:
        return poa.GetErrorString(poa.POAErrors(e))
    except Exception:
        return f"error code {e}"


# config IDs (see PlayerOneCamera.h)
_CFG_EXPOSURE_US = 0
_CFG_GAIN = 1
_CFG_TEMP = 3
_CFG_EXP_S = 31


@register("playerone")
class PlayerOneBackend(CameraBackend):
    vendor = "playerone"

    def __init__(self, retry_seconds=60):
        self._retry = retry_seconds
        self._cid = None
        self._props = None
        self._info = None
        self._buf = None
        self._w = self._h = 0

    # ---------- lifecycle ----------
    def open(self):
        t0 = time.time()
        props = None
        while time.time() - t0 < self._retry:
            try:
                n = poa.GetCameraCount()
                if n > 0:
                    e, props = poa.GetCameraProperties(0)
                    if e == poa.POAErrors.POA_OK:
                        break
            except Exception as exc:  # SDK can throw on a half-dead device
                print(f"  [playerone] enumeration hiccup: {exc}")
            time.sleep(2)
        if props is None:
            raise RuntimeError(
                "Player One camera not found after %ds. Check USB connection "
                "(prefer direct, no hub) and replug if needed." % self._retry
            )

        self._cid = props.cameraID
        self._props = props
        e = poa.OpenCamera(self._cid)
        if e != poa.POAErrors.POA_OK:
            raise RuntimeError(f"OpenCamera failed: {_errstr(e)}")
        e = poa.InitCamera(self._cid)
        if e != poa.POAErrors.POA_OK:
            raise RuntimeError(f"InitCamera failed: {_errstr(e)}")

        self._info = {
            "vendor": "playerone",
            "model": props.cameraModelName.decode().strip(),
            "serial": props.SN.decode().strip(),
            "sensor": props.sensorModelName.decode().strip(),
            "pixel_size_um": float(props.pixelSize),
            "bit_depth": int(props.bitDepth),
            "width": int(props.maxWidth),
            "height": int(props.maxHeight),
            "usb3": bool(props.isUSB3Speed),
        }
        return dict(self._info)

    def configure(self, gain=0):
        if float(gain) != int(gain):
            raise RuntimeError(
                f"Player One gain must be an integer, got {gain!r} (basler uses dB)"
            )
        p = self._props
        poa.SetImageStartPos(self._cid, 0, 0)
        poa.SetImageSize(self._cid, p.maxWidth, p.maxHeight)
        poa.SetImageBin(self._cid, 1)
        poa.SetImageFormat(self._cid, poa.POAImgFormat.POA_RAW16)
        e = _set_cfg(self._cid, _CFG_GAIN, _POAConfigValue(intValue=int(gain)))
        if e != poa.POAErrors.POA_OK:
            raise RuntimeError(f"SetGain failed: {_errstr(e)}")
        e, w, h = poa.GetImageSize(self._cid)
        self._w, self._h = int(w), int(h)
        self._buf = np.zeros(w * h * 2, dtype=np.uint8)
        self._info["gain"] = int(gain)
        return self._info

    def sensor_temp_c(self):
        e, t, _ = _get_cfg_float(self._cid, _CFG_TEMP)
        return float(t) if e == poa.POAErrors.POA_OK else float("nan")

    def _set_exposure_s(self, exposure_s):
        e = _set_cfg(
            self._cid, _CFG_EXP_S, _POAConfigValue(floatValue=float(exposure_s))
        )
        if e != poa.POAErrors.POA_OK:
            # fall back to integer microseconds
            e = _set_cfg(
                self._cid,
                _CFG_EXPOSURE_US,
                _POAConfigValue(intValue=int(exposure_s * 1e6)),
            )
        if e != poa.POAErrors.POA_OK:
            raise RuntimeError(f"SetExposure failed: {_errstr(e)}")

    def snap(self, exposure_s, n_frames):
        self._set_exposure_s(exposure_s)
        stack = np.empty((n_frames, self._h, self._w), dtype=np.uint16)
        for i in range(n_frames):
            e = poa.StartExposure(self._cid, True)  # snap mode: single frame
            if e != poa.POAErrors.POA_OK:
                raise RuntimeError(f"StartExposure failed: {_errstr(e)}")
            t0 = time.time()
            while True:
                e, st = poa.GetCameraState(self._cid)
                if st != poa.POACameraState.STATE_EXPOSING:
                    break
                if time.time() - t0 > exposure_s + 20:
                    raise RuntimeError("exposure did not complete in time")
                time.sleep(0.02)
            e, ready = poa.ImageReady(self._cid)
            e = poa.GetImageData(self._cid, self._buf, int(exposure_s * 1000) + 5000)
            if e != poa.POAErrors.POA_OK:
                raise RuntimeError(f"GetImageData failed: {_errstr(e)}")
            stack[i] = self._buf.view(np.uint16).reshape(self._h, self._w)
        return stack

    def close(self):
        """No-op on purpose: calling CloseCamera() on macOS wedges the device for
        the next process (needs a physical replug to recover). Process exit is the
        cleanup. See module docstring."""
