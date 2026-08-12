"""Camera backend interface."""

from abc import ABC, abstractmethod


class CameraBackend(ABC):
    """A camera acquisition backend.

    Lifecycle contract:
      - open(): find and open the camera (may retry; never raises before giving up)
      - configure(): full-frame, deepest raw format, bin 1, fixed gain
      - snap(exposure_s, n_frames): capture n_frames at exposure_s, return stack
      - close(): release resources. Backends where close() wedges the device
        (Player One on macOS) may implement close() as a no-op and document why:
        process exit performs the kernel cleanup instead.
    """

    vendor = "unknown"

    @abstractmethod
    def open(self):
        """Find and open the camera. Returns camera info dict."""

    @abstractmethod
    def configure(self, gain=0):
        """Configure imaging: full frame, raw format, bin 1, fixed gain."""

    @abstractmethod
    def snap(self, exposure_s, n_frames):
        """Capture n_frames at exposure_s (seconds). Returns uint16 stack (n, h, w)."""

    @abstractmethod
    def close(self):
        """Release the camera."""
