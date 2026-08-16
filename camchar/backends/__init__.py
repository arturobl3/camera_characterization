"""Backend abstraction: each camera vendor implements CameraBackend."""

from .base import CameraBackend

__all__ = ["CameraBackend", "get_backend", "register"]

_BACKENDS = {}


def register(name):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls

    return deco


def get_backend(name):
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[name]()


try:
    from . import playerone  # noqa: E402, F401  (registers the playerone backend on import)
except OSError as _e:
    # SDK binary for this OS is missing (vendor/playerone/lib/). Offline
    # commands (analyze) must still work; fail only if the backend is used.
    import warnings

    warnings.warn(
        f"playerone backend unavailable (SDK library not installed): {_e}",
        stacklevel=2,
    )

try:
    from . import basler  # noqa: E402, F401  (registers the basler backend on import)
except (ImportError, OSError) as _e:
    # basler.py self-guards the pypylon import (sets pylon = None); this only
    # fires for pypylon DLL-load failures (OSError) or unrelated import errors.
    # Offline commands must still work.
    import warnings

    warnings.warn(
        f"basler backend unavailable (pypylon not installed): {_e}", stacklevel=2
    )
