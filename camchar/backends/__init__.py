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


from . import playerone  # noqa: E402, F401  (registers the playerone backend on import)
