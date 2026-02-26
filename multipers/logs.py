"""Runtime warning classes and toggles for multipers.

Usage:

    import multipers as mp
    mp.logs.CopyWarning.enabled = False
"""

from __future__ import annotations

from warnings import warn


class MultipersWarning(UserWarning):
    enabled: bool = True


class CopyWarning(MultipersWarning):
    enabled: bool = False


class SuperfluousComputationWarning(MultipersWarning):
    enabled: bool = True


class AutodiffWarning(MultipersWarning):
    enabled: bool = True


class ExperimentalWarning(MultipersWarning):
    enabled: bool = True


class FallbackWarning(MultipersWarning):
    enabled: bool = True


class GeometryWarning(MultipersWarning):
    enabled: bool = True


_WARNING_CLASSES = {
    "copy": CopyWarning,
    "superfluous_computation": SuperfluousComputationWarning,
    "autodiff": AutodiffWarning,
    "experimental": ExperimentalWarning,
    "fallback": FallbackWarning,
    "geometry": GeometryWarning,
}

_DEFAULTS = {
    CopyWarning: False,
    SuperfluousComputationWarning: True,
    AutodiffWarning: True,
    ExperimentalWarning: True,
    FallbackWarning: True,
    GeometryWarning: True,
}

_TOGGLE_HINT_SHOWN = {key: False for key in _WARNING_CLASSES}


def _emit(kind: str, message: str, stacklevel: int = 2) -> None:
    warning_cls = _WARNING_CLASSES[kind]
    if not warning_cls.enabled:
        return
    if not _TOGGLE_HINT_SHOWN[kind]:
        message = (
            f"{message} "
            f"Set `multipers.logs.{warning_cls.__name__}.enabled = False` to disable this warning."
        )
        _TOGGLE_HINT_SHOWN[kind] = True
    warn(message, category=warning_cls, stacklevel=stacklevel)


def warn_copy(message: str, *, stacklevel: int = 2) -> None:
    _emit("copy", message, stacklevel=stacklevel)


def warn_superfluous_computation(message: str, *, stacklevel: int = 2) -> None:
    _emit("superfluous_computation", message, stacklevel=stacklevel)


def warn_autodiff(message: str, *, stacklevel: int = 2) -> None:
    _emit("autodiff", message, stacklevel=stacklevel)


def warn_experimental(message: str, *, stacklevel: int = 2) -> None:
    _emit("experimental", message, stacklevel=stacklevel)


def warn_fallback(message: str, *, stacklevel: int = 2) -> None:
    _emit("fallback", message, stacklevel=stacklevel)


def warn_geometry(message: str, *, stacklevel: int = 2) -> None:
    _emit("geometry", message, stacklevel=stacklevel)


def set_level(level: int) -> None:
    if level == 0:
        for warning_cls in _DEFAULTS:
            warning_cls.enabled = False
        return
    if level == 1:
        for warning_cls, enabled in _DEFAULTS.items():
            warning_cls.enabled = enabled
        return
    if level == 2:
        for warning_cls in _DEFAULTS:
            warning_cls.enabled = True
        return
    raise ValueError(f"Invalid level {level}. Expected 0, 1, or 2.")
