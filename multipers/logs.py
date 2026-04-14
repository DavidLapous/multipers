"""Runtime warning classes and toggles for multipers.

Usage:

    import multipers as mp
    mp.logs.CopyWarning.enabled = False
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from time import perf_counter
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
_EXT_LOG_ENABLED = False


def _emit(kind: str, message: str, stacklevel: int = 2) -> None:
    warning_cls = _WARNING_CLASSES[kind]
    if not warning_cls.enabled:
        return
    if not _TOGGLE_HINT_SHOWN[kind]:
        message = (
            f"{message} "
            f"\nSet `multipers.logs.{warning_cls.__name__}.enabled = False` to disable this warning."
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


def log_verbose(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


class _Timings:
    def __init__(
        self,
        name: str,
        *,
        enabled: bool,
        details: Mapping[str, object] | None = None,
        parent: "_Timings | None" = None,
        label: str | None = None,
    ):
        self.parent = parent
        self.label = name if label is None else label
        self.name = name if parent is None else f"{parent.name}:{self.label}"
        self.enabled = enabled
        merged_details = {}
        if parent is not None:
            merged_details.update(parent.details)
        if details is not None:
            merged_details.update({str(key): value for key, value in details.items()})
        self.details = merged_details
        self._start = perf_counter()
        self._prev = self._start
        self._substeps: list[tuple[str, float]] = []
        self._stats: dict[str, object] = {}

    def _format_details(self) -> str:
        if not self.details:
            return ""
        return " " + " ".join(f"{key}={value}" for key, value in self.details.items())

    def _format_substeps(self) -> str:
        if not self._substeps:
            return ""
        return " " + " ".join(f"{name}={seconds:.3f}s" for name, seconds in self._substeps)

    def _format_stats(self) -> str:
        if not self._stats:
            return ""
        return " " + " ".join(f"{key}={value}" for key, value in self._stats.items())

    def __enter__(self):
        if self.enabled:
            print(f"[{self.name}]{self._format_details()}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        finished_at = perf_counter()
        total = finished_at - self._start
        if self.enabled:
            print(
                f"[{self.name}][Done ({total:.3f}s)]{self._format_substeps()}{self._format_stats()}",
                flush=True,
            )
        if self.parent is not None:
            self.parent._substeps.append((self.label, total))
            self.parent._prev = finished_at
        return False

    def substep(self, label: str) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        self._substeps.append((label, now - self._prev))
        self._prev = now

    def step(self, label: str, *, details: Mapping[str, object] | None = None) -> "_Timings":
        return _Timings(label, enabled=self.enabled, details=details, parent=self, label=label)

    def add_stats(self, stats: Mapping[str, object] | None) -> None:
        if stats is None:
            return
        self._stats.update({str(key): value for key, value in stats.items()})

    def total(self) -> float:
        return perf_counter() - self._start


def timings(name: str, *, enabled: bool, details: Mapping[str, object] | None = None) -> _Timings:
    return _Timings(name, enabled=enabled, details=details)


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


def enable_ext_log(enabled: bool = True) -> None:
    """Enable or disable raw stdout coming from external backends.

    This flag is global to the current Python process. It is intended for
    debugging only. Do not toggle it while threaded backend computations are
    already running.
    """

    global _EXT_LOG_ENABLED
    _EXT_LOG_ENABLED = bool(enabled)
    for module_name in (
        "multipers._multi_critical_interface",
        "multipers._mpfree_interface",
    ):
        try:
            module = import_module(module_name)
        except Exception:
            continue
        setter = getattr(module, "_set_backend_stdout", None)
        if setter is not None:
            setter(_EXT_LOG_ENABLED)


def compiled_backend_log_flags() -> dict[str, bool | None]:
    """Return the backend log flags compiled into the installed native modules."""

    out: dict[str, bool | None] = {
        "mpfree": None,
        "2pac": None,
        "multi_critical": None,
        "function_delaunay": None,
    }
    for module_name in (
        "multipers._mpfree_interface",
        "multipers._2pac_interface",
        "multipers._multi_critical_interface",
        "multipers._function_delaunay_interface",
    ):
        try:
            module = import_module(module_name)
        except Exception:
            continue
        getter = getattr(module, "_compiled_log_flags", None)
        if getter is None:
            continue
        for key, value in dict(getter()).items():
            out[str(key)] = bool(value)
    return out


def ext_log_enabled() -> bool:
    return _EXT_LOG_ENABLED
