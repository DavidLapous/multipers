"""Runtime warning classes and toggles for multipers.

Usage:

    import multipers as mp
    mp.logs.CopyWarning.enabled = False
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from warnings import warn

from . import _slicer_nanobind


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
_BACKEND_LOG_BITS = {
    "mpfree": 1 << 0,
    "multi_critical": 1 << 1,
    "function_delaunay": 1 << 2,
    "2pac": 1 << 3,
}


def _get_backend_log_mask() -> int:
    return int(_slicer_nanobind._get_backend_log_mask())


def _set_backend_log_mask(mask: int) -> None:
    _slicer_nanobind._set_backend_log_mask(int(mask))


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
    """Enable or disable raw stdout coming from all external backends.

    This policy is global to the current Python process. It is intended for
    debugging only. Do not toggle it while threaded backend computations are
    already running.
    """

    set_ext_log_policy(
        mpfree=enabled,
        multi_critical=enabled,
        function_delaunay=enabled,
        twopac=enabled,
    )


def set_ext_log_policy(
    *,
    mpfree: bool | None = None,
    multi_critical: bool | None = None,
    function_delaunay: bool | None = None,
    twopac: bool | None = None,
) -> None:
    """Set raw backend log policy for external backends.

    Each argument updates one backend when not ``None``. This policy is process-global.
    It is intended for debugging only. Do not toggle it while threaded backend
    computations are already running.
    """

    mask = _get_backend_log_mask()
    updates = {
        "mpfree": mpfree,
        "multi_critical": multi_critical,
        "function_delaunay": function_delaunay,
        "2pac": twopac,
    }
    for backend, enabled in updates.items():
        if enabled is None:
            continue
        bit = _BACKEND_LOG_BITS[backend]
        if enabled:
            mask |= bit
        else:
            mask &= ~bit
    _set_backend_log_mask(mask)


def ext_log_policy() -> dict[str, bool]:
    mask = _get_backend_log_mask()
    return {backend: bool(mask & bit) for backend, bit in _BACKEND_LOG_BITS.items()}


def ext_log_enabled(backend: str | None = None) -> bool:
    policy = ext_log_policy()
    if backend is None:
        return any(policy.values())
    if backend not in policy:
        raise KeyError(f"Unknown backend {backend!r}. Expected one of {tuple(policy)}.")
    return policy[backend]
