from __future__ import annotations

from fractions import Fraction
from typing import Optional

import numpy as np

import multipers.logs as _mp_logs
import multipers.array_api.numpy as npapi
from multipers.invariants._utils import _as_slicer


def persistence_landscape(
    filtered_complex,
    degree: Optional[int] = None,
    ks=np.arange(5),
    box=None,
    resolution=(100, 100),
    grid=None,
    direction=None,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-12,
    n_jobs: int = 1,
    vineyard=None,
    plot: bool = False,
    **kwargs,
):
    """Evaluate directional multiparameter persistence landscapes on a 2D grid."""
    if _is_module_approximation(filtered_complex):
        return _module_persistence_landscape(
            filtered_complex,
            degree=degree,
            ks=ks,
            box=box,
            resolution=resolution,
            grid=grid,
            direction=direction,
            n_jobs=n_jobs,
            plot=plot,
            persistence_kwargs=kwargs,
        )

    slicer = _as_landscape_slicer(filtered_complex, degree)
    if slicer.is_squeezed:
        _mp_logs.warn_copy("Got a squeezed input; unsqueezing for direct landscape computation.")
        slicer = slicer.unsqueeze()
    if slicer.num_parameters != 2:
        raise ValueError(
            f"persistence_landscape is currently 2D-only. Got {slicer.num_parameters} parameters."
        )

    degree = _landscape_degree(slicer, degree)
    ks = _landscape_ks(ks)
    if np.any(ks > np.iinfo(np.int32).max):
        raise ValueError("`ks` values are too large.")
    ks = npapi.ascontiguous(ks.astype(np.int32, copy=False))
    persistence_kwargs = _landscape_persistence_kwargs(kwargs)
    ignore_inf = persistence_kwargs.pop("ignore_infinite_filtration_values", True)
    if persistence_kwargs:
        raise TypeError(f"Unsupported landscape persistence kwargs: {sorted(persistence_kwargs)}")
    if _landscape_direction_is_fast(direction):
        raise ValueError("`direction='fast'` is only supported for module approximation inputs.")

    direction = _landscape_direction(direction, slicer.num_parameters)
    explicit_grid = grid is not None
    if explicit_grid:
        grid = _landscape_grid(slicer, box=box, resolution=resolution, grid=grid)
        aligned_grid = _aligned_landscape_grid(grid, direction, rtol=rtol, atol=atol)
    else:
        landscape_box = _landscape_box(slicer, box)
        landscape_resolution = _landscape_resolution(resolution)
        grid, aligned_grid = _direction_aligned_landscape_grid(
            landscape_box,
            landscape_resolution,
            direction,
            rtol=rtol,
            atol=atol,
        )

    if aligned_grid is None:
        if explicit_grid:
            raise ValueError(
                "Landscape grid must be regular and aligned with `direction`; "
                "pass `box` and `resolution` to align by enlarging the box."
            )
        raise ValueError("Could not build a direction-aligned landscape grid.")

    out = np.zeros((len(ks), len(grid[0]), len(grid[1])), dtype=np.float64)
    if len(ks) == 0:
        return out

    worker = _landscape_worker_slicer(
        slicer,
        use_vineyard=_landscape_use_vineyard(slicer, vineyard),
    )
    value_dtype = np.dtype(worker.dtype)
    if not np.issubdtype(value_dtype, np.floating):
        raise ValueError("Direct landscapes require floating slicer filtrations.")
    direction = npapi.ascontiguous(np.asarray(direction, dtype=value_dtype))
    stride_i, stride_j, dt = aligned_grid
    out = np.asarray(
        worker._landscapes_on_grid(
            npapi.ascontiguous(np.asarray(grid[0], dtype=value_dtype)),
            npapi.ascontiguous(np.asarray(grid[1], dtype=value_dtype)),
            direction,
            int(stride_i),
            int(stride_j),
            float(dt),
            int(degree),
            ks,
            int(n_jobs),
            bool(ignore_inf),
        )
    )

    if plot:
        _plot_landscape(out, grid)

    return out


def _plot_landscape(out, grid):
    from multipers.plots import plot_surface

    plot_surface(grid=grid, hf=np.sum(out, axis=0))


def _is_module_approximation(obj):
    from multipers.multiparameter_module_approximation import available_pymodules

    return bool(available_pymodules) and isinstance(obj, available_pymodules)


def _module_persistence_landscape(
    module,
    *,
    degree,
    ks,
    box,
    resolution,
    grid,
    direction,
    n_jobs,
    plot,
    persistence_kwargs,
):
    if direction is not None and direction != "fast":
        raise ValueError(
            "Module approximation landscapes do not support `direction`; pass a filtered complex for direct directional landscapes."
        )
    if degree is None:
        raise ValueError("`degree` is required for module approximation inputs.")

    ks = _landscape_ks(ks)
    if np.any(ks > np.iinfo(np.int32).max):
        raise ValueError("`ks` values are too large.")
    ks = ks.astype(np.int32, copy=False)
    persistence_kwargs = _landscape_persistence_kwargs(persistence_kwargs)
    if persistence_kwargs:
        raise TypeError(f"Unsupported landscape persistence kwargs: {sorted(persistence_kwargs)}")

    if grid is not None:
        grid = _landscape_grid(module, box=box, resolution=resolution, grid=grid)
        out = np.asarray(module.landscapes(degree=degree, ks=ks, grid=grid, n_jobs=n_jobs))
        if plot:
            _plot_landscape(out, grid)
        return out

    return np.asarray(
        module.landscapes(
            degree=degree,
            ks=ks,
            box=box,
            resolution=resolution,
            n_jobs=n_jobs,
            plot=plot,
        )
    )


def _as_landscape_slicer(filtered_complex, degree):
    from multipers.slicer import is_slicer

    if isinstance(filtered_complex, (list, tuple)) and is_slicer(filtered_complex):
        if degree is None:
            if len(filtered_complex) != 1:
                raise ValueError("Provide `degree` when passing several minpres slicers.")
            return filtered_complex[0]
        matches = [
            slicer
            for slicer in filtered_complex
            if slicer.minpres_degree < 0 or slicer.minpres_degree == int(degree)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one minpres slicer for degree {degree}.")
        return matches[0]
    return _as_slicer(filtered_complex)


def _landscape_degree(slicer, degree):
    if slicer.is_minpres:
        if degree is None:
            degree = slicer.minpres_degree
        elif slicer.minpres_degree >= 0 and slicer.minpres_degree != int(degree):
            raise ValueError(
                "Cannot change degree of an already minimal-presentation slicer."
            )
        if degree is None or int(degree) < 0:
            raise ValueError("Minimal-presentation input has no valid `minpres_degree`.")
        return int(degree)
    if degree is None:
        raise ValueError("`degree` is inferred for minpres inputs, otherwise required.")
    return int(degree)


def _landscape_ks(ks):
    ks = np.asarray(ks, dtype=np.int64).reshape(-1)
    if np.any(ks < 0):
        raise ValueError("`ks` must contain nonnegative integers.")
    return ks


def _landscape_direction_is_fast(direction):
    if not isinstance(direction, str):
        return False
    if direction == "fast":
        return True
    raise ValueError("`direction` must be 'fast', None, or a positive direction vector.")


def _landscape_direction(direction, num_parameters):
    if direction is None:
        direction = np.ones(num_parameters, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    if direction.shape != (num_parameters,):
        raise ValueError(
            f"Expected direction with shape ({num_parameters},). Got {direction.shape}."
        )
    if not np.all(np.isfinite(direction)):
        raise ValueError("`direction` must contain finite values.")
    if np.any(direction <= 0):
        raise ValueError("`direction` must be strictly positive in every coordinate.")
    return direction


def _landscape_grid(slicer, *, box, resolution, grid):
    if grid is not None:
        if len(grid) != 2:
            raise ValueError("`grid` must contain two coordinate arrays.")
        out = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in grid)
        if any(axis.size == 0 for axis in out):
            raise ValueError("Landscape grid axes must be non-empty.")
        if any(not np.all(np.isfinite(axis)) for axis in out):
            raise ValueError("Landscape grid coordinates must be finite.")
        return out

    return _landscape_grid_from_box(_landscape_box(slicer, box), _landscape_resolution(resolution))


def _landscape_box(slicer, box):
    if box is None:
        box = slicer.filtration_bounds()
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (2, 2):
        raise ValueError(f"Expected `box` with shape (2, 2). Got {box.shape}.")
    if not np.all(np.isfinite(box)):
        raise ValueError("`box` must contain finite values.")
    return box


def _landscape_resolution(resolution):
    if np.ndim(resolution) == 0:
        size = int(np.asarray(resolution).item())
        resolution = (size, size)
    else:
        resolution = tuple(int(r) for r in resolution)
    if len(resolution) != 2 or any(r <= 0 for r in resolution):
        raise ValueError("`resolution` must be a positive integer or a pair of positive integers.")
    return resolution


def _landscape_grid_from_box(box, resolution):
    return tuple(
        box[0, axis]
        + (box[1, axis] - box[0, axis]) / resolution[axis] * np.arange(resolution[axis])
        for axis in range(2)
    )


def _direction_aligned_landscape_grid(box, resolution, direction, *, rtol, atol):
    lower = box[0]
    upper = box[1]
    widths = np.maximum(upper - lower, 0.0)
    t = float(np.max(widths / direction))
    if not np.isfinite(t) or t <= 0.0:
        return _landscape_grid_from_box(box, resolution), None

    aligned_box = np.vstack((lower, lower + t * direction))
    grid = _landscape_grid_from_box(aligned_box, resolution)
    cropped = []
    for axis, limit in zip(grid, upper, strict=True):
        stop = int(np.searchsorted(axis, limit + atol + abs(limit) * rtol, side="right"))
        cropped.append(axis[: max(stop, 1)])
    cropped = tuple(cropped)
    divisor = int(np.gcd.reduce(np.asarray(resolution, dtype=np.int64)))
    aligned_grid = (int(resolution[0]) // divisor, int(resolution[1]) // divisor, t / divisor)
    return cropped, aligned_grid


def _regular_grid_step(axis, *, rtol, atol):
    if axis.size <= 1:
        return None
    steps = np.diff(axis)
    if np.any(steps <= 0):
        return None
    if not np.allclose(steps, steps[0], rtol=rtol, atol=atol):
        return None
    return float(steps[0])


def _aligned_landscape_grid(grid, direction, *, rtol, atol):
    xgrid, ygrid = grid
    nx, ny = len(xgrid), len(ygrid)
    if nx == 1 or ny == 1:
        return None
    step_x = _regular_grid_step(xgrid, rtol=rtol, atol=atol)
    step_y = _regular_grid_step(ygrid, rtol=rtol, atol=atol)
    if step_x is None or step_y is None:
        return None

    return _landscape_stride(
        step_x,
        step_y,
        direction,
        max_denominator=max(nx, ny, 1),
        rtol=rtol,
        atol=atol,
    )


def _landscape_stride(step_x, step_y, direction, *, max_denominator, rtol, atol):
    target = direction[0] * step_y / (direction[1] * step_x)
    fraction = Fraction(float(target)).limit_denominator(max_denominator)
    stride_i = int(fraction.numerator)
    stride_j = int(fraction.denominator)
    if stride_i <= 0 or stride_j <= 0:
        return None
    dt_i = stride_i * step_x / direction[0]
    dt_j = stride_j * step_y / direction[1]
    if not np.isclose(dt_i, dt_j, rtol=rtol, atol=atol):
        return None
    return stride_i, stride_j, float((dt_i + dt_j) / 2.0)


def _landscape_persistence_kwargs(kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("keep_inf", None)
    kwargs.pop("full", None)
    return kwargs


def _landscape_use_vineyard(slicer, vineyard):
    if vineyard is None:
        return bool(slicer.is_vine)
    if vineyard in (True, False):
        return bool(vineyard)
    raise ValueError("`vineyard` must be True, False, or None.")


def _landscape_worker_slicer(slicer, *, use_vineyard, copy=False, warn_copy=True):
    from multipers._slicer_meta import Slicer

    if use_vineyard:
        if slicer.is_vine:
            return Slicer(slicer, vineyard=True, backend="Matrix") if copy else slicer
        if warn_copy:
            _mp_logs.warn_copy(
                "Got a non-vine slicer as an input. Use `vineyard=True` to remove this copy."
            )
        return slicer.astype(vineyard=True, pers_backend="Matrix")
    if not slicer.is_vine:
        return Slicer(slicer, vineyard=False) if copy else slicer

    return Slicer(slicer, vineyard=False)


__all__ = ["persistence_landscape"]
