from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
from joblib import Parallel, delayed

import multipers.logs as _mp_logs
from multipers._module_approximation_algorithms import (
    _compute_module_approximation_from_slicer,
)
from multipers.mma_structures import *  # noqa: F401,F403
import multipers.mma_structures as _mma
from multipers.simplex_tree_multi import SimplexTreeMulti_type, is_simplextree_multi
from multipers.slicer import Slicer_type, is_slicer


TRACE_MMA_OPS = False

_MMA_FROM_DUMP = {
    np.dtype(getattr(_mma, f"PyModule_{name[len('from_dump_') :]}")().dtype): value
    for name, value in vars(_mma).items()
    if name.startswith("from_dump_")
}

AVAILABLE_MMA_FLOAT_DTYPES = tuple(
    dtype for dtype in _MMA_FROM_DUMP if np.issubdtype(dtype, np.floating)
)


def module_approximation_from_slicer(
    slicer: Slicer_type,
    box: Optional[np.ndarray] = None,
    max_error=-1,
    complete: bool = True,
    threshold: bool = False,
    verbose: bool = False,
    direction: list[float] | np.ndarray = [],
    warnings: bool = True,
    unsqueeze_grid=None,
    n_jobs: int = -1,
) -> PyModule_type:
    if not slicer.is_vine:
        if warnings:
            _mp_logs.warn_copy(
                r"Got a non-vine slicer as an input. Use `vineyard=True` to remove this copy."
            )
        from multipers._slicer_meta import Slicer

        slicer = Slicer(slicer, vineyard=True, backend="matrix")

    direction_ = np.ascontiguousarray(direction, dtype=slicer.dtype)
    if box is None:
        box = slicer.filtration_bounds()

    dtype = np.dtype(slicer.dtype)
    if dtype not in _MMA_FROM_DUMP:
        supported = tuple(dt.name for dt in AVAILABLE_MMA_FLOAT_DTYPES)
        raise ValueError(
            f"Slicer must be float-like and enabled in options.py. Got {slicer.dtype}. Supported dtypes: {supported}."
        )

    approx_mod = _compute_module_approximation_from_slicer(
        slicer,
        direction_,
        max_error,
        np.asarray(box, dtype=dtype),
        threshold,
        complete,
        verbose,
        n_jobs,
    )

    if unsqueeze_grid is not None:
        if verbose:
            print("Reevaluating module in filtration grid...", end="", flush=True)
        approx_mod.evaluate_in_grid(unsqueeze_grid)
        from multipers.grids import compute_bounding_box

        if len(approx_mod):
            approx_mod.set_box(compute_bounding_box(approx_mod))
        if verbose:
            print("Done.", flush=True)

    return approx_mod


def module_approximation(
    input: Union[SimplexTreeMulti_type, Slicer_type, tuple],
    box: Optional[np.ndarray] = None,
    max_error: float = -1,
    nlines: int = 557,
    from_coordinates: bool = False,
    complete: bool = True,
    threshold: bool = False,
    verbose: bool = False,
    ignore_warnings: bool = False,
    direction: Iterable[float] = (),
    swap_box_coords: Iterable[int] = (),
    *,
    n_jobs: int = -1,
) -> PyModule_type:
    if isinstance(input, tuple) or isinstance(input, list):
        assert all(is_slicer(s) and (s.is_minpres or len(s) == 0) for s in input), (
            "Modules cannot be merged unless they are minimal presentations."
        )
        assert (
            np.unique([s.minpres_degree for s in input if len(s)], return_counts=True)[
                1
            ]
            <= 1
        ).all(), "Multiple modules are at the same degree, cannot merge modules"
        if len(input) == 0:
            return available_pymodules[0]()
        modules = tuple(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(module_approximation)(
                    input=slicer,
                    box=box,
                    max_error=max_error,
                    nlines=nlines,
                    from_coordinates=from_coordinates,
                    complete=complete,
                    threshold=threshold,
                    verbose=verbose,
                    ignore_warnings=ignore_warnings,
                    direction=direction,
                    swap_box_coords=swap_box_coords,
                    n_jobs=n_jobs,
                )
                for slicer in input
            )
        )
        box = np.array(
            [
                np.min([m.get_box()[0] for m in modules if len(m)], axis=0),
                np.max([m.get_box()[1] for m in modules if len(m)], axis=0),
            ]
        )
        module_dtype = np.dtype(modules[0].dtype)
        constructor = _MMA_FROM_DUMP.get(module_dtype)
        if constructor is None:
            raise ValueError(
                f"Unsupported module dtype {module_dtype} for module merge."
            )
        mod = constructor((box, tuple()))
        for i, m in enumerate(modules):
            mod.merge(m, input[i].minpres_degree)
        return mod

    if len(input) == 0:
        if verbose:
            print("Empty input, returning the trivial module.")
        return available_pymodules[0]()

    if is_simplextree_multi(input):
        from multipers._slicer_meta import Slicer

        input = Slicer(input, backend="matrix", vineyard=True)

    direction = np.asarray(direction, dtype=np.float64)
    is_degenerate = np.any(direction == 0) if direction.size else False
    if np.any(direction < 0):
        raise ValueError(f"Got an invalid negative direction. {direction=}")
    if is_degenerate and not ignore_warnings:
        _mp_logs.warn_geometry(
            "Got a degenerate direction. This function may fail if the first line is not generic."
        )

    if from_coordinates and not input.is_squeezed:
        if verbose:
            print("Preparing filtration (squeeze)... ", end="", flush=True)
        if not ignore_warnings:
            _mp_logs.warn_copy("Got a non-squeezed input with `from_coordinates=True`.")
        input = input.grid_squeeze()
        if verbose:
            print("Done.", flush=True)

    unsqueeze_grid = None
    if input.is_squeezed:
        if verbose:
            print("Preparing filtration (unsqueeze)... ", end="", flush=True)
        if from_coordinates:
            from multipers.grids import sanitize_grid

            unsqueeze_grid = sanitize_grid(
                input.filtration_grid, numpyfy=True, add_inf=True
            )
            input = input.astype(dtype=np.float64)
            if direction.size == 0:
                _direction = np.asarray(
                    [1 / g.size for g in unsqueeze_grid], dtype=np.float64
                )
                _direction /= np.sqrt((_direction**2).sum())
                direction = _direction
            if verbose:
                print(f"Updated  `{direction=}`, and `{max_error=}` ", end="")
        else:
            if not ignore_warnings:
                _mp_logs.warn_copy("Got a squeezed input.")
            input = input.unsqueeze()
        if verbose:
            print("Done.", flush=True)

    if box is None:
        if verbose:
            print("No box given. Using filtration bounds to infer it.")
        box = input.filtration_bounds()
        if verbose:
            print(f"Using {box=}.", flush=True)

    box = np.asarray(box, dtype=np.float64)
    if box.ndim != 2:
        raise ValueError(f"Invalid box dimension. Got {box.ndim=} != 2")
    scales = box[1] - box[0]
    scales /= scales.max()
    if np.any(scales < 0.1):
        _mp_logs.warn_geometry(
            f"Squewed filtration detected. Found {scales=}. Consider rescaling the filtration for interpretable results."
        )

    zero_idx = box[1] == box[0]
    if np.any(zero_idx):
        if not ignore_warnings:
            _mp_logs.warn_geometry(
                f"Got {(box[1] == box[0])=} trivial box coordinates."
            )
        box[1] += zero_idx

    for i in swap_box_coords:
        box[[0, 1], i] = box[[1, 0], i]
    num_parameters = box.shape[1]
    assert direction.size == 0 or direction.size == num_parameters, (
        f"Invalid line direction size, has to be 0 or {num_parameters=}"
    )

    prod = sum(
        np.abs(box[1] - box[0])[:i].prod() * np.abs(box[1] - box[0])[i + 1 :].prod()
        for i in range(0, num_parameters)
        if (direction.size == 0 or direction[i] != 0)
    )
    if max_error <= 0:
        max_error = (prod / nlines) ** (1 / (num_parameters - 1))

    estimated_nlines = prod / (max_error ** (num_parameters - 1))
    if not ignore_warnings and estimated_nlines >= 10_000:
        raise ValueError(
            f"""
Warning : the number of lines (around {np.round(estimated_nlines)}) may be too high.
This may be due to extreme box or filtration bounds :

{box=}

Try to increase the precision parameter, or set `ignore_warnings=True` to compute this module.
Returning the trivial module.
"""
        )

    if not is_slicer(input):
        raise ValueError("First argument must be a simplextree or a slicer !")

    return module_approximation_from_slicer(
        slicer=input,
        box=box,
        max_error=max_error,
        complete=complete,
        threshold=threshold,
        verbose=verbose,
        direction=direction,
        unsqueeze_grid=unsqueeze_grid,
        n_jobs=n_jobs,
    )


__all__ = ["module_approximation", "module_approximation_from_slicer", "TRACE_MMA_OPS"]
