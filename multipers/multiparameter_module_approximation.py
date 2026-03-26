from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
from joblib import Parallel, delayed

import multipers.logs as _mp_logs
from multipers._module_approximation_algorithms import (
    _compute_module_approximation_from_slicer,
    _module_approximation_single_input,
)
from multipers.mma_structures import *  # noqa: F401,F403
import multipers.mma_structures as _mma
from multipers.simplex_tree_multi import SimplexTreeMulti_type, is_simplextree_multi
from multipers.slicer import Slicer_type, is_slicer

_MMA_EMPTY = {np.dtype(cls().dtype): cls for cls in _mma.available_pymodules}

AVAILABLE_MMA_FLOAT_DTYPES = tuple(
    dtype for dtype in _MMA_EMPTY if np.issubdtype(dtype, np.floating)
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
    if dtype not in AVAILABLE_MMA_FLOAT_DTYPES:
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
        dtype = next((np.dtype(s.dtype) for s in input if hasattr(s, "dtype")), None)
    else:
        dtype = np.dtype(input.dtype) if hasattr(input, "dtype") else None
    constructor = _MMA_EMPTY.get(dtype, None)

    if isinstance(input, tuple) or isinstance(input, list):
        if not all(is_slicer(s) and (s.is_minpres or len(s) == 0) for s in input):
            raise ValueError(
                "Modules cannot be merged unless they are minimal presentations."
            )
        if not (
            np.unique([s.minpres_degree for s in input if len(s)], return_counts=True)[
                1
            ]
            <= 1
        ).all():
            raise ValueError(
                "Multiple modules are at the same degree, cannot merge modules"
            )
        if len(input) == 0:
            return (
                constructor() if constructor is not None else available_pymodules[0]()
            )
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
        non_empty_modules = tuple(m for m in modules if len(m))
        if len(non_empty_modules) == 0:
            return (
                constructor() if constructor is not None else available_pymodules[0]()
            )
        box = np.array(
            [
                np.min([m.get_box()[0] for m in non_empty_modules], axis=0),
                np.max([m.get_box()[1] for m in non_empty_modules], axis=0),
            ]
        )
        if constructor is None:
            raise ValueError(f"Unsupported module dtype {dtype} for module merge.")
        mod = constructor().set_box(box)
        for i, m in enumerate(modules):
            mod.merge(m, input[i].minpres_degree)
        return mod

    if len(input) == 0:
        if verbose:
            print("Empty input, returning the trivial module.")
        return constructor() if constructor is not None else available_pymodules[0]()

    direction = np.asarray(direction, dtype=np.float64)
    swap_box_coords = np.asarray(tuple(swap_box_coords), dtype=np.int32)
    if box is None:
        box = np.empty((0, 0), dtype=np.float64)
    else:
        box = np.asarray(box, dtype=np.float64)

    return _module_approximation_single_input(
        input=input,
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


__all__ = ["module_approximation", "module_approximation_from_slicer"]
