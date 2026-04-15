import os
from typing import Optional

import numpy as np

import multipers.slicer as mps
from multipers.simplex_tree_multi import is_simplextree_multi
from multipers.slicer import (
    _column_type,
    _filtration_container_type,
    _valid_dtype,
    _valid_pers_backend,
    is_slicer,
)


def _slicer_from_simplextree(st, backend, vineyard):
    backend = backend.lower() if isinstance(backend, str) else backend
    if vineyard:
        if backend == "matrix":
            slicer = mps._SlicerVineSimplicial(st)
        elif backend == "clement":
            raise ValueError("This one takes a minpres")
        elif backend == "graph":
            slicer = mps._SlicerVineGraph(st)
        else:
            raise ValueError(f"Inimplemented backend {backend}.")
    else:
        if backend == "matrix":
            slicer = mps._SlicerNoVineSimplicial(st)
        elif backend == "clement":
            raise ValueError("Clement is Vineyard")
        elif backend == "graph":
            raise ValueError("Graph is Vineyard")
        else:
            raise ValueError(f"Inimplemented backend {backend}.")
    return slicer


def Slicer(
    st=None,
    vineyard: Optional[bool] = None,
    reduce: bool = False,
    reduce_backend: Optional[str] = None,
    dtype: Optional[_valid_dtype] = None,
    kcritical: Optional[bool] = None,
    column_type: Optional[_column_type] = None,
    backend: Optional[_valid_pers_backend] = None,
    filtration_container: Optional[str] = None,
    max_dim: Optional[int] = None,
    return_type_only: bool = False,
    _shift_dimension: int = 0,
) -> mps.Slicer_type:
    """
    Given a simplextree, slicer, or SCC file path,
    returns a structure that can compute persistence on line (or more)
    slices, eventually vineyard update, etc.

    This can be used to compute interval-decomposable module approximations
    or signed measures, using, e.g.
     - `multipers.module_approximation(this, *args)`
     - `multipers.signed_measure(this, *args)`

    Input
    -----
     - st : SimplexTreeMulti, slicer, or path to an SCC file
     - backend: slicer backend, e.g, "matrix", "clement", "graph"
     - vineyard: vineyard capable (may slow down computations if true)
    Output
    ------
    The corresponding slicer.
    """
    if reduce:
        from warnings import warn

        warn(
            "Deprecated argument `reduce`. just reduce it afterwards.",
            DeprecationWarning,
        )
    if max_dim is not None:
        raise DeprecationWarning("deprecated parameter.")
    if is_slicer(st, allow_minpres=False) or is_simplextree_multi(st):
        dtype = st.dtype if dtype is None else dtype
        is_kcritical = st.is_kcritical if kcritical is None else kcritical
        filtration_container = (
            st.filtration_container
            if filtration_container is None
            else filtration_container
        )
    else:
        dtype = np.float64 if dtype is None else dtype
        is_kcritical = False if kcritical is None else kcritical
        filtration_container = (
            "contiguous" if filtration_container is None else filtration_container
        )

    if is_slicer(st, allow_minpres=False):
        vineyard = st.is_vine if vineyard is None else vineyard
        column_type = st.col_type if column_type is None else column_type
        backend = st.pers_backend if backend is None else backend
    else:
        vineyard = False if vineyard is None else vineyard
        column_type = mps.default_column_type if column_type is None else column_type
        backend = "matrix" if backend is None else backend

    _Slicer = mps.get_matrix_slicer(
        is_vineyard=vineyard,
        is_k_critical=is_kcritical,
        dtype=dtype,
        col=column_type,
        pers_backend=backend,
        filtration_container=filtration_container,
    )
    if return_type_only:
        return _Slicer

    if st is None:
        return _Slicer()
    elif mps.is_slicer(st):
        slicer = _Slicer(st)
    elif is_simplextree_multi(st) and backend == "graph":
        slicer = _slicer_from_simplextree(st, backend, vineyard)
        if st.is_squeezed:
            slicer.filtration_grid = st.filtration_grid

    elif isinstance(st, str) or isinstance(st, os.PathLike):
        slicer = _Slicer(os.fspath(st), _shift_dimension)
    elif is_simplextree_multi(st):
        slicer = _Slicer(st)
    elif backend == "Graph":
        raise ValueError(
            """
Graph is simplicial, incompatible with minpres.
You can try using `multipers.slicer.to_simplextree`."""
        )
    else:
        raise TypeError(
            "Slicer construction from Python SCC/block iterables has been removed. "
            "Construct from a SimplexTreeMulti, an existing slicer, an SCC file path, "
            "or explicit (generator_maps, generator_dimensions, filtration_values) data."
        )
    if reduce:
        from multipers.ops import minimal_presentation

        slicer = minimal_presentation(
            slicer,
            backend=reduce_backend,
            slicer_backend=backend,
            vineyard=vineyard,
        )
    return slicer
