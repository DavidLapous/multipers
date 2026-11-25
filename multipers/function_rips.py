from __future__ import annotations

from typing import Iterable

import numpy as np
from gudhi.simplex_tree import SimplexTree

from . import _function_rips_nanobind as _nb
from . import _simplex_tree_multi_nanobind as _st_nb


def _canonical_degree_rips_simplextree(st_multi):
    cls = _st_nb._SimplexTreeMulti_Flat_Kf64
    if isinstance(st_multi, cls):
        return st_multi
    out = cls()
    out._copy_from_any(st_multi)
    # out.filtration_grid = st_multi.filtration_grid
    return out


def get_degree_rips(st, degrees: Iterable[int]):
    assert isinstance(st, SimplexTree), "Input has to be a Gudhi simplextree for now."
    assert st.dimension() == 1, (
        "Simplextree has to be of dimension 1. You can use the `prune_above_dimension` method."
    )
    degree_rips_cls = _st_nb._SimplexTreeMulti_Flat_Kf64
    degree_rips_st = degree_rips_cls(num_parameters=2)
    gudhi_state = np.ascontiguousarray(st.__getstate__(), dtype=np.int8)
    degree_array = np.ascontiguousarray(np.asarray(degrees, dtype=np.int32).reshape(-1))
    _nb.get_degree_rips(degree_rips_st, gudhi_state, degree_array)
    return degree_rips_st

