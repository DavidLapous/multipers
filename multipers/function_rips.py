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
    out.filtration_grid = st_multi.filtration_grid
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


def function_rips_surface(
    st_multi,
    homological_degrees: Iterable[int],
    mobius_inversion=True,
    zero_pad=False,
    n_jobs=0,
):
    st_multi = _canonical_degree_rips_simplextree(st_multi)
    assert st_multi.is_squeezed, "Squeeze first !"
    degrees = np.ascontiguousarray(
        np.asarray(homological_degrees, dtype=np.int32).reshape(-1)
    )
    surface = _nb.function_rips_surface(
        st_multi,
        degrees,
        mobius_inversion=mobius_inversion,
        zero_pad=zero_pad,
        n_jobs=np.int32(n_jobs),
    )
    filtration_grid = list(st_multi.filtration_grid)
    if filtration_grid and filtration_grid[0][-1] == np.inf:
        filtration_grid[0] = np.asarray(filtration_grid[0]).copy()
        filtration_grid[0][-1] = filtration_grid[0][-2]
    return filtration_grid, np.asarray(surface)


def function_rips_signed_measure(
    st_multi,
    homological_degrees: Iterable[int],
    mobius_inversion=True,
    zero_pad=False,
    n_jobs=0,
    reconvert=True,
):
    st_multi = _canonical_degree_rips_simplextree(st_multi)
    assert st_multi.is_squeezed
    degrees = np.ascontiguousarray(
        np.asarray(homological_degrees, dtype=np.int32).reshape(-1)
    )
    pts, weights = _nb.function_rips_signed_measure(
        st_multi,
        degrees,
        mobius_inversion=mobius_inversion,
        zero_pad=zero_pad,
        n_jobs=np.int32(n_jobs),
    )
    pts = np.asarray(pts, dtype=np.int32).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.int32)

    degree_indices = [
        np.argwhere(pts[:, 0] == degree_index).flatten()
        for degree_index in range(len(degrees))
    ]
    sms = [(pts[idx, 1:], weights[idx]) for idx in degree_indices]
    if not reconvert:
        return sms

    grid_conversion = st_multi.filtration_grid
    converted = []
    for coords_idx, coord_weights in sms:
        coords = np.empty(shape=coords_idx.shape, dtype=float)
        for i in range(coords.shape[1]):
            coords[:, i] = np.asarray(grid_conversion[i])[coords_idx[:, i]]
        converted.append((coords, coord_weights))
    return converted
