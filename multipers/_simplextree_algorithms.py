from __future__ import annotations

import numpy as np

from multipers import _simplex_tree_multi_nanobind as _nb
from multipers.point_measure import rank_decomposition_by_rectangles, sparsify


def _hilbert_signed_measure(
    simplextree,
    degrees,
    mass_default=None,
    plot=False,
    n_jobs=0,
    verbose=False,
    expand_collapse=False,
):
    assert simplextree.is_squeezed, "Squeeze grid first."
    zero_pad = mass_default is not None
    grid_shape = np.array([len(f) for f in simplextree.filtration_grid], dtype=np.int32)
    if mass_default is not None:
        mass_default = np.asarray(mass_default)
        assert (
            mass_default.ndim == 1
            and mass_default.shape[0] == simplextree.num_parameters
        )
    if zero_pad:
        grid_shape = grid_shape + 1
    pts, weights = _nb._compute_hilbert_signed_measure(
        simplextree,
        grid_shape.tolist(),
        np.asarray(degrees, dtype=np.int32).tolist(),
        zero_pad,
        n_jobs,
        verbose,
        expand_collapse,
    )
    pts = np.asarray(pts, dtype=np.int32).reshape(-1, simplextree.num_parameters + 1)
    weights = np.asarray(weights, dtype=np.int32)
    slices = np.concatenate(
        [np.searchsorted(pts[:, 0], np.arange(len(degrees))), [pts.shape[0]]]
    )
    return [
        (pts[slices[i] : slices[i + 1], 1:], weights[slices[i] : slices[i + 1]])
        for i in range(slices.shape[0] - 1)
    ]


def _euler_signed_measure(simplextree, mass_default=None, verbose=False):
    if not simplextree.is_squeezed:
        raise ValueError("Squeeze grid first.")
    zero_pad = mass_default is not None
    grid_shape = np.array([len(f) for f in simplextree.filtration_grid], dtype=np.int32)
    if mass_default is not None:
        mass_default = np.asarray(mass_default)
        assert (
            mass_default.ndim == 1
            and mass_default.shape[0] == simplextree.num_parameters
        )
    if zero_pad:
        grid_shape = grid_shape + 1
    pts, weights = _nb._compute_euler_signed_measure(
        simplextree, grid_shape.tolist(), zero_pad, verbose
    )
    return np.asarray(pts, dtype=np.int32).reshape(
        -1, simplextree.num_parameters
    ), np.asarray(weights, dtype=np.int32)


def _rank_signed_measure(
    simplextree,
    degrees,
    mass_default=None,
    plot=False,
    n_jobs=0,
    verbose=False,
    expand_collapse=False,
):
    assert simplextree.is_squeezed, "Squeeze grid first."
    zero_pad = mass_default is not None
    grid_shape = np.array([len(f) for f in simplextree.filtration_grid], dtype=np.int32)
    flat_container, shape = _nb._compute_rank_tensor(
        simplextree,
        grid_shape.tolist(),
        np.asarray(degrees, dtype=np.int32).tolist(),
        n_jobs,
        expand_collapse,
    )
    rank = np.asarray(flat_container, dtype=np.int32).reshape(tuple(shape))
    rank = tuple(
        rank_decomposition_by_rectangles(rank_of_degree, threshold=zero_pad)
        for rank_of_degree in rank
    )
    out = []
    num_parameters = simplextree.num_parameters
    for rank_decomposition in rank:
        coords, weights = sparsify(np.ascontiguousarray(rank_decomposition))
        births = coords[:, :num_parameters]
        deaths = coords[:, num_parameters:]
        correct_indices = np.all(births <= deaths, axis=1)
        coords = coords[correct_indices]
        weights = weights[correct_indices]
        if len(correct_indices) == 0:
            coords, weights = np.empty((0, 2 * num_parameters)), np.empty((0))
        out.append((coords, weights))
    return out
