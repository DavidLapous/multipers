from __future__ import annotations

import numpy as np

from multipers import _slicer_nanobind as _nb
from multipers.point_measure import rank_decomposition_by_rectangles, sparsify


python_indices_type = np.int32
python_tensor_dtype = np.int32
_valid_mobius = {"auto", "sparse", "dense"}


def _validate_mobius(mobius):
    if mobius not in _valid_mobius:
        raise ValueError(
            f"Invalid mobius={mobius!r}. Expected 'auto', 'sparse', or 'dense'."
        )


def _hilbert_mobius_strategy(grid_shape):
    grid_shape = np.asarray(grid_shape, dtype=np.int64)
    num_parameters = grid_shape.size
    if num_parameters < 2 or num_parameters > 4:
        return "dense"
    if np.any(grid_shape <= 0):
        return "dense"

    dense_lines = int(np.prod(grid_shape[1:], dtype=np.int64))
    barcode_axis = int(np.argmax(grid_shape))
    sparse_lines = int(
        np.prod(np.delete(grid_shape, barcode_axis), dtype=np.int64)
    )
    dense_entries = int(np.prod(grid_shape, dtype=np.int64))
    line_gain = dense_lines / sparse_lines
    if num_parameters >= 3 and dense_entries >= 8_000_000 and line_gain >= 1.0:
        return "sparse"
    if (
        num_parameters == 2
        and dense_entries >= 8_000_000
        and dense_lines <= 512
        and line_gain >= 1.0
    ):
        return "sparse"
    if sparse_lines >= dense_lines:
        return "dense"

    if num_parameters == 2:
        return "sparse" if dense_lines >= 150 and line_gain >= 1.35 else "dense"
    return "sparse" if line_gain >= 2.0 else "dense"


def _hilbert_signed_measure(
    slicer,
    degrees,
    zero_pad=False,
    n_jobs=0,
    verbose=False,
    ignore_inf=True,
    mobius="auto",
):
    _validate_mobius(mobius)
    assert slicer.is_squeezed, "Squeeze grid first."
    if slicer.is_squeezed:
        grid_shape = np.array(
            [len(f) for f in slicer.filtration_grid], dtype=python_indices_type
        )
    else:
        grid_shape = (slicer.filtration_bounds()[1]).astype(python_indices_type) + 1
    if zero_pad:
        grid_shape = grid_shape + 1
    num_parameters = slicer.num_parameters
    if mobius == "auto":
        mobius = _hilbert_mobius_strategy(grid_shape)
    if mobius == "sparse" and 2 <= num_parameters <= 4:
        pts, weights = _nb._compute_hilbert_signed_measure_sparse(
            slicer,
            grid_shape.tolist(),
            np.asarray(degrees, dtype=np.int32).tolist(),
            zero_pad,
            n_jobs,
            ignore_inf,
        )
    else:
        pts, weights = _nb._compute_hilbert_signed_measure(
            slicer,
            grid_shape.tolist(),
            np.asarray(degrees, dtype=np.int32).tolist(),
            zero_pad,
            n_jobs,
            verbose,
            ignore_inf,
        )
    pts = np.asarray(pts, dtype=python_indices_type).reshape(
        -1, num_parameters + 1
    )
    weights = np.asarray(weights, dtype=python_tensor_dtype)
    slices = np.concatenate(
        [np.searchsorted(pts[:, 0], np.arange(len(degrees))), [pts.shape[0]]]
    )
    return [
        (pts[slices[i] : slices[i + 1], 1:], weights[slices[i] : slices[i + 1]])
        for i in range(slices.shape[0] - 1)
    ]


def _rank_from_slicer(
    slicer,
    degrees,
    verbose=False,
    n_jobs=1,
    zero_pad=False,
    grid_shape=None,
    plot=False,
    return_raw=False,
    ignore_inf=True,
    mobius="auto",
):
    _validate_mobius(mobius)
    if grid_shape is None:
        if slicer.is_squeezed:
            grid_shape = np.array(
                [len(f) for f in slicer.filtration_grid], dtype=python_indices_type
            )
        else:
            grid_shape = (slicer.filtration_bounds()[1]).astype(python_indices_type) + 1
    grid_shape = np.asarray(grid_shape, dtype=python_indices_type)
    num_parameters = len(grid_shape)
    if mobius == "auto":
        mobius = "sparse"
    if mobius == "sparse" and num_parameters >= 2 and not return_raw:
        pts, weights = _nb._compute_rank_signed_measure_sparse(
            slicer,
            grid_shape.tolist(),
            np.asarray(degrees, dtype=np.int32).tolist(),
            zero_pad,
            n_jobs,
            ignore_inf,
        )
        pts = np.asarray(pts, dtype=python_indices_type).reshape(
            -1, 1 + 2 * num_parameters
        )
        weights = np.asarray(weights, dtype=python_tensor_dtype)
        if pts.shape[0] == 0:
            return [
                (
                    np.empty((0, 2 * num_parameters), dtype=python_indices_type),
                    np.empty((0,), dtype=python_tensor_dtype),
                )
                for _ in degrees
            ]
        slices = np.concatenate(
            [np.searchsorted(pts[:, 0], np.arange(len(degrees))), [pts.shape[0]]]
        )
        return [
            (pts[slices[i] : slices[i + 1], 1:], weights[slices[i] : slices[i + 1]])
            for i in range(slices.shape[0] - 1)
        ]

    flat_container, shape = _nb._compute_rank_tensor(
        slicer,
        grid_shape.tolist(),
        np.asarray(degrees, dtype=np.int32).tolist(),
        n_jobs,
        ignore_inf,
    )
    rank = np.asarray(flat_container, dtype=python_tensor_dtype).reshape(tuple(shape))
    rank = tuple(
        rank_decomposition_by_rectangles(rank_of_degree, threshold=zero_pad)
        for rank_of_degree in rank
    )
    if return_raw:
        return rank

    def clean_rank(rank_decomposition):
        coords, weights = sparsify(np.ascontiguousarray(rank_decomposition))
        births = coords[:, :num_parameters]
        deaths = coords[:, num_parameters:]
        correct_indices = np.all(births <= deaths, axis=1)
        coords = coords[correct_indices]
        coords[:, num_parameters:] += 1
        weights = weights[correct_indices]
        return coords, weights

    return [clean_rank(rank_of_degree) for rank_of_degree in rank]
