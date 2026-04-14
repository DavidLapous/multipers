from __future__ import annotations

import numpy as np

import multipers.slicer as mps
from multipers import _slicer_nanobind as _nb
from multipers.point_measure import rank_decomposition_by_rectangles, sparsify


python_indices_type = np.int32
python_tensor_dtype = np.int32


def _hilbert_signed_measure(
    slicer,
    degrees,
    zero_pad=False,
    n_jobs=0,
    verbose=False,
    ignore_inf=True,
):
    assert slicer.is_squeezed, "Squeeze grid first."
    if slicer.is_squeezed:
        grid_shape = np.array(
            [len(f) for f in slicer.filtration_grid], dtype=python_indices_type
        )
    else:
        grid_shape = (slicer.filtration_bounds()[1]).astype(python_indices_type) + 1
    if zero_pad:
        grid_shape = grid_shape + 1
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
        -1, slicer.num_parameters + 1
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
):
    if grid_shape is None:
        if slicer.is_squeezed:
            grid_shape = np.array(
                [len(f) for f in slicer.filtration_grid], dtype=python_indices_type
            )
        else:
            grid_shape = (slicer.filtration_bounds()[1]).astype(python_indices_type) + 1
    grid_shape = np.asarray(grid_shape, dtype=python_indices_type)
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

    num_parameters = len(grid_shape)

    def clean_rank(rank_decomposition):
        coords, weights = sparsify(np.ascontiguousarray(rank_decomposition))
        births = coords[:, :num_parameters]
        deaths = coords[:, num_parameters:]
        correct_indices = np.all(births <= deaths, axis=1)
        coords = coords[correct_indices]
        weights = weights[correct_indices]
        return coords, weights

    return [clean_rank(rank_of_degree) for rank_of_degree in rank]


def from_bitmap(img, **kwargs):
    img = np.asarray(img)
    from multipers import Slicer

    dtype = kwargs.get("dtype", img.dtype)
    kwargs["dtype"] = dtype
    if img.dtype != dtype:
        raise ValueError(f"Invalid type matching. Got {dtype=} and {img.dtype=}.")

    _Slicer = Slicer(return_type_only=True, **kwargs)
    builder_name = f"_build_bitmap_{np.dtype(dtype).name.replace('float64', 'f64').replace('int32', 'i32')}"
    if not hasattr(_nb, builder_name):
        raise ValueError(
            f"Invalid dtype. Got {img.dtype=}, was expecting {mps.available_dtype=}."
        )
    flattened = np.ascontiguousarray(img.reshape(-1, img.shape[-1]))
    shape = np.ascontiguousarray(img.shape[:-1], dtype=np.uint32)
    base = getattr(_nb, builder_name)(flattened, shape)
    if type(base) is _Slicer:
        return base
    return _Slicer(base)
