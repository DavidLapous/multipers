from typing import cast

import numpy as np
import pytest


def test_compute_grid_drop_quantiles_accepts_1d_values():
    from multipers.grids import compute_grid

    grid = compute_grid(
        [np.arange(10, dtype=float)],
        resolution=3,
        strategy="quantile",
        drop_quantiles=(0.2, 0.8),
    )

    assert len(grid) == 1
    assert len(grid[0]) <= 3


def test_signed_measure_formatter_modes_are_exclusive():
    from multipers.ml.signed_measures import SignedMeasureFormatter

    with pytest.raises(ValueError, match="One post processing"):
        SignedMeasureFormatter(unsparse=True, integrate=True)


def test_sparse_convolution_rejects_iterable_bandwidth():
    from multipers.ml.signed_measures import SignedMeasure2Convolution

    signed_measures = [[(np.array([[0.0, 0.0]]), np.array([1]))]]
    transform = SignedMeasure2Convolution(
        filtration_grid=[np.array([0.0, 1.0]), np.array([0.0, 1.0])],
        bandwidth=[1.0, 1.0],
    ).fit(signed_measures)

    with pytest.raises(ValueError, match="scalar bandwidth"):
        transform.transform(signed_measures)


def test_get_simplextree_rejects_plain_object_cleanly():
    from multipers.ml.tools import get_simplextree

    with pytest.raises(TypeError, match="Not a valid SimplexTree"):
        get_simplextree(object())


def test_point_cloud_to_simplextree_honors_delayed_flag():
    from multipers.ml.one import PointCloud2SimplexTree

    clouds = cast(list, [np.zeros((2, 2))])
    delayed = PointCloud2SimplexTree(delayed=True).transform(clouds)

    assert callable(delayed[0])
    assert delayed[1] is clouds


def test_regular_closest_rejects_negative_resolution():
    from multipers._grid_helper_nanobind import regular_closest_1d_indices

    with pytest.raises(ValueError, match="resolution"):
        regular_closest_1d_indices(np.array([0.0, 1.0]), -1)


def test_get_simplices_of_dimension_rejects_negative_dimension():
    from multipers.simplex_tree_multi import SimplexTreeMulti

    st = SimplexTreeMulti(num_parameters=1)

    with pytest.raises(ValueError, match="Dimension"):
        st.get_simplices_of_dimension(-1)


def test_simplextree_rejects_missing_simplex_and_hides_raw_pointer_adoption():
    from multipers.simplex_tree_multi import SimplexTreeMulti

    st = SimplexTreeMulti(num_parameters=1)

    assert not hasattr(st, "_from_ptr")
    with pytest.raises(KeyError, match="Simplex"):
        st[[0]]


def test_integrate_measure_accepts_empty_grid_axis():
    from multipers._grid_helper_nanobind import integrate_measure

    out = integrate_measure(
        np.empty((0, 1), dtype=float),
        np.empty((0,), dtype=np.int64),
        (np.array([], dtype=float),),
    )

    assert out.shape == (0,)


def test_packed_slicer_rejects_invalid_boundary_indices():
    from multipers._slicer_nanobind import build_contiguous_f64_slicer_from_packed_f64

    with pytest.raises(RuntimeError, match="boundary_flat"):
        build_contiguous_f64_slicer_from_packed_f64(
            np.array([0, 1], dtype=np.int64),
            np.array([-1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([[0.0, 0.0]], dtype=float),
        )


def test_packed_slicer_rejects_nonempty_zero_cell_boundary():
    from multipers._slicer_nanobind import build_kcritical_contiguous_slicer_from_packed_f64

    with pytest.raises(RuntimeError, match="0-dimensional"):
        build_kcritical_contiguous_slicer_from_packed_f64(
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0, 1], dtype=np.int64),
            np.array([[0.0, 0.0]], dtype=float),
        )
