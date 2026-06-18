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
