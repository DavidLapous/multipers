import numpy as np
import pytest

import multipers as mp
from multipers.distances import matching_distance


def test_hera_matching_distance_with_unequal_essential_h0_ranks():
    hera = pytest.importorskip("multipers._hera_interface")
    if not hera.available():
        pytest.skip("Hera interface unavailable")

    left = mp.SimplexTreeMulti(num_parameters=2, dtype=np.float64)
    left.insert([0], [1.0, 1.0])
    right = mp.SimplexTreeMulti(num_parameters=2, dtype=np.float64)
    right.insert([0], [1.0, 1.0])
    right.insert([1], [2.0, 2.0])

    # The old Hera pin returned a finite estimate instead of infinity when
    # allowed to return at the iteration limit.
    distance = matching_distance(
        left,
        right,
        degree=0,
        strategy="hera",
        epsilon=0,
        delta=0.1,
        hera_initialization_depth=0,
        hera_max_depth=1,
        hera_stop_asap=False,
        hera_tolerate_max_iter_exceeded=True,
    )
    assert np.isposinf(distance)
