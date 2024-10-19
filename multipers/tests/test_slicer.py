import numpy as np
import pytest
from numpy import array

import multipers as mp
import multipers.slicer as mps
from multipers.tests import assert_sm


def test_1():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    for S in mps.available_slicers:
        if not S().is_vine or not S().col_type or S().is_kcritical:
            continue
        from multipers._slicer_meta import _blocks2boundary_dimension_grades

        generator_maps, generator_dimensions, filtration_values = (
            _blocks2boundary_dimension_grades(
                st._to_scc(),
                inplace=False,
            )
        )
        it = S(
            generator_maps, generator_dimensions, filtration_values
        ).persistence_on_line([0, 0])
        assert (
            len(it) == 2
        ), "There are simplices of dim 0 and 1, but no pers ? got {}".format(len(it))
        assert len(it[1]) == 0, "Pers of dim 1 is not empty ? got {}".format(it[1])
        for x in it[0]:
            if np.any(np.asarray(x)):
                continue
            assert x[0] == 1 and x[1] > 45, "pers should be [1,inf], got {}".format(x)


def test_2():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    st.insert([0, 1, 2], [2, 2])
    assert mp.slicer.to_simplextree(mp.Slicer(st, dtype=st.dtype)) == st


def test_3():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    sm = mp.signed_measure(st, invariant="rank", degree=0)
    sm2 = mp.signed_measure(mp.Slicer(st, dtype=np.int32), invariant="rank", degree=0)
    sm3 = mp.signed_measure(mp.Slicer(st, dtype=np.float64), invariant="rank", degree=0)
    it = [
        (
            array([[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
            array([1, 1, -1]),
        )
    ]
    assert_sm(sm, it)
    assert_sm(sm2, it)
    assert_sm(sm3, it)


def test_representative_cycles():
    truc = [
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {2, 3},
        {4, 5},
        {0, 2},
        {0, 1},
        {1, 3},
        {1, 2},
        {7, 11, 12},
        {9, 10, 12},
        {5, 6},
        {2, 4},
        {4, 6},
        {8, 15, 17},
        {3, 6},
    ]
    truc = [list(machin) for machin in truc]
    slicer = mp.slicer._Slicer_Matrix0_vine_f64(
        truc,
        np.array([max(len(x) - 1, 0) for x in truc]),
        np.array([list(range(len(truc))), list(range(len(truc)))]).T,
    )
    slicer.compute_persistence(one_filtration=list(range(len(truc))))
    cycles = slicer.get_representative_cycles()
    assert len(cycles) == 3, f"There should be 3 dimensions here, found {len(cycles)}"
    assert (
        np.asarray(cycles[0]).shape[0] == 7
    ), f"Invalid number of 0-cycles, got {np.asarray(cycles[0]).size}"
    for c in cycles[1]:
        assert (
            np.unique(cycles[1][0], return_counts=True)[1] == 2
        ).all(), f"Found a non-cycle, {cycles[1][0]}"
    assert len(cycles[2]) == 0, "Found a 2-cycle, which should not exist"


def test_pruning():
    from multipers.tests import random_st

    st = random_st(num_parameters=3, max_dim=4)
    s = mp.Slicer(st)
    s.prune_above_dimension(2)
    s2 = mp.Slicer(s, max_dim=2)
    assert s.get_dimensions()[-1] == 2, "Failed to prune dimension"
    st.prune_above_dimension(2)
    assert (
        s.get_dimensions() == s2.get_dimensions()
    ).all(), "pruned dimensions do not coincide"
    assert s.get_boundaries() == s2.get_boundaries(), "Boundaries changed"
    assert (
        s.get_filtrations_values() == s2.get_filtrations_values()
    ).all(), "Filtrations have changed"


def test_scc():
    import numpy as np

    import multipers as mp
    from multipers.distances import sm_distance
    from multipers.io import _check_available
    from multipers.tests import random_st

    st = random_st(npts=500, num_parameters=2)
    s = mp.Slicer(st, dtype=np.float64)
    s.to_scc("truc.scc", degree=1)
    s2 = mp.Slicer("truc.scc")
    (a,) = mp.signed_measure(s)
    (b,) = mp.signed_measure(s2)
    assert sm_distance(a, b) < np.finfo(np.float64).resolution * s.num_generators

    st = random_st(npts=500)
    s = mp.Slicer(st).grid_squeeze(inplace=True)
    s.filtration_grid = []
    s.to_scc("truc.scc", degree=1)
    s2 = mp.Slicer("truc.scc", dtype=np.float64)
    (a,) = mp.signed_measure(s)
    (b,) = mp.signed_measure(s2)
    assert sm_distance(a, b) == 0

    st = random_st(npts=200)
    s = mp.Slicer(st).grid_squeeze(inplace=True)
    s.filtration_grid = []
    s.to_scc(
        "truc.scc",
    )
    if _check_available("mpfree"):
        s2 = mp.Slicer("truc.scc", dtype=np.float64).minpres(1)
        (a,) = mp.signed_measure(s, degree=1)
        (b,) = mp.signed_measure(s2, degree=1)
        assert sm_distance(a, b) == 0
    else:
        pytest.skip("Skipped a test bc `mpfree` was not found.")


def test_bitmap():
    img = np.random.uniform(size=(12, 10, 11, 2))
    num_parameters = img.shape[-1]
    s = mp.slicer.from_bitmap(img)
    num_vertices = np.searchsorted(s.get_dimensions(), 1)
    assert num_vertices == np.prod(img.shape[:-1])
    assert s.num_parameters == num_parameters
    assert s.dimension == img.ndim - 1
    assert np.all(
        np.asarray(s.get_filtrations()[:num_vertices])
        == img.reshape(-1, num_parameters)
    )
