import numpy as np
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
