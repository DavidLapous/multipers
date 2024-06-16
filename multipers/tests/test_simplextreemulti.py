import gudhi as gd
import numpy as np
from numpy import array

import multipers as mp
from multipers.tests import assert_st_simplices

mp.simplex_tree_multi.SAFE_CONVERSION = True


def test_1():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    it = [([0, 1], [1.0, 1.0]), ([0], [0.0, 1.0]), ([1], [1.0, 0.0])]
    assert_st_simplices(st, it)


def test_2():
    from gudhi.rips_complex import RipsComplex

    st2 = RipsComplex(points=[[0, 1], [1, 0], [0, 0]]).create_simplex_tree()
    st2 = mp.SimplexTreeMulti(
        st2, num_parameters=3, default_values=[1, 2]
    )  # the gudhi filtration is placed on axis 0

    it = (
        ([0, 1], [np.sqrt(2), 1.0, 2.0]),
        ([0, 2], [1.0, 1.0, 2.0]),
        ([0], [0.0, 1.0, 2.0]),
        ([1, 2], [1.0, 1.0, 2.0]),
        ([1], [0.0, 1.0, 2.0]),
        ([2], [0.0, 1.0, 2.0]),
    )
    assert_st_simplices(st2, it)


def test_3():
    st = gd.SimplexTree()  # usual gudhi simplextree
    st.insert([0, 1], 1)
    st.insert([1], 0)
    # converts the simplextree into a multiparameter simplextree
    for dtype in [np.int32, np.int64, np.float32, np.float64]:
        try:
            st_multi = mp.SimplexTreeMulti(st, num_parameters=4, dtype=dtype)
        except KeyError:
            import sys

            print(f"type {dtype} not compiled, skipping.", file=sys.stderr)
            continue  ## dtype not compiled
        minf = -np.inf if isinstance(dtype(1), np.floating) else np.iinfo(dtype).min
        it = [
            (array([0, 1]), array([1.0, minf, minf, minf])),
            (array([0]), array([1.0, minf, minf, minf])),
            (array([1]), array([0.0, minf, minf, minf])),
        ]
        assert_st_simplices(st_multi, it)


def test_4():
    st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)
    st.insert([0, 1, 2], [0, 1])
    st.insert([0, 1, 2], [1, 0])
    st.remove_maximal_simplex([0, 1, 2])
    st.insert([0, 1, 2], [1, 2])
    st.insert([0, 1, 2], [2, 1])
    st.insert([0, 1, 2], [1.5, 1.5])
    st.insert([0, 1, 2], [2.5, 0.5])
    st.insert([0, 1, 2], [0.5, 2.5])

    s = mp.Slicer(st, is_kcritical=True)

    assert np.array_equal(
        s.get_filtrations_values(),
        array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 2.0],
                [2.0, 1.0],
                [1.5, 1.5],
                [2.5, 0.5],
                [0.5, 2.5],
            ]
        ),
    ), "Invalid conversion from kcritical st to kcritical slicer."
    death_curve = np.asarray(
        mp.module_approximation(s, box=[[0, 0], [3, 3]])
        .get_module_of_degree(1)[0]
        .get_death_list()
    )
    assert np.array_equal(
        death_curve,
        array(
            [
                [2.0, 1.5],
                [2.5, 1.0],
                [np.inf, 0.5],
                [1.5, 2.0],
                [1.0, 2.5],
                [0.5, np.inf],
            ]
        ),
    )


def test_make_filtration_non_decreasing():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [1, 2])
    st.insert([1], [2, 3])
    st.insert([2], [3, 2])

    st.insert([0, 1], [10, 21])
    st.insert([2, 0], [11, 20])

    st.insert([0, 1, 2], [-np.inf, -np.inf])
    assert (st.filtration([1, 2]) == -np.inf).all()
    st.make_filtration_non_decreasing()
    assert (st.filtration([1, 2]) == 3).all()
    assert (st.filtration([0, 1, 2]) == [11, 21]).all()


def test_flagify():
    st = mp.SimplexTreeMulti(num_parameters=2)

    st.insert([0], [1, 4])
    st.insert([1], [2, 3])
    st.insert([2], [3, 2])

    st.insert([0, 1], [21, 20])
    st.insert([2, 0], [20, 21])
    st.insert([1, 2])
    st.insert([0, 1, 2], [41, 55])

    st.flagify(2)
    assert (st.filtration([0, 1, 2]) == 21).all()
    st.flagify(1)
    assert (np.array([f for s, f in st]).max(axis=0) == [3, 4]).all()
    st.flagify(0)
    assert (np.array([f for s, f in st]) == -np.inf).all()
