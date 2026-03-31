import pickle as pkl

import gudhi as gd
import numpy as np
import pytest
from numpy import array

import multipers as mp
from multipers.tests import assert_st_simplices, random_st
from multipers.simplex_tree_multi import available_dtype, available_simplextrees


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
    for dtype in available_dtype:
        st_multi = mp.SimplexTreeMulti(st, num_parameters=4, dtype=dtype)
        minf = -np.inf if isinstance(dtype(1), np.floating) else np.iinfo(dtype).min
        it = [
            (array([0, 1]), array([1.0, minf, minf, minf])),
            (array([0]), array([1.0, minf, minf, minf])),
            (array([1]), array([0.0, minf, minf, minf])),
        ]
        assert_st_simplices(st_multi, it)


has_kcritical = np.any(
    [a().is_kcritical for a in mp.simplex_tree_multi.available_simplextrees]
)


@pytest.mark.skipif(
    not has_kcritical,
    reason="kcritical simplextree not compiled, skipping this test",
)
def test_kcritical_insert_without_filtration_uses_first_possible_time():
    st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)
    st.insert([0], [1, 2])
    st.insert([1], [3, 0])

    st.insert([0, 1])

    assert np.array_equal(np.asarray(st[[0]]), np.array([[1.0, 2.0]]))
    assert np.array_equal(np.asarray(st[[1]]), np.array([[3.0, 0.0]]))
    assert np.array_equal(np.asarray(st[[0, 1]]), np.array([[3.0, 2.0]]))


@pytest.mark.skipif(
    not has_kcritical,
    reason="kcritical simplextree not compiled, skipping this test",
)
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

    expected_edge_filtration = array([[0.0, 1.0]])
    for simplex in ([0], [1], [2], [0, 1], [0, 2], [1, 2]):
        assert np.array_equal(np.asarray(st[simplex]), expected_edge_filtration), (
            f"Unexpected filtration update on lower-dimensional simplex {simplex}"
        )

    s = mp.Slicer(st, vineyard=True)

    assert np.array_equal(
        s.get_filtrations_values(),
        array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.5, 2.5],
                [1.0, 2.0],
                [1.5, 1.5],
                [2.0, 1.0],
                [2.5, 0.5],
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
                [0.5, np.inf],
                [1.0, 2.5],
                [1.5, 2.0],
                [2.0, 1.5],
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
    assert (st.filtration([1, 2]) == 3).all()
    assert (st.filtration([0, 1, 2]) == [11, 21]).all()
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


def test_distance_matrix_filling():
    X = np.random.uniform(size=(200, 2))
    D = np.sqrt(((X[None] - X[:, None]) ** 2).sum(axis=-1))
    st = gd.RipsComplex(distance_matrix=D).create_simplex_tree()
    st = mp.SimplexTreeMulti(st, num_parameters=1)
    st2 = mp.SimplexTreeMulti(st)
    assert st2.num_parameters == 1
    st2.fill_lowerstar(np.zeros(st.num_vertices), 0)
    assert st2 != st
    st2.fill_distance_matrix(D, 0)
    assert st2 == st


def test_serialize():
    stm = random_st(num_parameters=4)
    stm2 = pkl.loads(pkl.dumps(stm))
    assert stm == stm2
    stm2[[0]][:] = stm2[[0]] + 1
    assert not stm == stm2
    st1 = stm.project_on_line(parameter=0)
    stm = mp.SimplexTreeMulti(st1, num_parameters=3)
    assert st1 == stm.project_on_line(parameter=0), (
        "Gudhi<->Multipers conversion failed"
    )


def test_simplextree_unsqueeze_roundtrip_random_st():
    np.random.seed(0)
    st = random_st(npts=12, num_parameters=3, max_dim=2)
    assert st.grid_squeeze().unsqueeze() == st


@pytest.mark.skipif(
    not has_kcritical,
    reason="kcritical simplextree not compiled, skipping this test",
)
def test_simplextree_unsqueeze_roundtrip_kcritical():
    np.random.seed(0)
    st = mp.filtrations.CoreDelaunay(
        points=np.random.uniform(size=(12, 2)), ks=[1, 2, 3]
    )
    assert st.grid_squeeze().unsqueeze() == st


def test_project_on_line_does_not_require_gudhi_thisptr(monkeypatch):
    stm = random_st(num_parameters=3)
    reference = stm.project_on_line(parameter=0)
    original_cls = gd.SimplexTree

    class NoThisPtrSimplexTree:
        def __init__(self, other=None):
            assert other is None
            self._tree = original_cls()

        def __getattr__(self, name):
            if name == "thisptr":
                raise AssertionError(
                    "project_on_line accessed gudhi.SimplexTree.thisptr"
                )
            return getattr(self._tree, name)

        def __eq__(self, other):
            return self._tree == getattr(other, "_tree", other)

    monkeypatch.setattr(gd, "SimplexTree", NoThisPtrSimplexTree)
    projected = stm.project_on_line(parameter=0)

    assert projected == reference


def test_linear_projections_does_not_require_gudhi_thisptr(monkeypatch):
    stm = random_st(num_parameters=3)
    linear_forms = np.asarray([[1.0, 0.0, 0.0], [0.5, 1.0, 0.25]])
    reference = stm.linear_projections(linear_forms)
    original_cls = gd.SimplexTree

    class NoThisPtrSimplexTree:
        def __init__(self, other=None):
            assert other is None
            self._tree = original_cls()

        def __getattr__(self, name):
            if name == "thisptr":
                raise AssertionError(
                    "linear_projections accessed gudhi.SimplexTree.thisptr"
                )
            return getattr(self._tree, name)

        def __eq__(self, other):
            return self._tree == getattr(other, "_tree", other)

    monkeypatch.setattr(gd, "SimplexTree", NoThisPtrSimplexTree)
    projected = stm.linear_projections(linear_forms)

    assert len(projected) == len(reference)
    for out, ref in zip(projected, reference):
        assert out == ref


def test_astypes():
    for cls in available_simplextrees:
        sample = cls()
        st = mp.SimplexTreeMulti(
            num_parameters=2,
            dtype=sample.dtype,
            kcritical=sample.is_kcritical,
            ftype=sample.filtration_container,
        )
        st.insert([0], [0, 0])

        assert st.thisptr == st.astype().thisptr
        assert st.thisptr == st.astype(dtype=st.dtype).thisptr
        assert st.thisptr == st.astype(kcritical=st.is_kcritical).thisptr
        assert (
            st.thisptr
            == st.astype(filtration_container=st.filtration_container.lower()).thisptr
        )

    st = random_st()
    for cls in available_simplextrees:
        sample = cls()
        out = st.astype(
            dtype=sample.dtype,
            kcritical=sample.is_kcritical,
            filtration_container=sample.filtration_container,
        )
        assert np.dtype(out.dtype) == np.dtype(sample.dtype)
        assert out.is_kcritical == sample.is_kcritical
        assert out.filtration_container == sample.filtration_container


@pytest.mark.parametrize(
    "builder",
    [
        lambda: random_st(npts=8, num_parameters=2, max_dim=2),
        lambda: mp.filtrations.DegreeRips(
            points=mp.data.three_annulus(8, 0),
            ks=[1, 2, 3],
            threshold_radius=0.8,
            squeeze=False,
        ),
        lambda: mp.filtrations.CoreDelaunay(
            points=mp.data.three_annulus(8, 4),
            ks=[1, 2, 3],
        ),
    ],
)
def test_simplextree_to_scc_matches_slicer_export(builder, tmp_path):
    np.random.seed(0)
    st = builder()
    slicer = mp.Slicer(st, dtype=st.dtype)

    st_path = tmp_path / "simplextree.scc"
    slicer_path = tmp_path / "slicer.scc"

    st.to_scc(st_path, degree=1, strip_comments=True)
    slicer.to_scc(slicer_path, degree=1, strip_comments=True, unsqueeze=False)

    assert st_path.read_text() == slicer_path.read_text()


@pytest.mark.skipif(
    not has_kcritical,
    reason="kcritical simplextree not compiled, skipping this test",
)
def test_kcritical_simplextree_to_scc_matches_slicer_export(tmp_path):
    st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)
    for simplex in ([0], [1]):
        st.insert(simplex, [0.0, 0.0])
    st.insert([0, 1], [1.0, 2.0])
    st.insert([0, 1], [2.0, 1.0])

    st_path = tmp_path / "simplextree_kcritical.scc"
    slicer_path = tmp_path / "slicer_kcritical.scc"

    st.to_scc(st_path, degree=1, strip_comments=True)
    mp.Slicer(st, dtype=st.dtype).to_scc(
        slicer_path,
        degree=1,
        strip_comments=True,
        unsqueeze=False,
    )

    assert st_path.read_text() == slicer_path.read_text()


@pytest.mark.skipif(
    not has_kcritical,
    reason="kcritical simplextree not compiled, skipping this test",
)
def test_kcritical_batch_insert():
    st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)

    vertices = [[0, 1]]
    vertices_filtrations = np.array([[[-1, -2]], [[-2, -1]]])
    st.insert_batch(vertices, vertices_filtrations)

    edges = np.array([[0, 1], [1, 2], [2, 0]]).T
    edges_filtrations = np.array(
        [
            [[1, 0], [0, 1], [np.inf, np.inf]],
            [[1, 0], [0, 1], [np.inf, np.inf]],
            [[1, 0], [0, 1], [-1, 3]],
        ]
    )
    st.insert_batch(edges, edges_filtrations)

    triangle = np.array([[0, 1, 2]]).T
    triangle_filration = [[[2, 2]]]
    st.insert_batch(triangle, triangle_filration)

    from numpy import array

    goal = [
        (array([0, 1, 2]), [array([2.0, 2.0])]),
        (array([0, 1]), [array([0.0, 1.0]), array([1.0, 0.0])]),
        (array([0, 2]), [array([-1.0, 3.0]), array([0.0, 1.0]), array([1.0, 0.0])]),
        (array([0]), [array([-1.0, -2.0])]),
        (array([1, 2]), [array([0.0, 1.0]), array([1.0, 0.0])]),
        (array([1]), [array([-2.0, -1.0])]),
        (array([2]), [array([0.0, 1.0]), array([1.0, 0.0])]),
    ]
    assert_st_simplices(st, goal)


def test_5():
    st3 = random_st(num_parameters=3)
    st5 = mp.SimplexTreeMulti(st3, num_parameters=5)
    assert st5.num_parameters == 5
    for s, f in st5:
        assert len(f) == 5
    assert mp.SimplexTreeMulti(st5, num_parameters=3) == st3
