import pickle as pkl

import gudhi as gd
import numpy as np
import pytest
from numpy import array

import multipers as mp
import multipers.slicer as mps
from multipers.tests import assert_sm

mpfree_flag = mp.io._check_available("mpfree")
fd_flag = mp.io._check_available("function_delaunay")


def test_1():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    for S in mps.available_slicers:
        # TODO : investigate gudhi ??
        if (
            S().pers_backend.lower() == "gudhicohomology"
            or S().is_kcritical
            or not np.issubdtype(S().dtype, np.floating)
        ):
            continue
        from multipers._slicer_meta import _blocks2boundary_dimension_grades

        generator_maps, generator_dimensions, filtration_values = (
            _blocks2boundary_dimension_grades(
                st._to_scc(),
                inplace=False,
            )
        )
        s = S(generator_maps, generator_dimensions, filtration_values)
        print(type(s), s.col_type, s.pers_backend)

        s.info
        it = s.persistence_on_line(
            [0, 0], [1, 1], ignore_infinite_filtration_values=False
        )
        assert len(it) == 2, (
            "There are simplices of dim 0 and 1, but no pers ? got {}".format(len(it))
        )
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
    sm3 = mp.signed_measure(mp.Slicer(st, dtype=np.float64), invariant="rank", degree=0)

    ((pts, w),) = mp.signed_measure(
        mp.Slicer(st, dtype=np.int32), invariant="rank", degree=0
    )
    pts = np.array(pts, dtype=float)
    pts[pts == mp.grids._inf_value(np.int32)] = np.inf
    sm2 = ((pts, w),)

    it = [
        (
            array(
                [
                    [0.0, 1.0, np.inf, np.inf],
                    [1.0, 0.0, np.inf, np.inf],
                    [1.0, 1.0, np.inf, np.inf],
                ]
            ),
            array([1, 1, -1]),
        )
    ]
    assert_sm(sm, it)
    assert_sm(sm2, it)
    assert_sm(sm3, it)


def test_rank_custom():
    B = [[], [0], [0], [0], [0]]
    F = [[0, 0], [2, 1], [1, 2], [3, 0], [0, 3]]
    D = [0, 1, 1, 1, 1]
    s = mp.Slicer(return_type_only=True, dtype=np.int32)(B, D, F)
    ((pts, w),) = mp.signed_measure(s, invariant="rank", degree=0)
    assert np.array_equal(
        pts,
        [
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 3.0, 1.0],
        ],
    )
    assert np.array_equal(w, [-1, 1, -1, 1, 1])
    B = [[]]
    D = [0]
    F = [[0, 0]]
    s = mp.Slicer(return_type_only=True)(B, D, F)
    ((a, b),) = mp.signed_measure(
        s,
        invariant="rank",
        degree=0,
    )
    assert np.array_equal(a, [[0, 0, np.inf, np.inf]])
    assert np.array_equal(b, [1])


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
    slicer = mp.slicer._ContiguousSlicer_Matrix0_vine_f64(
        truc,
        np.array([max(len(x) - 1, 0) for x in truc]),
        np.array([list(range(len(truc))), list(range(len(truc)))]).T,
    )
    slicer.compute_persistence(one_filtration=list(range(len(truc))))
    cycles = slicer.get_representative_cycles()
    assert len(cycles) == 3, f"There should be 3 dimensions here, found {len(cycles)}"
    assert len(cycles[0]) == 7, (
        f"Invalid number of 0-cycles, got {np.asarray(cycles[0]).size}"
    )
    for c in cycles[1]:
        assert (np.unique(cycles[1][0], return_counts=True)[1] == 2).all(), (
            f"Found a non-cycle, {cycles[1][0]}"
        )
    assert len(cycles[2]) == 0, "Found a 2-cycle, which should not exist"


# def test_pruning():
#     from multipers.tests import random_st
#
#     st = random_st(num_parameters=3, max_dim=4)
#     s = mp.Slicer(st)
#     s.prune_above_dimension(2)
#     s2 = mp.Slicer(s, max_dim=2)
#     assert s.get_dimensions()[-1] == 2, "Failed to prune dimension"
#     st.prune_above_dimension(2)
#     assert (s.get_dimensions() == s2.get_dimensions()).all(), (
#         "pruned dimensions do not coincide"
#     )
#     assert s.get_boundaries() == s2.get_boundaries(), "Boundaries changed"
#     assert (s.get_filtrations_values() == s2.get_filtrations_values()).all(), (
#         "Filtrations have changed"
#     )


def test_scc():
    import numpy as np

    import multipers as mp
    from multipers.distances import sm_distance
    import multipers.ext_interface._mpfree_interface as _mpfree_interface
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

        if _mpfree_interface._is_available():
            np.random.seed(3)
            st = random_st(npts=200)
            s = mp.Slicer(st).grid_squeeze(inplace=True)
            s.filtration_grid = []
            s.to_scc("truc.scc")

            previous_is_available = _mpfree_interface._is_available
            try:
                _mpfree_interface._is_available = lambda: True
                s_in_memory = mp.Slicer("truc.scc", dtype=np.float64).minpres(1)

                _mpfree_interface._is_available = lambda: False
                s_external = mp.Slicer("truc.scc", dtype=np.float64).minpres(1)
            finally:
                _mpfree_interface._is_available = previous_is_available

            (m,) = mp.signed_measure(s_in_memory, degree=1)
            (e,) = mp.signed_measure(s_external, degree=1)
            assert sm_distance(m, e) == 0
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


def test_get_filtrations_view_flag_one_critical():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0.0, 1.0])
    st.insert([1], [2.0, 3.0])
    st.insert([0, 1], [4.0, 5.0])
    s = mp.Slicer(st)

    copied = s.get_filtrations(view=False)
    assert isinstance(copied, np.ndarray)
    assert copied.shape == (len(s), s.num_parameters)

    viewed = s.get_filtrations(view=True)
    viewed[0][0] = viewed[0][0] + 10.0
    assert np.isclose(s.get_filtrations(view=True)[0][0], viewed[0][0])

    copied_after = s.get_filtrations(view=False)
    copied_after[0, 0] = copied_after[0, 0] + 5.0
    assert not np.isclose(copied_after[0, 0], s.get_filtrations(view=True)[0][0])

    copied_alias = s.get_filtrations(copy=True)
    viewed_alias = s.get_filtrations(copy=False)
    assert isinstance(copied_alias, np.ndarray)
    assert isinstance(viewed_alias, list)


def test_get_filtration_single_view_one_critical():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0.0, 1.0])
    st.insert([1], [2.0, 3.0])
    st.insert([0, 1], [4.0, 5.0])
    s = mp.Slicer(st)

    f0 = s.get_filtration(0)
    f0[0] = f0[0] + 10.0
    assert np.isclose(s.get_filtration(0)[0], f0[0])
    assert np.allclose(s.get_filtration(-1), s.get_filtrations(view=True)[-1])

    with pytest.raises(IndexError):
        s.get_filtration(len(s))


def test_get_boundaries_copy_and_packed_one_critical():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0.0, 1.0])
    st.insert([1], [2.0, 3.0])
    st.insert([0, 1], [4.0, 5.0])
    s = mp.Slicer(st)

    boundaries = s.get_boundaries()
    assert isinstance(boundaries, tuple)
    assert all(isinstance(row, np.ndarray) for row in boundaries)

    boundaries[0][:] = 123
    refreshed = s.get_boundaries()
    assert refreshed[0].size == 0

    indptr, indices = s.get_boundaries(packed=True)
    rebuilt = tuple(indices[indptr[i] : indptr[i + 1]] for i in range(len(indptr) - 1))
    assert len(rebuilt) == len(boundaries)
    for a, b in zip(refreshed, rebuilt):
        assert np.array_equal(a, b)


def test_get_filtrations_packed_one_critical_raises():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0.0, 1.0])
    st.insert([1], [2.0, 3.0])
    st.insert([0, 1], [4.0, 5.0])
    s = mp.Slicer(st)

    with pytest.raises(ValueError):
        s.get_filtrations(packed=True)


def test_get_filtrations_packed_multicritical():
    x = np.random.uniform(size=(100, 2))
    st = mp.filtrations.CoreDelaunay(x, ks=np.arange(1, 8).tolist())
    s = mp.Slicer(st)
    if not s.is_kcritical:
        pytest.skip("Expected k-critical slicer for packed multicritical test.")

    indptr, grades = s.get_filtrations(packed=True)
    dense = s.get_filtrations(view=False)

    assert indptr.shape == (len(s) + 1,)
    assert indptr[0] == 0
    assert indptr[-1] == grades.shape[0]
    assert grades.shape[1] == s.num_parameters

    num_check = min(len(s), 30)
    for i in range(num_check):
        a = np.asarray(dense[i], dtype=np.float64).reshape(-1, s.num_parameters)
        b = grades[indptr[i] : indptr[i + 1]]
        assert np.allclose(a, b)

    with pytest.raises(ValueError):
        s.get_filtrations(view=True, packed=True)


def test_get_filtration_single_view_multicritical():
    x = np.random.uniform(size=(100, 2))
    st = mp.filtrations.CoreDelaunay(x, ks=np.arange(1, 8).tolist())
    s = mp.Slicer(st)
    if not s.is_kcritical:
        pytest.skip(
            "Expected k-critical slicer for multicritical single filtration test."
        )

    all_views = s.get_filtrations(view=True)
    for i in [0, min(len(s) - 1, 5), -1]:
        got = np.asarray(s.get_filtration(i), dtype=np.float64).reshape(
            -1, s.num_parameters
        )
        ref = np.asarray(all_views[i], dtype=np.float64).reshape(-1, s.num_parameters)
        assert np.allclose(got, ref)


def test_get_boundaries_packed_multicritical():
    x = np.random.uniform(size=(100, 2))
    st = mp.filtrations.CoreDelaunay(x, ks=np.arange(1, 8).tolist())
    s = mp.Slicer(st)
    if not s.is_kcritical:
        pytest.skip("Expected k-critical slicer for packed boundaries test.")

    boundaries = s.get_boundaries()
    indptr, indices = s.get_boundaries(packed=True)

    assert indptr.shape == (len(s) + 1,)
    assert indptr[0] == 0
    assert indptr[-1] == indices.shape[0]

    num_check = min(len(s), 30)
    for i in range(num_check):
        expected = boundaries[i]
        got = indices[indptr[i] : indptr[i + 1]]
        assert np.array_equal(expected, got)


def test_get_filtrations_unsqueeze_multicritical():
    from multipers.grids import evaluate_in_grid

    x = np.random.uniform(size=(100, 2))
    st = mp.filtrations.CoreDelaunay(x, ks=np.arange(1, 8).tolist())
    s = mp.Slicer(st)
    if not s.is_kcritical:
        pytest.skip("Expected k-critical slicer for multicritical unsqueeze test.")

    ss = s.grid_squeeze()
    out = ss.get_filtrations(unsqueeze=True)

    idx, grades = ss.get_filtrations(packed=True)
    unsqueezed_grades = evaluate_in_grid(np.asarray(grades).copy(), ss.filtration_grid)

    num_check = min(len(ss), 30)
    for i in range(num_check):
        expected = unsqueezed_grades[idx[i] : idx[i + 1]]
        got = np.asarray(out[i], dtype=np.float64).reshape(-1, ss.num_parameters)
        assert np.allclose(got, expected)


@pytest.mark.skipif(
    not fd_flag or not mpfree_flag,
    reason="Skipped external test as `function_delaunay`, `mpfree` were not found.",
)
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_external(dim, degree):
    if degree >= dim:
        return
    X = np.random.uniform(size=(100, dim))
    f = np.random.uniform(size=(X.shape[0]))
    fd = mps.from_function_delaunay(X, f)
    assert mp.simplex_tree_multi.is_simplextree_multi(fd)
    assert np.array_equal(np.unique(mp.Slicer(fd).get_dimensions()), np.arange(dim + 2))
    fd_ = mps.from_function_delaunay(X, f, degree=degree)
    assert mps.is_slicer(fd_)
    assert np.array_equal(
        np.unique(fd_.get_dimensions()), [degree, degree + 1]
    )  ## only does presentations
    fd_ = mp.Slicer(fd).minpres(degree=degree)
    assert np.array_equal(
        np.unique(fd_.get_dimensions()), [degree, degree + 1, degree + 2]
    )  ## resolution by default


def test_pkl():
    for num_params in range(1, 4):
        img = np.random.uniform(size=(20, 20, num_params))
        img2 = np.array(img)
        img2[0, 0, 0] += 1
        s = mp.slicer.from_bitmap(img)
        s2 = mp.slicer.from_bitmap(img2)
        s3 = type(s)(
            s.get_boundaries()[:-1], s.get_dimensions()[:-1], s.get_filtrations()[:-1]
        )
        s4 = type(s)(s.get_boundaries(), s.get_dimensions() + 1, s.get_filtrations())
        assert len(s) > np.prod(img.shape[:-1]) + 1
        assert s == s.copy()
        assert s != s2
        assert s != s3
        assert s != s4
        assert s == pkl.loads(pkl.dumps(s))


def test_colexical():
    from multipers.distances import sm_distance
    from multipers.tests import random_st

    s = mp.Slicer(random_st())
    (s1,) = mp.signed_measure(s, degree=1)
    (s2,) = mp.signed_measure(s.to_colexical(), degree=1)
    assert np.isclose(
        sm_distance(
            s1,
            s2,
        ),
        0.0,
    )

    _s = s.to_colexical()
    F = np.array(_s.get_filtrations())
    dims = _s.get_dimensions()
    dims_shift = np.searchsorted(dims, np.unique(dims)[1:])
    G = F[1:] - F[:-1]
    num_parameters = G.shape[1]
    idx = np.arange(G.shape[0])
    for i in range(num_parameters - 1, 0, -1):
        idx = np.argwhere(G[idx, i] < 0).squeeze()
    for stuff in idx:
        assert stuff + 1 in dims_shift, "Colexical sort issue"


def test_clean_filtration_grid():
    x = np.random.uniform(size=(100, 2))
    from multipers.filtrations import RipsCodensity

    st = RipsCodensity(x, bandwidth=0.1)
    st = st.grid_squeeze()
    st.collapse_edges(-1, auto_clean=False)
    st.expansion(2)
    s = [len(f) for f in st.filtration_grid]
    st2 = st.copy()
    st2._clean_filtration_grid()
    s2 = [len(f) for f in st2.filtration_grid]
    assert s2[0] < s[0] and s2[1] <= s[1]
    (sm1,) = mp.signed_measure(st, degree=1)
    (sm2,) = mp.signed_measure(st2, degree=1)
    assert_sm(sm1, sm2)

    s = mp.Slicer(st)
    s2 = mp.Slicer(st)
    s2._clean_filtration_grid()
    a = [len(f) for f in s.filtration_grid]
    b = [len(f) for f in s2.filtration_grid]
    assert b[0] < a[0] and b[1] <= a[1]
    (sm1,) = mp.signed_measure(s, degree=1)
    (sm2,) = mp.signed_measure(s2, degree=1)
    assert_sm(sm1, sm2)


@pytest.mark.skipif(
    not np.any(
        [a().is_kcritical for a in mp.simplex_tree_multi.available_simplextrees]
    ),
    reason="kcritical simplextree not compiled, skipping this test",
)
def test_slicer_grid_squeeze_roundtrip_on_gudhi_and_multipers_simplextree():
    pts = np.random.uniform(size=(100, 2))
    st_gudhi = gd.AlphaComplex(points=pts).create_simplex_tree()
    st1 = mp.SimplexTreeMulti(st_gudhi)
    st2 = mp.filtrations.CoreDelaunay(points=pts, ks=np.arange(1, 50))

    for st_ in [st1, st2]:
        grid = mp.grids.compute_grid(st_)
        st_sq = st_.grid_squeeze(grid)

        assert mp.Slicer(st_sq) == mp.Slicer(st_).grid_squeeze()
        assert mp.Slicer(st_) == mp.Slicer(st_sq).unsqueeze()


def test_astypes():
    from multipers.tests import random_st

    st = random_st()

    for vine in [True, False]:
        for kcritical in [True, False]:
            for dtype in mp.slicer.available_dtype:
                # for dtype in [np.int32,np.float64]:
                for ftype in mp.slicer.available_filtration_container:
                    for col in mp.slicer.available_columns:
                        for pers_backend in mp.slicer.available_pers_backend:
                            if pers_backend == "GudhiCohomology" and (
                                vine
                                or col != next(iter(mp.slicer.available_columns))[0]
                            ):
                                continue
                            if ftype == "Flat" and not kcritical:
                                continue
                            s = mp.Slicer(st).astype(
                                kcritical=kcritical,
                                vineyard=vine,
                                dtype=dtype,
                                filtration_container=ftype,
                                col=col,
                                pers_backend=pers_backend,
                            )
                            assert s.is_kcritical == kcritical
                            assert s.is_vine == vine
                            assert s.dtype == dtype
                            assert s.filtration_container == ftype
                            assert s.col_type == col
                            assert s.pers_backend == pers_backend
    s = mp.Slicer(st)
    assert s.get_ptr() == s.astype().get_ptr()
