import gudhi as gd
import numpy as np
import pytest

import multipers as mp
import multipers.filtrations as mpf

import multipers._function_delaunay_interface as _function_delaunay_interface

nptss = [50]
ress = [1, 10]
dims = [1, 2, 4]
nparamss = [1, 2, 3]
betas = [0.0, 0.5, 1.0]
kss = [[1, 2, 3], np.arange(1, 10, 2)]


@pytest.mark.parametrize("npts", nptss)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("num_parameters", nparamss)
def test_ripslowerstar(npts, dim, num_parameters):
    points = np.random.uniform(size=(npts, dim))
    f = np.random.uniform(size=(npts, num_parameters - 1))
    from scipy.spatial.distance import cdist

    distance_matrix = cdist(points, points)
    st1 = mpf.RipsLowerstar(points=points, function=f)
    st2 = mpf.RipsLowerstar(distance_matrix=distance_matrix, function=f)
    assert st1 == st2, "Distance vs points are different"

    assert st1.num_parameters == num_parameters, "Bad number of parameters"
    assert np.array_equal([f[1:] for s, f in st1.get_skeleton(0)], f)
    assert np.all(
        [
            (f[1] == distance_matrix[s[0], s[1]])
            for s, f in st1.get_skeleton(0)
            if len(s) == 2
        ]
    )


@pytest.mark.parametrize("npts", nptss)
@pytest.mark.parametrize("dim", dims)
def test_ripscodensity(npts, dim):
    np.random.seed(0)
    points = np.random.uniform(size=(npts, dim))
    s = mpf.RipsCodensity(points=points, dtm_mass=0.4)
    assert s.num_parameters == 2, "Bad number of parameters"
    s = mpf.RipsCodensity(points=points, bandwidth=0.1)
    assert s.num_parameters == 2, "Bad number of parameters"


@pytest.mark.parametrize("npts", nptss)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.skipif(
    not _function_delaunay_interface.available(),
    reason="Skipped test because the function_delaunay backend is unavailable.",
)
def test_delaunaylowerstar(npts, dim):
    np.random.seed(0)
    points = np.random.uniform(size=(npts, dim))
    f = np.random.uniform(size=(npts))
    from scipy.spatial.distance import cdist

    distance_matrix = cdist(points, points)
    s = mpf.DelaunayLowerstar(points=points, function=f)

    st = (
        s
        if mp.simplex_tree_multi.is_simplextree_multi(s)
        else mp.slicer.to_simplextree(s)
    )

    assert st.num_parameters == 2, "Bad number of parameters"
    F1 = np.asarray([g[1] for s, g in st.get_skeleton(0)])
    p1 = np.argsort(F1)
    p2 = np.argsort(f)
    assert np.allclose(F1[p1], f[p2])
    assert np.all(
        [
            np.isclose(st.filtration(p1[s])[1], distance_matrix[p2[s][0], p2[s][1]])
            for s, f in st.get_skeleton(0)
            if len(s) == 2
        ]
    )


@pytest.mark.parametrize("npts", nptss)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.skipif(
    not _function_delaunay_interface.available(),
    reason="Skipped test because the function_delaunay backend is unavailable.",
)
def test_delaunaycodensity(npts, dim):
    np.random.seed(0)
    points = np.random.uniform(size=(npts, dim))
    s = mpf.DelaunayCodensity(points=points, dtm_mass=0.4)  # bandwidth requires pykeops

    assert s.num_parameters == 2, "Bad number of parameters"


@pytest.mark.parametrize("res", ress)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("num_parameters", nparamss)
def test_cubical(res, dim, num_parameters):
    image = np.random.uniform(size=(*[res] * dim, num_parameters))
    s = mpf.Cubical(image)
    assert s.num_parameters == num_parameters


has_kcritical = np.any(
    [a().is_kcritical for a in mp.simplex_tree_multi.available_simplextrees]
)


def _python_coredelaunay_reference(
    points,
    *,
    beta,
    ks,
    precision="safe",
    max_alpha_square=float("inf"),
    positive_degree=False,
):
    from scipy.spatial import KDTree

    points = np.asarray(points, dtype=np.float64)
    ks = np.asarray(ks, dtype=np.int64)
    alpha_complex = gd.AlphaComplex(points=points, precision=precision).create_simplex_tree(
        max_alpha_square=max_alpha_square
    )
    knn_distances = KDTree(points).query(points, k=ks)[0]

    simplex_tree_multi = mp.SimplexTreeMulti(
        num_parameters=2, kcritical=True, dtype=np.float64
    )
    vertex_arrays_in_dimension = [[] for _ in range(alpha_complex.dimension() + 1)]
    squared_alphas_in_dimension = [[] for _ in range(alpha_complex.dimension() + 1)]
    for simplex, alpha_squared in alpha_complex.get_simplices():
        dim = len(simplex) - 1
        squared_alphas_in_dimension[dim].append(alpha_squared)
        vertex_arrays_in_dimension[dim].append(simplex)

    for vertex_array, alpha_squared in zip(
        vertex_arrays_in_dimension, squared_alphas_in_dimension
    ):
        if not vertex_array:
            continue
        vertex_array = np.asarray(vertex_array, dtype=np.int32)
        alphas = np.sqrt(np.asarray(alpha_squared, dtype=np.float64))
        max_knn_distances = np.max(knn_distances[vertex_array], axis=1)
        critical_radii = np.maximum(alphas[:, None], beta * max_knn_distances)
        filtrations = np.stack(
            (
                critical_radii,
                (ks[-1] - ks if positive_degree else -ks) * np.ones_like(critical_radii),
            ),
            axis=-1,
        )
        simplex_tree_multi.insert_batch(vertex_array.T, filtrations)

    return simplex_tree_multi


def _normalized_kcritical_filtration(filtration):
    rows = np.asarray([np.asarray(row, dtype=np.float64) for row in filtration])
    if rows.size == 0:
        return rows.reshape(0, 2)
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[order]


@pytest.mark.skipif(
    not has_kcritical,
    reason="kcritical simplextree not compiled, skipping this test",
)
@pytest.mark.parametrize("npts", nptss)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("beta", betas)
@pytest.mark.parametrize("ks", kss)
def test_coredelaunay(npts, dim, beta, ks):
    np.random.seed(0)
    points = np.random.uniform(size=(npts, dim))
    s = mpf.CoreDelaunay(points=points, beta=beta, ks=ks)
    ac = gd.AlphaComplex(points=points).create_simplex_tree(
        default_filtration_value=True
    )

    assert s.num_parameters == 2, "Bad number of parameters"
    assert s.is_kcritical, "Bifiltration is not k-critical"
    assert s.dimension == dim, "Bad dimension"
    assert set(tuple(spx) for spx, _ in s.get_simplices()) == set(
        tuple(spx) for spx, _ in ac.get_simplices()
    ), "Simplices differs from the Delaunay Complex"


@pytest.mark.skipif(
    not has_kcritical,
    reason="kcritical simplextree not compiled, skipping this test",
)
@pytest.mark.parametrize("positive_degree", [False, True])
def test_coredelaunay_matches_python_reference(positive_degree):
    np.random.seed(0)
    points = np.random.uniform(size=(12, 3))
    ks = np.asarray([1, 2, 4, 6], dtype=np.int64)
    got = mpf.CoreDelaunay(
        points=points,
        beta=1.3,
        ks=ks,
        positive_degree=positive_degree,
    )
    ref = _python_coredelaunay_reference(
        points,
        beta=1.3,
        ks=ks,
        positive_degree=positive_degree,
    )

    got_filtrations = {
        tuple(simplex): _normalized_kcritical_filtration(filtration)
        for simplex, filtration in got.get_simplices()
    }
    ref_filtrations = {
        tuple(simplex): _normalized_kcritical_filtration(filtration)
        for simplex, filtration in ref.get_simplices()
    }

    assert got_filtrations.keys() == ref_filtrations.keys()
    for simplex in got_filtrations:
        assert np.allclose(got_filtrations[simplex], ref_filtrations[simplex])
