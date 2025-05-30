import gudhi as gd
import numpy as np
import pytest

import multipers as mp
import multipers.filtrations as mpf

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
    not mp.io._check_available("function_delaunay"),
    reason="Skipped external test as `function_delaunay` was not found.",
)
def test_delaunaylowerstar(npts, dim):
    np.random.seed(0)
    points = np.random.uniform(size=(npts, dim))
    f = np.random.uniform(size=(npts))
    from scipy.spatial.distance import cdist

    distance_matrix = cdist(points, points)
    s = mpf.DelaunayLowerstar(points=points, function=f)

    st = mp.slicer.to_simplextree(s)

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
    not mp.io._check_available("function_delaunay"),
    reason="Skipped external test as `function_delaunay` was not found.",
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
