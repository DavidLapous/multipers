import numpy as np
import pytest

import multipers as mp
import multipers.ml.mma as mma
from multipers.tests import random_st


def test_1():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    mma_pymodule = mp.module_approximation(st)
    assert np.array_equal(mma_pymodule[0].get_birth_list(), [[0.0, 1.0], [1.0, 0.0]])
    assert np.array_equal(mma_pymodule[0].get_death_list(), [[np.inf, np.inf]])


def test_img():
    simplextree = mp.SimplexTreeMulti(num_parameters=4)
    simplextree.insert([0], [1, 2, 3, 4])
    mod = mp.module_approximation(simplextree,
        box=[[0, 0, 0, 0], [5, 5, 5, 5]], max_error=1.0
    )
    img = mod.representation(resolution=6, kernel="linear")
    assert np.isclose(img[0, 1, 2, 3, 4], 0.5)
    assert np.isclose(img[0, 1, 1, 1, 1], 0)
    assert np.isclose(img[0, 3, 4, 5, 5], 1)


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("prune_degrees_above", [0, 1, None])
def test_pipeline1(prune_degrees_above, n_jobs):
    args = locals()
    st = random_st(npts=50, max_dim=1).collapse_edges(-2)
    st.expansion(2)
    (truc1,) = mma.FilteredComplex2MMA(**args).fit_transform([[st]])[0]
    (truc2,) = mma.FilteredComplex2MMA(**args).fit_transform([[mp.Slicer(st)]])[0]
    box = st.filtration_bounds()
    st_copy = st.copy()
    if prune_degrees_above is not None:
        st_copy.prune_above_dimension(prune_degrees_above)
    output = mp.module_approximation(st_copy, box=box).representation(
        bandwidth=0.1, kernel="linear"
    )
    assert np.sum(output) > 0, "Invalid mma rpz"
    assert np.array_equal(
        truc1.representation(bandwidth=0.1, kernel="linear"),
        truc2.representation(bandwidth=0.1, kernel="linear"),
    ), "Slicer == Simplextree not satisfied"
    assert np.array_equal(truc1.representation(bandwidth=0.1, kernel="linear"), output)

    st = [random_st(npts=50).collapse_edges(-2, ignore_warning=True) for _ in range(5)]
    some_fited_pipeline = mma.FilteredComplex2MMA(**args).fit([st])
    truc1 = some_fited_pipeline.transform([st])
    truc2 = mma.FilteredComplex2MMA(**args).fit_transform(
        [[mp.Slicer(truc) for truc in st]]
    )
    for a, b in zip(truc1, truc2):
        assert np.array_equal(
            a[0].representation(bandwidth=0.01, kernel="linear"),
            b[0].representation(bandwidth=0.01, kernel="linear"),
        ), "Slicer == Simplextree not satisfied"


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("prune_degrees_above", [1, None])
@pytest.mark.parametrize("expand_dim", [None, 2, 3])
def test_pipeline2(prune_degrees_above, n_jobs, expand_dim):
    args = locals()
    st = random_st(max_dim=1)
    st.collapse_edges(-2)
    truc = mma.FilteredComplex2MMA(**args).fit_transform([[st]])[0]
    box = st.filtration_bounds()
    st_copy = st.copy()
    # if prune_degrees_above is not None:
    #     st_copy.prune_above_dimension(prune_degrees_above)
    if expand_dim is not None:
        st_copy.expansion(expand_dim)

    output = mp.module_approximation(st_copy, box=box).representation(bandwidth=0.01)
    assert np.array_equal(truc[0].representation(bandwidth=-0.01), output)


def test_dump_load():
    import pickle as pkl

    from multipers.tests import random_st

    mod = mp.module_approximation(random_st())
    _mod = pkl.loads(pkl.dumps(mod))
    assert mod == _mod
    mod = mp.module_approximation(random_st())
    assert _mod != mod
