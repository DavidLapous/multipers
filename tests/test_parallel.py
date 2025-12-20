import numpy as np
import pytest
from joblib import Parallel, delayed

import multipers as mp
import multipers.io as mio
from multipers.tests import assert_sm_pair

np.random.seed(0)


def io_fd_mpfree(x):
    s = mp.filtrations.DelaunayCodensity(points=x, bandwidth=0.2)
    s = s.minpres(1).to_colexical()
    return mp.signed_measure(s, degree=1, invariant="hilbert")[0]


def io_fd_mpfree2(x):
    s = mp.filtrations.DelaunayCodensity(points=x, bandwidth=0.2)
    return mp.signed_measure(s, degree=1, invariant="hilbert")[0]


@pytest.mark.skipif(
    not (mio._check_available("mpfree") and mio._check_available("function_delaunay")),
    reason="Skipped external test as `function_delaunay`, `mpfree` were not found.",
)
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("backend", ["loky", "threading"])
def test_io_parallel(backend):
    x = mp.data.three_annulus(100, 50)
    X = [x] * 15
    ground_truth = io_fd_mpfree(x)
    ground_truth = io_fd_mpfree2(x)
    dgms = Parallel(n_jobs=-1, backend=backend)(delayed(io_fd_mpfree)(x) for x in X)

    for dgm in dgms:
        assert_sm_pair(ground_truth, dgm, exact=False, reg=0, max_error=1e-12)


x = mp.data.three_annulus(50, 10)
st = mp.filtrations.RipsCodensity(points=x, bandwidth=0.1)
st = st.collapse_edges(-2)
st.expansion(2)


def get_sm_st(n_jobs=1, to_slicer=False, invariant="hilbert"):
    if to_slicer:
        return mp.signed_measure(
            mp.Slicer(st), degree=1, n_jobs=n_jobs, invariant=invariant
        )[0]
    return mp.signed_measure(st, degree=1, n_jobs=n_jobs, invariant=invariant)[0]


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("backend", ["loky", "threading"])
@pytest.mark.parametrize("slicer", [False, True])
@pytest.mark.parametrize("invariant", ["hilbert"])
def test_st_sm_parallel(backend, slicer, invariant):
    ground_truth = get_sm_st(n_jobs=1, to_slicer=slicer, invariant=invariant)
    ground_truth2 = get_sm_st(n_jobs=1, to_slicer=not slicer, invariant=invariant)
    assert_sm_pair(ground_truth, ground_truth2, reg=0, exact=False, max_error=1e-12)
    sms = Parallel(n_jobs=-1, backend=backend)(
        delayed(get_sm_st)(-1, slicer, invariant) for _ in range(15)
    )
    for sm in sms:
        assert_sm_pair(ground_truth, sm, exact=False, reg=0, max_error=1e-12)
