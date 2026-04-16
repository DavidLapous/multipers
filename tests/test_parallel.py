import platform
import os
import threading
import warnings

import numpy as np
import pytest
from joblib import Parallel, delayed
import multipers.distances as mpdist
from multipers.distances import matching_distance
from threadpoolctl import threadpool_limits

import multipers as mp
from multipers.data import three_annulus
from multipers.filtrations import CoreDelaunay
from multipers.tests import assert_sm_pair

from multipers import (
    _mpfree_interface,
    _multi_critical_interface,
    _function_delaunay_interface,
)

np.random.seed(0)


_is_macos_intel = platform.system() == "Darwin" and platform.machine() in {
    "x86_64",
    "i386",
}


def _skip_bugged_loky_on_macos_intel(backend: str) -> None:
    if _is_macos_intel and backend == "loky":
        pytest.skip("Skipped on macOS Intel bc loky bugged.")


def io_fd_mpfree(x):
    s = mp.filtrations.DelaunayCodensity(points=x, bandwidth=0.2)
    s = mp.Slicer(s).minpres(1).to_colexical()
    return mp.signed_measure(s, degree=1, invariant="hilbert")[0]


def io_fd_mpfree2(x):
    s = mp.filtrations.DelaunayCodensity(points=x, bandwidth=0.2)
    return mp.signed_measure(s, degree=1, invariant="hilbert")[0]


@pytest.mark.skipif(
    not _mpfree_interface.available()
    or not _function_delaunay_interface.available(),
    reason="Skipped bridge pipeline test because the mpfree or function_delaunay backend is unavailable.",
)
@pytest.mark.parametrize("backend", ["loky", "threading"])
@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_io_parallel(backend, n_jobs):
    _skip_bugged_loky_on_macos_intel(backend)
    x = mp.data.three_annulus(100, 50)
    X = [x] * 15
    ground_truth = io_fd_mpfree(x)
    ground_truth = io_fd_mpfree2(x)
    dgms = Parallel(n_jobs=n_jobs, backend=backend)(delayed(io_fd_mpfree)(x) for x in X)

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

@pytest.mark.parametrize("backend", ["threading", "loky"])
@pytest.mark.parametrize("slicer", [False, True])
@pytest.mark.parametrize("invariant", ["hilbert"])
@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_st_sm_parallel(backend, slicer, invariant, n_jobs):
    _skip_bugged_loky_on_macos_intel(backend)
    ground_truth = get_sm_st(n_jobs=1, to_slicer=slicer, invariant=invariant)
    ground_truth2 = get_sm_st(n_jobs=1, to_slicer=not slicer, invariant=invariant)
    assert_sm_pair(ground_truth, ground_truth2, reg=0, exact=False, max_error=1e-12)
    sms = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(get_sm_st)(n_jobs, slicer, invariant) for _ in range(15)
    )
    for sm in sms:
        assert_sm_pair(ground_truth, sm, exact=False, reg=0, max_error=1e-12)
