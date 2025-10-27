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
@pytest.mark.parametrize("backend", ["loky", "threading"])
def test_io_parallel(backend):
    x = mp.data.three_annulus(100, 50)
    X = [x] * 100
    ground_truth = io_fd_mpfree2(x)
    dgms = Parallel(n_jobs=-1, backend=backend)(delayed(io_fd_mpfree)(x) for x in X)

    for dgm in dgms:
        assert_sm_pair(ground_truth, dgm, exact=False, reg=0, max_error= 1e-10)
