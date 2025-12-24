import multipers as mp
from joblib import Parallel, delayed
from multipers.point_measure import add_sms
from multipers.tests import random_st
from multipers.tests import assert_sm
import pytest

mpfree_flag = mp.io._check_available("mpfree")
mpv = pytest.importorskip(
    "multipers.vector_interface",
    reason="AIDA wasn't compiled. Check the blacklist in `setup.py`.",
)


def test_indecomposable():
    B = [[], [], [0, 1]]
    D = [7, 7, 8]
    F = [[0, 1], [1, 0], [2, 2]]

    # an indecomposable
    s = mp.Slicer(return_type_only=True)(B, D, F)
    s.minpres_degree = 7
    s = s.to_colexical()
    s2 = mpv.aida(s)
    assert len(s2) == 1
    s2 = s2[0].to_colexical()
    assert s == s2


@pytest.mark.skipif(
    not mpfree_flag,
    reason="Skipped external test as  `mpfree` was not found.",
)
def test_equality():
    st = random_st()
    degree = 1
    s = mp.Slicer(st).minpres(degree)
    (sm1,) = mp.signed_measure(s, degree=degree, invariant="hilbert")
    s_ = mpv.aida(s)
    sms = Parallel(n_jobs=-1, backend="threading")(
        delayed(
            lambda x: mp.signed_measure(
                x.minpres(degree), degree=degree, invariant="hilbert"
            )[0]
        )(x)
        for x in s_
    )
    sm2 = add_sms(sms)
    assert_sm(sm1, sm2, exact=False, max_error=1e-10, reg=0)
