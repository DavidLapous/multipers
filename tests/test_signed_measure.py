import numpy as np
import pytest

import multipers as mp
import multipers.io as mio
from multipers.tests import assert_sm, random_st

np.random.seed(0)

st = random_st(npts=50).collapse_edges(-2, ignore_warning=True)

invariants = ["euler", "hilbert", "rank"]
degrees = [0, 1]
mass_defaults = [None, "auto"]
strats = [("regular_closest", 20), ("quantile", 20), ("regular", 17)]

mio._init_external_softwares()
mpfree_flag = mio._check_available("mpfree")


@pytest.mark.parametrize("invariant", invariants)
@pytest.mark.parametrize("degree", degrees)
@pytest.mark.parametrize("mass_default", mass_defaults)
@pytest.mark.parametrize("S", strats)
def test_backends(invariant, degree, mass_default, S):
    degree = None if invariant == "euler" else degree
    strat, r = S
    sms = []
    sms.append(
        mp.signed_measure(
            st,
            degree=degree,
            grid_strategy=strat,
            resolution=r,
            mass_default=mass_default,
            invariant=invariant,
        )[0]
    )
    sms.append(
        mp.signed_measure(
            st.grid_squeeze(grid_strategy=strat, resolution=r),
            degree=degree,
            mass_default=mass_default,
            invariant=invariant,
        )[0]
    )
    snv = mp.Slicer(st, vineyard=False)
    sv = mp.Slicer(st, vineyard=True)
    for s in [sv, snv]:
        sms.append(
            mp.signed_measure(
                s,
                degree=1,
                grid_strategy=strat,
                resolution=r,
                mass_default=mass_default,
                invariant=invariant,
            )[0]
        )
    if invariant != "euler":
        if not mpfree_flag:
            pytest.skip(r"Skipping next test, as `mpfree` was not found.")
        else:
            for s in [sv, snv]:
                assert s.minpres(degree=degree).is_minpres, "minpres is not minpres"
                sms.append(
                    mp.signed_measure(
                        s.minpres(degree=degree),
                        degree=degree,
                        grid_strategy=strat,
                        resolution=r,
                        mass_default=mass_default,
                        invariant=invariant,
                    )[0]
                )
                sms.append(
                    mp.signed_measure(
                        st,
                        grid_strategy=strat,
                        degree=degree,
                        resolution=r,
                        mass_default=mass_default,
                        backend="mpfree",
                        invariant=invariant,
                    )[0]
                )
    if mass_default is not None and invariant != "rank":
        assert sms[0][1].sum() == 0, "Did not remove all of the mass"
    assert_sm(*sms, exact=False, max_error=0.5)
