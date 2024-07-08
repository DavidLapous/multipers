import numpy as np
import multipers as mp
import multipers.ml.point_clouds as mmp
from multipers.data import noisy_annulus
from multipers.tests import assert_sm
import pytest
import multipers.io as mio

np.random.seed(0)
x = noisy_annulus(50, 1)
(st,) = mmp.PointCloud2SimplexTree(
    masses=[0.3], expand_dim=2, num_collapses=-2
).fit_transform([x])[0]

invariants = ["euler", "hilbert", "rank"]
degrees = [0, 1]
mass_defaults = [None, "auto"]
strats = [("regular_closest", 20), ("quantile", 20), ("regular", 17)]

mio._init_external_softwares()
mpfree_flag = mio.pathes["mpfree"] is not None


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
    if invariant != "euler" and mpfree_flag:
        for s in [sv, snv]:
            sms.append(
                mp.signed_measure(
                    s.grid_squeeze(grid_strategy=strat, resolution=r),
                    degree=degree,
                    mass_default=mass_default,
                    invariant=invariant,
                )[0]
            )
            assert s.minpres(degree=degree).is_minpres, "minpres is not minpres"
            sms.append(
                mp.signed_measure(
                    s.minpres(degree=degree),
                    degree=1,
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
    assert_sm(*sms, exact=False, max_error=1)
