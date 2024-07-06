import multipers as mp
import multipers.ml.point_clouds as mmp
from multipers.data import noisy_annulus
from multipers.tests import assert_sm
import pytest
import multipers.io as mio

x = noisy_annulus(50, 1)
(st,) = mmp.PointCloud2SimplexTree(
    bandwidths=[0.2], expand_dim=2, num_collapses=-2
).fit_transform([x])[0]

invariants = ["euler", "hilbert", "rank"]
degrees = [0, 1]
mass_defaults = [None, "auto"]
strats = [("regular_closest", 100), ("exact", None)]

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
    if (invariant == "rank" and mass_default is None) or (
        invariant == "hilbert" and mpfree_flag
    ):
        for s in [sv, snv]:
            sms.append(
                mp.signed_measure(
                    s,
                    degree=degree,
                    grid_strategy=strat,
                    resolution=r,
                    mass_default=mass_default,
                    invariant=invariant,
                )[0]
            )
    # if invariant != "euler" and mpfree_flag:
    #     for s in [sv, snv]:
    #         s = s.minpres(degree=degree)
    #         sms.append(
    #             mp.signed_measure(
    #                 s,
    #                 grid_strategy=strat,
    #                 resolution=r,
    #                 mass_default=mass_default,
    #                 invariant=invariant,
    #             )[0]
    #         )
    #         sms.append(
    #             mp.signed_measure(
    #                 st,
    #                 grid_strategy=strat,
    #                 resolution=r,
    #                 mass_default=mass_default,
    #                 backend="mpfree",
    #                 invariant=invariant,
    #             )[0]
    #         )
    if mass_default is not None and invariant != "rank":
        assert sms[0][1].sum() == 0, "Did not remove all of the mass"
    assert_sm(*sms, exact=False, max_error=0.1)
