import numpy as np
import pytest
from gudhi.wasserstein import wasserstein_distance
from multipers.point_measure import clean_sms

import multipers as mp
import multipers.io as mio
from multipers.tests import assert_sm, random_st

np.random.seed(0)

st = random_st(npts=50).collapse_edges(-2, ignore_warning=True)

invariants = ["euler", "hilbert", "rank"]
degrees = [0, 1]
mass_defaults = [None, "auto", "inf"]
strats = [("regular_closest", 20), ("quantile", 20), ("regular", 17)]

mpfree_flag = mio._check_available("mpfree")


@pytest.mark.parametrize("invariant", invariants)
@pytest.mark.parametrize("degree", degrees)
@pytest.mark.parametrize("mass_default", mass_defaults)
@pytest.mark.parametrize("S", strats)
def test_backends(invariant, degree, mass_default, S):
    degree = None if invariant == "euler" else degree
    strat, r = S
    grid = mp.grids.compute_grid(st, strategy=strat, resolution=r)
    sms = []
    sms.append(
        mp.signed_measure(
            st,
            degree=degree,
            grid=grid,
            mass_default=mass_default,
            invariant=invariant,
            clean=True,
        )[0]
    )
    sms.append(
        mp.signed_measure(
            st.grid_squeeze(grid),
            degree=degree,
            mass_default=mass_default,
            invariant=invariant,
            clean=True,
        )[0]
    )
    snv = mp.Slicer(st, vineyard=False)
    sv = mp.Slicer(st, vineyard=True)
    for s in [sv, snv]:
        sms.append(
            mp.signed_measure(
                s,
                degree=degree,
                grid=grid,
                mass_default=mass_default,
                invariant=invariant,
                clean=True,
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
                        grid=grid,
                        mass_default=mass_default,
                        invariant=invariant,
                        clean=True,
                    )[0]
                )
    if mass_default is not None and invariant != "rank":
        assert sms[0][1].sum() == 0, "Did not remove all of the mass"
    assert_sm(*sms, exact=True)


@pytest.mark.parametrize("degree", degrees)
def test_rank(degree):
    s = mp.Slicer(
        random_st(npts=50, max_dim=1).collapse_edges(-2).expansion(degree + 1)
    )
    (rank_sm,) = mp.signed_measure(s, invariant="rank", degree=degree)
    for _ in range(5):
        bp = np.random.uniform(size=(2))
        d = np.random.uniform(size=(2))
        bc1 = mp.point_measure.barcode_from_rank_sm(rank_sm, bp, d)
        bc2 = s.persistence_on_line(bp, d)[degree]
        # assert_sm(bc1,bc2, max_error=.1)
        assert np.isclose(wasserstein_distance(bc1, bc2), 0)


def test_hook_decomposition():
    from multipers.point_measure import rectangle_to_hook_minimal_signed_barcode

    B = [[], [0]]
    D = [0, 1]
    F = [[0, 0], [1, 1]]
    s = mp.Slicer(return_type_only=True)(B, D, F)
    (sm_rank,) = mp.signed_measure(
        s,
        invariant="rank",
        degree=0,
    )
    sm_hook = rectangle_to_hook_minimal_signed_barcode(*sm_rank)
    pts_hook, w_hook = sm_hook
    assert np.array_equal(pts_hook, [[0, 0, 1, 1]]), pts_hook
    assert np.array_equal(w_hook, [1]), w_hook
    B = [[], [0], [0]]
    D = [0, 1, 1]
    F = [[0, 0], [1, 0], [0, 1]]
    s = mp.Slicer(return_type_only=True)(B, D, F)
    (sm_rank,) = mp.signed_measure(
        s,
        invariant="rank",
        degree=0,
    )
    sm_hook = rectangle_to_hook_minimal_signed_barcode(*sm_rank)
    pts_hook, w_hook = sm_hook
    assert np.array_equal(pts_hook, [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]), (
        pts_hook
    )
    assert np.array_equal(w_hook, [1, 1, -1]), w_hook

    B = [
        [],
        [0],
    ]
    D = [
        0,
        1,
    ]
    F = [[0, 0], [1, 0]]
    s = mp.Slicer(return_type_only=True)(B, D, F)
    (sm_rank,) = mp.signed_measure(
        s,
        invariant="rank",
        degree=0,
    )
    sm_hook = rectangle_to_hook_minimal_signed_barcode(*sm_rank)
    pts_hook, w_hook = sm_hook
    assert np.array_equal(pts_hook, [[0, 0, 1, 0]]), pts_hook
    assert np.array_equal(w_hook, [1]), w_hook

    B = [[], [0]]
    D = [0, 1]
    F = [[0, 0], [0, 1]]
    s = mp.Slicer(return_type_only=True)(B, D, F)
    (sm_rank,) = mp.signed_measure(
        s,
        invariant="rank",
        degree=0,
    )
    sm_hook = rectangle_to_hook_minimal_signed_barcode(*sm_rank)
    pts_hook, w_hook = sm_hook
    assert np.array_equal(pts_hook, [[0, 0, 0, 1]]), pts_hook
    assert np.array_equal(w_hook, [1]), w_hook


@pytest.mark.parametrize("degree", degrees)
def test_hilbert_mass_default_matches_inside_domain(degree):
    st_local = random_st(npts=100).collapse_edges(-2, ignore_warning=True)
    grid = mp.grids.compute_grid(st_local, strategy="regular_closest", resolution=11)
    mass_default = np.array([1.1 * np.max(f) - 0.1 * np.min(f) for f in grid])
    backends = [
        st_local,
        st_local.grid_squeeze(grid),
        mp.Slicer(st_local, vineyard=True),
        mp.Slicer(st_local, vineyard=False),
    ]
    if mpfree_flag:
        backends.extend(
            [
                mp.Slicer(st_local, vineyard=True).minpres(degree=degree),
                mp.Slicer(st_local, vineyard=False).minpres(degree=degree),
            ]
        )

    for backend in backends:
        sm_none = mp.signed_measure(
            backend,
            degree=degree,
            grid=grid,
            mass_default=None,
            invariant="hilbert",
            clean=True,
        )[0]
        sm_auto = mp.signed_measure(
            backend,
            degree=degree,
            grid=grid,
            mass_default="auto",
            invariant="hilbert",
            clean=True,
        )[0]

        pts_none, weights_none = sm_none
        pts_auto, weights_auto = sm_auto
        restricted_none = clean_sms(
            [
                (
                    pts_none[np.all(pts_none < mass_default, axis=1)],
                    weights_none[np.all(pts_none < mass_default, axis=1)],
                )
            ]
        )[0]
        restricted_auto = clean_sms(
            [
                (
                    pts_auto[np.all(pts_auto < mass_default, axis=1)],
                    weights_auto[np.all(pts_auto < mass_default, axis=1)],
                )
            ]
        )[0]
        assert_sm(
            restricted_none,
            restricted_auto,
            exact=True,
        )
