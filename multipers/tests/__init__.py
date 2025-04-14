import numpy as np


def assert_st_simplices(st, dump):
    """
    Checks that the simplextree has the same
    filtration as the dump.
    """

    assert np.all(
        [
            np.isclose(a, b).all()
            for x, y in zip(st.get_simplices(), dump, strict=True)
            for a, b in zip(x, y, strict=True)
        ]
    )


def sort_sm(sms):
    idx = np.argsort([sm[0][:, 0] for sm in sms])
    return tuple((sm[0][idx], sm[1][idx]) for sm in sms)


def assert_sm_pair(sm1, sm2, exact=True, max_error=1e-3, reg=0.1, threshold=None):
    if not exact:
        from multipers.distances import sm_distance
        if threshold is not None:
            _inf_value_fix = threshold
            sm1[0][sm1[0] >threshold] = _inf_value_fix
            sm2[0][sm2[0] >threshold] = _inf_value_fix

        d = sm_distance(sm1, sm2, reg=reg)
        assert d < max_error, f"Failed comparison:\n{sm1}\n{sm2},\n with distance {d}."
        return
    assert np.all(
        [
            np.isclose(a, b).all()
            for x, y in zip(sm1, sm2, strict=True)
            for a, b in zip(x, y, strict=True)
        ]
    ), f"Failed comparison:\n-----------------\n{sm1}\n-----------------\n{sm2}"


def assert_sm(*args, exact=True, max_error=1e-5, reg=0.1, threshold=None):
    sms = tuple(args)
    for i in range(len(sms) - 1):
        print(i)
        assert_sm_pair(sms[i], sms[i + 1], exact=exact, max_error=max_error, reg=reg, threshold=threshold)


def random_st(npts=100, num_parameters=2, max_dim=2):
    import gudhi as gd

    import multipers as mp
    from multipers.data import noisy_annulus

    x = noisy_annulus(npts // 2, npts - npts // 2, dim=max_dim)
    st = gd.AlphaComplex(points=x).create_simplex_tree()
    st = mp.SimplexTreeMulti(st, num_parameters=num_parameters)
    for p in range(num_parameters):
        st.fill_lowerstar(np.random.uniform(size=npts), p)
    return st
