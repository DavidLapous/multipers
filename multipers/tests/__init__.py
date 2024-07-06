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


def assert_sm_pair(sm1, sm2, exact=True, max_error=1e-3, reg=0.1):
    if not exact:
        from multipers.distances import sm_distance

        d = sm_distance(sm1, sm2, reg=0.1)
        assert d < max_error, f"Failed comparison:\n{sm1}\n{sm2},\n with distance {d}."
        return
    assert np.all(
        [
            np.isclose(a, b).all()
            for x, y in zip(sm1, sm2, strict=True)
            for a, b in zip(x, y, strict=True)
        ]
    ), f"Failed comparison:\n-----------------\n{sm1}\n-----------------\n{sm2}"


def assert_sm(*args, exact=True, max_error=1e-5, reg=0.1):
    sms = tuple(args)
    for i in range(len(sms) - 1):
        assert_sm_pair(sms[i], sms[i + 1], exact=exact, max_error=max_error, reg=reg)
