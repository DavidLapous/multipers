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


def assert_sm(sm1, sm2, exact=True, max_error=1e-5):
    if not exact:
        from multipers.distances import sm_distance

        for sm1_, sm2_ in zip(sm1, sm2, strict=True):
            d = sm_distance(sm1_, sm2_)
            assert (
                d < max_error
            ), f"Failed comparison:\n{sm1_}\n{sm2_},\n with distance {d}."
        return
    assert np.all(
        [
            np.isclose(a, b).all()
            for x, y in zip(sm1, sm2, strict=True)
            for a, b in zip(x, y, strict=True)
        ]
    ), f"Failed comparison:\n-----------------\n{sm1}\n-----------------\n{sm2}"
