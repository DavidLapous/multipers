def available():
    return False


def _is_available():
    return available()


def require():
    raise RuntimeError(
        "mpfree interface is not available in this build. "
        "Rebuild multipers with mpfree support to enable this backend."
    )


def minimal_presentation(*args, **kwargs):
    require()
