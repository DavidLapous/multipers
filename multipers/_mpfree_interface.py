def _is_available():
    return False


def minimal_presentation(*args, **kwargs):
    raise RuntimeError("mpfree in-memory interface is not available.")
