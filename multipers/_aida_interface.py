def _is_available():
    return False


def aida(s, sort=True, verbose=False, progress=False):
    raise RuntimeError(
        "AIDA in-memory interface is not available in this build. "
        "Rebuild multipers with AIDA support to enable this backend."
    )
