import numpy as np
from typing import Iterable, Literal, Optional
import multipers.logs as _mp_logs


def aida(s, bool sort=True, bool verbose=False, bool progress=False):
    import importlib.util

    if importlib.util.find_spec("multipers.ext_interface._aida_interface") is None:
        raise RuntimeError(
            "AIDA in-memory interface is not available in this build. "
            "Rebuild multipers with AIDA support to enable this backend."
        )
    from multipers.ext_interface import _aida_interface

    if not _aida_interface._is_available():
        raise RuntimeError(
            "AIDA in-memory interface is not available in this build. "
            "Rebuild multipers with AIDA support to enable this backend."
        )
    return _aida_interface.aida(s, sort=sort, verbose=verbose, progress=progress)


def one_criticalify(
    slicer,
    bool reduce=False,
    degree: Optional[int] = None,
    bool clear=True,
    swedish: Optional[bool] = None,
    bool verbose=False,
    bool kcritical=False,
    str algo: Literal["path", "tree"] = "path",
    str filtration_container="contiguous",
):
    """
    Computes a free implicit representation of a given multi-critical
    multifiltration of a given homological degree (i.e., for a given
    homological degree, a quasi-isomorphic 1-critical filtration), or free
    resolution of the multifiltration (i.e., quasi-isomorphic 1-critical chain
    complex).

    From [Fast free resolutions of bifiltered chain complexes](https://doi.org/10.48550/arXiv.2512.08652),
    whose code is available here: https://bitbucket.org/mkerber/multi_critical
    """
    from multipers.io import _multi_critical_from_slicer
    from multipers.slicer import is_slicer

    if not is_slicer(slicer):
        raise ValueError(f"Invalid input. Expected `SlicerType` got {type(slicer)=}.")
    if not slicer.is_kcritical:
        return slicer
    if slicer.is_squeezed:
        F = slicer.filtration_grid
    else:
        F = None
    out = _multi_critical_from_slicer(
        slicer,
        reduce=reduce,
        algo=algo,
        degree=degree,
        clear=clear,
        swedish=swedish,
        verbose=verbose,
        kcritical=kcritical,
        filtration_container=filtration_container,
    )
    if is_slicer(out, allow_minpres=False):
        out.filtration_grid = F
    else:
        for stuff in out:
            stuff.filtration_grid = F
    return out


def minimal_presentation(
    slicer,
    int degree=-1,
    degrees: Iterable[int] = [],
    str backend: Literal["mpfree", "2pac", ""] = "mpfree",
    int n_jobs=-1,
    bool force=False,
    bool auto_clean=True,
    bool verbose=False,
):
    """
    Computes a minimal presentation of a (1-critical) multifiltered complex.

    From [Fast minimal presentations of bi-graded persistence modules](https://doi.org/10.1137/1.9781611976472.16),
    whose code is available here: https://bitbucket.org/mkerber/mpfree
    """
    from multipers.io import _minimal_presentation_from_slicer
    from joblib import Parallel, delayed
    from multipers.slicer import is_slicer

    if is_slicer(slicer) and slicer.is_minpres and not force:
        _mp_logs.warn_superfluous_computation(
            f"The slicer seems to be already reduced, "
            f"from homology of degree {slicer.minpres_degree}."
        )
        return slicer
    if len(degrees) > 0:

        def todo(int degree):
            return minimal_presentation(
                slicer,
                degree=degree,
                backend=backend,
                force=force,
                auto_clean=auto_clean,
            )

        return tuple(
            Parallel(n_jobs=n_jobs, backend="threading")(delayed(todo)(d) for d in degrees)
        )
    assert degree >= 0, "Degree not provided."
    dimensions = np.asarray(slicer.get_dimensions(), dtype=np.int32)
    idx = np.searchsorted(dimensions, degree)
    if idx >= dimensions.shape[0] or dimensions[idx] != degree:
        return type(slicer)()

    return _minimal_presentation_from_slicer(
        slicer,
        degree=degree,
        backend=backend,
        auto_clean=auto_clean,
        verbose=verbose,
    )
