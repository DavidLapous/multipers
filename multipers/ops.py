import numpy as np
from typing import Iterable, Literal, Optional

import multipers.logs as _mp_logs
from multipers._stdout_silence import call_silencing_stdout


def aida(s, sort=True, verbose=False, progress=False):
    import importlib.util

    if importlib.util.find_spec("multipers._aida_interface") is None:
        raise RuntimeError(
            "AIDA in-memory interface is not available in this build. "
            "Rebuild multipers with AIDA support to enable this backend."
        )
    from multipers import _aida_interface

    if not _aida_interface._is_available():
        raise RuntimeError(
            "AIDA in-memory interface is not available in this build. "
            "Rebuild multipers with AIDA support to enable this backend."
        )
    return _aida_interface.aida(s, sort=sort, verbose=verbose, progress=progress)


def one_criticalify(
    slicer,
    reduce: Optional[bool] = None,
    degree: Optional[int] = None,
    clear=True,
    swedish: Optional[bool] = None,
    verbose=False,
    kcritical=False,
    algo: Literal["path", "tree"] = "path",
    filtration_container="contiguous",
    force_resolution=True,
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
    from multipers.simplex_tree_multi import is_simplextree_multi

    if is_simplextree_multi(slicer):
        from multipers import Slicer

        _mp_logs.warn_copy(
            f"[One criticalify] Had a simplextree as an input. Copy needed for slicer conversion."
        )
        slicer = Slicer(slicer)

    if not is_slicer(slicer):
        raise ValueError(f"Invalid input. Expected `SlicerType` got {type(slicer)=}.")
    if not slicer.is_kcritical:
        return slicer
    working_slicer = slicer.astype(dtype=np.float64)

    if working_slicer.is_squeezed:
        F = working_slicer.filtration_grid
    else:
        F = None
    if reduce is None and degree is not None:
        reduce = True
    out = call_silencing_stdout(
        _multi_critical_from_slicer,
        working_slicer,
        reduce=reduce,
        algo=algo,
        degree=degree,
        clear=clear,
        swedish=swedish,
        verbose=verbose,
        kcritical=kcritical,
        filtration_container=filtration_container,
        enabled=not _mp_logs.ext_log_enabled(),
    )
    if not reduce and is_slicer(out):
        out = out.astype(
            vineyard=slicer.is_vine,
            kcritical=False,
            dtype=slicer.dtype,
            col=slicer.col_type,
            pers_backend=slicer.pers_backend,
            filtration_container=filtration_container,
        )
    if not reduce:
        return out

    def _todo(x, i):
        x.filtration_grid = F
        x.minpres_degree = i
        if reduce and force_resolution:
            x = minimal_presentation(x, degree=i, force=True)
        return x

    if isinstance(out, tuple):
        return tuple(_todo(out[i], i) for i in range(len(out)))
    return _todo(out, degree)


def minimal_presentation(
    slicer,
    degree=-1,
    degrees: Iterable[int] = [],
    backend: Literal["mpfree", "2pac", ""] = "mpfree",
    n_jobs=-1,
    force=False,
    auto_clean=True,
    verbose=False,
    full_resolution=True,
    use_chunk=True,
    use_clearing=True,
    keep_generators: bool = False,
):
    """
    Computes a minimal presentation of a (1-critical) multifiltered complex.

    From [Fast minimal presentations of bi-graded persistence modules](https://doi.org/10.1137/1.9781611976472.16),
    whose code is available here: https://bitbucket.org/mkerber/mpfree

    Available backends include `mpfree` and `2pac`.
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

        def todo(degree):
            return minimal_presentation(
                slicer,
                degree=degree,
                backend=backend,
                force=force,
                auto_clean=auto_clean,
                full_resolution=full_resolution,
                use_chunk=use_chunk,
                use_clearing=use_clearing,
                keep_generators=keep_generators,
            )

        return tuple(
            Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(todo)(d) for d in degrees
            )
        )
    assert degree >= 0, "Degree not provided."
    dimensions = np.asarray(slicer.get_dimensions(), dtype=np.int32)
    idx = np.searchsorted(dimensions, degree)
    if idx >= dimensions.shape[0] or dimensions[idx] != degree:
        return type(slicer)()

    return call_silencing_stdout(
        _minimal_presentation_from_slicer,
        slicer,
        degree=degree,
        backend=backend,
        auto_clean=auto_clean,
        verbose=verbose,
        full_resolution=full_resolution,
        use_chunk=use_chunk,
        use_clearing=use_clearing,
        keep_generators=keep_generators,
        enabled=not _mp_logs.ext_log_enabled(),
    )
