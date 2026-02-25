import numpy as np
from typing import Iterable, Literal, Optional


def aida(s, bool sort=True, bool verbose=False, bool progress=False):
    from multipers.ext_interface import _aida_interface

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
    from multipers.io import _init_external_softwares, scc_reduce_from_str_to_slicer
    from joblib import Parallel, delayed
    from multipers.slicer import is_slicer
    from multipers.ext_interface import _mpfree_interface
    import os
    import tempfile

    if is_slicer(slicer) and slicer.is_minpres and not force:
        from warnings import warn

        warn(
            f"(unnecessary computation) The slicer seems to be already reduced, "
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
    if not np.any(slicer.get_dimensions() == degree):
        return type(slicer)()

    if backend == "mpfree":
        if _mpfree_interface._is_available():
            new_slicer = _mpfree_interface.minimal_presentation(
                slicer,
                degree=degree,
                verbose=verbose,
                use_chunk=True,
                use_clearing=True,
                full_resolution=True,
            )
            new_slicer.minpres_degree = degree
            new_slicer.filtration_grid = slicer.filtration_grid if slicer.is_squeezed else None
            if new_slicer.is_squeezed and auto_clean:
                new_slicer = new_slicer._clean_filtration_grid()
            return new_slicer

    _init_external_softwares(requires=[backend])
    dimension = slicer.dimension - degree
    with tempfile.TemporaryDirectory(prefix="multipers") as tmpdir:
        tmp_path = os.path.join(tmpdir, "multipers.scc")
        slicer.to_scc(path=tmp_path, strip_comments=True, degree=degree - 1, unsqueeze=False)
        new_slicer = type(slicer)()
        if backend == "mpfree":
            shift_dimension = degree - 1
        else:
            shift_dimension = degree
        scc_reduce_from_str_to_slicer(
            path=tmp_path,
            slicer=new_slicer,
            dimension=dimension,
            backend=backend,
            shift_dimension=shift_dimension,
            verbose=verbose,
        )

        new_slicer.minpres_degree = degree
        new_slicer.filtration_grid = slicer.filtration_grid if slicer.is_squeezed else None
        if new_slicer.is_squeezed and auto_clean:
            new_slicer = new_slicer._clean_filtration_grid()
        return new_slicer
