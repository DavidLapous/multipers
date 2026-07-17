import numpy as np
from operator import index as _index
from typing import Literal, Optional, Sequence

import multipers.logs as _mp_logs
from multipers.array_api import api_from_tensors
from multipers.slicer import is_slicer
from multipers.simplex_tree_multi import is_simplextree_multi


def _graph_mph0_minimal_presentation(slicer, degree, full_resolution):
    if is_simplextree_multi(slicer):
        from multipers import Slicer

        slicer = Slicer(slicer)
    if not is_slicer(slicer):
        raise ValueError(f"Expected a Slicer or SimplexTreeMulti, got {type(slicer)=}.")
    if slicer.is_kcritical:
        raise ValueError("graph requires a one-critical filtration.")
    if slicer.num_parameters != 2:
        raise ValueError("graph requires exactly two filtration parameters.")
    if slicer.is_squeezed:
        slicer = slicer.unsqueeze()

    from multipers import _slicer_nanobind

    return _slicer_nanobind._graph_mph0_minimal_presentation(
        slicer, degree, full_resolution
    )


def _normalize_degree(source, target, degree):
    if degree is None:
        source_degree = _index(source.minpres_degree)
        target_degree = _index(target.minpres_degree)
        if source_degree >= 0 and source_degree == target_degree:
            return source_degree
        raise ValueError("Expected degree= unless source and target have the same minpres_degree.")
    if isinstance(degree, bool):
        raise ValueError("Expected an integral non-bool degree.")
    try:
        degree = _index(degree)
    except TypeError as exc:
        raise ValueError("Expected an integral non-bool degree.") from exc
    if degree < 0:
        raise ValueError("Expected a non-negative degree.")
    return degree


def _degree_block_filtrations(slicer, degree):
    bounds = np.searchsorted(np.asarray(slicer.get_dimensions(), dtype=np.int32), [degree, degree + 1])
    return np.asarray(slicer.get_filtrations())[bounds[0] : bounds[1]]


def _packed_block_boundaries(slicer, row_degree, col_degree):
    dimensions = np.asarray(slicer.get_dimensions(), dtype=np.int32)
    row_bounds = np.searchsorted(dimensions, [row_degree, row_degree + 1])
    col_bounds = np.searchsorted(dimensions, [col_degree, col_degree + 1])
    indptr, indices = slicer.get_boundaries(packed=True)
    indptr = np.asarray(indptr, dtype=np.uint64)
    indices = np.asarray(indices, dtype=np.uint32)
    start = indptr[col_bounds[0]]
    stop = indptr[col_bounds[1]]
    raw_indices = indices[start:stop]
    if np.any((raw_indices < row_bounds[0]) | (raw_indices >= row_bounds[1])):
        raise ValueError("Morphism slicer boundaries must point from source generators to target generators.")
    return (
        np.ascontiguousarray(indptr[col_bounds[0] : col_bounds[1] + 1] - start, dtype=np.uint64),
        np.ascontiguousarray(raw_indices.astype(np.int64, copy=False) - row_bounds[0], dtype=np.uint32),
    )


def _packed_morphism_from_slicer(morphism, source, target, degree):
    if morphism.is_kcritical:
        raise ValueError("Algebra ops expect a one-critical morphism slicer.")
    source_grades = _degree_block_filtrations(source, degree)
    target_grades = _degree_block_filtrations(target, degree)
    map_rows = _degree_block_filtrations(morphism, degree)
    map_cols = _degree_block_filtrations(morphism, degree + 1)
    if len(map_rows) != len(target_grades) or len(map_cols) != len(source_grades):
        raise ValueError("Morphism slicer must have target generators in degree and source generators in degree + 1.")
    if not np.array_equal(map_rows, target_grades) or not np.array_equal(map_cols, source_grades):
        raise ValueError("Morphism slicer grades must match target/source generator grades.")
    return _packed_block_boundaries(morphism, degree, degree + 1)


def _same_local_relation_boundaries(source, target, degree):
    def relation_block(slicer):
        dimensions = np.asarray(slicer.get_dimensions(), dtype=np.int32)
        row_bounds = np.searchsorted(dimensions, [degree, degree + 1])
        col_bounds = np.searchsorted(dimensions, [degree + 1, degree + 2])
        indptr, indices = slicer.get_boundaries(packed=True)
        indptr = np.asarray(indptr, dtype=np.uint64)
        indices = np.asarray(indices, dtype=np.uint32)
        start = indptr[col_bounds[0]]
        stop = indptr[col_bounds[1]]
        block_indices = indices[start:stop]
        if np.any((block_indices < row_bounds[0]) | (block_indices >= row_bounds[1])):
            return None
        return indptr[col_bounds[0] : col_bounds[1] + 1] - start, block_indices.astype(np.int64) - row_bounds[0]

    source_block = relation_block(source)
    target_block = relation_block(target)
    return (
        source_block is not None
        and target_block is not None
        and np.array_equal(source_block[0], target_block[0])
        and np.array_equal(source_block[1], target_block[1])
    )


def _implicit_identity_columns(source, target, degree):
    source_bounds = np.searchsorted(np.asarray(source.get_dimensions(), dtype=np.int32), [degree, degree + 1])
    target_bounds = np.searchsorted(np.asarray(target.get_dimensions(), dtype=np.int32), [degree, degree + 1])
    rank = source_bounds[1] - source_bounds[0]
    if rank != target_bounds[1] - target_bounds[0] or not _same_local_relation_boundaries(source, target, degree):
        raise ValueError("Implicit identity morphism requires source and target to have the same boundary structure.")
    if rank > np.iinfo(np.uint32).max:
        raise ValueError("Implicit identity morphism rank exceeds supported uint32 range.")
    return np.arange(rank + 1, dtype=np.uint64), np.arange(rank, dtype=np.uint32)


def _parse_morphism(morphism, source, target, degree):
    if isinstance(morphism, dict):
        source = morphism.get("source", source)
        target = morphism.get("target", target)
        degree = morphism.get("degree", degree)
        morphism = morphism.get("map", morphism.get("slicer"))
    if source is None or target is None:
        raise ValueError("Expected source= and target= for algebra operations.")
    if not is_slicer(source, allow_minpres=False) or not is_slicer(target, allow_minpres=False):
        raise ValueError("Expected source= and target= to be slicers for algebra operations.")
    degree = _normalize_degree(source, target, degree)
    _validate_algebra_inputs(source, target)
    if is_slicer(morphism, allow_minpres=False):
        columns = _packed_morphism_from_slicer(morphism, source, target, degree)
    elif morphism is None:
        columns = _implicit_identity_columns(source, target, degree)
    else:
        raise ValueError("Expected a morphism slicer, a dict with a map/slicer key, or morphism=None.")
    return source, target, degree, columns


def _same_grid(left, right):
    if left is None or right is None or len(left) != len(right):
        return False
    if len(left) == 0:
        return True
    try:
        api = api_from_tensors(*left, *right)
    except ValueError:
        return False
    return all(
        np.array_equal(api.asnumpy(api.astensor(a)), api.asnumpy(api.astensor(b)))
        for a, b in zip(left, right)
    )


def _validate_squeezed_grids(source, target):
    source_squeezed = bool(source.is_squeezed)
    target_squeezed = bool(target.is_squeezed)
    if source_squeezed != target_squeezed:
        raise ValueError("Squeezed source/target grids must both be present and identical for algebra ops.")
    if source_squeezed and not _same_grid(source.filtration_grid, target.filtration_grid):
        raise ValueError("Squeezed source/target filtration grids must be identical for algebra ops.")


def _validate_free_slicers(source, target):
    for label, slicer in (("source", source), ("target", target)):
        if slicer.is_kcritical:
            raise ValueError(
                f"Algebra ops expect free/one-critical {label} slicers; "
                "call multipers.ops.one_criticalify on multicritical inputs first."
            )


def _validate_algebra_inputs(source, target):
    _validate_free_slicers(source, target)
    _validate_squeezed_grids(source, target)


def normalize(filtered_complex, box=None):
    """Normalize filtration values in place into the affine coordinates of ``box``."""
    if is_slicer(filtered_complex, allow_minpres=False) or is_simplextree_multi(filtered_complex):
        return filtered_complex.normalize_filtrations(box=box)
    raise ValueError(f"Expected a Slicer or SimplexTreeMulti, got {type(filtered_complex)=}.")


def _algebra_op(op, morphism, *, source=None, target=None, degree=None, backend="persistence-algebra"):
    """Run an algebra operation on an explicit degree-wise morphism."""
    if backend == "muphasa":
        return _muphasa_op(op, morphism, source=source, target=target, degree=degree)
    if backend not in {"persistence-algebra", "pa"}:
        raise ValueError("Algebra operations support backend='persistence-algebra' or backend='muphasa'.")
    source, target, degree, columns = _parse_morphism(morphism, source, target, degree)
    from multipers import _persistence_algebra_interface

    out = _persistence_algebra_interface.algebra_operation(op, source, target, *columns, degree)
    out._mark_minpres(degree, is_minres=False)
    owner = source if op in {"kernel", "coimage"} else target
    if owner.is_squeezed:
        out.filtration_grid = owner.filtration_grid
    return out


def _muphasa_op(op, morphism, *, source=None, target=None, degree=None):
    if op not in {"kernel", "image"}:
        raise NotImplementedError("Muphasa kernel/image are supported but cokernel/coimage are not yet bound.")
    source, target, degree, columns = _parse_morphism(morphism, source, target, degree)
    from multipers import _muphasa_interface

    _muphasa_interface.require()
    owner = source if op == "kernel" else target
    auto_grid = None
    if source.is_squeezed:
        source_arg, target_arg = source, target
    else:
        grid = source.get_filtration_grid("exact")
        if not _same_grid(grid, target.get_filtration_grid("exact")):
            raise ValueError("Muphasa source/target exact filtration grids must be identical for algebra ops.")
        auto_grid = grid
        source_arg = source.grid_squeeze(grid)
        target_arg = target.grid_squeeze(grid)
    try:
        out = _muphasa_interface.algebra_operation(op, source_arg, target_arg, *columns, degree)
    except RuntimeError as exc:
        if "only bound when the result is provably free" in str(exc):
            raise NotImplementedError(str(exc)) from exc
        raise
    out._mark_minpres(degree, is_minres=False)
    if owner.is_squeezed:
        out.filtration_grid = owner.filtration_grid
    elif auto_grid is not None:
        out.filtration_grid = auto_grid
    return out


def kernel(morphism=None, *, source=None, target=None, degree=None, backend="persistence-algebra"):
    """Kernel of a 2-parameter finitely presented module morphism.

    Parameters
    ----------
    morphism : Slicer, dict, or None
        A morphism slicer stores target generators in ``degree`` and source
        generators in ``degree + 1``; its boundaries are the matrix columns.
        ``None`` means the implicit identity/change-of-filtration map, allowed
        only when source and target have the same boundary structure. A dict may
        carry ``source``, ``target``, ``degree``, and ``map`` keys. Duplicate
        boundary entries cancel in F2. Nonzero entries must be coordinatewise
        grade-compatible and source relations must map into the target relation
        submodule.
    source, target : Slicer
        Free/one-critical presentation slicers of the source and target.
        Multicritical inputs must first be converted with ``one_criticalify``.
        Dimension ``degree`` stores generators; ``degree + 1`` stores relations.
    degree : int, optional
        Presented module degree. If omitted, source and target must be marked as
        minimal presentations with the same ``minpres_degree``.
    backend : {'persistence-algebra', 'muphasa'}
        Persistence-Algebra is the default. Muphasa computes kernel/image for
        free degree blocks; quotient/coimage are not yet bound.

    Returns
    -------
    Slicer
        Presentation of the kernel, with generators in ``degree`` and relations
        in ``degree + 1``. Kernel/coimage inherit source metadata; image/cokernel
        inherit target metadata. Squeezed source/target grids must match.
    """
    return _algebra_op("kernel", morphism, source=source, target=target, degree=degree, backend=backend)


def image(morphism=None, *, source=None, target=None, degree=None, backend="persistence-algebra"):
    """Image of a 2-parameter finitely presented module morphism.

    Inputs and output use the morphism/source/target/degree API
    documented in :func:`kernel`.
    """
    return _algebra_op("image", morphism, source=source, target=target, degree=degree, backend=backend)


def cokernel(morphism=None, *, source=None, target=None, degree=None, backend="persistence-algebra"):
    """Cokernel of a 2-parameter finitely presented module morphism.

    Inputs and output use the morphism/source/target/degree API
    documented in :func:`kernel`.
    """
    return _algebra_op("cokernel", morphism, source=source, target=target, degree=degree, backend=backend)


def coimage(morphism=None, *, source=None, target=None, degree=None, backend="persistence-algebra"):
    """Coimage of a 2-parameter finitely presented module morphism.

    Inputs and output use the morphism/source/target/degree API
    documented in :func:`kernel`.
    """
    return _algebra_op("coimage", morphism, source=source, target=target, degree=degree, backend=backend)


def _minimal_presentation_from_slicer(
    slicer,
    degree,
    backend="mpfree",
    auto_clean=True,
    verbose=False,
    full_resolution=True,
    use_clearing=True,
    use_chunk=True,
    keep_generators=False,
):
    if backend == "graph":
        if keep_generators:
            raise ValueError("graph does not support keep_generators.")
        return _graph_mph0_minimal_presentation(slicer, degree, full_resolution)

    if backend == "muphasa":
        from multipers import _muphasa_interface

        if full_resolution:
            raise ValueError("Muphasa backend currently supports only full_resolution=False.")
        if keep_generators:
            raise ValueError("Muphasa backend does not support keep_generators yet.")
        if slicer.num_parameters < 2:
            raise ValueError("Muphasa backend expects at least 2-parameter slicers.")
        _muphasa_interface.require()
        if not slicer.is_squeezed:
            slicer = slicer.grid_squeeze(slicer.get_filtration_grid("exact"))
        with _mp_logs.timings(
            "minimal_presentation",
            enabled=verbose,
            details={"backend": "muphasa", "mode": "cpp_interface", "degree": degree},
        ) as timing:
            new_slicer = _muphasa_interface.minimal_presentation(
                slicer,
                degree=degree,
                verbose=verbose,
                full_resolution=False,
                keep_generators=False,
            )
            timing.substep("backend_call")
        new_slicer._mark_minpres(degree, is_minres=False)
        new_slicer.filtration_grid = slicer.filtration_grid if slicer.is_squeezed else None
        if new_slicer.is_squeezed and auto_clean:
            new_slicer = new_slicer._clean_filtration_grid()
        return new_slicer

    if backend == "mpfree":
        from multipers import _mpfree_interface

        _mpfree_interface.require()
        with _mp_logs.timings(
            "minimal_presentation",
            enabled=verbose,
            details={"backend": "mpfree", "mode": "cpp_interface", "degree": degree},
        ) as timing:
            new_slicer = _mpfree_interface.minimal_presentation(
                slicer,
                degree=degree,
                verbose=verbose,
                use_chunk=use_chunk,
                use_clearing=use_clearing,
                full_resolution=full_resolution,
                keep_generators=keep_generators,
            )
            timing.substep("backend_call")
        new_slicer._mark_minpres(degree, is_minres=full_resolution)
        new_slicer.filtration_grid = slicer.filtration_grid if slicer.is_squeezed else None
        if new_slicer.is_squeezed and auto_clean:
            new_slicer = new_slicer._clean_filtration_grid()
        return new_slicer

    if backend in {"2pac", "2pac-homology"}:
        from multipers import _2pac_interface

        _2pac_interface.require()
        use_cohomology = backend == "2pac"
        with _mp_logs.timings(
            "minimal_presentation",
            enabled=verbose,
            details={
                "backend": backend,
                "mode": "cpp_interface",
                "degree": degree,
                "keep_generators": keep_generators,
                "algorithm": "cohomology" if use_cohomology else "homology",
            },
        ) as timing:
            new_slicer = _2pac_interface.minimal_presentation(
                slicer,
                degree=degree,
                verbose=verbose,
                use_chunk=use_chunk,
                use_clearing=use_clearing,
                full_resolution=full_resolution,
                keep_generators=keep_generators,
                use_cohomology=use_cohomology,
            )
            timing.substep("backend_call")
        new_slicer._mark_minpres(degree, is_minres=full_resolution)
        new_slicer.filtration_grid = slicer.filtration_grid if slicer.is_squeezed else None
        if new_slicer.is_squeezed and auto_clean:
            new_slicer = new_slicer._clean_filtration_grid()
        return new_slicer

    raise ValueError(
        f"Unsupported backend {backend!r}. Minimal presentation supports only `mpfree`, `muphasa`, `2pac`, "
        "`2pac-homology`, and `graph`."
    )


def _multi_critical_from_slicer(
    slicer,
    reduce=False,
    algo: Literal["path", "tree"] = "path",
    degree: Optional[int] = None,
    clear=True,
    swedish=None,
    verbose=False,
    kcritical=False,
    filtration_container="contiguous",
    **slicer_kwargs,
):
    del clear
    from multipers import _multi_critical_interface

    _multi_critical_interface.require()

    reduce = False if reduce is None else reduce
    swedish = degree is not None if swedish is None else swedish
    if reduce:
        out_kcritical = kcritical
        out_filtration_container = filtration_container
    else:
        out_kcritical = False
        out_filtration_container = "contiguous"

    with _mp_logs.timings(
        "one_criticalify",
        enabled=verbose,
        details={
            "backend": "multi_critical",
            "mode": "cpp_interface",
            "algo": algo,
            "reduce": reduce,
            "degree": degree,
            "swedish": swedish,
        },
    ) as timing:
        out = _multi_critical_interface.one_criticalify(
            slicer,
            reduce=reduce,
            algo=algo,
            degree=degree,
            swedish=swedish,
            verbose=verbose,
            kcritical=out_kcritical,
            filtration_container=out_filtration_container,
            **slicer_kwargs,
        )
        timing.substep("backend_call")
        return out


def aida(s, sort=True, verbose=False, progress=False):
    from multipers import _aida_interface

    _aida_interface.require()
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
    out = _multi_critical_from_slicer(
        working_slicer,
        reduce=reduce,
        algo=algo,
        degree=degree,
        clear=clear,
        swedish=swedish,
        verbose=verbose,
        kcritical=kcritical,
        filtration_container=filtration_container,
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
        out.filtration_grid = F
        return out

    def _todo(x, i):
        x.filtration_grid = F
        x._mark_minpres(i, is_minres=False)
        if reduce and force_resolution:
            x = minimal_presentation(x, degree=i, force=True)
        return x

    if isinstance(out, tuple):
        return tuple(_todo(out[i], i) for i in range(len(out)))
    return _todo(out, degree)


def minimal_presentation(
    slicer,
    degree=-1,
    degrees: Sequence[int] = (),
    backend: Literal["mpfree", "muphasa", "2pac", "2pac-homology", "graph", ""] = "mpfree",
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

    Backend references:

    - `mpfree`: [Fast minimal presentations of bi-graded persistence modules](https://doi.org/10.1137/1.9781611976472.16),
      with code at https://bitbucket.org/mkerber/mpfree
    - `2pac`: [Efficient Two-Parameter Persistence Computation via Cohomology](https://doi.org/10.4230/LIPIcs.SoCG.2023.15)
    - `muphasa`: https://github.com/olivergafvert/muphasa
    - `graph`: [Computing Betti Tables and Minimal Presentations of Zero-Dimensional Persistent Homology](https://doi.org/10.4230/LIPIcs.SoCG.2025.69)

    Available backends include `mpfree`, `muphasa` (Muphasa backend, currently
    at least 2-parameter and full_resolution=False only), `2pac` (2pac cohomology / dual
    transpose route, with 2pac's bounded-support assumptions), and
    `2pac-homology` (the original direct homology route), and `graph` (for
    graph-shaped presentations).
    """
    from joblib import Parallel, delayed
    full_resolution = bool(full_resolution)

    if is_simplextree_multi(slicer):
        from multipers import Slicer

        slicer = Slicer(slicer)

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
    if backend == "graph":
        return _minimal_presentation_from_slicer(
            slicer,
            degree=degree,
            backend=backend,
            auto_clean=auto_clean,
            verbose=verbose,
            full_resolution=full_resolution,
            use_chunk=use_chunk,
            use_clearing=use_clearing,
            keep_generators=keep_generators,
        )
    if is_slicer(slicer) and slicer.is_minpres and not force and (not full_resolution or slicer.is_minres):
        _mp_logs.warn_superfluous_computation(
            f"The slicer seems to be already reduced, "
            f"from homology of degree {slicer.minpres_degree}."
        )
        return slicer
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
        full_resolution=full_resolution,
        use_chunk=use_chunk,
        use_clearing=use_clearing,
        keep_generators=keep_generators,
    )
