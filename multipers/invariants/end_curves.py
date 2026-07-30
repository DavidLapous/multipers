from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np

from multipers.grids import Lstrategies, compute_grid

from ._utils import _as_slicer


def end_curves(*args, **kwargs):
    """Two-parameter end-curve invariant, not implemented yet.

    End-curves collect both birth-curves and death-curves.  In the 2-parameter
    theory these curves determine Betti tables and give a positive curve count
    for finite-grid modules.

    Parameters
    ----------
    *args, **kwargs:
        Reserved for the future paired end-curve API.

    Raises
    ------
    NotImplementedError
        Always raised until the invariant is implemented.

    Output
    ------
    None
        No value is returned because this invariant is not implemented yet.

    References
    ----------
    Brüstle, Oudot, Scoccola, and Thomas, "Counts and end-curves in
    two-parameter persistence", arXiv:2505.13412, 2025.
    """
    raise NotImplementedError("end_curves is not implemented yet.")


def _grid_inf_indices(grid) -> np.ndarray:
    lengths = np.fromiter((len(axis) for axis in grid), dtype=np.int64)
    has_inf = np.fromiter(
        (
            len(axis) > 0 and np.isinf(np.asarray(axis)[-1])
            for axis in grid
        ),
        dtype=bool,
    )
    return lengths - has_inf.astype(np.int64)


def _packed_row_positions(starts: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(row_ids, positions)`` enumerating packed entries.

    ``row_ids[i]`` is the row owning entry ``i`` and ``positions[i]`` is the
    corresponding index into the underlying flat buffer.
    """
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    row_ids = np.repeat(np.arange(starts.size, dtype=np.int64), lengths)
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64))
    )
    row_offsets = np.repeat(cumulative, lengths)
    positions = np.repeat(starts.astype(np.int64, copy=False), lengths) + (
        np.arange(total, dtype=np.int64) - row_offsets
    )
    return row_ids, positions


def _birth_curve_presentation(presentation, inf_indices: np.ndarray):
    if not presentation.is_minpres:
        if not presentation.is_pres:
            raise ValueError("birth-curve input must be a presentation.")
        presentation = presentation.minpres(
            degree=presentation.pres_degree, full_resolution=False
        )
    degree = presentation.minpres_degree
    boundary_indptr, boundary_flat = presentation.get_boundaries(packed=True)
    boundary_indptr = np.asarray(boundary_indptr, dtype=np.int64)
    boundary_flat = np.asarray(boundary_flat)
    dimensions = np.asarray(presentation.get_dimensions(), dtype=np.int32)
    num_parameters = int(len(inf_indices))
    filtrations = np.asarray(presentation.get_filtrations(), dtype=np.int64)
    if filtrations.ndim != 2:
        filtrations = filtrations.reshape(-1, num_parameters)

    generators = np.flatnonzero(dimensions == degree)
    relations = np.flatnonzero(dimensions == degree + 1)
    old_to_new = np.full(dimensions.shape[0], -1, dtype=np.int64)
    old_to_new[generators] = np.arange(generators.size, dtype=np.int64)

    relation_starts = boundary_indptr[relations]
    relation_raw_lengths = boundary_indptr[relations + 1] - relation_starts
    relation_ids_full, relation_positions = _packed_row_positions(
        relation_starts, relation_raw_lengths
    )
    if relation_positions.size:
        relation_boundary_flat = old_to_new[boundary_flat[relation_positions]]
        keep = relation_boundary_flat >= 0
        relation_lengths = np.bincount(
            relation_ids_full[keep],
            minlength=relations.size,
        ).astype(np.int64, copy=False)
        relation_boundary_flat = relation_boundary_flat[keep].astype(
            np.int32,
            copy=False,
        )
    else:
        relation_lengths = np.zeros(relations.size, dtype=np.int64)
        relation_boundary_flat = np.empty(0, dtype=np.int32)
    boundary_lengths = np.concatenate(
        (
            np.zeros(generators.size, dtype=np.int64),
            relation_lengths,
            np.ones(generators.size, dtype=np.int64),
        )
    )
    new_boundary_indptr = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(boundary_lengths, dtype=np.int64))
    )
    new_boundary_flat = np.concatenate(
        (relation_boundary_flat, np.arange(generators.size, dtype=np.int32))
    )

    new_dimensions = np.concatenate(
        (
            np.full(generators.size, degree, dtype=np.int32),
            np.full(relations.size + generators.size, degree + 1, dtype=np.int32),
        )
    )
    new_filtrations = np.empty(
        (generators.size + relations.size + generators.size, num_parameters),
        dtype=np.float64,
    )
    if generators.size:
        gen_filtrations = filtrations[generators]
        new_filtrations[: generators.size] = gen_filtrations
        new_filtrations[generators.size + relations.size :] = np.minimum(
            gen_filtrations + 1, inf_indices
        )
    if relations.size:
        new_filtrations[generators.size : generators.size + relations.size] = (
            filtrations[relations]
        )

    from multipers.slicer import get_matrix_slicer

    return get_matrix_slicer(
        is_vineyard=False,
        is_k_critical=False,
        dtype=np.double,
        col="UNORDERED_SET",
        pers_backend="Matrix",
        filtration_container="Contiguous",
    )(
        new_boundary_indptr,
        new_boundary_flat,
        new_dimensions,
        new_filtrations,
    )


def _to_grid_coordinates(
    points: np.ndarray,
    grid,
    inf_indices: np.ndarray,
    include_infinite: bool,
) -> np.ndarray:
    if not include_infinite:
        points = points[np.all(points < inf_indices, axis=1)]
    out = np.empty(points.shape, dtype=np.float64)
    for axis, values in enumerate(grid):
        values = np.asarray(values, dtype=np.float64)
        coordinate = points[:, axis]
        finite = coordinate < values.size
        out[finite, axis] = values[coordinate[finite]]
        out[~finite, axis] = np.inf
    return out


def _plot_box_from_grid(grid, coordinates: bool) -> np.ndarray:
    if not coordinates:
        return np.asarray([[0.0, 0.0], _grid_inf_indices(grid)], dtype=np.float64)

    mins = []
    maxs = []
    for axis in grid:
        values = np.asarray(axis, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            mins.append(0.0)
            maxs.append(1.0)
            continue
        lo = float(finite[0])
        if finite.size > 1:
            step = float(finite[-1] - finite[-2])
            if not np.isfinite(step) or step <= 0:
                step = 1.0
        else:
            step = 1.0
        mins.append(lo)
        maxs.append(float(finite[-1] + step))
    return np.asarray([mins, maxs], dtype=np.float64)


def birth_curves(
    filtered_complex,
    degree: Optional[int] = None,
    *,
    grid: Optional[Iterable] = None,
    grid_strategy: Lstrategies = "exact",
    coordinates: bool = True,
    include_infinite: bool = True,
    sort: bool = True,
    aida_sort: bool = True,
    verbose: bool = False,
    progress: bool = False,
    minpres_kwargs: Optional[dict] = None,
    plot: bool = False,
    min_length: float = -1,
    plot_kwargs: Optional[dict] = None,
    **infer_grid_kwargs,
) -> list[np.ndarray]:
    """Compute two-parameter birth-curves.

    Birth-curves are the spread-curve indecomposable summands obtained from
    the birth part of the end-curve construction for a two-parameter module.
    Returns one ``(k, 2)`` array per birth-curve. By default points are mapped
    back from squeezed grid indices to filtration coordinates, with one
    ``np.inf`` sentinel per axis for curves reaching infinity. If the input is
    a minimal-presentation slicer, ``degree`` is inferred from
    ``minpres_degree``. ``min_length`` only filters plotted curves when
    ``plot=True``; returned curves are not filtered.

    Parameters
    ----------
    filtered_complex : Slicer or SimplexTreeMulti-like
        Two-parameter filtered complex, module presentation, or object
        convertible to a slicer.
    degree : int, optional
        Homological degree. Inferred from ``minpres_degree`` when
        ``filtered_complex`` is already a minimal-presentation slicer.
    grid : iterable of array-like, optional
        Filtration grid used before squeezing. If omitted, ``compute_grid`` is
        called with ``grid_strategy`` and ``infer_grid_kwargs``.
    grid_strategy : str, default="exact"
        Strategy passed to ``compute_grid`` when ``grid`` is omitted.
    coordinates : bool, default=True
        If true, convert squeezed grid indices back to filtration coordinates.
        If false, return grid-index vertices.
    include_infinite : bool, default=True
        Whether to include vertices representing curves reaching infinity.
    sort : bool, default=True
        Whether to lexicographically sort vertices along each returned curve.
    aida_sort : bool, default=True
        Sort option forwarded to the C++ AIDA decomposition used internally.
    verbose : bool, default=False
        Verbosity forwarded to the C++ AIDA decomposition used internally.
    progress : bool, default=False
        Progress-bar option forwarded to the C++ AIDA decomposition used internally.
    minpres_kwargs : dict, optional
        Keyword arguments forwarded to ``slicer.minpres``. ``full_resolution``
        is forced to ``False``.
    plot : bool, default=False
        If true, plot the curves with ``multipers.plots.plot_end_curve``.
    min_length : float, default=-1
        Minimum plotted curve length. Does not filter returned curves.
    plot_kwargs : dict, optional
        Keyword arguments forwarded to ``plot_end_curve``.
    **infer_grid_kwargs : object
        Additional keyword arguments forwarded to ``compute_grid``.

    Output
    ------
    list[numpy.ndarray]
        One ``(k, 2)`` array per birth-curve. Rows are curve vertices in
        filtration coordinates when ``coordinates=True`` and squeezed grid
        indices otherwise.

    References
    ----------
    Brüstle, Oudot, Scoccola, and Thomas, "Counts and end-curves in
    two-parameter persistence", arXiv:2505.13412, 2025.
    """
    slicer = _as_slicer(filtered_complex)
    if slicer.num_parameters != 2:
        raise ValueError("birth_curves is only defined for 2-parameter modules.")
    from multipers import _end_curves_interface

    _end_curves_interface.require_birth()
    if slicer.is_pres:
        if degree is None:
            degree = slicer.pres_degree
        elif int(degree) != slicer.pres_degree:
            raise ValueError("Cannot change degree of an existing presentation.")

    requested_grid = grid is not None
    grid = (
        tuple(grid)
        if grid is not None
        else tuple(compute_grid(slicer, strategy=grid_strategy, **infer_grid_kwargs))
    )
    if not slicer.is_squeezed or requested_grid:
        slicer = slicer.grid_squeeze(grid)
    grid = tuple(slicer.filtration_grid)
    inf_indices = _grid_inf_indices(grid)

    if degree is None or degree < 0:
        raise ValueError("`degree` is inferred for presentation inputs, otherwise required.")
    degree = int(degree)

    if slicer.is_minpres:
        if slicer.minpres_degree != degree:
            raise ValueError(
                "Cannot change degree of an already minimal-presentation slicer."
            )
        presentation = slicer
    else:
        minpres_kwargs = {} if minpres_kwargs is None else dict(minpres_kwargs)
        minpres_kwargs["full_resolution"] = False
        presentation = slicer.minpres(degree=degree, **minpres_kwargs)

    birth_presentation = _birth_curve_presentation(presentation, inf_indices)
    birth_presentation = birth_presentation.minpres(
        degree=degree,
        full_resolution=False,
    )

    index_curves = _end_curves_interface.birth_curve_indices(
        birth_presentation,
        inf_indices.tolist(),
        include_infinite=include_infinite,
        sort=sort,
        aida_sort=aida_sort,
        verbose=verbose,
        progress=progress,
    )

    curves = []
    for curve in index_curves:
        curve = np.asarray(curve, dtype=np.int64)
        if coordinates:
            curve = _to_grid_coordinates(curve, grid, inf_indices, include_infinite)
        elif not include_infinite:
            curve = curve[np.all(curve < inf_indices, axis=1)]
        curves.append(curve)
    if plot:
        from multipers.plots import plot_end_curve

        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
        plot_kwargs["min_length"] = min_length
        plot_kwargs.setdefault("box", _plot_box_from_grid(grid, coordinates))
        plot_kwargs.setdefault("title", "Birth curves")
        plot_end_curve(curves, **plot_kwargs)
    return curves


def death_curves(
    filtered_complex,
    degree: Optional[int] = None,
    *,
    grid: Optional[Iterable] = None,
    grid_strategy: Lstrategies = "exact",
    coordinates: bool = True,
    include_infinite: bool = True,
    sort: bool = True,
    aida_sort: bool = True,
    verbose: bool = False,
    progress: bool = False,
    minpres_kwargs: Optional[dict] = None,
    plot: bool = False,
    min_length: float = -1,
    plot_kwargs: Optional[dict] = None,
    **infer_grid_kwargs,
) -> list[np.ndarray]:
    """Compute two-parameter death-curves.

    Death-curves are the spread-curve indecomposable summands obtained from
    the death part of the end-curve construction for a two-parameter module.
    Returns one ``(k, 2)`` array per death-curve. By default points are mapped
    back from squeezed grid indices to filtration coordinates, with one
    ``np.inf`` sentinel per axis for curves reaching infinity. If the input is
    a minimal-presentation slicer, ``degree`` is inferred from
    ``minpres_degree``. ``min_length`` only filters plotted curves when
    ``plot=True``; returned curves are not filtered.

    Parameters
    ----------
    filtered_complex : Slicer or SimplexTreeMulti-like
        Two-parameter filtered complex, module presentation, or object
        convertible to a slicer.
    degree : int, optional
        Homological degree. Inferred from ``minpres_degree`` when
        ``filtered_complex`` is already a minimal-presentation slicer.
    grid : iterable of array-like, optional
        Filtration grid used before squeezing. If omitted, ``compute_grid`` is
        called with ``grid_strategy`` and ``infer_grid_kwargs``.
    coordinates : bool, default=True
        If true, return filtration coordinates; otherwise return squeezed
        grid-index vertices.
    include_infinite : bool, default=True
        Whether to include vertices representing curves reaching infinity.
    sort : bool, default=True
        Whether to lexicographically sort vertices along each returned curve.
    aida_sort, verbose, progress
        Options forwarded to the C++ AIDA decomposition used internally.
    minpres_kwargs : dict, optional
        Keyword arguments forwarded to ``slicer.minpres``. ``full_resolution``
        is forced to ``False``.
    plot, min_length, plot_kwargs
        Plotting options forwarded to ``multipers.plots.plot_end_curve``;
        ``min_length`` does not filter returned curves.
    **infer_grid_kwargs : object
        Additional keyword arguments forwarded to ``compute_grid``.

    Availability
    ------------
    Requires the optional Persistence-Algebra and AIDA backends.

    Output
    ------
    list[numpy.ndarray]
        One ``(k, 2)`` array per death-curve. Rows are curve vertices in
        filtration coordinates when ``coordinates=True`` and squeezed grid
        indices otherwise.

    References
    ----------
    Brüstle, Oudot, Scoccola, and Thomas, "Counts and end-curves in
    two-parameter persistence", arXiv:2505.13412, 2025.
    """
    slicer = _as_slicer(filtered_complex)
    if slicer.num_parameters != 2:
        raise ValueError("death_curves is only defined for 2-parameter modules.")
    from multipers import _end_curves_interface

    _end_curves_interface.require_death()
    if slicer.is_pres:
        if degree is None:
            degree = slicer.pres_degree
        elif int(degree) != slicer.pres_degree:
            raise ValueError("Cannot change degree of an existing presentation.")

    requested_grid = grid is not None
    grid = (
        tuple(grid)
        if grid is not None
        else tuple(compute_grid(slicer, strategy=grid_strategy, **infer_grid_kwargs))
    )
    if not slicer.is_squeezed or requested_grid:
        slicer = slicer.grid_squeeze(grid)
    grid = tuple(slicer.filtration_grid)
    inf_indices = _grid_inf_indices(grid)

    if degree is None or degree < 0:
        raise ValueError("`degree` is inferred for presentation inputs, otherwise required.")
    degree = int(degree)

    if slicer.is_minpres:
        if slicer.minpres_degree != degree:
            raise ValueError(
                "Cannot change degree of an already minimal-presentation slicer."
            )
        presentation = slicer
    else:
        minpres_kwargs = {} if minpres_kwargs is None else dict(minpres_kwargs)
        minpres_kwargs["full_resolution"] = False
        presentation = slicer.minpres(degree=degree, **minpres_kwargs)

    index_curves = _end_curves_interface.death_curve_indices(
        presentation,
        degree,
        inf_indices.tolist(),
        include_infinite=include_infinite,
        sort=sort,
        aida_sort=aida_sort,
        verbose=verbose,
        progress=progress,
    )

    curves = []
    for curve in index_curves:
        curve = np.asarray(curve, dtype=np.int64)
        if coordinates:
            curve = _to_grid_coordinates(curve, grid, inf_indices, include_infinite)
        elif not include_infinite:
            curve = curve[np.all(curve < inf_indices, axis=1)]
        curves.append(curve)
    if plot:
        from multipers.plots import plot_end_curve

        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
        plot_kwargs["min_length"] = min_length
        plot_kwargs.setdefault("box", _plot_box_from_grid(grid, coordinates))
        plot_kwargs.setdefault("title", "Death curves")
        plot_end_curve(curves, **plot_kwargs)
    return curves
