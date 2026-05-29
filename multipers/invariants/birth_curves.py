from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np

from multipers.grids import Lstrategies, compute_grid

from ._utils import _as_slicer


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
        raise ValueError("birth-curve input presentation must be minimal.")
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

    from multipers._slicer_nanobind import build_contiguous_f64_slicer_from_packed_f64

    return build_contiguous_f64_slicer_from_packed_f64(
        new_boundary_indptr,
        new_boundary_flat,
        new_dimensions,
        new_filtrations,
    )


def _curve_vertices(
    summand,
    degree: int,
    inf_indices: np.ndarray,
    include_infinite: bool,
) -> np.ndarray:
    boundary_indptr, boundary_flat = summand.get_boundaries(packed=True)
    boundary_indptr = np.asarray(boundary_indptr, dtype=np.int64)
    boundary_flat = np.asarray(boundary_flat)
    dimensions = np.asarray(summand.get_dimensions(), dtype=np.int32)
    num_parameters = len(inf_indices)
    filtrations = np.asarray(summand.get_filtrations(), dtype=np.int64)
    if filtrations.ndim != 2:
        filtrations = filtrations.reshape(-1, num_parameters)

    generators = np.flatnonzero(dimensions == degree)
    relations = np.flatnonzero(dimensions == degree + 1)
    finite_limit = np.maximum(inf_indices - 1, 0)
    unbounded_limit = inf_indices if include_infinite else finite_limit
    generator_position = np.full(dimensions.shape[0], -1, dtype=np.int64)
    generator_position[generators] = np.arange(generators.size, dtype=np.int64)
    terminated_axes = np.zeros((generators.size, num_parameters), dtype=bool)

    chunks = []
    if generators.size:
        chunks.append(filtrations[generators])

    relation_starts = boundary_indptr[relations]
    relation_raw_lengths = boundary_indptr[relations + 1] - relation_starts
    relation_ids_full, relation_positions = _packed_row_positions(
        relation_starts, relation_raw_lengths
    )
    if relation_positions.size:
        boundary = boundary_flat[relation_positions]
        boundary_generator_positions = generator_position[boundary]
        keep = boundary_generator_positions >= 0
        relation_ids = relation_ids_full[keep]
        boundary = boundary[keep]
        boundary_generator_positions = boundary_generator_positions[keep]
        relation_lengths = np.bincount(
            relation_ids,
            minlength=relations.size,
        ).astype(np.int64, copy=False)
    else:
        relation_ids = np.empty(0, dtype=np.int64)
        boundary = np.empty(0, dtype=np.int64)
        boundary_generator_positions = np.empty(0, dtype=np.int64)
        relation_lengths = np.zeros(relations.size, dtype=np.int64)

    if boundary.size:
        relation_offsets = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(relation_lengths[:-1], dtype=np.int64),
            )
        )

        single_relation_ids = np.flatnonzero(relation_lengths == 1)
        if single_relation_ids.size:
            single_offsets = relation_offsets[single_relation_ids]
            single_boundary = boundary[single_offsets]
            single_generator_positions = boundary_generator_positions[single_offsets]
            single_relations = relations[single_relation_ids]
            active = filtrations[single_relations] > filtrations[single_boundary]
            valid = active.sum(axis=1) == 1
            if np.any(valid):
                axes = active[valid].argmax(axis=1)
                vertices = filtrations[single_relations[valid]].copy()
                vertices[np.arange(axes.size), axes] -= 1
                chunks.append(vertices)
                terminated_axes[single_generator_positions[valid], axes] = True

        # ``np.maximum.reduceat`` reduces ``filtrations[boundary]`` over each
        # nonempty relation block. Reduceat skips zero-length blocks because
        # consecutive starts would match. We pre-allocate ``joins`` per relation
        # for direct ``joins[relation_id]`` indexing.
        nonempty_relation_ids = np.flatnonzero(relation_lengths > 0)
        joins = np.empty((relations.size, num_parameters), dtype=filtrations.dtype)
        if nonempty_relation_ids.size:
            joins[nonempty_relation_ids] = np.maximum.reduceat(
                filtrations[boundary],
                relation_offsets[nonempty_relation_ids],
            )
        multi_relation_ids = np.flatnonzero(relation_lengths >= 2)
        if multi_relation_ids.size:
            chunks.append(joins[multi_relation_ids])
            multi_entries = relation_lengths[relation_ids] >= 2
            multi_generator_positions = boundary_generator_positions[multi_entries]
            active = (
                joins[relation_ids[multi_entries]]
                > filtrations[boundary[multi_entries]]
            )
            valid = active.sum(axis=1) == 1
            if np.any(valid):
                terminated_axes[
                    multi_generator_positions[valid], active[valid].argmax(axis=1)
                ] = True

    if generators.size:
        unbounded_generators, unbounded_axes = np.nonzero(~terminated_axes)
        if unbounded_generators.size:
            unbounded_vertices = filtrations[generators[unbounded_generators]].copy()
            unbounded_vertices[
                np.arange(unbounded_axes.size), unbounded_axes
            ] = unbounded_limit[unbounded_axes]
            chunks.append(unbounded_vertices)

    if not chunks:
        return np.empty((0, num_parameters), dtype=np.int64)
    return np.unique(np.vstack(chunks).astype(np.int64, copy=False), axis=0)


def _sort_spread_curve(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    order = np.lexsort((-points[:, 1], points[:, 0]))
    return points[order]


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

    Returns one ``(k, 2)`` array per birth-curve. By default points are mapped
    back from squeezed grid indices to filtration coordinates, with one
    ``np.inf`` sentinel per axis for curves reaching infinity. If the input is
    a minimal-presentation slicer, ``degree`` is inferred from
    ``minpres_degree``. ``min_length`` only filters plotted curves when
    ``plot=True``; returned curves are not filtered.
    """
    slicer = _as_slicer(filtered_complex)
    if slicer.num_parameters != 2:
        raise ValueError("birth_curves is only defined for 2-parameter modules.")
    if degree is None and slicer.is_minpres:
        degree = slicer.minpres_degree

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
        raise ValueError("`degree` is inferred for minpres inputs, otherwise required.")
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

    from multipers import ops

    curves = []
    for summand in ops.aida(
        birth_presentation,
        sort=aida_sort,
        verbose=verbose,
        progress=progress,
    ):
        curve = _curve_vertices(summand, degree, inf_indices, include_infinite)
        if sort:
            curve = _sort_spread_curve(curve)
        if coordinates:
            curve = _to_grid_coordinates(curve, grid, inf_indices, include_infinite)
        elif not include_infinite:
            curve = curve[np.all(curve < inf_indices, axis=1)]
        curves.append(curve)
    if plot:
        from multipers.plots import plot_birth_curve

        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
        plot_kwargs["min_length"] = min_length
        plot_birth_curve(curves, **plot_kwargs)
    return curves
