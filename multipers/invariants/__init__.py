from __future__ import annotations

from collections.abc import Iterable, Sequence
from operator import index as _index
from typing import Optional

import numpy as np

import multipers.logs as _mp_logs
from multipers._signed_measure_meta import signed_measure
from multipers.array_api import api_from_tensor, api_from_tensors
from multipers.grids import Lstrategies, compute_grid
from multipers.invariants.landscapes import persistence_landscape
from multipers.multiparameter_module_approximation import module_approximation
from multipers.point_measure import barcode_from_rank_sm as barcode_from_rank_signed_measure
from ._utils import _as_slicer, _normalize_degrees
from .end_curves import birth_curves, death_curves
from .projected_barcode import projected_barcode
from .skyscraper import SkyscraperInvariant, skyscraper_invariant


def fibered_barcode(
    filtered_complex,
    basepoint,
    direction=None,
    *,
    degree: Optional[int] = None,
    min_persistence: float = 0.0,
    **kwargs,
):
    """Compute the barcode of one affine-line restriction.

    This is the fibered-barcode descriptor: restrict the multiparameter module
    to a positive-direction affine line and compute ordinary one-parameter
    persistence on that line.

    Parameters
    ----------
    filtered_complex : Slicer or SimplexTreeMulti-like
        Slicer-like filtered complex or object convertible to a slicer.
    basepoint : array-like
        Line basepoint.
    direction : array-like, optional
        Optional line direction.  Passed to ``persistence_on_line``.
    degree : int, optional
        If provided, return only this homological degree.  If omitted, return all
        degree barcodes for the line.
    min_persistence : float, default=0.0
        Keep only bars with strict persistence larger than this value.  The
        default removes zero-length bars; pass a negative value to keep them.
    **kwargs : object
        Forwarded to ``persistence_on_line``.

    Returns
    -------
    tuple or numpy.ndarray
        All degree barcodes for the line, or the selected degree barcode.

    Output
    ------
    tuple or numpy.ndarray
        All degree barcodes for the line, or the selected degree barcode.

    References
    ----------
    Carlsson and Zomorodian, "The theory of multidimensional persistence",
    Discrete & Computational Geometry, 2009. DOI: 10.1007/s00454-009-9176-0.
    """
    barcode = _as_slicer(filtered_complex).persistence_on_line(
        basepoint,
        direction=direction,
        **kwargs,
    )
    if degree is None:
        return _filter_barcodes_by_persistence(barcode, min_persistence)
    degree = int(degree)
    return _filter_barcode_by_persistence(barcode[degree], min_persistence)


def fibered_barcodes(
    filtered_complex,
    basepoints,
    directions=None,
    *,
    degree: Optional[int] = None,
    min_persistence: float = 0.0,
    **kwargs,
):
    """Compute barcodes of several affine-line restrictions.

    Parameters
    ----------
    filtered_complex : Slicer or SimplexTreeMulti-like
        Slicer-like filtered complex or object convertible to a slicer.
    basepoints : array-like
        One basepoint per line.
    directions : array-like, optional
        Optional direction per line.  Passed to ``persistence_on_lines``.
    degree : int, optional
        If provided, return only this homological degree for each line.  If
        omitted, return all degree barcodes for each line.
    min_persistence : float, default=0.0
        Keep only bars with strict persistence larger than this value.  The
        default removes zero-length bars; pass a negative value to keep them.
    **kwargs : object
        Forwarded to ``persistence_on_lines``.

    Returns
    -------
    tuple
        One barcode object per line.  If ``degree`` is ``None``, each entry
        contains all homological degrees; otherwise each entry is the selected
        degree barcode.

    Output
    ------
    tuple
        One barcode object per line.  If ``degree`` is ``None``, each entry
        contains all homological degrees; otherwise each entry is the selected
        degree barcode.

    References
    ----------
    Carlsson and Zomorodian, "The theory of multidimensional persistence",
    Discrete & Computational Geometry, 2009. DOI: 10.1007/s00454-009-9176-0.
    """
    barcodes = _as_slicer(filtered_complex).persistence_on_lines(
        basepoints,
        directions=directions,
        **kwargs,
    )
    if degree is None:
        return tuple(
            _filter_barcodes_by_persistence(barcode, min_persistence)
            for barcode in barcodes
        )
    degree = int(degree)
    return tuple(
        _filter_barcode_by_persistence(barcode[degree], min_persistence)
        for barcode in barcodes
    )


def _filter_barcode_by_persistence(barcode, min_persistence: float):
    api = api_from_tensor(barcode)
    bars = api.reshape(api.astensor(barcode), (-1, 2))
    if api.size(bars) == 0:
        return bars
    persistence = bars[:, 1] - bars[:, 0]
    return bars[persistence > float(min_persistence)]


def _filter_barcodes_by_persistence(barcodes, min_persistence: float):
    return tuple(
        _filter_barcode_by_persistence(barcode, min_persistence)
        for barcode in barcodes
    )


def hilbert_function(
    filtered_complex,
    degree: Optional[int] = None,
    degrees: Sequence[int] = (),
    *,
    grid: Optional[Iterable] = None,
    grid_strategy: Lstrategies = "exact",
    resolution: Optional[int | Iterable[int]] = None,
    unique: bool = True,
    drop_quantiles: Iterable[float] = (0, 0),
    threshold_min=None,
    threshold_max=None,
    n_jobs: int = -1,
    ignore_infinite_filtration_values: bool = True,
    mobius: str = "auto",
    plot: bool = False,
    colorbar: bool = True,
    plot_kwargs: Optional[dict] = None,
    **kwargs,
):
    """Evaluate Hilbert functions on a grid.

    For a persistence module ``M`` this is the pointwise dimension function
    ``h_M(x) = dim M(x)``.  The computation uses the Hilbert signed measure
    ``mu`` characterized by ``h_M(x) = mu({y | y <= x})`` and integrates it over
    the requested grid, so the output is a dense tensor with one axis per
    parameter.

    Use ``degree`` for a single dense tensor.  Use ``degrees`` for a tuple of
    tensors in the same degree order as ``signed_measure``.  If
    ``filtered_complex`` is already a minimal-presentation slicer, its
    ``minpres_degree`` is used when neither argument is given.

    Parameters
    ----------
    filtered_complex : Slicer or SimplexTreeMulti-like
        Filtered complex, module presentation, or object accepted by
        ``signed_measure``.
    degree : int, optional
        Homological degree to evaluate.  Returns a single dense tensor and is
        mutually exclusive with ``degrees``.
    degrees : sequence of int, optional
        Homological degrees to evaluate.  Returns a tuple of dense tensors and
        must not contain duplicates.
    grid : iterable of array-like, optional
        Filtration grid on which to evaluate.  If omitted, ``compute_grid`` is
        called with the grid options below.
    grid_strategy : str, default="exact"
        Strategy passed to ``compute_grid`` when ``grid`` is omitted.
    resolution : int or iterable of int, optional
        Grid resolution passed to ``compute_grid``.
    unique : bool, default=True
        Whether inferred grid axes should keep unique values.
    drop_quantiles : iterable of float, default=(0, 0)
        Quantiles dropped by ``compute_grid`` when inferring axes.
    threshold_min, threshold_max : scalar or array-like, optional
        Lower and upper grid thresholds passed to ``compute_grid``.
    n_jobs : int, default=-1
        Parallelism passed to ``signed_measure``.
    ignore_infinite_filtration_values : bool, default=True
        Whether the signed-measure backend should ignore infinite filtration
        values.
    mobius : str, default="auto"
        Mobius-inversion backend choice passed to ``signed_measure``.
    plot : bool, default=False
        If true, plot the dense Hilbert tensor(s) as discrete surfaces and still
        return the tensor data.
    colorbar : bool, default=True
        Whether plotted Hilbert surfaces include a colorbar.
    plot_kwargs : dict, optional
        Keyword arguments forwarded to ``multipers.plots.plot_surface`` or
        ``plot_surfaces``.
    **kwargs : object
        Forwarded to ``signed_measure(..., invariant="hilbert")``.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, ...]
        Dense Hilbert tensor for ``degree``, or one dense tensor per entry of
        ``degrees``.  Each tensor has shape ``tuple(len(axis) for axis in
        grid)``.

    Output
    ------
    numpy.ndarray or tuple[numpy.ndarray, ...]
        Dense Hilbert tensor for ``degree``, or one dense tensor per entry of
        ``degrees``. Each tensor has shape ``tuple(len(axis) for axis in grid)``.
    """
    filtered_complex = _as_slicer(filtered_complex)
    inferred_degree = None
    if filtered_complex.is_minpres:
        inferred_degree = filtered_complex.minpres_degree
        if inferred_degree is None or inferred_degree < 0:
            inferred_degree = None
        else:
            inferred_degree = int(inferred_degree)

    degrees, single_output = _normalize_degrees(
        degree,
        degrees,
        inferred_degree=inferred_degree,
    )
    if grid is None:
        grid = compute_grid(
            filtered_complex,
            strategy=grid_strategy,
            resolution=resolution,
            unique=unique,
            drop_quantiles=drop_quantiles,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    grid = tuple(grid)
    sms = signed_measure(
        filtered_complex,
        degree=None,
        degrees=degrees,
        grid=grid,
        invariant="hilbert",
        n_jobs=n_jobs,
        ignore_infinite_filtration_values=ignore_infinite_filtration_values,
        mobius=mobius,
        **kwargs,
    )

    from multipers.point_measure import integrate_measure

    out = tuple(
        integrate_measure(points, weights, filtration_grid=grid)
        for points, weights in sms
    )
    if plot:
        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
        plot_kwargs.setdefault("discrete_surface", True)
        plot_kwargs.setdefault("has_negative_values", False)
        plot_kwargs.setdefault("threshold_min", 0)
        plot_kwargs.setdefault("threshold_max", 10)
        plot_kwargs.setdefault("colorbar", colorbar)
        if single_output or len(out) == 1:
            from multipers.plots import plot_surface

            plot_surface(grid=grid, hf=out[0], **plot_kwargs)
        else:
            from multipers.plots import plot_surfaces

            plot_surfaces((grid, np.asarray(out)), **plot_kwargs)
    return out[0] if single_output else out


def rank_invariant(
    filtered_complex,
    degree: Optional[int] = None,
    degrees: Sequence[int] = (),
    *,
    grid: Optional[Iterable] = None,
    grid_strategy: Lstrategies = "exact",
    resolution: Optional[int | Iterable[int]] = None,
    unique: bool = True,
    drop_quantiles: Iterable[float] = (0, 0),
    threshold_min=None,
    threshold_max=None,
    n_jobs: int = -1,
    ignore_infinite_filtration_values: bool = True,
    mobius: str = "auto",
):
    """Evaluate dense rank-invariant tensors on a grid.

    For a persistence module ``M`` this is the two-point invariant
    ``rho_M(a, b) = rank(M(a) -> M(b))`` for comparable grades ``a <= b``.
    The output for each degree is a dense tensor of shape
    ``grid_shape + grid_shape``: first the birth-coordinate axes, then the
    death-coordinate axes.  Entries with incomparable birth/death pairs are
    zero.

    The computation reconstructs the tensor from the rank signed measure.  Death
    coordinates in the signed measure use the opposite poset convention, so they
    are switched before integration and flipped back afterwards.

    Use ``degree`` for a single dense tensor.  Use ``degrees`` for a tuple of
    tensors in the same degree order as ``signed_measure``.  If
    ``filtered_complex`` is already a minimal-presentation slicer, its
    ``minpres_degree`` is used when neither argument is given.

    Parameters
    ----------
    filtered_complex : Slicer or SimplexTreeMulti-like
        Filtered complex, module presentation, or object accepted by
        ``signed_measure``.
    degree : int, optional
        Homological degree to evaluate.  Returns a single dense tensor and is
        mutually exclusive with ``degrees``.
    degrees : sequence of int, optional
        Homological degrees to evaluate.  Returns a tuple of dense tensors and
        must not contain duplicates.
    grid : iterable of array-like, optional
        Filtration grid for both birth and death coordinates.  If omitted,
        ``compute_grid`` is called with the grid options below.
    grid_strategy : str, default="exact"
        Strategy passed to ``compute_grid`` when ``grid`` is omitted.
    resolution : int or iterable of int, optional
        Grid resolution passed to ``compute_grid``.
    unique : bool, default=True
        Whether inferred grid axes should keep unique values.
    drop_quantiles : iterable of float, default=(0, 0)
        Quantiles dropped by ``compute_grid`` when inferring axes.
    threshold_min, threshold_max : scalar or array-like, optional
        Lower and upper grid thresholds passed to ``compute_grid``.
    n_jobs : int, default=-1
        Parallelism passed to ``signed_measure``.
    ignore_infinite_filtration_values : bool, default=True
        Whether the signed-measure backend should ignore infinite filtration
        values.
    mobius : str, default="auto"
        Mobius-inversion backend choice passed to ``signed_measure``.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, ...]
        Dense rank tensor for ``degree``, or one dense rank tensor per entry of
        ``degrees``.  Each tensor has shape ``grid_shape + grid_shape``.

    Output
    ------
    numpy.ndarray or tuple[numpy.ndarray, ...]
        Dense rank tensor for ``degree``, or one dense rank tensor per entry of
        ``degrees``. Each tensor has shape ``grid_shape + grid_shape``.

    References
    ----------
    Carlsson and Zomorodian, "The theory of multidimensional persistence",
    Discrete & Computational Geometry, 2009. DOI: 10.1007/s00454-009-9176-0.
    """
    filtered_complex = _as_slicer(filtered_complex)
    inferred_degree = None
    if filtered_complex.is_minpres:
        inferred_degree = filtered_complex.minpres_degree
        if inferred_degree is None or inferred_degree < 0:
            inferred_degree = None
        else:
            inferred_degree = int(inferred_degree)

    degrees, single_output = _normalize_degrees(
        degree,
        degrees,
        inferred_degree=inferred_degree,
    )
    if grid is None:
        grid = compute_grid(
            filtered_complex,
            strategy=grid_strategy,
            resolution=resolution,
            unique=unique,
            drop_quantiles=drop_quantiles,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    grid = tuple(grid)

    sms = signed_measure(
        filtered_complex,
        degree=None,
        degrees=degrees,
        grid=grid,
        invariant="rank",
        n_jobs=n_jobs,
        ignore_infinite_filtration_values=ignore_infinite_filtration_values,
        mobius=mobius,
    )

    from multipers.point_measure import integrate_measure

    num_parameters = len(grid)
    grid_shape = tuple(len(axis) for axis in grid)
    death_axes = tuple(range(num_parameters, 2 * num_parameters))
    death_slice = slice(num_parameters, None)
    valid_pairs = np.ones(grid_shape + grid_shape, dtype=bool)
    for axis, values in enumerate(grid):
        values = np.asarray(values)
        birth_shape = [1] * (2 * num_parameters)
        death_shape = [1] * (2 * num_parameters)
        birth_shape[axis] = len(values)
        death_shape[num_parameters + axis] = len(values)
        valid_pairs &= values.reshape(birth_shape) <= values.reshape(death_shape)
    switched_death_grid = tuple(
        -np.concatenate((np.asarray(axis, dtype=np.float64)[1:], [np.inf]))[::-1]
        for axis in grid
    )
    rank_grid = tuple(grid) + switched_death_grid

    out = []
    for points, weights in sms:
        points = np.asarray(points)
        weights = np.asarray(weights)
        if points.size == 0:
            out.append(np.zeros(grid_shape + grid_shape, dtype=weights.dtype))
            continue

        switched_points = points.copy()
        np.negative(switched_points[:, death_slice], out=switched_points[:, death_slice])
        rank = integrate_measure(
            switched_points,
            weights,
            filtration_grid=rank_grid,
        )
        rank = np.flip(rank, axis=death_axes)
        np.multiply(rank, valid_pairs, out=rank, casting="unsafe")
        out.append(rank)
    out = tuple(out)
    return out[0] if single_output else out


def graphcode(
    filtered_complex,
    degree: Optional[int] = None,
    *,
    primary_parameter: int = 1,
    slices: int = 0,
    include_infinite_bars: bool = True,
    filter_out_disjoint_pairs: bool = True,
    do_exhaustive_reduction: bool = False,
    relevance_threshold: float = -1.0,
    secondary_threshold: float = np.inf,
    compressed: bool = True,
    double_edges: bool = True,
    verbose: bool = False,
):
    """Compute graphcode of a 2-parameter module, in memory.

    Input must be a presentation slicer. Nonminimal presentations are minimized
    before computing the ``vertices``, ``edges``, and ``slice_values`` output.
    """
    from multipers import _graphcode_interface

    slicer = _as_slicer(filtered_complex)
    if slicer.num_parameters != 2:
        raise ValueError("graphcode expects a 2-parameter slicer.")
    if not slicer.is_pres or slicer.pres_degree < 0:
        raise ValueError("graphcode expects a presentation slicer.")
    if degree is None:
        degree = int(slicer.pres_degree)
    else:
        if isinstance(degree, bool):
            raise ValueError("`degree` must be a non-negative integer.")
        degree = _index(degree)
        if degree < 0:
            raise ValueError("`degree` must be a non-negative integer.")
        if slicer.pres_degree != degree:
            raise ValueError("Cannot change degree of an existing presentation slicer.")

    if not slicer.is_minpres:
        slicer = slicer.minpres(degree=degree, full_resolution=False)

    _graphcode_interface.require()

    grid_api = None
    grid = None
    coordinate_output = False
    backend_slicer = slicer
    if slicer.is_squeezed:
        grid = slicer.filtration_grid
        grid_api = api_from_tensors(*grid)
        coordinate_output = (
            slices == 0
            and np.isinf(float(secondary_threshold))
            and float(relevance_threshold) <= 0.0
        )
        if not coordinate_output:
            if any(grid_api.has_grad(axis) for axis in grid):
                _mp_logs.warn_autodiff(
                    "graphcode preserves filtration gradients only for squeezed slicers "
                    "with slices=0 and default thresholds."
                )
            backend_slicer = slicer.unsqueeze()

    with _mp_logs.timings(
        "graphcode",
        enabled=verbose,
        details={"degree": degree, "primary_parameter": primary_parameter, "slices": slices},
    ) as timing:
        out = _graphcode_interface.graphcode(
            backend_slicer,
            primary_parameter=primary_parameter,
            slices=slices,
            include_infinite_bars=include_infinite_bars,
            filter_out_disjoint_pairs=filter_out_disjoint_pairs,
            do_exhaustive_reduction=do_exhaustive_reduction,
            relevance_threshold=relevance_threshold,
            secondary_threshold=secondary_threshold,
            compressed_graphcode=compressed,
            double_edges=double_edges,
        )
        timing.substep("backend_call")
    return _graphcode_array_output(
        out,
        api=grid_api,
        grid=grid,
        primary_parameter=primary_parameter,
        coordinate_output=coordinate_output,
    )


def _graphcode_array_output(
    out,
    *,
    api=None,
    grid=None,
    primary_parameter: int = 1,
    coordinate_output: bool = False,
):
    vertices = np.ascontiguousarray(np.asarray(out["vertices"], dtype=np.float64).reshape(-1, 4))
    edges = np.ascontiguousarray(np.asarray(out["edges"], dtype=np.int64).reshape(-1, 2))
    slice_values = np.ascontiguousarray(np.asarray(out["slice_values"], dtype=np.float64).reshape(-1))
    if api is None or grid is None or not coordinate_output:
        api = api_from_tensor(vertices)
        return {
            "vertices": api.astensor(vertices, contiguous=True),
            "edges": api.astensor(edges, dtype=api.int64, contiguous=True),
            "slice_values": api.astensor(slice_values, contiguous=True),
        }

    primary_parameter = _index(primary_parameter)
    secondary_parameter = 1 - primary_parameter
    birth = _graphcode_take_grid_values(api, grid[secondary_parameter], vertices[:, 0])
    death = _graphcode_take_grid_values(api, grid[secondary_parameter], vertices[:, 1])
    slice_index = api.astensor(
        np.ascontiguousarray(vertices[:, 2]),
        dtype=getattr(birth, "dtype", None),
        device=api.device(grid[primary_parameter]),
    )
    vertex_slice_values = _graphcode_take_grid_values(api, grid[primary_parameter], vertices[:, 3])
    return {
        "vertices": api.ascontiguous(
            api.moveaxis(api.stack([birth, death, slice_index, vertex_slice_values]), 0, 1)
        ),
        "edges": api.astensor(
            edges,
            dtype=api.int64,
            contiguous=True,
            device=api.device(grid[primary_parameter]),
        ),
        "slice_values": api.ascontiguous(
            _graphcode_take_grid_values(api, grid[primary_parameter], slice_values)
        ),
    }
def _graphcode_take_grid_values(api, axis, coordinates):
    axis = api.astensor(axis, contiguous=True)
    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1)
    finite = np.isfinite(coordinates) & (0 <= coordinates) & (coordinates < len(axis))
    indices = np.zeros(coordinates.shape, dtype=np.int64)
    indices[finite] = np.rint(coordinates[finite]).astype(np.int64)
    index_tensor = api.astensor(indices, dtype=api.int64, device=api.device(axis))
    values = axis[index_tensor]
    if np.all(finite):
        return api.ascontiguous(values)
    infinity = api.astensor(
        np.full(coordinates.shape, np.finfo(np.float64).max),
        dtype=getattr(values, "dtype", None),
        device=api.device(axis),
    )
    mask = api.astensor(finite, device=api.device(axis))
    return api.ascontiguous(api.where(mask, values, infinity))


def betti_degrees(resolution, degree: Optional[int] = None):
    """Return generator grades grouped by free-resolution index.

    For a minimal free resolution
    ``... -> F_2 -> F_1 -> F_0 -> H_d -> 0`` encoded as a slicer, the returned
    tuple has one array per free term ``F_i``.  Its rows are the multigrades of
    the rank-one free summands in that term, so repeated rows represent Betti
    multiplicity.

    Slicer dimensions store the resolution index shifted by homological degree:
    generators of ``F_i`` live in slicer dimension ``d + i``.  If ``resolution``
    is a minimal-presentation slicer and ``degree`` is omitted, its
    ``minpres_degree`` is used.

    Parameters
    ----------
    resolution : Slicer
        Slicer encoding a minimal presentation or free resolution.
    degree : int, optional
        Homological degree ``d`` of the represented module.  Required for
        non-minpres slicers; inferred from ``minpres_degree`` for minpres input.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        ``out[i]`` contains the multigrades of the free generators in ``F_i``.
        Empty input returns an empty tuple.

    Output
    ------
    tuple[numpy.ndarray, ...]
        ``out[i]`` contains the multigrades of the free generators in ``F_i``.
        Empty input returns an empty tuple.

    References
    ----------
    Carlsson and Zomorodian, "The theory of multidimensional persistence",
    Discrete & Computational Geometry, 2009. DOI: 10.1007/s00454-009-9176-0.
    """
    from multipers.slicer import is_slicer

    if not is_slicer(resolution):
        raise TypeError(f"Expected a Slicer. Got {type(resolution)!r}.")
    if resolution.is_minpres:
        if degree is None:
            degree = resolution.minpres_degree
        elif resolution.minpres_degree != int(degree):
            raise ValueError(
                "Cannot change degree of an already minimal-presentation slicer."
            )
    if degree is None or degree < 0:
        raise ValueError("`degree` must be provided for non-minpres slicers.")
    degree = int(degree)

    dimensions = np.asarray(resolution.get_dimensions(), dtype=np.int32)
    filtrations = np.asarray(
        resolution.get_filtrations(unsqueeze=resolution.is_squeezed)
    )
    if dimensions.size == 0:
        return tuple()

    term_dimensions = np.arange(degree, int(dimensions[-1]) + 1, dtype=dimensions.dtype)
    starts = np.searchsorted(dimensions, term_dimensions, side="left")
    ends = np.searchsorted(dimensions, term_dimensions, side="right")
    present = starts < ends
    if not np.all(present):
        first_missing = int(np.flatnonzero(~present)[0])
        starts = starts[:first_missing]
        ends = ends[:first_missing]

    return tuple(filtrations[start:end] for start, end in zip(starts, ends, strict=True))


def betti_table(resolution, degree: Optional[int] = None):
    """Return sparse Betti tables grouped by free-resolution index.

    This compresses ``betti_degrees`` by aggregating equal multigrades.  For each
    free-resolution term ``F_i``, the output contains ``(grades,
    multiplicities)`` where ``grades[j]`` has Betti multiplicity
    ``multiplicities[j]``.

    Parameters
    ----------
    resolution : Slicer
        Slicer encoding a minimal presentation or free resolution.
    degree : int, optional
        Homological degree ``d`` of the represented module.  Required for
        non-minpres slicers; inferred from ``minpres_degree`` for minpres input.

    Returns
    -------
    tuple[tuple[numpy.ndarray, numpy.ndarray], ...]
        For each free term ``F_i``, a pair ``(grades, multiplicities)``.  The
        rows of ``grades`` are unique multigrades and ``multiplicities`` gives
        their Betti counts.

    Output
    ------
    tuple[tuple[numpy.ndarray, numpy.ndarray], ...]
        For each free term ``F_i``, a pair ``(grades, multiplicities)``. The
        rows of ``grades`` are unique multigrades and ``multiplicities`` gives
        their Betti counts.

    References
    ----------
    Carlsson and Zomorodian, "The theory of multidimensional persistence",
    Discrete & Computational Geometry, 2009. DOI: 10.1007/s00454-009-9176-0.
    """
    table = []
    for grades in betti_degrees(resolution, degree=degree):
        if len(grades) == 0:
            table.append((grades, np.empty((0,), dtype=np.int32)))
            continue
        unique, inverse = np.unique(grades, axis=0, return_inverse=True)
        table.append((unique, np.bincount(inverse).astype(np.int32, copy=False)))
    return tuple(table)


__all__ = [
    "barcode_from_rank_signed_measure",
    "betti_degrees",
    "betti_table",
    "birth_curves",
    "death_curves",
    "fibered_barcode",
    "fibered_barcodes",
    "graphcode",
    "hilbert_function",
    "module_approximation",
    "persistence_landscape",
    "projected_barcode",
    "rank_invariant",
    "signed_measure",
    "SkyscraperInvariant",
    "skyscraper_invariant",
]
