from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np

from multipers._signed_measure_meta import signed_measure
from multipers.grids import Lstrategies, compute_grid
from multipers.multiparameter_module_approximation import module_approximation
from multipers.point_measure import barcode_from_rank_sm as barcode_from_rank_signed_measure
from ._utils import _as_slicer, _normalize_degrees
from .birth_curves import birth_curves
from .projected_barcode import projected_barcode


def death_curves(*args, **kwargs):
    """Two-parameter death-curve invariant, not implemented yet.

    Death-curves are the indecomposable summands of ``coker(xy)`` for a
    2-parameter module over a grid.  They are ephemeral modules and are
    classified by spread curves, dual to the birth-curves from ``ker(xy)``.

    Parameters
    ----------
    *args, **kwargs:
        Reserved for the future death-curve API.

    Raises
    ------
    NotImplementedError
        Always raised until the invariant is implemented.
    """
    raise NotImplementedError("death_curves is not implemented yet.")


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
    """
    raise NotImplementedError("end_curves is not implemented yet.")


def fibered_barcode(
    filtered_complex,
    basepoint,
    direction=None,
    *,
    degree: Optional[int] = None,
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
    **kwargs : object
        Forwarded to ``persistence_on_line``.

    Returns
    -------
    tuple or numpy.ndarray
        All degree barcodes for the line, or the selected degree barcode.
    """
    barcode = _as_slicer(filtered_complex).persistence_on_line(
        basepoint,
        direction=direction,
        **kwargs,
    )
    if degree is None:
        return barcode
    degree = int(degree)
    return barcode[degree]


def fibered_barcodes(
    filtered_complex,
    basepoints,
    directions=None,
    *,
    degree: Optional[int] = None,
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
    **kwargs : object
        Forwarded to ``persistence_on_lines``.

    Returns
    -------
    tuple
        One barcode object per line.  If ``degree`` is ``None``, each entry
        contains all homological degrees; otherwise each entry is the selected
        degree barcode.
    """
    barcodes = _as_slicer(filtered_complex).persistence_on_lines(
        basepoints,
        directions=directions,
        **kwargs,
    )
    if degree is None:
        return barcodes
    degree = int(degree)
    return tuple(barcode[degree] for barcode in barcodes)


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
    "end_curves",
    "fibered_barcode",
    "fibered_barcodes",
    "hilbert_function",
    "module_approximation",
    "projected_barcode",
    "rank_invariant",
    "signed_measure",
]
