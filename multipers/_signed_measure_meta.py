from collections.abc import Iterable, Sequence
from typing import Optional, Union

import numpy as np

import multipers as mp
from multipers.grids import compute_grid, sms_in_grid
from multipers.plots import plot_signed_measures
from multipers.point_measure import clean_sms, zero_out_sms
from multipers.simplex_tree_multi import (
    SimplexTreeMulti_type,
    _available_strategies,
    is_simplextree_multi,
)
from multipers.slicer import (
    Slicer_type,
    _hilbert_signed_measure,
    _rank_from_slicer,
    is_slicer,
)


def signed_measure(
    filtered_complex: Union[SimplexTreeMulti_type, Slicer_type],
    degree: Optional[int] = None,
    degrees: Sequence[int | None] = [],
    mass_default=None,
    grid_strategy: _available_strategies = "exact",
    invariant: Optional[str] = None,
    plot: bool = False,
    verbose: bool = False,
    n_jobs: int = -1,
    expand_collapse: bool = False,
    backend: Optional[str] = None,
    thread_id: str = "",
    grid: Optional[Iterable] = None,
    coordinate_measure: bool = False,
    num_collapses: int = 0,
    clean: Optional[bool] = None,
    vineyard: bool = False,
    grid_conversion: Optional[Iterable] = None,
    **infer_grid_kwargs,
) -> list[tuple[np.ndarray, np.ndarray]]:
    r"""
    Computes the signed measures given by the decomposition of the hilbert
    function or the euler characteristic, or the rank invariant.

    Input
    -----
     - filtered_complex: given by a simplextree or a slicer.
     - degree:int|None / degrees:list[int] the degrees to compute.
       None represents the euler characteristic.
     - mass_default: Either None, or 'auto' or 'inf', or array-like of floats.
       Where to put the default mass to get a zero-mass measure.
       This corresponds to zero-out the filtered complex outside of $\{ x\in \mathbb R^n \mid x\le `mass_default`\}$
     - invariant: The invariant to use, either "hilbert", "rank", or "euler".
     - plot:bool, plots the computed measures if true.
     - n_jobs:int, number of jobs. Defaults to #cpu.
     - verbose:bool, prints c++ logs.
     - expand_collapse: when the input is a simplextree,
       only expands the complex when computing 1-dimensional slices.
       Meant to reduce memory footprint at some computational expense.
     - backend:str  reduces first the filtered complex using an external library `backend`,
         see ``backend`` in :func:`multipers.io.reduce_complex`.
     - grid: If given, the computations will be done on the restriction of the filtered complex to this grid.
        It can also be used for auto-differentiation, i.e., if the grid is a list of pytorch tensors,
        then the output measure will be pytorch-differentiable.
     - grid_strategy: If not squeezed yet, and no grid is given,
       the strategy to coarsen the grid; see ``strategy`` in :func:`multipers.grids.compute_grid`.
     - coordinate_measure: bool, if True, compute the signed measure as a coordinates given in grid.
     - num_collapses: int, if `filtered_complex` is a simplextree, does some collapses if possible.
     - clean: if True, reduces the measure. It is not necessary in general.

    Output
    ------

    `[signed_measure_of_degree for degree in degrees]`
    with `signed_measure_of_degree` of the form `(dirac location, dirac weights)`.

    Notes on computational backends
    -------------------------------
    There are several backends for each of these computations.
    The backend for computations used can be displayed with `verbose=True`, use it!
    Also note that if `backend` is given, then the input will be converted to a slicer.
     - Euler: is always computed by summing the weights of the simplices
     - Hilbert: is computed by computing persistence on slices, and a Möbius inversion,
       unless the detected input is a minimal presentation (i.e., `filtered_complex.is_minpres`),
       which in that case, doesn't need any computation.
       - If the input is a simplextree, this is done via a the standard Gudhi implementation,
         with parallel (TBB) computations of slices.
       - If the input is a slicer then
         - If the input is vineyard-capable, then slices are computed via vineyards updates.
           It is slower in general, but faster if single threaded.
           In particular, it is usually faster to use this backend if you want to compute the
           signed measure of multiple datasets in a parallel context.
         - Otherwise, slices are computed in parallel.
           It is usually faster to use this backend if not in a parallel context.
     - Rank: Same as Hilbert.
    """
    ## TODO : add timings in verbose
    if grid_conversion is not None:
        grid = tuple(f for f in grid_conversion)
        raise DeprecationWarning(
            """
                Parameter `grid_conversion` is deprecated. Use `grid` instead. 
                Most of the time there is no conversion anymore.
                """
        )

    if degree is not None or len(degrees) == 0:
        degrees = list(degrees) + [degree]
    if None in degrees:
        assert (
            len(degrees) == 1
        ), f"Can only compute one invariant at the time. Got {degrees=}, {invariant=}."
        assert invariant is None or not (
            "hilbert" in invariant or "rank" in invariant
        ), f"Hilbert and Rank cannot compute `None` degree. got {degrees=}, {invariant=}."
        invariant = "euler"
    if clean is None:
        clean = True if None in degrees else False

    assert invariant is None or invariant in [
        "hilbert",
        "rank_invariant",
        "euler",
        "rank",
        "euler_characteristic",
        "hilbert_function",
    ]

    assert (
        not plot or filtered_complex.num_parameters == 2
    ), "Can only plot 2d measures."

    if grid is None:
        if not filtered_complex.is_squeezed:
            grid = compute_grid(
                filtered_complex, strategy=grid_strategy, **infer_grid_kwargs
            )
        else:
            grid = tuple(np.asarray(f) for f in filtered_complex.filtration_grid)

    if mass_default is None:
        mass_default = mass_default
    elif isinstance(mass_default, str):
        if mass_default == "auto":
            mass_default = np.array([1.1 * np.max(f) - 0.1 * np.min(f) for f in grid])
        elif mass_default == "inf":
            mass_default = np.array([np.inf] * filtered_complex.num_parameters)
        else:
            raise NotImplementedError
    else:
        mass_default = np.asarray(mass_default)
        assert (
            mass_default.ndim == 1
            and mass_default.shape[0] == filtered_complex.num_parameters
        )

    if not filtered_complex.is_squeezed:
        if verbose:
            print("Coarsening complex...", end="")
        filtered_complex_ = filtered_complex.grid_squeeze(grid)
        if verbose:
            print("Done.")
    else:
        filtered_complex_ = filtered_complex.copy()

    # assert filtered_complex_.is_squeezed
    if None not in degrees:
        if is_slicer(filtered_complex_) and filtered_complex_.is_minpres:
            pass
        else:
            max_degree = np.max(degrees) + 1
            if verbose:
                print(f"Pruning simplicies up to {max_degree}...", end="")
            if filtered_complex_.dimension > max_degree:
                filtered_complex_.prune_above_dimension(max_degree)
            if verbose:
                print("Done.")

    num_parameters = filtered_complex.num_parameters
    assert num_parameters == len(
        grid
    ), f"Number of parameter do not coincide. Got (grid) {len(grid)} and (filtered complex) {num_parameters}."

    if is_simplextree_multi(filtered_complex_):
        if num_collapses != 0:
            if verbose:
                print("Collapsing edges...", end="")
            filtered_complex_.collapse_edges(num_collapses)
            if verbose:
                print("Done.")
        if backend is not None:
            filtered_complex_ = mp.Slicer(filtered_complex_, vineyard=vineyard)

    fix_mass_default = mass_default is not None
    if is_slicer(filtered_complex_):
        if verbose:
            print("Input is a slicer.")
        if backend is not None and not filtered_complex_.is_minpres:
            from multipers.slicer import minimal_presentation

            assert (
                invariant != "euler"
            ), "Euler Characteristic cannot be speed up by a backend"
            # This returns a list of reduced complexes
            if verbose:
                print("Reducing complex...", end="")
            reduced_complex = minimal_presentation(
                filtered_complex_,
                degrees=degrees,
                backend=backend,
                vineyard=vineyard,
                verbose=verbose,
            )
            if verbose:
                print("Done.")
            if invariant is not None and "rank" in invariant:
                if verbose:
                    print("Computing rank...", end="")
                sms = [
                    _rank_from_slicer(
                        s,
                        degrees=[d],
                        n_jobs=n_jobs,
                        # grid_shape=tuple(len(g) for g in grid),
                        zero_pad=fix_mass_default,
                    )[0]
                    for s, d in zip(reduced_complex, degrees)
                ]
                fix_mass_default = False
                if verbose:
                    print("Done.")
            else:
                if verbose:
                    print("Reduced slicer. Retrieving measure from it...", end="")
                sms = [
                    _signed_measure_from_slicer(
                        s,
                        shift=(
                            reduced_complex.minpres_degree % 2 if d is None else d % 2
                        ),
                    )[0]
                    for s, d in zip(reduced_complex, degrees)
                ]
                if verbose:
                    print("Done.")
        else:  # No backend
            if invariant is not None and "rank" in invariant:
                degrees = np.asarray(degrees, dtype=int)
                if verbose:
                    print("Computing rank...", end="")
                sms = _rank_from_slicer(
                    filtered_complex_,
                    degrees=degrees,
                    n_jobs=n_jobs,
                    zero_pad=fix_mass_default,
                    # grid_shape=tuple(len(g) for g in grid),
                )
                fix_mass_default = False
                if verbose:
                    print("Done.")
            elif filtered_complex_.is_minpres:
                if verbose:
                    print("Reduced slicer. Retrieving measure from it...", end="")
                sms = [
                    _signed_measure_from_slicer(
                        filtered_complex_,
                        shift=(
                            filtered_complex_.minpres_degree % 2 if d is None else d % 2
                        ),
                    )[0]
                    for d in degrees
                ]
                if verbose:
                    print("Done.")
            elif (invariant is None or "euler" in invariant) and (
                len(degrees) == 1 and degrees[0] is None
            ):
                if verbose:
                    print("Retrieving measure from slicer...", end="")
                sms = _signed_measure_from_slicer(
                    filtered_complex_,
                    shift=0,  # no minpres
                )
                if verbose:
                    print("Done.")
            else:
                if verbose:
                    print("Computing Hilbert function...", end="")
                sms = _hilbert_signed_measure(
                    filtered_complex_,
                    degrees=degrees,
                    zero_pad=fix_mass_default,
                    n_jobs=n_jobs,
                    verbose=verbose,
                )
                fix_mass_default = False
                if verbose:
                    print("Done.")

    elif is_simplextree_multi(filtered_complex_):
        if verbose:
            print("Input is a simplextree.")
        ## we still have a simplextree here
        if invariant in ["rank_invariant", "rank"]:
            if verbose:
                print("Computing rank invariant...", end="")
            assert (
                num_parameters == 2
            ), "Rank invariant only implemented for 2-parameter modules."
            assert not coordinate_measure, "Not implemented"
            from multipers.simplex_tree_multi import _rank_signed_measure as smri

            sms = smri(
                filtered_complex_,
                mass_default=mass_default,
                degrees=degrees,
                expand_collapse=expand_collapse,
            )
            fix_mass_default = False
            if verbose:
                print("Done.")
        elif len(degrees) == 1 and degrees[0] is None:
            if verbose:
                print("Computing Euler Characteristic...", end="")
            assert invariant is None or invariant in [
                "euler",
                "euler_characteristic",
            ], "Provide a degree to compute hilbert function."
            # assert not coordinate_measure, "Not implemented"
            from multipers.simplex_tree_multi import _euler_signed_measure

            sms = [
                _euler_signed_measure(
                    filtered_complex_,
                    mass_default=mass_default,
                    verbose=verbose,
                )
            ]
            fix_mass_default = False
            if verbose:
                print("Done.")
        else:
            if verbose:
                print("Computing Hilbert Function...", end="")
            assert invariant is None or invariant in [
                "hilbert",
                "hilbert_function",
            ], "Found homological degrees for euler computation."
            from multipers.simplex_tree_multi import (
                _hilbert_signed_measure as hilbert_signed_measure,
            )

            sms = hilbert_signed_measure(
                filtered_complex_,
                degrees=degrees,
                mass_default=mass_default,
                verbose=verbose,
                n_jobs=n_jobs,
                expand_collapse=expand_collapse,
            )
            fix_mass_default = False
            if verbose:
                print("Done.")
    else:
        raise ValueError("Filtered complex has to be a SimplexTree or a Slicer.")

    if clean:
        if verbose:
            print("Cleaning measure...", end="")
        sms = clean_sms(sms)
        if verbose:
            print("Done.")
    if grid is not None and not coordinate_measure:
        if verbose:
            print("Pushing back the measure to the grid...", end="")
        sms = sms_in_grid(
            sms,
            grid=grid,
            mass_default=mass_default,
            num_parameters=num_parameters,
        )
        if verbose:
            print("Done.")

    if fix_mass_default:
        # TODO : some methods need to use this, this could be optimized
        if verbose:
            print("Seems that fixing mass default is necessary...", end="")
        sms = zero_out_sms(sms, mass_default=mass_default)
        if verbose:
            print("Done.")
    if plot:
        plot_signed_measures(sms)
    return sms


def _signed_measure_from_scc(
    minimal_presentation,
) -> list[tuple[np.ndarray, np.ndarray]]:
    pts = np.concatenate([b[0] for b in minimal_presentation])
    weights = np.concatenate(
        [
            (1 - 2 * (i % 2)) * np.ones(len(b[0]))
            for i, b in enumerate(minimal_presentation)
        ]
    )
    sm = [(pts, weights)]
    return sm


def _signed_measure_from_slicer(
    slicer: Slicer_type,
    shift: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    assert not slicer.is_kcritical, "Not implemented for k-critical filtrations yet."
    pts = np.array(slicer.get_filtrations())
    dims = slicer.get_dimensions()
    if shift:
        dims += shift
    weights = 1 - 2 * (dims % 2)
    sm = [(pts, weights)]
    return sm
