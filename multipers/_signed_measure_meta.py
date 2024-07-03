from typing import Iterable, Optional, Union

import numpy as np

from multipers.grids import compute_grid, sms_in_grid
from multipers.plots import plot_signed_measures
from multipers.point_measure_integration import clean_sms, zero_out_sms
from multipers.rank_invariant import rank_from_slicer
from multipers.simplex_tree_multi import (SimplexTreeMulti_type,
                                          _available_strategies,
                                          is_simplextree_multi)
from multipers.slicer import Slicer_type, is_slicer


def signed_measure(
    filtered_complex: Union[SimplexTreeMulti_type, Slicer_type],
    degree: Optional[int] = None,
    degrees: Iterable[int | None] = [None],
    mass_default=None,
    grid_strategy: _available_strategies = "exact",
    invariant: Optional[str] = None,
    plot: bool = False,
    verbose: bool = False,
    n_jobs: int = -1,
    expand_collapse: bool = False,
    backend: Optional[str] = None,
    thread_id: str = "",
    grid_conversion: Optional[list] = None,
    coordinate_measure: bool = False,
    num_collapses: int = 0,
    clean: Optional[bool] = None,
    vineyard:bool=False,
    **infer_grid_kwargs,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Computes the signed measures given by the decomposition of the hilbert
    function or the euler characteristic, or the rank invariant.

    Input
    -----
     - filtered_complex: given by a simplextree or a slicer.
     - degree:int|None / degrees:list[int] the degrees to compute.
       None represents the euler characteristic.
     - mass_default: Either None, or 'auto' or 'inf', or array-like of floats.
       Where to put the default mass to get a zero-mass measure.
     - grid_strategy: If not squeezed yet, the strategy to coarsen the grid; see ``strategy`` in :func:`multipers.grids.compute_grid`.
     - invariant: The invariant to use, either "hilbert", "rank", or "euler".
     - plot:bool, plots the computed measures if true.
     - n_jobs:int, number of jobs.
       Defaults to #cpu, but when doing parallel computations of signed measures, we recommend setting this to 1.
     - verbose:bool, prints c++ logs.
     - expand_collapse: when the input is a simplextree, only expands the complex when computing 1-dimensional slices. Meant to reduce memory footprint at some computational expense.
     - backend:str when the input is a simplextree, reduce first the filtered complex using an external library
     see ``backend`` in :func:`multipers.io.reduce_complex`.
     - grid_conversion: If given, re-evaluates the final signed measure in this grid.
     - coordinate_measure: bool, if True, compute the signed measure as a coordinates given in grid_conversion.
     - num_collapses: int, if `filtered_complex` is a simplextree, does some collapses if possible.
     - clean: reduces the output signed measure. Only useful for euler computations.

    Output
    ------

    `[signed_measure_of_degree for degree in degrees]`
    with `signed_measure_of_degree` of the form `(dirac location, dirac weights)`.
    """

    if len(degrees) == 1 and degrees[0] is None and degree is not None:
        degrees = [degree]
    if None in degrees:
        assert len(degrees) == 1
        invariant = "euler"
    if len(degrees) == 0:
        return []
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

    if grid_conversion is None and not filtered_complex.is_squeezed:
        grid_conversion = compute_grid(
            filtered_complex, strategy=grid_strategy, **infer_grid_kwargs
        )
    if filtered_complex.is_squeezed and grid_conversion is None:
        grid_conversion = tuple(np.asarray(f) for f in filtered_complex.filtration_grid)

    if mass_default is None:
        mass_default = mass_default
    elif mass_default == "inf":
        mass_default = np.array([np.inf] * filtered_complex.num_parameters)
    elif mass_default == "auto": # this will not work with torch, but we don't want to autodiff this anyway
        mass_default = np.array(
            [1.1 * np.max(f) - 0.1 * np.min(f) for f in grid_conversion]
        )
    else:
        mass_default = np.asarray(mass_default)
        assert (
            mass_default.ndim == 1
            and mass_default.shape[0] == filtered_complex.num_parameters
        )
    
    INPUT_ARGS = locals()
    if not filtered_complex.is_squeezed:
        filtered_complex_ = filtered_complex.grid_squeeze(
            grid_conversion, coordinates=True
        )
    else:
        filtered_complex_ = filtered_complex

    num_parameters = filtered_complex.num_parameters
    assert num_parameters == len(
        grid_conversion
    ), f"Number of parameter do not coincide. Got (grid_conversion) {len(grid_conversion)} and (filtered complex) {num_parameters}."

    fix_mass_default = mass_default is not None
    if is_slicer(filtered_complex_):
        if backend is not None:
            from multipers.slicer import minimal_presentation
            assert (
                invariant != "euler"
            ), "Euler Characteristic cannot be speed up by a backend"
            # This returns a list of reduced complexes
            reduced_complex = minimal_presentation(filtered_complex_, degrees=degrees, backend=backend, vineyard = vineyard)
            if invariant in ("rank", "rank_invariant"):
                sms = [
                    rank_from_slicer(
                        s,
                        degrees=[1],
                        n_jobs=n_jobs,
                        grid_shape=tuple(len(g) for g in grid_conversion),
                        plot=plot,
                    )
                    for s, d in zip(reduced_complex, degrees)
                ]
            else:
                sms = [
                    _signed_measure_from_slicer(s)[0]
                    for s in reduced_complex
                ]
        else: # No backend
            if invariant in ("rank", "rank_invariant"):  # TODO Hilbert from slicer
                degrees = np.asarray(degrees, dtype=int)
                return rank_from_slicer(
                    filtered_complex_,
                    degrees=degrees,
                    n_jobs=n_jobs,
                    grid_shape=tuple(len(g) for g in grid_conversion),
                    plot=plot,
                )
            elif invariant is None or "euler" in invariant:
                sms = _signed_measure_from_slicer(
                    filtered_complex_,
                )
            else:
                if filtered_complex_.is_minpres:
                    assert len(degrees) == 1
                    sms = _signed_measure_from_slicer(filtered_complex_)

                else:
                    from multipers.slicer import minimal_presentation
                    reduced_complex = minimal_presentation(filtered_complex_, degrees=degrees, backend=backend, vineyard=vineyard)
                    sms = [_signed_measure_from_slicer(s)[0]
                        for s in reduced_complex
                    ]
        

    elif is_simplextree_multi(filtered_complex_):
        if num_collapses != 0:
            filtered_complex_.collapse_edges(num_collapses)
        if backend is not None:
            from multipers.slicer import minimal_presentation
            reduced_complex = minimal_presentation(filtered_complex_, degrees=degrees, backend=backend, vineyard=vineyard)
            # this is a list of slicer -> need to go back 
            INPUT_ARGS.pop('filtered_complex')
            INPUT_ARGS.pop('degrees')
            INPUT_ARGS.pop('degree')
            INPUT_ARGS.pop('backend')
            plot = INPUT_ARGS.pop('plot') ## plot has to be dealt with later
            output = [_signed_measure_from_slicer(s, degree=d, **INPUT_ARGS)[0] for s,d in zip(reduced_complex, degrees)]
            if plot:
                plot_signed_measures(output)
            return output
        ## we still have a simplextree here
        if invariant in ["rank_invariant", "rank"]:
            assert (
                filtered_complex.num_parameters == 2
            ), "Rank invariant only implemented for 2-parameter modules."
            assert not coordinate_measure, "Not implemented"
            from multipers.simplex_tree_multi import \
                _rank_signed_measure as smri

            sms = smri(
                filtered_complex_,
                mass_default=mass_default,
                degrees=degrees,
                plot=plot,
                expand_collapse=expand_collapse,
            )
            fix_mass_default = False
        elif len(degrees) == 1 and degrees[0] is None:
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
        else:
            assert invariant is None or invariant in [
                "hilbert",
                "hilbert_function",
            ], "Found homological degrees for euler computation."
            from multipers.simplex_tree_multi import \
                _hilbert_signed_measure as hilbert_signed_measure

            sms = hilbert_signed_measure(
                filtered_complex_,
                degrees=degrees,
                mass_default=mass_default,
                verbose=verbose,
                n_jobs=n_jobs,
                expand_collapse=expand_collapse,
            )
            fix_mass_default = False
    else:
        raise ValueError("Filtered complex has to be a SimplexTree or a Slicer.")

    if clean:
        sms = clean_sms(sms)
    if grid_conversion is not None and not coordinate_measure:
        sms = sms_in_grid(sms, grid_conversion=grid_conversion, mass_default=mass_default, num_parameters=num_parameters)

    if fix_mass_default:
        # TODO : some methods need to use this, this could be optimized
        sms = zero_out_sms(sms, mass_default=mass_default)
    if plot:
        plot_signed_measures(sms)
    return sms


def _signed_measure_from_scc(
    minimal_presentation, grid_conversion=None
) -> list[tuple[np.ndarray, np.ndarray]]:
    pts = np.concatenate([b[0] for b in minimal_presentation if len(b[0]) > 0])
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
) -> list[tuple[np.ndarray, np.ndarray]]:
    assert not slicer.is_kcritical, "Not implemented for k-critical filtrations yet."
    pts = np.array(slicer.get_filtrations())
    dims = slicer.get_dimensions()
    weights = 1 - 2 * (
        (1 + dims) % 2
    )  # dim 0 is always empty : TODO : make that more clean
    sm = [(pts, weights)]
    return sm
