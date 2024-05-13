from typing import Optional, Union

import numpy as np

from multipers.grids import compute_grid, sms_in_grid
from multipers.plots import plot_signed_measures
from multipers.point_measure_integration import clean_sms
from multipers.rank_invariant import rank_from_slicer
from multipers.simplex_tree_multi import (
    SimplexTreeMulti_type,
    _available_strategies,
    is_simplextree_multi,
)
from multipers.slicer import Slicer_type, is_slicer


def signed_measure(
    simplextree: Union[SimplexTreeMulti_type, Slicer_type],
    degree: Optional[int] = None,
    degrees=[None],
    mass_default=None,
    grid_strategy: _available_strategies = "exact",
    invariant: Optional[str] = None,
    plot: bool = False,
    verbose: bool = False,
    n_jobs: int = -1,
    expand_collapse: bool = False,
    backend: str = "multipers",
    thread_id: str = "",
    input_path: Optional[str] = None,
    grid_conversion: Optional[list] = None,
    coordinate_measure: bool = False,
    num_collapses: int = 0,
    clean: bool = False,
    **infer_grid_kwargs,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Computes the signed measures given by the decomposition of the hilbert
    function or the euler characteristic.

    Input
    -----
     - simplextree:SimplexTreeMulti, the multifiltered simplicial complex.
       Its recommended to squeeze the simplextree first.
     - mass_default: Either None, or 'auto' or 'inf', or array-like of floats.
       Where to put the default mass to get a zero-mass measure.
     - degree:int|None / degrees:list[int] the degrees to compute.
       None represents the euler characteristic.
     - plot:bool, plots the computed measures if true.
     - n_jobs:int, number of jobs.
       Defaults to #cpu, but when doing parallel computations of signed measures, we recommend setting this to 1.
     - verbose:bool, prints c++ logs.

    Output
    ------
    `[signed_measure_of_degree for degree in degrees]`
    with `signed_measure_of_degree` of the form `(dirac location, dirac weights)`.
    """
    if len(degrees) == 1 and degrees[0] is None and degree is not None:
        degrees = [degree]
    if None in degrees:
        assert len(degrees) == 1
    if len(degrees) == 0:
        return []
    if is_slicer(simplextree):
        if grid_conversion is None and not simplextree.is_squeezed:
            grid_conversion = compute_grid(
                simplextree.get_filtrations_values().T,
                strategy=grid_strategy,
                **infer_grid_kwargs,
            )
        if not simplextree.is_squeezed:
            simplextree_ = simplextree.grid_squeeze(grid_conversion, coordinates=True)
        else:
            simplextree_ = simplextree
            if grid_conversion is None:
                grid_conversion = tuple(
                    np.asarray(f, dtype=np.float64) for f in simplextree.filtration_grid
                )
        if invariant == "rank":  # TODO Hilbert from slicer
            degrees = np.asarray(degrees, dtype=int)
            return rank_from_slicer(
                simplextree_,
                degrees=degrees,
                n_jobs=n_jobs,
                grid_shape=tuple(len(g) for g in grid_conversion),
                grid_conversion=grid_conversion,
                plot=plot,
            )
        return _signed_measure_from_slicer(
            simplextree_,
            plot=plot,
            grid_conversion=grid_conversion,
            clean=clean,
        )
    assert is_simplextree_multi(simplextree), "Input has to be simplextree or slicer."
    assert invariant is None or invariant in [
        "hilbert",
        "rank_invariant",
        "euler",
        "rank",
        "euler_characteristic",
        "hilbert_function",
    ]
    assert not plot or simplextree.num_parameters == 2, "Can only plot 2d measures."

    if not simplextree._is_squeezed:
        if grid_conversion is None:
            grid_conversion = simplextree.get_filtration_grid(
                grid_strategy=grid_strategy,
                **infer_grid_kwargs,
            )  # put a warning ?
        simplextree_ = simplextree.grid_squeeze(
            grid_conversion,
            coordinate_values=True,
            inplace=False,
            **infer_grid_kwargs,
        )
        if num_collapses != 0:
            simplextree_.collapse_edges(num_collapses)
    else:
        simplextree_ = simplextree
        if grid_conversion is None:
            grid_conversion = [np.asarray(f) for f in simplextree_.filtration_grid]
    if coordinate_measure:
        grid_conversion = None

    if backend != "multipers":
        if input_path is not None:
            import multipers.io as mio

            mio.input_path = input_path
        assert (
            len(degrees) == 1
            and mass_default is None
            and (invariant is None or "hilbert" in invariant)
        )
        from multipers.io import reduce_complex

        minimal_presentation = reduce_complex(
            simplextree_,
            full_resolution=True,
            dimension=degrees[0],
            id=thread_id,
            backend=backend,
            verbose=verbose,
        )
        sms = _signed_measure_from_scc(
            minimal_presentation, grid_conversion=grid_conversion
        )
        if plot:
            from multipers.plots import plot_signed_measures

            plot_signed_measures(sms)
        return sms
    # assert simplextree.num_parameters == 2
    if mass_default is None:
        mass_default = mass_default
    elif mass_default == "inf":
        mass_default = np.array([np.inf] * simplextree.num_parameters)
    elif mass_default == "auto":
        grid_conversion = [np.asarray(f) for f in simplextree_.filtration_grid]
        mass_default = np.array(
            [1.1 * np.max(f) - 0.1 * np.min(f) for f in grid_conversion]
        )
    else:
        mass_default = np.asarray(mass_default)
        assert (
            mass_default.ndim == 1
            and mass_default.shape[0] == simplextree.num_parameters
        )
    # assert not coordinate_measure or grid_conversion is None

    if invariant in ["rank_invariant", "rank"]:
        assert (
            simplextree.num_parameters == 2
        ), "Rank invariant only implemented for 2-parameter modules."
        assert not coordinate_measure, "Not implemented"
        from multipers.simplex_tree_multi import _rank_signed_measure as smri

        sms = smri(
            simplextree_,
            mass_default=mass_default,
            degrees=degrees,
            plot=plot,
            expand_collapse=expand_collapse,
        )
    elif len(degrees) == 1 and degrees[0] is None:
        assert invariant is None or invariant in [
            "euler",
            "euler_characteristic",
        ], "Provide a degree to compute hilbert function."
        # assert not coordinate_measure, "Not implemented"
        from multipers.simplex_tree_multi import _euler_signed_measure

        sms = [
            _euler_signed_measure(
                simplextree_,
                mass_default=mass_default,
                verbose=verbose,
                plot=plot,
                grid_conversion=grid_conversion,
            )
        ]
    else:
        assert invariant is None or invariant in [
            "hilbert",
            "hilbert_function",
        ], "Found homological degrees for euler computation."
        from multipers.simplex_tree_multi import (
            _hilbert_signed_measure as hilbert_signed_measure,
        )

        sms = hilbert_signed_measure(
            simplextree_,
            degrees=degrees,
            mass_default=mass_default,
            verbose=verbose,
            plot=plot,
            n_jobs=n_jobs,
            expand_collapse=expand_collapse,
            grid_conversion=grid_conversion,
        )

    if clean:
        sms = clean_sms(sms)
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
    if grid_conversion is not None:
        sm = sms_in_grid(sm, grid_conversion)
    return sm


def _signed_measure_from_slicer(
    slicer: Slicer_type,
    plot: bool = False,
    grid_conversion=None,
    clean: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    pts = slicer.get_filtrations()
    dims = slicer.get_dimensions()
    weights = 1 - 2 * (
        (1 + dims) % 2
    )  # dim 0 is always empty : TODO : make that more clean
    sm = [(pts, weights)]
    if slicer.is_squeezed and grid_conversion is None:
        grid_conversion = [
            np.asarray(f, dtype=np.float64) for f in slicer.filtration_grid
        ]
    if grid_conversion is not None:
        sm = sms_in_grid(sm, grid_conversion)
    if clean:
        sm = clean_sms(sm)
    if plot:
        plot_signed_measures(sm)
    return sm
