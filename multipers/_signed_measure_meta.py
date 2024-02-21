from typing import Optional, Union

from collections import defaultdict
import numpy as np

from multipers.grids import sms_in_grid
from multipers.plots import plot_signed_measures
from multipers.simplex_tree_multi import (
    SimplexTreeMulti,  # Typing hack
    _available_strategies,
)
from multipers.slicer import (
    Slicer,
    SlicerClement,
    SlicerVineGraph,
    SlicerVineSimplicial,
)


def signed_measure(
    simplextree: Union[
        SimplexTreeMulti, Slicer, SlicerClement, SlicerVineGraph, SlicerVineSimplicial
    ],
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
):
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
    if not isinstance(simplextree, SimplexTreeMulti):
        return _signed_measure_from_slicer(
            simplextree, plot=plot, grid_conversion=grid_conversion, clean=clean
        )
    assert invariant is None or invariant in [
        "hilbert",
        "rank_invariant",
        "euler",
        "rank",
        "euler_characteristic",
        "hilbert_function",
    ]
    assert not plot or simplextree.num_parameters == 2, "Can only plot 2d measures."
    if len(degrees) == 1 and degrees[0] is None and degree is not None:
        degrees = [degree]
    if None in degrees:
        assert len(degrees) == 1

    if len(degrees) == 0:
        return []

    if not simplextree._is_squeezed:
        simplextree_ = SimplexTreeMulti(simplextree)
        if grid_conversion is None:
            grid_conversion = simplextree_.get_filtration_grid(
                grid_strategy=grid_strategy,
                **infer_grid_kwargs,
            )  # put a warning ?
        simplextree_.grid_squeeze(
            grid_conversion,
            coordinate_values=True,
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
        from multipers.rank_invariant import signed_measure as smri

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
        from multipers.euler_characteristic import euler_signed_measure

        sms = [
            euler_signed_measure(
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
        from multipers.hilbert_function import hilbert_signed_measure

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
        sms = clean_signed_measure(sms)
    return sms


def _signed_measure_from_scc(minimal_presentation, grid_conversion=None):
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
    slicer: Union[Slicer, SlicerClement, SlicerVineGraph, SlicerVineSimplicial],
    plot: bool = False,
    grid_conversion=None,
    clean: bool = False,
):
    pts = slicer.get_filtrations()
    dims = slicer.get_dimensions()
    weights = 1 - 2 * (
        (1 + dims) % 2
    )  # dim 0 is always empty : TODO : make that more clean
    sm = [(pts, weights)]
    if grid_conversion is not None:
        sm = sms_in_grid(sm, grid_conversion)
    if plot:
        plot_signed_measures(sm)
    if clean:
        sm = clean_signed_measure(sm)
    return sm


def clean_signed_measure(sms):
    """
    Sum the diracs at the same locations. i.e.,
    returns the minimal sized measure to represent the input.
    Mostly useful for, e.g., euler_characteristic from simplical complexes.
    """
    new_sms = []
    for pts, weights in sms:
        out = defaultdict(lambda: 0)
        for pt, w in zip(
            pts, weights
        ):  ## this is slow. but not a bottleneck TODO: optimize
            out[tuple(pt)] += w
        pts = np.fromiter(out.keys(), dtype=np.dtype((np.float32, 2)))
        weights = np.fromiter(out.values(), dtype=int)
        idx = np.nonzero(weights)
        pts = pts[idx]
        weights = weights[idx]
        new_sms.append(tuple((pts, weights)))
    return new_sms
