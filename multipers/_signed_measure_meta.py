from multipers.plots import plot_signed_measures
from multipers.simplex_tree_multi import SimplexTreeMulti  # Typing hack
from typing import Optional, Union
import numpy as np
from multipers.simplex_tree_multi import _available_strategies

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
    mpfree_path: Optional[str] = None,
    grid_conversion: Optional[list] = None,
    coordinate_measure: bool = False,
    num_collapses: int = 0,
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
            simplextree,
            plot=plot,
            grid_conversion=grid_conversion,
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

    if backend == "mpfree":
        if mpfree_path is not None:
            import multipers.io as mio

            mio.mpfree_path = mpfree_path
        assert (
            len(degrees) == 1
            and mass_default is None
            and (invariant is None or "hilbert" in invariant)
        )
        from multipers.io import minimal_presentation_from_mpfree

        minimal_presentation = minimal_presentation_from_mpfree(
            simplextree_,
            True,
            degrees[0],
            id=thread_id,
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

    return sms


def _signed_measure_from_scc(minimal_presentation, grid_conversion=None):
    pts = np.concatenate([b[0] for b in minimal_presentation if len(b[0]) > 0])
    weights = np.concatenate(
        [
            (1 - 2 * (i % 2)) * np.ones(len(b[0]))
            for i, b in enumerate(minimal_presentation)
            if len(b[0]) > 0
        ]
    )
    if grid_conversion is not None:
        pts = np.asarray(pts, dtype=int)
        coords = np.empty(shape=pts.shape, dtype=float)
        for i in range(coords.shape[1]):
            coords[:, i] = np.asarray(grid_conversion[i])[pts[:, i]]
        sm = [(coords, weights)]
    else:
        sm = [(pts, weights)]
    return sm


def _signed_measure_from_slicer(
    slicer: Union[Slicer, SlicerClement, SlicerVineGraph, SlicerVineSimplicial],
    plot: bool = False,
    grid_conversion=None,
):
    pts = slicer.get_filtrations()
    dims = slicer.get_dimensions()
    weights = 1 - 2 * (dims % 2)
    if grid_conversion is not None:
        pts = np.asarray(pts, dtype=int)
        coords = np.empty(shape=pts.shape, dtype=float)
        for i in range(coords.shape[1]):
            coords[:, i] = np.asarray(grid_conversion[i])[pts[:, i]]
        sm = [(coords, weights)]
    else:
        sm = [(pts, weights)]
    if plot:
        plot_signed_measures(sm)
    return sm
