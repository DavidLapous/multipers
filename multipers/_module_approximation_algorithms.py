from __future__ import annotations

from multipers import _slicer_nanobind as _nb


def _compute_module_approximation_from_slicer(
    slicer,
    direction,
    max_error,
    box,
    threshold,
    complete,
    verbose,
    n_jobs,
):
    return _nb._compute_module_approximation_from_slicer(
        slicer,
        direction,
        max_error,
        box,
        threshold,
        complete,
        verbose,
        n_jobs,
    )
