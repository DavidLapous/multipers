from __future__ import annotations

from typing import Optional

import numpy as np

from ._utils import _as_slicer


def projected_barcode(
    filtered_complex,
    direction,
    *,
    degree: Optional[int] = None,
    minpres_kwargs: Optional[dict] = None,
    ignore_infinite_filtration_values: bool = True,
):
    """Gamma-linear projected barcode of one homology module.

    The input may be a filtered complex or an already computed full resolution.
    For a filtered complex, this first computes a full free resolution of
    ``H_degree`` and then projects that resolution. This is not ordinary
    persistence of the original chain complex after scalarizing its cell grades.

    One or several directions may be passed. A single direction returns the
    barcode of the projected resolution; a 2D array of directions returns one
    such barcode per row.

    References:
    - Fernandes, Oudot, Petit, "Computation of gamma-linear projected barcodes
      for multiparameter persistence", DOI: 10.1007/s41468-025-00209-9.
    - "Projected distances for multi-parameter persistence modules", DOI:
      10.5802/aif.3752.
    """
    slicer = _as_slicer(filtered_complex)
    if slicer.is_minpres:
        if degree is None:
            degree = slicer.minpres_degree
        elif slicer.minpres_degree != int(degree):
            raise ValueError(
                "Cannot change degree of an already minimal-presentation slicer."
            )
        resolution = slicer
    else:
        if degree is None or degree < 0:
            raise ValueError("`degree` is inferred for minpres inputs, otherwise required.")
        minpres_kwargs = {} if minpres_kwargs is None else dict(minpres_kwargs)
        if not minpres_kwargs.get("full_resolution", True):
            raise ValueError("projected_barcode requires `full_resolution=True`.")
        minpres_kwargs["full_resolution"] = True
        resolution = slicer.minpres(degree=int(degree), **minpres_kwargs)

    directions = np.asarray(direction, dtype=np.float64)
    grades = np.asarray(
        resolution.get_filtrations(unsqueeze=resolution.is_squeezed),
        dtype=np.float64,
    )
    num_parameters = resolution.num_parameters
    if directions.ndim == 1:
        if directions.shape[0] != num_parameters:
            raise ValueError(
                f"Expected direction length {num_parameters}. Got {directions.shape[0]}."
            )
        projected_grades = grades @ directions
    elif directions.ndim == 2:
        if directions.shape[1] != num_parameters:
            raise ValueError(
                f"Expected directions with {num_parameters} columns. "
                f"Got {directions.shape[1]}."
            )
        projected_grades = directions @ grades.T
    else:
        raise ValueError(
            "Expected `direction` with shape (num_parameters,) or "
            "(num_directions, num_parameters)."
        )

    return resolution.compute_persistence(
        projected_grades,
        ignore_infinite_filtration_values=ignore_infinite_filtration_values,
    )


__all__ = ["projected_barcode"]
