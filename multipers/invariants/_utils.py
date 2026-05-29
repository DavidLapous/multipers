from __future__ import annotations

import numpy as np


def _as_slicer(filtered_complex):
    from multipers._slicer_meta import Slicer
    from multipers.slicer import is_slicer
    from multipers.simplex_tree_multi import is_simplextree_multi

    if is_slicer(filtered_complex):
        return filtered_complex
    if is_simplextree_multi(filtered_complex):
        return Slicer(filtered_complex, dtype=filtered_complex.dtype)
    raise TypeError(
        "Expected a Slicer or SimplexTreeMulti. "
        f"Got {type(filtered_complex)!r}."
    )
def _normalize_degrees(degree=None, degrees=(), inferred_degree=None):
    degrees_array = np.asarray(() if degrees is None else degrees, dtype=int).reshape(-1)
    if inferred_degree is not None:
        inferred_degree = int(inferred_degree)
        if degree is None and degrees_array.size == 0:
            degree = inferred_degree
    single_output = degree is not None
    if single_output:
        if degrees_array.size != 0:
            raise ValueError("Provide either `degree` or `degrees`, not both.")
        degrees_array = np.asarray(degree, dtype=int).reshape(-1)
        if degrees_array.size != 1:
            raise ValueError("`degree` must be scalar. Use `degrees` for several outputs.")
    if degrees_array.size == 0:
        raise ValueError("Provide `degree` or a non-empty `degrees`.")
    if np.unique(degrees_array).size != degrees_array.size:
        raise ValueError("`degrees` must contain unique values.")
    if inferred_degree is not None and not np.array_equal(
        degrees_array,
        np.asarray([inferred_degree], dtype=int),
    ):
        raise ValueError("Cannot change degree of an already minimal-presentation slicer.")
    return tuple(int(d) for d in degrees_array), single_output
