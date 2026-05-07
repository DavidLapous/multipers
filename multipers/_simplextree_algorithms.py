from __future__ import annotations

import numpy as np

from multipers import _simplex_tree_multi_nanobind as _nb


def _euler_signed_measure(simplextree, mass_default=None, verbose=False):
    if not simplextree.is_squeezed:
        raise ValueError("Squeeze grid first.")
    zero_pad = mass_default is not None
    grid_shape = np.array([len(f) for f in simplextree.filtration_grid], dtype=np.int32)
    if mass_default is not None:
        mass_default = np.asarray(mass_default)
        assert (
            mass_default.ndim == 1
            and mass_default.shape[0] == simplextree.num_parameters
        )
    if zero_pad:
        grid_shape = grid_shape + 1
    pts, weights = _nb._compute_euler_signed_measure(
        simplextree, grid_shape.tolist(), zero_pad, verbose
    )
    return np.asarray(pts, dtype=np.int32).reshape(
        -1, simplextree.num_parameters
    ), np.asarray(weights, dtype=np.int32)
