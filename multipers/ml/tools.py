from types import FunctionType
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

import multipers as mp
from multipers.simplex_tree_multi import SimplexTreeMulti


def get_simplex_tree_from_delayed(x) -> mp.SimplexTreeMulti:
    f, args, kwargs = x
    return f(*args, **kwargs)


def get_simplextree(x) -> mp.SimplexTreeMulti:
    if isinstance(x, mp.SimplexTreeMulti):
        return x
    if len(x) == 3 and isinstance(x[0], FunctionType):
        return get_simplex_tree_from_delayed(x)
    else:
        raise TypeError("Not a valid SimplexTree !")


def filtration_grid_to_coordinates(F, return_resolution):
    # computes the mesh as a coordinate list
    mesh = np.meshgrid(*F)
    coordinates = np.concatenate([stuff.flatten()[:, None] for stuff in mesh], axis=1)
    if return_resolution:
        return coordinates, tuple(len(f) for f in F)
    return coordinates


def get_filtration_weights_grid(
    num_parameters: int = 2,
    resolution: int | Iterable[int] = 3,
    *,
    min: float = 0,
    max: float = 20,
    dtype=float,
    remove_homothetie: bool = True,
    weights=None,
):
    """
    Provides a grid of weights, for filtration rescaling.
     - num parameter : the dimension of the grid tensor
     - resolution :  the size of each coordinate
     - min : minimum weight
     - max : maximum weight
     - weights : custom weights (instead of linspace between min and max)
     - dtype : the type of the grid values (useful for int weights)
    """
    from itertools import product

    # if isinstance(resolution, int):
    try:
        float(resolution)
        resolution = [resolution] * num_parameters
    except:
        pass
    if weights is None:
        weights = [
            np.linspace(start=min, stop=max, num=r, dtype=dtype) for r in resolution
        ]
    try:
        float(weights[0])  # same weights for each filtrations
        weights = [weights] * num_parameters
    except:
        None
    out = np.asarray(list(product(*weights)))
    if remove_homothetie:
        _, indices = np.unique(
            [x / x.max() for x in out if x.max() != 0], axis=0, return_index=True
        )
        out = out[indices]
    return list(out)


class SimplexTreeEdgeCollapser(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_collapses: int = 0,
        full: bool = False,
        max_dimension: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__()
        self.full = full
        self.num_collapses = num_collapses
        self.max_dimension = max_dimension
        self.n_jobs = n_jobs
        return

    def fit(self, X: np.ndarray | list, y=None):
        return self

    def transform(self, X):
        edges_list = Parallel(n_jobs=-1, prefer="threads")(
            delayed(mp.SimplextreeMulti.get_edge_list)(x) for x in X
        )
        collapsed_edge_lists = Parallel(n_jobs=self.n_jobs)(
            delayed(mp._collapse_edge_list)(
                edges, full=self.full, num=self.num_collapses
            )
            for edges in edges_list
        )
        collapsed_simplextrees = Parallel(n_jobs=-1, prefer="threads")(
            delayed(mp.SimplexTreeMulti._reconstruct_from_edge_list)(
                collapsed_edge_lists, swap=True, expand_dim=self.max_dimension
            )
        )
        return collapsed_simplextrees
