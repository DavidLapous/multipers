from typing import Literal

import numpy as np
import torch
from pykeops.torch import LazyTensor


def get_grid(strategy: Literal["exact", "regular_closest", "regular_left", "quantile"]):
    """
    Given a strategy, returns a function of signature
    `(num_pts, num_parameter), int --> Iterable[1d array]`
    that generates a torch-differentiable grid from a set of points,
    and a resolution.
    """
    match strategy:
        case "exact":
            return _exact_grid
        case "regular":
            return _regular_grid
        case "regular_closest":
            return _regular_closest_grid
        case "regular_left":
            return _regular_left_grid
        case "quantile":
            return _quantile_grid
        case _:
            raise ValueError(
                f"""
Unimplemented strategy {strategy}.
Available ones : exact, regular_closest, regular_left, quantile.
"""
            )


def todense(grid: list[torch.Tensor]):
    return torch.cartesian_prod(*grid)


def _exact_grid(filtration_values, r=None):
    assert r is None
    grid = tuple(_unique_any(f) for f in filtration_values)
    return grid


def _regular_closest_grid(filtration_values, res):
    grid = tuple(_regular_closest(f, r) for f,r in zip(filtration_values, res))
    return grid

def _regular_grid(filtration_values, res):
    grid = tuple(_regular(g,r) for g,r in zip(filtration_values, res))
    return grid

def _regular(x, r:int):
    if x.ndim != 1:
        raise ValueError(f"Got ndim!=1. {x=}")
    return torch.linspace(start=torch.min(x), end=torch.max(x), steps=r, dtype=x.dtype) 

def _regular_left_grid(filtration_values, res):
    grid = tuple(_regular_left(f, r) for f,r in zip(filtration_values,res))
    return grid


def _quantile_grid(filtration_values, res):
    grid = tuple(_quantile(f, r) for f,r in zip(filtration_values,res))
    return grid
def _quantile(x, r):
    if x.ndim != 1:
        raise ValueError(f"Got ndim!=1. {x=}")
    qs = torch.linspace(0, 1, r, dtype=x.dtype)
    return _unique_any(torch.quantile(x, q=qs))




def _unique_any(x, assume_sorted=False, remove_inf: bool = True):
    if x.ndim != 1:
        raise ValueError(f"Got ndim!=1. {x=}")
    if not assume_sorted:
        x, _ = x.sort()
    if remove_inf and x[-1] == torch.inf:
        x = x[:-1]
    with torch.no_grad():
        y = x.unique()
        idx = torch.searchsorted(x, y)
    x = torch.cat([x, torch.tensor([torch.inf])])
    return x[idx]


def _regular_left(f, r: int, unique: bool = True):
    if f.ndim != 1:
        raise ValueError(f"Got ndim!=1. {f=}")
    f = _unique_any(f)
    with torch.no_grad():
        f_regular = torch.linspace(f[0].item(), f[-1].item(), r, device=f.device)
        idx = torch.searchsorted(f, f_regular)
    f = torch.cat([f, torch.tensor([torch.inf])])
    if unique:
        return _unique_any(f[idx])
    return f[idx]


def _regular_closest(f, r: int, unique: bool = True):
    if f.ndim != 1:
        raise ValueError(f"Got ndim!=1. {f=}")
    f = _unique_any(f)
    with torch.no_grad():
        f_reg = torch.linspace(
            f[0].item(), f[-1].item(), steps=r, dtype=f.dtype, device=f.device
        )
        _f = LazyTensor(f[:, None, None])
        _f_reg = LazyTensor(f_reg[None, :, None])
        indices = (_f - _f_reg).abs().argmin(0).ravel()
    f = torch.cat([f, torch.tensor([torch.inf])])
    f_regular_closest = f[indices]
    if unique:
        f_regular_closest = _unique_any(f_regular_closest)
    return f_regular_closest


def evaluate_in_grid(pts, grid):
    """Evaluates points (assumed to be coordinates) in this grid.
    Input
    -----
     - pts: (num_points, num_parameters) array
     - grid: Iterable of 1-d array, for each parameter

    Returns
    -------
     - array of shape like points of dtype like grid.
    """
    # grid = [torch.cat([g, torch.tensor([torch.inf])]) for g in grid]
    # new_pts = torch.empty(pts.shape, dtype=grid[0].dtype, device=grid[0].device)
    # for parameter, pt_of_parameter in enumerate(pts.T):
    #     new_pts[:, parameter] = grid[parameter][pt_of_parameter]
    return torch.cat(
        [
            grid[parameter][pt_of_parameter][:, None]
            for parameter, pt_of_parameter in enumerate(pts.T)
        ],
        dim=1,
    )


def evaluate_mod_in_grid(mod, grid, box=None):
    """Given an MMA module, pushes it into the specified grid.
    Useful for e.g., make it differentiable.

    Input
    -----
     - mod: PyModule
     - grid: Iterable of 1d array, for num_parameters
    Ouput
    -----
    torch-compatible module in the format:
    (num_degrees) x (num_interval of degree) x ((num_birth, num_parameter), (num_death, num_parameters))

    """
    if box is not None:
        grid = tuple(
            torch.cat(
                [
                    box[0][[i]],
                    _unique_any(
                        grid[i].clamp(min=box[0][i], max=box[1][i]), assume_sorted=True
                    ),
                    box[1][[i]],
                ]
            )
            for i in range(len(grid))
        )
    (birth_sizes, death_sizes), births, deaths = mod.to_flat_idx(grid)
    births = evaluate_in_grid(births, grid)
    deaths = evaluate_in_grid(deaths, grid)
    diff_mod = tuple(
        zip(
            births.split_with_sizes(birth_sizes.tolist()),
            deaths.split_with_sizes(death_sizes.tolist()),
        )
    )
    return diff_mod


def evaluate_mod_in_grid__old(mod, grid, box=None):
    """Given an MMA module, pushes it into the specified grid.
    Useful for e.g., make it differentiable.

    Input
    -----
     - mod: PyModule
     - grid: Iterable of 1d array, for num_parameters
    Ouput
    -----
    torch-compatible module in the format:
    (num_degrees) x (num_interval of degree) x ((num_birth, num_parameter), (num_death, num_parameters))

    """
    from pykeops.numpy import LazyTensor

    with torch.no_grad():
        if box is None:
            # box = mod.get_box()
            box = np.asarray([[g[0] for g in grid], [g[-1] for g in grid]])
        S = mod.dump()[1]

        def get_idx_parameter(A, G, p):
            g = G[p].numpy() if isinstance(G[p], torch.Tensor) else np.asarray(G[p])
            la = LazyTensor(np.asarray(A, dtype=g.dtype)[None, :, [p]])
            lg = LazyTensor(g[:, None, None])
            return (la - lg).abs().argmin(0)

        Bdump = np.concatenate([s[0] for s in S], axis=0).clip(box[[0]], box[[1]])
        B = np.concatenate(
            [get_idx_parameter(Bdump, grid, p) for p in range(mod.num_parameters)],
            axis=1,
            dtype=np.int64,
        )
        Ddump = np.concatenate([s[1] for s in S], axis=0, dtype=np.float32).clip(
            box[[0]], box[[1]]
        )
        D = np.concatenate(
            [get_idx_parameter(Ddump, grid, p) for p in range(mod.num_parameters)],
            axis=1,
            dtype=np.int64,
        )

    BB = evaluate_in_grid(B, grid)
    DD = evaluate_in_grid(D, grid)

    b_idx = tuple((len(s[0]) for s in S))
    d_idx = tuple((len(s[1]) for s in S))
    BBB = BB.split_with_sizes(b_idx)
    DDD = DD.split_with_sizes(d_idx)

    splits = np.concatenate([[0], mod.degree_splits(), [len(BBB)]])
    splits = torch.from_numpy(splits)
    out = [
        list(zip(BBB[splits[i] : splits[i + 1]], DDD[splits[i] : splits[i + 1]]))
        for i in range(len(splits) - 1)
    ]  ## For some reasons this kills the gradient ???? pytorch bug
    return out
