from typing import Iterable, Literal, Optional

import numpy as np

import multipers.logs as _mp_logs
from multipers.array_api import api_from_tensor, api_from_tensors
from multipers.array_api import numpy as npapi

from . import _grid_helper_nanobind as _mg_nb

available_strategies = [
    "exact",
    "regular",
    "regular_closest",
    "regular_left",
    "partition",
    "quantile",
    "precomputed",
]
Lstrategies = Literal[
    "exact",
    "regular",
    "regular_closest",
    "regular_left",
    "partition",
    "quantile",
    "precomputed",
]


def sanitize_grid(grid, numpyfy=False, add_inf=False, api = None):
    num_parameters = len(grid)
    if num_parameters == 0:
        raise ValueError("empty filtration grid")
    api = api_from_tensors(*grid) if api is None else api
    if numpyfy:
        grid = tuple(api.asnumpy(grid[i]) for i in range(num_parameters))
    else:
        # copy here may not be necessary, but cheap
        grid = [
            api.astensor(grid[i], contiguous=True) for i in range(num_parameters)
        ]
    if add_inf:
        api = api_from_tensors(grid[0])
        inf = api.astensor(_inf_value(grid[0]))
        grid = tuple(
            grid[i] if grid[i][-1] == inf else api.cat([grid[i], inf[None]])
            for i in range(num_parameters)
        )
    assert np.all([g.ndim == 1 for g in grid])
    return grid


def threshold_slice(a, m, M):
    if m is not None:
        a = a[a >= m]
    if M is not None:
        a = a[a <= M]
    return a


def _exact_grid(x, api, _mean):
    return [api.unique(api.astensor(f), _mean=_mean) for f in x]


def get_exact_grid(
    x,
    threshold_min=None,
    threshold_max=None,
    return_api=False,
    _mean=False,
    api=None,
):
    """
    Computes an initial exact grid
    """
    from multipers.mma_structures import is_mma
    from multipers.simplex_tree_multi import is_simplextree_multi
    from multipers.slicer import is_slicer

    if (is_slicer(x) or is_simplextree_multi(x)) and x.is_squeezed:
        initial_grid = x.filtration_grid
        api = api_from_tensors(*initial_grid) if api is None else api
        initial_grid = _exact_grid(initial_grid, api, _mean)
    elif is_slicer(x):
        initial_grid = x.get_filtrations_values().T
        api = npapi if api is None else api
        initial_grid = _exact_grid(initial_grid, api, _mean)
    elif is_simplextree_multi(x):
        initial_grid = x._get_filtration_values()[0]
        api = npapi if api is None else api
        initial_grid = _exact_grid(initial_grid, api, _mean)
    elif is_mma(x):
        initial_grid = x.get_filtration_values()
        api = npapi if api is None else api
        initial_grid = _exact_grid(initial_grid, api, _mean)
    elif isinstance(x, np.ndarray):
        api = npapi if api is None else api
        initial_grid = _exact_grid(x, api, _mean)
    else:
        if len(x) == 0:
            return [], npapi
        api = api_from_tensors(*x) if api is None else api
        initial_grid = _exact_grid(x, api, _mean)

    num_parameters = len(initial_grid)

    if threshold_min is not None or threshold_max is not None:
        if threshold_min is None:
            threshold_min = [None] * num_parameters
        if threshold_max is None:
            threshold_max = [None] * num_parameters

        initial_grid = [
            threshold_slice(xx, a, b)
            for xx, a, b in zip(initial_grid, threshold_min, threshold_max)
        ]
    for i in range(num_parameters):
        initial_grid[i] = api.ascontiguous(initial_grid[i])
    if return_api:
        return initial_grid, api
    return initial_grid


def compute_grid(
    x,
    resolution: Optional[int | Iterable[int]] = None,
    strategy: Lstrategies = "exact",
    unique=True,
    _q_factor=1.0,
    drop_quantiles=[0, 0],
    dense=False,
    threshold_min=None,
    threshold_max=None,
    _mean=False,
    force_contiguous=True,
    api=None,
):
    """
    Computes a grid from filtration values, using some strategy.

    Input
    -----

    - `filtrations_values`: `Iterable[filtration of parameter for parameter]`
       where `filtration_of_parameter` is a array[float, ndim=1]
     - `resolution`:Optional[int|tuple[int]]
     - `strategy`: either exact, regular, regular_closest, regular_left, partition, quantile, or precomputed.
     - `unique`: if true, doesn't repeat values in the output grid.
     - `drop_quantiles` : drop some filtration values according to these quantiles
    Output
    ------

    Iterable[array[float, ndim=1]] : the 1d-grid for each parameter.
    """

    # Extract initial_grid and api using the helper function
    initial_grid, api = get_exact_grid(
        x,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        return_api=True,
        api=api,
    )
    if len(initial_grid) == 0:
        return initial_grid
    num_parameters = len(initial_grid)
    try:
        int(resolution)
        resolution = [resolution] * num_parameters
    except TypeError:
        pass

    grid = _compute_grid_numpy(
        initial_grid,
        resolution=resolution,
        strategy=strategy,
        unique=unique,
        _q_factor=_q_factor,
        drop_quantiles=drop_quantiles,
        api=api,
    )
    if force_contiguous:
        grid = tuple(api.astensor(x, contiguous=True) for x in grid)
    if dense:
        grid = todense(grid)
    return grid


def compute_grid_from_iterable(
    xs,
    resolution: Optional[int | Iterable[int]] = None,
    strategy: Lstrategies = "exact",
    unique=True,
    _q_factor=1.0,
    drop_quantiles=[0, 0],
    dense=False,
    threshold_min=None,
    threshold_max=None,
):
    initial_grids = tuple(
        get_exact_grid(
            x,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
        for x in xs
    )
    api = api_from_tensors(*initial_grids[0])
    num_parameters = len(initial_grids[0])
    grid = tuple(api.cat([x[i] for x in initial_grids]) for i in range(num_parameters))
    return compute_grid(
        grid,
        resolution=resolution,
        strategy=strategy,
        unique=unique,
        _q_factor=_q_factor,
        drop_quantiles=drop_quantiles,
        dense=dense,
    )


def todense(grid, api=None):
    if len(grid) == 0:
        return np.empty(0)
    if api is None:
        api = api_from_tensors(*grid)
    return api.cartesian_product(*grid)


def _compute_grid_numpy(
    filtrations_values,
    resolution=None,
    strategy: Lstrategies = "exact",
    unique=True,
    _q_factor=1.0,
    drop_quantiles=[0, 0],
    api=None,
):
    """
    Computes a grid from filtration values, using some strategy.

    Input
    -----
     - `filtrations_values`: `Iterable[filtration of parameter for parameter]`
       where `filtration_of_parameter` is a array[float, ndim=1]
     - `resolution`:Optional[int|tuple[int]]
     - `strategy`: either exact, regular, regular_closest, regular_left, partition, quantile, or precomputed.
     - `unique`: if true, doesn't repeat values in the output grid.
     - `drop_quantiles` : drop some filtration values according to these quantiles
    Output
    ------
    Iterable[array[float, ndim=1]] : the 1d-grid for each parameter.
    """
    num_parameters = len(filtrations_values)
    if api is None:
        api = api_from_tensors(*filtrations_values)
    try:
        a, b = drop_quantiles
    except:
        a, b = drop_quantiles, drop_quantiles

    ## match doesn't work with cython BUG
    if strategy == "exact":
        if resolution is not None:
            raise ValueError(
                f"Exact strategy cannot have a resolution. Got {resolution=}."
            )
        return filtrations_values
    if resolution is None:
        raise ValueError(f"Strategy {strategy} need a resolution. Got {resolution=}.")
    if strategy == "regular_closest":
        return tuple(
            _todo_regular_closest(f, r, unique, api)
            for f, r in zip(filtrations_values, resolution)
        )
    if strategy == "regular":
        return tuple(
            _todo_regular(f, r, api) for f, r in zip(filtrations_values, resolution)
        )
    if strategy == "regular_left":
        return tuple(
            _todo_regular_left(f, r, unique, api)
            for f, r in zip(filtrations_values, resolution)
        )
    if strategy == "partition":
        return tuple(
            _todo_partition(f, r, unique, api)
            for f, r in zip(filtrations_values, resolution)
        )
    if strategy == "precomputed":
        return filtrations_values
    if a != 0 or b != 0:
        boxes = api.astensor(
            [
                api.quantile_closest(filtration, [a, b], axis=1)
                for filtration in filtrations_values
            ]
        )
        min_filtration, max_filtration = (
            api.minvalues(boxes, axis=(0, 1)),
            api.maxvalues(boxes, axis=(0, 1)),
        )  # box, birth/death, filtration
        filtrations_values = [
            filtration[(m < filtration) * (filtration < M)]
            for filtration, m, M in zip(
                filtrations_values, min_filtration, max_filtration
            )
        ]

    if strategy == "quantile":
        F = filtrations_values
        # F = tuple(api.unique(f) for f in filtrations_values)
        max_resolution = [min(len(f), r) for f, r in zip(F, resolution)]
        F = tuple(
            api.quantile_closest(
                f,
                q=api.linspace(0, 1, int(r * _q_factor)),
                axis=0,
            )
            for f, r in zip(F, resolution)
        )

        if unique:
            F = tuple(api.unique(f) for f in F)
            if np.all(np.asarray(max_resolution) > np.asarray([len(f) for f in F])):
                return _compute_grid_numpy(
                    filtrations_values=filtrations_values,
                    resolution=resolution,
                    strategy="quantile",
                    _q_factor=1.5 * _q_factor,
                )
        return F
    raise ValueError(
        f"Invalid strategy {strategy}. Pick something in {available_strategies}."
    )


def _todo_regular(f, r, api):
    if api.has_grad(f):
        _mp_logs.warn_autodiff(
            "`strategy=regular` is not differentiable. Removing grad."
        )
    with api.no_grad():
        return api.linspace(api.min(f), api.max(f), r)


def _todo_regular_closest(f, r, unique, api=None):
    if api is None:
        api = api_from_tensor(f)
    f = api.astensor(f)
    if f.ndim != 1:
        raise ValueError(f"Got ndim!=1. {f=}")
    sorted_f = api.sort(f)
    sorted_f_np = np.ascontiguousarray(api.asnumpy(sorted_f))
    indices = _mg_nb.regular_closest_1d_indices(sorted_f_np, int(r), unique)
    indices = np.ascontiguousarray(indices)
    return sorted_f[indices]


def _todo_regular_left(f, r, unique, api):
    sorted_f = api.sort(f)
    with api.no_grad():
        f_regular = api.linspace(
            sorted_f[0],
            sorted_f[-1],
            r,
            dtype=sorted_f.dtype,
            device=api.device(sorted_f),
        )
        idx = api.searchsorted(sorted_f, f_regular)
    f_regular_closest = sorted_f[idx]
    if unique:
        f_regular_closest = api.unique(f_regular_closest)
    return f_regular_closest



def _todo_partition(x, resolution, unique, api):
    if api.has_grad(x):
        _mp_logs.warn_autodiff(
            "`strategy=partition` is not differentiable. Removing grad."
        )
    out = _todo_partition_(api.asnumpy(x), resolution, unique)
    return api.from_numpy(out)


def _todo_partition_(data, resolution, unique):
    if data.shape[0] < resolution:
        resolution = data.shape[0]
    k = data.shape[0] // resolution
    partitions = np.partition(data, k)
    f = partitions[[i * k for i in range(resolution)]]
    if unique:
        f = np.unique(f)
    return f


def compute_bounding_box(stuff, inflate=0.0):
    r"""
    Returns a array of shape (2, num_parameters)
    such that for any filtration value $y$ of something in stuff,
    then if (x,z) is the output of this function, we have
    $x\le y \le z$.
    """
    grid = compute_grid(stuff, strategy="regular", resolution=2)
    api = api_from_tensors(*grid)
    box = api.moveaxis(api.stack(grid), 0, 1)
    if inflate:
        box[0] -= inflate
        box[1] += inflate
    return box


def push_to_grid(points, grid, return_coordinate=False):
    """
    Given points and a grid (list of one parameter grids),
    pushes the points onto the grid.
    """
    api = api_from_tensors(points, *grid)
    points = api.astensor(points, contiguous=True)
    grid = tuple(api.astensor(g, contiguous=True) for g in grid)
    coordinates = _mg_nb.push_to_grid_coordinates(
        np.ascontiguousarray(api.asnumpy(points)),
        tuple(np.ascontiguousarray(api.asnumpy(g)) for g in grid),
    )
    coordinates = np.ascontiguousarray(coordinates)
    if return_coordinate:
        return coordinates
    return evaluate_in_grid(coordinates, grid)


def coarsen_points(points, strategy="exact", resolution=-1, coordinate=False):
    grid = _compute_grid_numpy(points.T, strategy=strategy, resolution=resolution)
    if coordinate:
        return push_to_grid(points, grid, coordinate), grid
    return push_to_grid(points, grid, coordinate)


def _inf_value(array):
    if isinstance(array, np.dtype):
        return npapi.inf_value(array)

    try:
        api = api_from_tensor(array, strict=True)
    except Exception:
        try:
            import torch
        except Exception:
            torch = None
        if torch is not None and isinstance(array, torch.dtype):
            import multipers.array_api.torch as torchapi

            return torchapi.inf_value(array)
        try:
            return npapi.inf_value(array)
        except Exception as exc:
            raise ValueError(f"Unsupported dtype object: {array!r}") from exc

    return api.inf_value(array)


def evaluate_in_grid(
    pts, grid, mass_default=None, input_inf_value=None, output_inf_value=None, api=None,
):
    """
    Input
    -----
     - pts: of the form array[int, ndim=2]
     - grid of the form Iterable[array[float, ndim=1]]
    """
    if pts.ndim != 2:
        raise ValueError(f"`pts` must have ndim == 2. Got {pts.ndim}.")
    first_filtration = grid[0]
    dtype = first_filtration.dtype
    if api is None:
        api = api_from_tensors(*grid)
    if mass_default is not None:
        grid = tuple(
            api.cat([g, api.astensor(m)[None]]) for g, m in zip(grid, mass_default)
        )

    def empty_like(x):
        return api.empty(x.shape, dtype=dtype, device=api.device(first_filtration))

    coords = empty_like(pts)
    dim = coords.shape[1]
    pts_inf = _inf_value(pts) if input_inf_value is None else input_inf_value
    coords_inf = _inf_value(coords) if output_inf_value is None else output_inf_value
    idx = np.argwhere(pts == pts_inf)
    inf_idx = (idx[:, 0], idx[:, 1])
    pts = api_from_tensor(pts).set_at(pts, inf_idx, 0)
    for i in range(dim):
        coords = api.set_at(coords, (slice(None), i), grid[i][pts[:, i]])
    coords = api.set_at(coords, inf_idx, coords_inf)
    return coords


_push_pts_to_lines_jit = {}


def _get_push_pts_to_lines_kernel(api, with_directions):
    key = (api, with_directions)
    if key in _push_pts_to_lines_jit:
        return _push_pts_to_lines_jit[key]

    if with_directions:

        def kernel(pts, basepoints, directions):
            delta = pts[None, :, :] - basepoints[:, None, :]
            positive_idx = directions > 0
            safe_directions = api.where(positive_idx, directions, directions * 0 + 1)
            positive_terms = api.where(
                positive_idx[:, None, :],
                delta / safe_directions[:, None, :],
                -np.inf,
            )
            zero_terms = api.where(
                (directions == 0)[:, None, :],
                api.where(delta <= 0, -np.inf, np.inf),
                -np.inf,
            )
            return api.maxvalues(
                api.where(zero_terms > positive_terms, zero_terms, positive_terms),
                axis=2,
            )

    else:

        def kernel(pts, basepoints):
            delta = pts[None, :, :] - basepoints[:, None, :]
            return api.maxvalues(delta, axis=2)

    kernel = api.jit(kernel)
    _push_pts_to_lines_jit[key] = kernel
    return kernel


def sm_in_grid(pts, weights, grid, mass_default=None):
    """Given a measure whose points are coordinates,
    pushes this measure in this grid.
    Input
    -----
     - pts: of the form array[int, ndim=2]
     - weights: array[int, ndim=1]
     - grid of the form Iterable[array[float, ndim=1]]
     - num_parameters: number of parameters
    """
    if pts.ndim != 2:
        raise ValueError(f"invalid dirac locations. got {pts.ndim=} != 2")
    if len(grid) == 0:
        raise ValueError(f"Empty grid given. Got {grid=}")
    num_parameters = pts.shape[1]
    if mass_default is None:
        api = api_from_tensors(*grid)
    else:
        api = api_from_tensors(*grid, mass_default)

    _grid = list(grid)
    _mass_default = None if mass_default is None else api.astensor(mass_default)
    while len(_grid) < num_parameters:
        _grid += [
            api.cat(
                [
                    (gt := api.astensor(g))[1:],
                    api.astensor(_inf_value(api.asnumpy(gt))).reshape(1),
                ]
            )
            for g in grid
        ]
        if mass_default is not None:
            _mass_default = api.cat([_mass_default, mass_default])
    grid = tuple(_grid)
    mass_default = (
        None if _mass_default is None else api.to_device(_mass_default, api.device(pts))
    )

    coords = evaluate_in_grid(np.asarray(pts, dtype=int), grid, mass_default)
    return (coords, weights)


# TODO : optimize with memoryviews / typing
def sms_in_grid(sms, grid, mass_default=None):
    """Given a measure whose points are coordinates,
    pushes this measure in this grid.
    Input
    -----
     - sms: of the form (signed_measure_like for num_measures)
       where signed_measure_like = tuple(array[int, ndim=2], array[int])
     - grid of the form Iterable[array[float, ndim=1]]
    """
    sms = tuple(
        sm_in_grid(pts, weights, grid=grid, mass_default=mass_default)
        for pts, weights in sms
    )
    return sms


def _push_pts_to_line(pts, basepoint, direction=None, api=None, return_coordinate=False):
    basepoint = api_from_tensors(basepoint).astensor(basepoint)
    if basepoint.ndim != 1:
        raise ValueError(
            f"Expected a basepoint shape of the form (num_parameters,). Got {basepoint.shape=}"
        )
    if direction is not None:
        direction = api_from_tensors(direction).astensor(direction)
        if direction.ndim != 1:
            raise ValueError(
                f"Expected a direction shape of the form (num_parameters,). Got {direction.shape=}"
            )
        direction = direction[None]
    out = _push_pts_to_lines(
        pts,
        basepoint[None],
        directions=direction,
        api=api,
        return_coordinate=return_coordinate,
    )
    if not return_coordinate:
        return out[0]
    projected, coordinates = out
    return projected[0], coordinates[0]


def _push_pts_to_lines(pts, basepoints, directions=None, api=None, return_coordinate=False):
    if api is None:
        api = api_from_tensors(pts, basepoints, jit_promote=True)

    pts = api.astensor(pts)
    basepoints = api.astensor(basepoints)

    if directions is None:
        kernel = _get_push_pts_to_lines_kernel(api, with_directions=False)
        out = kernel(pts, basepoints)
    else:
        directions = api.astensor(directions)
        invalid_rows = api.sum(directions > 0, axis=1) <= 0
        if api.any(invalid_rows):
            invalid_direction = directions[invalid_rows][0]
            raise ValueError(f"Got invalid direction {invalid_direction}")

        kernel = _get_push_pts_to_lines_kernel(api, with_directions=True)
        out = kernel(pts, basepoints, directions)

    if not return_coordinate:
        return out

    order = np.argsort(np.ascontiguousarray(api.asnumpy(out)), axis=1, kind="stable")
    coordinates = np.empty_like(order)
    coordinates[np.arange(order.shape[0])[:, None], order] = np.arange(
        order.shape[1], dtype=order.dtype
    )
    coordinates = api.astensor(coordinates, dtype=api.int64)
    coordinates = api.to_device(coordinates, api.device(out))
    return out, coordinates


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
    (birth_sizes, death_sizes), births, deaths = mod.to_flat_idx(grid)
    births = evaluate_in_grid(births, grid)
    deaths = evaluate_in_grid(deaths, grid)
    api = api_from_tensors(births, deaths)
    diff_mod = tuple(
        zip(
            api.split_with_sizes(births, birth_sizes.tolist()),
            api.split_with_sizes(deaths, death_sizes.tolist()),
        )
    )
    return diff_mod
