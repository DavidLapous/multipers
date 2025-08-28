
from libc.stdint cimport intptr_t, int32_t, int64_t
from libcpp cimport bool,int, float

cimport numpy as cnp
import numpy as np
cnp.import_array()


from typing import Iterable,Literal,Optional
from itertools import product
from multipers.array_api import api_from_tensor, api_from_tensors
from multipers.array_api import numpy as npapi
from multipers.array_api import check_keops

available_strategies = ["regular","regular_closest", "regular_left", "partition", "quantile", "precomputed"]
Lstrategies = Literal["regular","regular_closest", "regular_left", "partition", "quantile", "precomputed"]

ctypedef fused some_int:
    int32_t
    int64_t

ctypedef fused some_float:
    float
    double

def sanitize_grid(grid, bool numpyfy=False):
    if len(grid) == 0:
        raise ValueError("empty filtration grid")
    api = api_from_tensors(*grid)
    if numpyfy:
        grid = tuple(api.asnumpy(g) for g in grid)
    else:
        # copy here may not be necessary, but cheap
        grid = tuple(api.astensor(g) for g in grid) 
    assert np.all([g.ndim==1 for g in grid])
    return grid

def compute_grid(
        x,
        resolution:Optional[int|Iterable[int]]=None, 
        strategy:Lstrategies="exact", 
        bool unique=True, 
        some_float _q_factor=1., 
        drop_quantiles=[0,0],
        bool dense = False,
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

    from multipers.slicer import is_slicer
    from multipers.simplex_tree_multi import is_simplextree_multi
    from multipers.mma_structures import is_mma

    if resolution is not None and strategy == "exact":
        raise ValueError("The 'exact' strategy does not support resolution.")
    if strategy != "exact":
        assert resolution is not None, "A resolution is required for non-exact strategies"


    cdef bool is_numpy_compatible = True
    if (is_slicer(x) or is_simplextree_multi(x)) and x.is_squeezed:
        initial_grid = x.filtration_grid
        api = api_from_tensors(*initial_grid)
    elif is_slicer(x):
        initial_grid = x.get_filtrations_values().T
        api = npapi
    elif is_simplextree_multi(x):
        initial_grid = x.get_filtration_grid()
        api = npapi
    elif is_mma(x):
        initial_grid = x.get_filtration_values()
        api = npapi
    elif isinstance(x, np.ndarray):
        initial_grid = x
        api = npapi
    else:
        x = tuple(x)
        if len(x) == 0: return []
        first = x[0]
        ## is_sm, i.e., iterable tuple(pts,weights)
        if isinstance(first, tuple) and getattr(first[0], "shape", None) is not None:
            initial_grid = tuple(f[0].T for f in x)
            api = api_from_tensors(*initial_grid)
            initial_grid = api.cat(initial_grid, axis=1)
            # if isinstance(initial_grid[0], np.ndarray):
            #     initial_grid = np.concatenate(initial_grid, axis=1)
            # else:
            #     is_numpy_compatible = False
            #     import torch
            #     assert isinstance(first[0], torch.Tensor), "Only numpy and torch are supported ftm."
            #     initial_grid = torch.cat(initial_grid, axis=1)
        ## is grid-like (num_params, num_pts)
        else:
            api = api_from_tensors(*x)
            initial_grid = tuple(api.astensor(f) for f in x)
            # elif isinstance(first,list) or isinstance(first, tuple) or isinstance(first, np.ndarray):
            #     initial_grid = tuple(f for f in x)
            # else:
            #     is_numpy_compatible = False
            #     import torch
            #     assert isinstance(first, torch.Tensor), "Only numpy and torch are supported ftm."
            #     initial_grid = x

    num_parameters = len(initial_grid)
    try:
        int(resolution)
        resolution = [resolution]*num_parameters
    except TypeError:
        pass

    grid = _compute_grid_numpy(
        initial_grid,
        resolution=resolution, 
        strategy = strategy, 
        unique = unique, 
        _q_factor=_q_factor, 
        drop_quantiles=drop_quantiles,
        dense = dense,
    )
    # from multipers.torch.diff_grids import get_grid
    # grid = get_grid(strategy)(initial_grid,resolution)
    if dense:
        grid = todense(grid)
    return grid





def _compute_grid_numpy(
        filtrations_values,
        resolution=None, 
        strategy:Lstrategies="exact", 
        bool unique=True, 
        some_float _q_factor=1., 
        drop_quantiles=[0,0],
        bool dense = False,
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
    api = api_from_tensors(*filtrations_values)
    try:
        a,b=drop_quantiles
    except:
        a,b=drop_quantiles,drop_quantiles

    if a != 0 or b != 0:
        boxes = api.astensor([api.quantile_closest(filtration, [a, b], axis=1) for filtration in filtrations_values])
        min_filtration, max_filtration = api.minvalues(boxes, axis=(0,1)), api.maxvalues(boxes, axis=(0,1)) # box, birth/death, filtration
        filtrations_values = [
                filtration[(m<filtration) * (filtration <M)] 
                for filtration, m,M in zip(filtrations_values, min_filtration, max_filtration)
                ]

    ## match doesn't work with cython BUG
    if strategy == "exact":
        F=tuple(api.unique(f) for f in filtrations_values)
    elif strategy == "quantile":
        F = tuple(api.unique(f) for f in filtrations_values)
        max_resolution = [min(len(f),r) for f,r in zip(F,resolution)]
        F = tuple( api.quantile_closest(f, q=api.linspace(0,1,int(r*_q_factor)), axis=0) for f,r in zip(F, resolution) )
        if unique:
            F = tuple(api.unique(f) for f in F)
            if np.all(np.asarray(max_resolution) > np.asarray([len(f) for f in F])):
                return _compute_grid_numpy(filtrations_values=filtrations_values, resolution=resolution, strategy="quantile",_q_factor=1.5*_q_factor)
    elif strategy == "regular":
        F = tuple(_todo_regular(f,r,api) for f,r in zip(filtrations_values, resolution))
    elif strategy == "regular_closest":
        F = tuple(_todo_regular_closest(f,r, unique,api) for f,r in zip(filtrations_values, resolution))
    elif strategy == "regular_left":
        F = tuple(_todo_regular_left(f,r, unique,api) for f,r in zip(filtrations_values, resolution))
    # elif strategy == "torch_regular_closest":
    #     F = tuple(_torch_regular_closest(f,r, unique) for f,r in zip(filtrations_values, resolution))
    elif strategy == "partition":
        F = tuple(_todo_partition(f,r, unique, api) for f,r in zip(filtrations_values, resolution))
    elif strategy == "precomputed":
        F=filtrations_values
    else:
        raise ValueError(f"Invalid strategy {strategy}. Pick something in {available_strategies}.")
    if dense:
        return todense(F)
    return F

def todense(grid, bool product_order=False):
    if len(grid) == 0:
        return np.empty(0)
    api = api_from_tensors(*grid)
    # if product_order:
    #     if not api.backend ==np:
    #         raise NotImplementedError("only numpy here.")
    #     return np.fromiter(product(*grid), dtype=np.dtype((dtype, len(grid))), count=np.prod([len(f) for f in grid]))
    return api.cartesian_product(*grid)
    # if not isinstance(grid[0], np.ndarray):
    #     import torch
    #     assert isinstance(grid[0], torch.Tensor)
    #     from multipers.torch.diff_grids import todense
    #     return todense(grid)
    # dtype = grid[0].dtype
    # if product_order:
    #     return np.fromiter(product(*grid), dtype=np.dtype((dtype, len(grid))), count=np.prod([len(f) for f in grid]))
    # mesh = np.meshgrid(*grid)
    # coordinates = np.stack(mesh, axis=-1).reshape(-1, len(grid)).astype(dtype)
    # return coordinates



def _todo_regular(f, int r, api):
    if api.has_grad(f):
        from warnings import warn
        warn("`strategy=regular` is not differentiable. Removing grad.")
    with api.no_grad():
        return api.linspace(api.min(f), api.max(f), r)

def _project_on_1d_grid(f,grid, bool unique, api):
    # api=api_from_tensors(f,grid)
    if f.ndim != 1:
        raise ValueError(f"Got ndim!=1. {f=}")
    f = api.unique(f)
    with api.no_grad():
        _f = api.LazyTensor(f[:, None, None])
        _f_reg = api.LazyTensor(grid[None, :, None])
        indices = (_f - _f_reg).abs().argmin(0).ravel()
    f = api.cat([f, api.tensor([api.inf], dtype=f.dtype)])
    f_proj = f[indices]
    if unique:
        f_proj = api.unique(f_proj)
    return f_proj

def _todo_regular_closest_keops(f, int r, bool unique, api):
    f = api.astensor(f)
    with api.no_grad():
        f_regular = api.linspace(api.min(f), api.max(f), r, device = api.device(f),dtype=f.dtype)
    return _project_on_1d_grid(f,f_regular,unique,api)

def _todo_regular_closest_old(some_float[:] f, int r, bool unique, api=None):
    f_array = np.asarray(f)
    f_regular = np.linspace(np.min(f), np.max(f),num=r, dtype=f_array.dtype)
    f_regular_closest = np.asarray([f[<int64_t>np.argmin(np.abs(f_array-f_regular[i]))] for i in range(r)], dtype=f_array.dtype)
    if unique: f_regular_closest = np.unique(f_regular_closest)
    return f_regular_closest

def _todo_regular_left(f, int r, bool unique,api):
    sorted_f = api.sort(f)
    with api.no_grad():
        f_regular = api.linspace(sorted_f[0],sorted_f[-1],r, dtype=sorted_f.dtype, device=api.device(sorted_f))
        idx=api.searchsorted(sorted_f,f_regular)
    f_regular_closest = sorted_f[idx]
    if unique: f_regular_closest = api.unique(f_regular_closest)
    return f_regular_closest

def _todo_regular_left_old(some_float[:] f, int r, bool unique):
    sorted_f = np.sort(f)
    f_regular = np.linspace(sorted_f[0],sorted_f[-1],num=r, dtype=sorted_f.dtype)
    f_regular_closest = sorted_f[np.searchsorted(sorted_f,f_regular)]
    if unique: f_regular_closest = np.unique(f_regular_closest)
    return f_regular_closest

def _todo_partition(x, int resolution, bool unique, api):
    if api.has_grad(x):
        from warnings import warn
        warn("`strategy=partition` is not differentiable. Removing grad.")
    out = _todo_partition_(api.asnumpy(x), resolution, unique)
    return api.from_numpy(out)

def _todo_partition_(some_float[:] data,int resolution, bool unique):
    if data.shape[0] < resolution: resolution=data.shape[0]
    k = data.shape[0] // resolution
    partitions = np.partition(data, k)
    f = partitions[[i*k for i in range(resolution)]]
    if unique: f= np.unique(f)
    return f


if check_keops():
    _todo_regular_closest = _todo_regular_closest_keops
else:
    _todo_regular_closest = _todo_regular_closest_old


def compute_bounding_box(stuff, inflate = 0.):
    r"""
    Returns a array of shape (2, num_parameters)
    such that for any filtration value $y$ of something in stuff,
    then if (x,z) is the output of this function, we have
    $x\le y \le z$.
    """
    box = np.array(compute_grid(stuff,strategy="regular",resolution=2)).T
    if inflate:
        box[0] -= inflate
        box[1] += inflate
    return box

def push_to_grid(some_float[:,:] points, grid, bool return_coordinate=False):
    """
    Given points and a grid (list of one parameter grids),
    pushes the points onto the grid.
    """
    num_points, num_parameters = points.shape[0], points.shape[1]
    cdef cnp.ndarray[int64_t,ndim=2] coordinates = np.empty((num_points, num_parameters),dtype=np.int64)
    for parameter in range(num_parameters):
        coordinates[:,parameter] = np.searchsorted(grid[parameter],points[:,parameter])
    if return_coordinate:
        return coordinates
    out = np.empty((num_points,num_parameters), grid[0].dtype)
    for parameter in range(num_parameters):
        out[:,parameter] = grid[parameter][coordinates[:,parameter]]
    return out


def coarsen_points(some_float[:,:] points, strategy="exact", int resolution=-1, bool coordinate=False):
    grid = _compute_grid_numpy(points.T, strategy=strategy, resolution=resolution)
    if coordinate:
        return push_to_grid(points, grid, coordinate), grid
    return push_to_grid(points, grid, coordinate)

def _inf_value(array):
    if isinstance(array, type|np.dtype):
        dtype = np.dtype(array) # torch types are not types
    elif isinstance(array, np.ndarray):
        dtype = np.dtype(array.dtype)
    else:
        import torch
        if isinstance(array, torch.Tensor):
            dtype=array.dtype
        elif isinstance(array, torch.dtype):
            dtype=array
        else:
            raise ValueError(f"unknown input of type {type(array)=} {array=}")

    if isinstance(dtype, np.dtype):
        if dtype.kind == 'f':
            return np.asarray(np.inf,dtype=dtype)
        if dtype.kind == 'i':
            return np.iinfo(dtype).max
    # torch only here.
    if dtype.is_floating_point:
        return torch.tensor(torch.inf, dtype=dtype)
    else:
        return torch.iinfo(dtype).max
    raise ValueError(f"Dtype must be integer or floating like (got {dtype})")

def evaluate_in_grid(pts, grid, mass_default=None, input_inf_value=None, output_inf_value=None):
    """    
    Input
    -----
     - pts: of the form array[int, ndim=2]
     - grid of the form Iterable[array[float, ndim=1]]
    """
    assert pts.ndim == 2
    first_filtration = grid[0]
    dtype = first_filtration.dtype
    api = api_from_tensors(*grid)
    if mass_default is not None:
        grid = tuple(api.cat([g, api.astensor(m)[None]]) for g,m in zip(grid, mass_default))
    def empty_like(x):
        return api.empty(x.shape, dtype=dtype)

    coords=empty_like(pts)
    cdef int dim = coords.shape[1]
    pts_inf = _inf_value(pts) if input_inf_value is None else input_inf_value
    coords_inf = _inf_value(coords) if output_inf_value is None else output_inf_value
    idx = np.argwhere(pts == pts_inf)
    pts[idx[:,0],idx[:,1]] = 0
    for i in range(dim):
        coords[:,i] = grid[i][pts[:,i]]
    coords[idx[:,0],idx[:,1]] = coords_inf
    return coords

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
    cdef int num_parameters  = pts.shape[1]
    if mass_default is None:
        api = api_from_tensors(*grid)
    else:
        api = api_from_tensors(*grid, mass_default)

    _grid = list(grid)
    _mass_default = None if mass_default is None else api.astensor(mass_default)
    while len(_grid) < num_parameters:
        _grid += [api.cat([
                (gt:=api.astensor(g))[1:],
                api.astensor(_inf_value(api.asnumpy(gt))).reshape(1)
            ]) for g in grid]
        if mass_default is not None:
            _mass_default = api.cat([_mass_default,mass_default])
    grid = tuple(_grid)
    mass_default = _mass_default

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
    sms = tuple(sm_in_grid(pts,weights,grid=grid, mass_default=mass_default) for pts,weights in sms)
    return sms


def _push_pts_to_line(pts, basepoint, direction=None, api=None):
    if api is None:
        api = api_from_tensors(pts, basepoint)
    pts = api.astensor(pts)
    basepoint = api.astensor(basepoint)
    num_parameters = basepoint.shape[0]
    if direction is not None:
        if not api.is_promotable(direction):
            raise ValueError(f"Incompatible input types. Got {type(pts)=}, {type(basepoint)=}, {type(direction)=}")

        direction = api.astensor(direction)
        ok_idx = direction > 0
        if ok_idx.sum() == 0:
            raise ValueError(f"Got invalid direction {direction}")
        zero_idx = None if ok_idx.all() else direction == 0
    else:
        direction = api.tensor([1], dtype=int)
        ok_idx = slice(None)
        zero_idx = None
    xa = api.maxvalues(
        (pts[:, ok_idx] - basepoint[ok_idx]) / direction[ok_idx], axis=1, keepdims=True
    )
    if zero_idx is not None:
        xb = api.where(pts[:, zero_idx] <= basepoint[zero_idx], -np.inf, np.inf)
        xs = api.maxvalues(api.cat([xa, xb], axis=1), axis=1, keepdims=True)
    else:
        xs = xa
    return xs.squeeze()

def _push_pts_to_lines(pts, basepoints, directions=None, api=None):
    if api is None:
        api = api_from_tensors(pts,basepoints)
    cdef int num_lines = len(basepoints)
    cdef int num_pts = len(pts)

    pts = api.astensor(pts)
    basepoints = api.astensor(basepoints)
    if directions is None:
        directions = [None]*num_lines
    else:
        directions = api.astensor(directions)

    out = api.empty((num_lines, num_pts), dtype=pts.dtype)
    for i in range(num_lines):
        out[i] = _push_pts_to_line(pts, basepoints[i], directions[i], api=api)[None]
    return out


