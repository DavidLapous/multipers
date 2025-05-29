
from libc.stdint cimport intptr_t, int32_t, int64_t
from libcpp cimport bool,int, float

cimport numpy as cnp
import numpy as np
cnp.import_array()


from typing import Iterable,Literal,Optional
from itertools import product
from multipers.array_api import api_from_tensor, api_from_tensors
from multipers.array_api import numpy as npapi

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

    if api is npapi:
        return _compute_grid_numpy(
        initial_grid,
        resolution=resolution, 
        strategy = strategy, 
        unique = unique, 
        _q_factor=_q_factor, 
        drop_quantiles=drop_quantiles,
        dense = dense,
        )
    from multipers.torch.diff_grids import get_grid
    grid = get_grid(strategy)(initial_grid,resolution)
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
    try:
        a,b=drop_quantiles
    except:
        a,b=drop_quantiles,drop_quantiles

    if a != 0 or b != 0:
        boxes = np.asarray([np.quantile(filtration, [a, b], axis=1, method='closest_observation') for filtration in filtrations_values])
        min_filtration, max_filtration = np.min(boxes, axis=(0,1)), np.max(boxes, axis=(0,1)) # box, birth/death, filtration
        filtrations_values = [
                filtration[(m<filtration) * (filtration <M)] 
                for filtration, m,M in zip(filtrations_values, min_filtration, max_filtration)
                ]

    to_unique = lambda f : np.unique(f) if isinstance(f,np.ndarray) else f.unique()
    ## match doesn't work with cython BUG
    if strategy == "exact":
        F=tuple(to_unique(f) for f in filtrations_values)
    elif strategy == "quantile":
        F = tuple(to_unique(f) for f in filtrations_values)
        max_resolution = [min(len(f),r) for f,r in zip(F,resolution)]
        F = tuple( np.quantile(f, q=np.linspace(0,1,num=int(r*_q_factor)), axis=0, method='closest_observation') for f,r in zip(F, resolution) )
        if unique:
            F = tuple(to_unique(f) for f in F)
            if np.all(np.asarray(max_resolution) > np.asarray([len(f) for f in F])):
                return _compute_grid_numpy(filtrations_values=filtrations_values, resolution=resolution, strategy="quantile",_q_factor=1.5*_q_factor)
    elif strategy == "regular":
        F = tuple(np.linspace(np.min(f),np.max(f),num=r, dtype=np.asarray(f).dtype) for f,r in zip(filtrations_values, resolution))
    elif strategy == "regular_closest":
        F = tuple(_todo_regular_closest(f,r, unique) for f,r in zip(filtrations_values, resolution))
    elif strategy == "regular_left":
        F = tuple(_todo_regular_left(f,r, unique) for f,r in zip(filtrations_values, resolution))
    elif strategy == "torch_regular_closest":
        F = tuple(_torch_regular_closest(f,r, unique) for f,r in zip(filtrations_values, resolution))
    elif strategy == "partition":
        F = tuple(_todo_partition(f,r, unique) for f,r in zip(filtrations_values, resolution))
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
    if not isinstance(grid[0], np.ndarray):
        import torch
        assert isinstance(grid[0], torch.Tensor)
        from multipers.torch.diff_grids import todense
        return todense(grid)
    dtype = grid[0].dtype
    if product_order:
        return np.fromiter(product(*grid), dtype=np.dtype((dtype, len(grid))), count=np.prod([len(f) for f in grid]))
    mesh = np.meshgrid(*grid)
    coordinates = np.concatenate(tuple(stuff.ravel()[:,None] for stuff in mesh), axis=1, dtype=dtype)
    return coordinates



## TODO : optimize. Pykeops ?
def _todo_regular_closest(some_float[:] f, int r, bool unique):
    f_array = np.asarray(f)
    f_regular = np.linspace(np.min(f), np.max(f),num=r, dtype=f_array.dtype)
    f_regular_closest = np.asarray([f[<int64_t>np.argmin(np.abs(f_array-f_regular[i]))] for i in range(r)])
    if unique: f_regular_closest = np.unique(f_regular_closest)
    return f_regular_closest

def _todo_regular_left(some_float[:] f, int r, bool unique):
    sorted_f = np.sort(f)
    f_regular = np.linspace(sorted_f[0],sorted_f[-1],num=r, dtype=sorted_f.dtype)
    f_regular_closest = sorted_f[np.searchsorted(sorted_f,f_regular)]
    if unique: f_regular_closest = np.unique(f_regular_closest)
    return f_regular_closest

def _torch_regular_closest(f, int r, bool unique=True):
    import torch
    f_regular = torch.linspace(f.min(),f.max(), r, dtype=f.dtype)
    f_regular_closest =torch.tensor([f[(f-x).abs().argmin()] for x in f_regular]) 
    if unique: f_regular_closest = f_regular_closest.unique()
    return f_regular_closest

def _todo_partition(some_float[:] data,int resolution, bool unique):
    if data.shape[0] < resolution: resolution=data.shape[0]
    k = data.shape[0] // resolution
    partitions = np.partition(data, k)
    f = partitions[[i*k for i in range(resolution)]]
    if unique: f= np.unique(f)
    return f


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

def evaluate_in_grid(pts, grid, mass_default=None):
    """    
    Input
    -----
     - pts: of the form array[int, ndim=2]
     - grid of the form Iterable[array[float, ndim=1]]
    """
    assert pts.ndim == 2
    first_filtration = grid[0]
    dtype = first_filtration.dtype
    if isinstance(first_filtration, np.ndarray):
        if mass_default is not None:
            grid = tuple(np.concatenate([g, [m]]) for g,m in zip(grid, mass_default))
        def empty_like(x):
            return np.empty_like(x, dtype=dtype)
    else: 
        import torch
        # assert isinstance(first_filtration, torch.Tensor), f"Invalid grid type. Got {type(grid[0])}, expected numpy or torch array."
        if mass_default is not None:
            grid = tuple(torch.cat([g, torch.tensor(m)[None]]) for g,m in zip(grid, mass_default))
        def empty_like(x):
            return torch.empty(x.shape,dtype=dtype)

    coords=empty_like(pts)
    cdef int dim = coords.shape[1]
    pts_inf = _inf_value(pts)
    coords_inf = _inf_value(coords)
    idx = np.argwhere(pts == pts_inf)
    pts[idx] == 0
    for i in range(dim):
        coords[:,i] = grid[i][pts[:,i]]
    coords[idx] = coords_inf
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


def _push_pts_to_line(pts, basepoint, direction=None):
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
