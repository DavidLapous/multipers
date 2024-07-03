
from libc.stdint cimport intptr_t, int32_t, int64_t
from libcpp cimport bool,int,long, float

cimport numpy as cnp
import numpy as np
cnp.import_array()


from typing import Iterable,Literal,Optional


available_strategies = ["regular","regular_closest", "regular_left", "partition", "quantile", "precomputed"]
Lstrategies = Literal["regular","regular_closest", "regular_left", "partition", "quantile", "precomputed"]

ctypedef fused some_int:
    int32_t
    int64_t
    int
    long

ctypedef fused some_float:
    float
    double


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

    if resolution is not None and strategy == "exact":
        raise ValueError("The 'exact' strategy does not support resolution.")
    if strategy != "exact":
        assert resolution is not None, "A resolution is required for non-exact strategies"


    cdef bool is_numpy_compatible = True
    if is_slicer(x):
        initial_grid = x.get_filtrations_values().T
    elif is_simplextree_multi(x):
        initial_grid = x.get_filtration_grid()
    elif isinstance(x, np.ndarray):
        initial_grid = x
    else:
        x = tuple(x)
        if len(x) == 0: return []
        first = x[0]
        if isinstance(first,list) or isinstance(first, tuple) or isinstance(first, np.ndarray):
            initial_grid = tuple(np.asarray(f) for f in x)
        else:
            is_numpy_compatible = False
            import torch
            assert isinstance(first, torch.Tensor), "Only numpy and torch are supported ftm."
            initial_grid = x

    if is_numpy_compatible:
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
    return get_grid(strategy)(initial_grid,resolution)






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
    if resolution is None and strategy not in ["exact", "precomputed"]:
        raise ValueError("Resolution must be provided for this strategy.")
    elif resolution is not None:
        try:
            int(resolution)
            resolution = [resolution]*num_parameters
        except:
            pass
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
        F = tuple(np.linspace(f.min(),f.max(),num=r, dtype=f.dtype) for f,r in zip(filtrations_values, resolution))
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
        mesh = np.meshgrid(*F)
        coordinates = np.concatenate(tuple(stuff.ravel()[:,None] for stuff in mesh), axis=1)
        return coordinates 
    return F

def todense(grid):
    if len(grid) == 0:
        return np.empty(0)
    dtype = grid[0].dtype
    mesh = np.meshgrid(*grid)
    coordinates = np.concatenate(tuple(stuff.ravel()[:,None] for stuff in mesh), axis=1, dtype=dtype)
    return coordinates



## TODO : optimize. Pykeops ?
def _todo_regular_closest(some_float[:] f, int r, bool unique):
    f_array = np.asarray(f)
    f_regular = np.linspace(np.min(f), np.max(f),num=r, dtype=f_array.dtype)
    f_regular_closest = np.asarray([f[<long>np.argmin(np.abs(f_array-f_regular[i]))] for i in range(r)])
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


def push_to_grid(some_float[:,:] points, grid, bool return_coordinate=False):
    """
    Given points and a grid (list of one parameter grids),
    pushes the points onto the grid.
    """
    num_points, num_parameters = points.shape[0], points.shape[1]
    cdef cnp.ndarray[long,ndim=2] coordinates = np.empty((num_points, num_parameters),dtype=np.int64)
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



def sm_in_grid(pts, weights, grid_conversion, int num_parameters=-1, mass_default=None):
    """Given a measure whose points are coordinates,
    pushes this measure in this grid.
    Input
    -----
     - pts: of the form array[int, ndim=2]
     - weights: array[int, ndim=1]
     - grid_conversion of the form Iterable[array[float, ndim=1]]
     - num_parameters: number of parameters
    """
    first_filtration = grid_conversion[0]
    dtype = first_filtration.dtype
    def to_int(x):
        return np.asarray(x,dtype=np.int64)
    if isinstance(first_filtration, np.ndarray):
        if mass_default is not None:
            grid_conversion = tuple(np.concatenate([g, [m]]) for g,m in zip(grid_conversion, mass_default))
        def empty_like(x, weights):
            return np.empty_like(x, dtype=dtype), np.asarray(weights)
    else: 
        import torch
        # assert isinstance(first_filtration, torch.Tensor), f"Invalid grid type. Got {type(grid_conversion[0])}, expected numpy or torch array."
        if mass_default is not None:
            grid_conversion = tuple(torch.cat([g, torch.tensor(m)[None]]) for g,m in zip(grid_conversion, mass_default))
        def empty_like(x, weights):
            return torch.empty(x.shape,dtype=dtype), torch.from_numpy(weights)

    pts = to_int(pts)
    coords,weights = empty_like(pts,weights)
    for i in range(coords.shape[1]):
        if num_parameters > 0:
            coords[:,i] = grid_conversion[i%num_parameters][pts[:,i]]
        else:
            coords[:,i] = grid_conversion[i][pts[:,i]]
    return (coords, weights)

# TODO : optimize with memoryviews / typing
def sms_in_grid(sms, grid_conversion, int num_parameters=-1, mass_default=None):
    """Given a measure whose points are coordinates,
    pushes this measure in this grid.
    Input
    -----
     - sms: of the form (signed_measure_like for num_measures)
       where signed_measure_like = tuple(array[int, ndim=2], array[int])
     - grid_conversion of the form Iterable[array[float, ndim=1]]
    """
    sms = tuple(sm_in_grid(pts,weights,grid_conversion=grid_conversion,num_parameters=num_parameters, mass_default=mass_default) for pts,weights in sms)
    return sms
