
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
        filtrations_values,
        resolutions=None, 
        strategy:Lstrategies="exact", 
        bool unique=True, 
        some_float _q_factor=1., 
        drop_quantiles=[0,0]
        ):
    """
    Computes a grid from filtration values, using some strategy.

    Input
    -----
     - `filtrations_values`: `Iterable[filtration of parameter for parameter]`
       where `filtration_of_parameter` is a array[float, ndim=1]
     - `resolutions`:Optional[int|tuple[int]]
     - `strategy`: either exact, regular, regular_closest, regular_left, partition, quantile, or precomputed.
     - `unique`: if true, doesn't repeat values in the output grid.
     - `drop_quantiles` : drop some filtration values according to these quantiles
    Output
    ------
    Iterable[array[float, ndim=1]] : the 1d-grid for each parameter.
    """
    num_parameters = len(filtrations_values)
    if resolutions is None and strategy not in ["exact", "precomputed"]:
        raise ValueError("Resolutions must be provided for this strategy.")
    elif resolutions is not None:
        try:
            int(resolutions)
            resolutions = [resolutions]*num_parameters
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

    ## match doesn't work with cython BUG
    if strategy == "exact":
        to_unique = lambda f : np.unique(f) if isinstance(f,np.ndarray) else f.unique()
        F=[to_unique(f) for f in filtrations_values]
    elif strategy == "quantile":
        F = [f.unique() for f in filtrations_values]
        max_resolution = [min(len(f),r) for f,r in zip(F,resolutions)]
        F = [np.quantile(f, q=np.linspace(0,1,num=int(r*_q_factor)), axis=0, method='closest_observation') for f,r in zip(F, resolutions)]
        if unique:
            F = [np.unique(f) for f in F]
            if np.all(np.asarray(max_resolution) > np.asarray([len(f) for f in F])):
                return compute_grid(filtrations_values=filtrations_values, resolutions=resolutions, strategy="quantile",_q_factor=1.5*_q_factor)
    elif strategy == "regular":
        F = [np.linspace(f.min(),f.max(),num=r) for f,r in zip(filtrations_values, resolutions)]
    elif strategy == "regular_closest":
        F = [_todo_regular_closest(f,r, unique) for f,r in zip(filtrations_values, resolutions)]
    elif strategy == "regular_left":
        F = [_todo_regular_left(f,r, unique) for f,r in zip(filtrations_values, resolutions)]
    elif strategy == "torch_regular_closest":
        F = [_torch_regular_closest(f,r, unique) for f,r in zip(filtrations_values, resolutions)]
    elif strategy == "partition":
        F = [_todo_partition(f,r, unique) for f,r in zip(filtrations_values, resolutions)]
    elif strategy == "precomputed":
        F=filtrations_values
    else:
        raise ValueError(f"Invalid strategy {strategy}. Pick something in {available_strategies}.")
    return F



def _todo_regular_closest(some_float[:] f, int r, bool unique):
    f_array = np.asarray(f)
    cdef float[:] f_regular = np.linspace(np.min(f), np.max(f),num=r, dtype=np.float32)
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
    f_regular = torch.linspace(f.min(),f.max(), r)
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


def coarsen_points(some_float[:,:] points, strategy="exact", int resolutions=-1, bool coordinate=False):
    grid = compute_grid(points.T, strategy=strategy, resolutions=resolutions)
    if coordinate:
        return push_to_grid(points, grid, coordinate), grid
    return push_to_grid(points, grid, coordinate)



# TODO : optimize with memoryviews / typing
def sms_in_grid(sms, grid_conversion):
    """Given a measure whose points are coordinates,
    pushes this measure in this grid.
    Input
    -----
     - sms: of the form (signed_measure_like for num_measures)
       where signed_measure_like = tuple(array[int, ndim=2], array[int])
     - grid_conversion of the form Iterable[array[float, ndim=1]]
    """
    first_filtration = grid_conversion[0]
    dtype = first_filtration.dtype
    def to_int(x):
        return np.asarray(x,dtype=np.int64)
    def empty_like(x, weights):
        return np.empty_like(x, dtype=dtype), np.asarray(weights)

    for degree_index,(pts,weights) in enumerate(sms):
        # print(pts.shape,weights.shape)
        # assert (pts>=0).all(), f"{degree_index=}, {pts=}, {weights=}"
        pts = to_int(pts)
        coords,weights = empty_like(pts,weights)
        for i in range(coords.shape[1]):
            coords[:,i] = grid_conversion[i][pts[:,i]]
        sms[degree_index]=(coords, weights)
    return sms
