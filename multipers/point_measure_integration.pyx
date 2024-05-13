# cimport multipers.tensor as mt
from libc.stdint cimport intptr_t, uint16_t, uint32_t, int32_t, int64_t
from libcpp.vector cimport vector
from libcpp cimport bool, int, float
import numpy as np
cimport numpy as cnp

from collections import defaultdict
cnp.import_array()
from scipy import sparse


import multipers.grids as mpg

ctypedef fused some_int:
    int32_t
    int64_t
    int

ctypedef fused some_float:
    float
    double


import cython
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def integrate_measure(
        some_float[:,:] pts, 
        some_int[:] weights, 
        filtration_grid:Optional[list[np.ndarray]]=None,
        grid_strategy:str="regular",
        resolution:int|list[int]=100,
        bool return_grid=False,
        **get_fitration_kwargs,
        ):
    """
    Integrate a point measure on a grid.
    Measure is a sum of diracs, based on points `pts` and weights `weights`.
    For instance, if the signed measure comes from the hilbert signed measure,
    this integration will return the hilbert function on this grid.
     - pts : array of points (num_pts, D)
     - weights : array of weights (num_pts,)
     - filtration_grid (optional) : list of 1d arrays
     - resolution : int or list of int
     - return_grid : return the grid of the measure
     - **get_fitration_kwargs : arguments to compute the grid,
        if the grid is not given.
    """
    if filtration_grid is None:
        import multipers.simplex_tree_multi
        filtration_grid = mpg.compute_grid(
                np.asarray(pts).T,
                strategy=grid_strategy,
                resolution=resolution,
                **get_fitration_kwargs
                )
    resolution = np.asarray([len(f) for f in filtration_grid])
    cdef int num_pts = pts.shape[0]
    cdef int num_parameters = pts.shape[1]
    assert weights.shape[0] == num_pts
    out = np.zeros(shape=resolution, dtype=np.int32) ## dim cannot be known at compiletime
    # cdef some_float[:] filtration_of_parameter
    # cdef cnp.ndarray indices = np.zeros(shape=num_parameters, dtype=int)
    #
    pts_coords = np.empty((num_parameters, num_pts), dtype=np.int64)
    for parameter in range(num_parameters):
        pts_coords[parameter] = np.searchsorted(filtration_grid[parameter], pts[:,parameter]) 
    for i in range(num_pts): 
        cone = tuple(slice(c,r) for r,c in zip(resolution,pts_coords[:,i]))
        out[cone] += weights[i] 
    if return_grid:
        return out,filtration_grid
    return out

## for benchmark purposes
def integrate_measure_python(pts, weights, filtrations):
    resolution = tuple([len(f) for f in filtrations])
    out = np.zeros(shape=resolution, dtype=pts.dtype)
    num_pts = pts.shape[0]
    num_parameters = pts.shape[1]
    for i in range(num_pts): #this is slow.
        indices = (filtrations[parameter]>=pts[i][parameter] for parameter in range(num_parameters))
        out[np.ix_(*indices)] += weights[i]
    return out


def sparsify(x):
    """
    Given an arbitrary dimensional numpy array, returns (coordinates,data).
    --
    cost : scipy sparse + num_points*num_parameters^2 divisions
    """
    num_parameters = x.ndim
    sx = sparse.coo_array(x.ravel())
    idx = sx.col
    data = sx.data
    coords = np.empty((data.shape[0], num_parameters), dtype=np.int64)
    for parameter in range(num_parameters-1,-1,-1):
        idx,coord_of_parameter = np.divmod(idx, x.shape[parameter])
        coords[:, parameter] = coord_of_parameter
    return coords,data




@cython.boundscheck(False)
@cython.wraparound(False)
def clean_signed_measure(some_float[:,:] pts, some_int[:] weights, dtype = np.float32):
    """
    Sum the diracs at the same locations. i.e.,
    returns the minimal sized measure to represent the input.
    Mostly useful for, e.g., euler_characteristic from simplical complexes.
    """
    cdef dict[tuple, int] out = {}
    cdef int num_diracs
    cdef int num_parameters
    num_diracs, num_parameters = pts.shape[:2]
    for i in range(num_diracs):
        key = tuple(pts[i]) # size cannot be known at compiletime
        out[tuple(pts[i])] = out.get(key,0)+ weights[i]
    num_keys = len(out)
    new_pts = np.fromiter(out.keys(), dtype=np.dtype((dtype,num_parameters)), count=num_keys) 
    new_weights = np.fromiter(out.values(), dtype=np.int32, count=num_keys)
    idx = np.nonzero(new_weights)
    new_pts = new_pts[idx]
    new_weights = new_weights[idx]
    return (new_pts, new_weights)

def clean_sms(sms):
    """
    Sum the diracs at the same locations. i.e.,
    returns the minimal sized measure to represent the input.
    Mostly useful for, e.g., euler_characteristic from simplical complexes.
    """
    return tuple(clean_signed_measure(pts,weights) for pts,weights in sms)


