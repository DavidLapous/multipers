# cimport multipers.tensor as mt
from libc.stdint cimport intptr_t, uint16_t, uint32_t, int32_t, int64_t
from libcpp.vector cimport vector
from libcpp cimport bool, int, float
import numpy as np
cimport numpy as cnp
import itertools
from typing import Optional,Iterable

from collections import defaultdict
cnp.import_array()
from scipy import sparse


import multipers.grids as mpg

ctypedef fused some_int:
    int32_t
    int64_t
    int

ctypedef fused some_float:
    int32_t
    int64_t
    int
    float
    double


import cython
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def integrate_measure(
        some_float[:,:] pts, 
        some_int[:] weights, 
        filtration_grid:Optional[Iterable[np.ndarray]]=None,
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
        out[key] = out.get(key,0)+ weights[i]
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

def zero_out_sm(pts,weights, mass_default):
    """
    Zeros out the modules outside of \f$ \{ x\in \mathbb R^n \mid x \le \mathrm{mass_default}\}\f$.
    """
    from itertools import product
    # merge pts, weights
    PTS = np.concatenate([pts,weights[:,None]], axis=1)[None]
    
    num_diracs, num_parameters=pts.shape
    
    # corners of the square of dim num_parameters. shape : (num_corners,num_parameters)
    C = np.fromiter(product(*([[1,0]]*num_parameters)), dtype=np.dtype((np.bool_, num_parameters)),)
    Cw = 1-2*((~C).sum(axis=1)%2)
    # add 1 in the end to copy the shape 
    C = np.concatenate([C, np.ones((C.shape[0],1))], axis=1).astype(np.bool_)
    
    # signs of each corner
    Cw = 1-2*((~C).sum(axis=1)%2)

    # something useless in the end to ensure that the shape is correct (as we added 1 in pts)
    mass_default = np.concatenate([mass_default,[np.nan]]) 
    mass_default = mass_default[None,None,:] ## num c, num_pts, num_params
    # each point `pt` becomes a corner of the square [`pt`, `mass_default`]
    new= np.where(C[:,None,:], PTS,mass_default).reshape(-1,(num_parameters+1))
    new_pts = new[:,:-1]
    new_masses = new[:,-1]*np.repeat(Cw,num_diracs)
    ## a bunch of stuff are in the same place, better clean it
    return clean_signed_measure(new_pts.astype(np.float64), new_masses.astype(np.int64), dtype=np.float64)
 
def zero_out_sms(sms, mass_default):
    """
    Zeros out the modules outside of \f$ \{ x\in \mathbb R^n \mid x \le \mathrm{mass_default}\}\f$.
    """
    return tuple(zero_out_sm(pts,weights, mass_default) for pts,weights in sms)

@cython.boundscheck(False)
@cython.wraparound(False)
def barcode_from_rank_sm(
    sm: tuple[np.ndarray, np.ndarray],
    basepoint: np.ndarray,
    direction: Optional[np.ndarray] = None,
    bool full = False,
):
    """
    Given a rank signed measure `sm` and a line with basepoint `basepoint` (1darray) and
    direction `direction` (1darray), projects the rank signed measure on the given line,
    and returns the associated estimated barcode.
    If full is True, the barcode is given as coordinates in R^{`num_parameters`} instead
    of coordinates w.r.t. the line.
    """
    basepoint = np.asarray(basepoint)
    num_parameters = basepoint.shape[0]
    x, w = sm
    assert (
        x.shape[1] // 2 == num_parameters
    ), f"Incoherent number of parameters. sm:{x.shape[1]//2} vs {num_parameters}"
    x, y = x[:, :num_parameters], x[:, num_parameters:]
    if direction is not None:
        direction = np.asarray(direction)
        ok_idx = direction > 0
        if ok_idx.sum() == 0:
            raise ValueError(f"Got invalid direction {direction}")
        zero_idx = None if np.all(ok_idx) else direction == 0
    else:
        direction = np.asarray([1], dtype=int)
        ok_idx = slice(None)
        zero_idx = None
    xa = np.max(
        (x[:, ok_idx] - basepoint[ok_idx]) / direction[ok_idx], axis=1, keepdims=1
    )
    ya = np.min(
        (y[:, ok_idx] - basepoint[ok_idx]) / direction[ok_idx], axis=1, keepdims=1
    )
    if zero_idx is not None:
        xb = np.where(x[:, zero_idx] <= basepoint[zero_idx], -np.inf, np.inf)
        yb = np.where(y[:, zero_idx] <= basepoint[zero_idx], -np.inf, np.inf)
        xs = np.max(np.concatenate([xa, xb], axis=1), axis=1, keepdims=1)
        ys = np.min(np.concatenate([ya, yb], axis=1), axis=1, keepdims=1)
    else:
        xs = xa
        ys = ya
    out = np.concatenate([xs, ys], axis=1,dtype=np.float64)

    ## TODO: check if this is faster than doing a np.repeat on x ?
    cdef dict[dict,tuple] d = {}
    cdef double[:,:] c_out = out # view
    cdef int64_t[:] c_w = np.asarray(w,dtype=np.int64)
    cdef int num_pts = out.shape[0]
    # for i, stuff in enumerate(out):
    #     if stuff[0] < np.inf:
    #         d[tuple(stuff)] = d.get(tuple(stuff), 0) + w[i]
    for i in range(num_pts):
        if c_out[i][0] < np.inf:
            machin = tuple(c_out[i])
            d[machin] = d.get(machin, 0) + c_w[i]

    out = np.fromiter(
        itertools.chain.from_iterable(([x] * w for x, w in d.items() if x[0] < x[1])),
        dtype=np.dtype((np.float64, 2)),
    )
    if full:
        out = basepoint[None, None] + out[..., None] * direction[None, None]
    return out


def estimate_rank_from_rank_sm(sm:tuple, a, b) -> int:
    """
    Given a rank signed measure (sm) and two points (a) and (b),
    estimates the rank between these two points.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if not (a <= b).all():
        return 0
    x, w = sm
    num_parameters = x.shape[1] // 2
    assert (
        a.shape[0] == b.shape[0] == num_parameters
    ), f"Incoherent number of parameters. sm:{num_parameters} vs {a.shape[0]} and {b.shape[0]}"
    idx = (
        (x[:, :num_parameters] <= a[None]).all(1)
        * (x[:, num_parameters:] >= b[None]).all(1)
    ).ravel()
    return w[idx].sum()


