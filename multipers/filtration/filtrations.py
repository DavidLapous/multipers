import gudhi as gd
import numpy as np
from scipy.spatial.distance import cdist
from numpy.typing import ArrayLike
from typing import Optional

import multipers as mp
import multipers.slicer as mps


def RipsLowerstar( *, 
        points:Optional[ArrayLike] = None,
        distance_matrix:Optional[ArrayLike]=None,
        function=None,
        threshold_radius=None
    ):
    """
    Computes the Rips complex, with the usual rips filtration as a first parameter,
    and the lower star multi filtration as other parameter.

    Input:
     - points or distance_matrix: ArrayLike
     - function : ArrayLike of shape (num_data, num_parameters -1)
     - threshold_radius:  max edge length of the rips. Defaults at min(max(distance_matrix, axis=1)).
    """
    assert points is not None or distance_matrix is not None, "`points` or `distance_matrix` has to be given."
    if distance_matrix is None:
        distance_matrix = cdist(points, points) # this may be slow...
    if threshold_radius is None:
        threshold_radius = np.min(np.max(distance_matrix, axis=1))
    st = gd.RipsComplex(distance_matrix = distance_matrix, max_edge_length=threshold_radius).create_simplex_tree()
    if function is None:
        return mp.SimplexTreeMulti(st, num_parameters = 1)

    function = np.asarray(function)
    if function.ndim == 1:
        function = function[:,None]
    num_parameters = function.shape[1] +1 
    st = mp.SimplexTreeMulti(st, num_parameters = num_parameters)
    for i in range(function.shape[1]):
        st.fill_lowerstar(function[:,i], parameter = 1+i)
    return st


def DelaunayLowerstar(
        points:ArrayLike,
        function:ArrayLike,
        *,
        distance_matrix:Optional[ArrayLike]=None,
        threshold_radius:Optional[float]=None,
        reduce_degree:int=-1, 
        vineyard:Optional[bool]=None, 
        dtype=np.float64, 
        verbose:bool=False, 
        clear:bool=True 
    ):
    """
    Computes the Function Delaunay bifiltration. Similar to RipsLowerstar, but most suited for low-dimensional euclidean data.

    Input:
     - points or distance_matrix: ArrayLike
     - function : ArrayLike of shape (num_data, ) 
     - threshold_radius:  max edge length of the rips. Defaults at min(max(distance_matrix, axis=1)).
    """
    assert distance_matrix is None, "Delaunay cannot be built from distance matrices"
    if threshold_radius is not None:
        raise NotImplementedError("Delaunay with threshold not implemented yet.")
    points = np.asarray(points)
    function = np.asarray(function).squeeze()
    assert function.ndim == 1, "Delaunay Lowerstar is only compatible with 1 additional parameter." 
    return mps.from_function_delaunay(points, function, degree=reduce_degree, vineyard=vineyard, dtype=dtype, verbose=verbose, clear=clear)

def Cubical(image:ArrayLike, **slicer_kwargs):
    """
    Computes the cubical filtration of an image. 
    The last axis dimention is interpreted as the number of parameters.

    Input:
     - image: ArrayLike of shape (*image_resolution, num_parameters)
     - ** args : specify non-default slicer parameters
    """
    return mps.from_bitmap(image, **slicer_kwargs)

def DegreeRips(*, points = None, distance_matrix=None, ks=None, threshold_radius=None):
    """
    The DegreeRips filtration.
    """

    raise NotImplementedError("Use the default implentation ftm.")

def CoreDelaunay():
    pass




