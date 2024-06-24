from typing import Callable, Literal, Optional

import numpy as np
import torch
from gudhi.rips_complex import RipsComplex

import multipers as mp
from multipers.ml.convolutions import DTM, KDE
from multipers.simplex_tree_multi import _available_strategies
from multipers.torch.diff_grids import get_grid


def function_rips_signed_measure_old(
    x,
    theta: Optional[float] = None,
    function: Literal["dtm", "gaussian", "exponential"] | Callable = "dtm",
    threshold: float = np.inf,
    grid_strategy: _available_strategies = "regular_closest",
    resolution: int = 100,
    return_original: bool = False,
    return_st: bool = False,
    safe_conversion: bool = False,
    num_collapses: int = -1,
    expand_collapse: bool = False,
    dtype=torch.float32,
    **sm_kwargs,
):
    """
    Computes a torch-differentiable function-rips signed measure.

    Input
    -----
     - x (num_pts, dim) : The point cloud
     - theta: For density-like functions : the bandwidth
     - threshold : rips threshold
     - function : Either "dtm", "gaussian", or "exponenetial" or Callable.
       Function to compute the second parameter.
     - grid_strategy: grid coarsenning strategy.
     - resolution : when coarsenning, the target resolution,
     - return_original : Also returns the non-differentiable signed measure.
     - safe_conversion : Activate this if you encounter crashes.
     - **kwargs : for the signed measure computation.
    """
    assert isinstance(x, torch.Tensor)
    if function == "dtm":
        assert theta is not None, "Provide a theta to compute DTM"
        codensity = DTM(masses=[theta]).fit(x).score_samples_diff(x)[0].type(dtype)
    elif function in ["gaussian", "exponential"]:
        assert theta is not None, "Provide a theta to compute density estimation"
        codensity = (
            -KDE(
                bandwidth=theta,
                kernel=function,
                return_log=True,
            )
            .fit(x)
            .score_samples(x)
            .type(dtype)
        )
    else:
        assert callable(function), "Function has to be callable"
        if theta is None:
            codensity = function(x).type(dtype)
        else:
            codensity = function(x, theta=theta).type(dtype)

    distance_matrix = torch.cdist(x, x).type(dtype)
    if threshold < np.inf:
        distance_matrix[distance_matrix > threshold] = np.inf

    st = RipsComplex(
        distance_matrix=distance_matrix.detach(), max_edge_length=threshold
    ).create_simplex_tree()
    # detach makes a new (reference) tensor, without tracking the gradient
    st = mp.SimplexTreeMulti(st, num_parameters=2, safe_conversion=safe_conversion)
    st.fill_lowerstar(
        codensity.detach(), parameter=1
    )  # fills the codensity in the second parameter of the simplextree

    # simplificates the simplextree for computation, the signed measure will be recovered from the copy afterward
    st_copy = st.grid_squeeze(
        grid_strategy=grid_strategy, resolution=resolution, coordinate_values=True
    )
    if sm_kwargs.get("degree", None) is None and sm_kwargs.get("degrees", [None]) == [
        None
    ]:
        expansion_degree = st.num_vertices
    else:
        expansion_degree = (
            max(np.max(sm_kwargs.get("degrees", 1)), sm_kwargs.get("degree", 1)) + 1
        )
    st.collapse_edges(num=num_collapses)
    if not expand_collapse:
        st.expansion(expansion_degree)  # edge collapse
    sms = mp.signed_measure(st, **sm_kwargs)  # computes the signed measure
    del st

    simplices_list = tuple(
        s for s, _ in st_copy.get_simplices()
    )  # not optimal, we may want to do that in cython to get edges and nodes
    sms_diff = []
    for sm, weights in sms:
        indices, not_found_indices = st_copy.pts_to_indices(
            sm, simplices_dimensions=[1, 0]
        )
        if sm_kwargs.get("verbose", False):
            print(
                f"Found {(1-(indices == -1).mean()).round(2)} indices. \
                Out : {(indices == -1).sum()}, {len(not_found_indices)}"
            )
        sm_diff = torch.empty(sm.shape).type(dtype)
        # sim_dim = sm_diff.shape[1] // 2

        # fills the Rips-filtrations of the signed measure.
        # the loop is for the rank invariant
        for i in range(0, sm_diff.shape[1], 2):
            idxs = indices[:, i]
            if (idxs == -1).all():
                continue
            useful_idxs = idxs != -1
            # Retrieves the differentiable values from the distance_matrix
            if useful_idxs.size > 0:
                edges_filtrations = torch.cat(
                    [
                        distance_matrix[*simplices_list[idx], None]
                        for idx in idxs[useful_idxs]
                    ]
                )
                # fills theses values into the signed measure
                sm_diff[:, i][useful_idxs] = edges_filtrations
        # same for the other axis
        for i in range(1, sm_diff.shape[1], 2):
            idxs = indices[:, i]
            if (idxs == -1).all():
                continue
            useful_idxs = idxs != -1
            if useful_idxs.size > 0:
                nodes_filtrations = torch.cat(
                    [codensity[simplices_list[idx]] for idx in idxs[useful_idxs]]
                )
                sm_diff[:, i][useful_idxs] = nodes_filtrations

        # fills not-found values as constants
        if len(not_found_indices) > 0:
            not_found_indices = indices == -1
            sm_diff[indices == -1] = torch.from_numpy(sm[indices == -1]).type(dtype)

        sms_diff.append((sm_diff, torch.from_numpy(weights)))
    flags = [True, return_original, return_st]
    if np.sum(flags) == 1:
        return sms_diff
    return tuple(stuff for stuff, flag in zip([sms_diff, sms, st_copy], flags) if flag)


def function_rips_signed_measure(
    x,
    theta: Optional[float] = None,
    function: Literal["dtm", "gaussian", "exponential"] | Callable = "gaussian",
    threshold: Optional[float] = None,
    grid_strategy: Literal["regular_closest", "exact", "quantile", "regular_left"] = "exact",
    complex:Literal["rips", "delaunay"] = "rips",
    resolution: int = 100,
    safe_conversion: bool = False,
    num_collapses: Optional[int] = None,
    expand_collapse: bool = False,
    dtype=torch.float32,
    plot=False,
    # return_st: bool = False,
    *,
    log_density: bool = True,
    vineyard:bool = False,
    **sm_kwargs,
):
    """
    Computes a torch-differentiable function-rips signed measure.

    Input
    -----
     - x (num_pts, dim) : The point cloud
     - theta: For density-like functions : the bandwidth
     - threshold : rips threshold
     - function : Either "dtm", "gaussian", or "exponenetial" or Callable.
       Function to compute the second parameter.
     - grid_strategy: grid coarsenning strategy.
     - resolution : when coarsenning, the target resolution,
     - return_original : Also returns the non-differentiable signed measure.
     - safe_conversion : Activate this if you encounter crashes.
     - **kwargs : for the signed measure computation.
    """
    if num_collapses is None:
        num_collapses = -1 if complex == "rips" else None
    assert isinstance(x, torch.Tensor)
    if function == "dtm":
        assert theta is not None, "Provide a theta to compute DTM"
        codensity = DTM(masses=[theta]).fit(x).score_samples_diff(x)[0].type(dtype)
    elif function in ["gaussian", "exponential"]:
        assert theta is not None, "Provide a theta to compute density estimation"
        codensity = (
            -KDE(
                bandwidth=theta,
                kernel=function,
                return_log=log_density,
            )
            .fit(x)
            .score_samples(x)
            .type(dtype)
        )
    elif isinstance(function, torch.Tensor):
        assert function.ndim == 1 and codensity.shape[0] == x.shape[0], """
        When function is a tensor, it is interpreted as the value of some function over x. 
        """
        codensity = function
    else:
        assert callable(function), "Function has to be callable"
        if theta is None:
            codensity = function(x).type(dtype)
        else:
            codensity = function(x, theta=theta).type(dtype)

    
    distance_matrix = torch.cdist(x, x).type(dtype)
    distances = distance_matrix.ravel()
    if  complex == "rips":
        threshold = distance_matrix.max(axis=1).values.min() if threshold is None else threshold
        distances = distances[distances <= threshold]
    elif complex == "delaunay":
        distances /= 2
    else:
        raise ValueError(f"Unimplemented with complex {complex}. You can use rips or delaunay ftm.")
    
    # simplificates the simplextree for computation, the signed measure will be recovered from the copy afterward
    reduced_grid = get_grid(strategy=grid_strategy)((distances, codensity), resolution)

    degrees = sm_kwargs.pop("degrees", [])
    if sm_kwargs.get("degree", None) is not None:
        degrees = [sm_kwargs.pop("degree", None)] + degrees
    if complex == "rips":
        st = RipsComplex(
            distance_matrix=distance_matrix.detach(), max_edge_length=threshold
        ).create_simplex_tree()
        # detach makes a new (reference) tensor, without tracking the gradient
        st = mp.SimplexTreeMulti(st, num_parameters=2, safe_conversion=safe_conversion)
        st.fill_lowerstar(
            codensity.detach(), parameter=1
        )  # fills the codensity in the second parameter of the simplextree
        st = st.grid_squeeze(reduced_grid, coordinate_values=True)
        if None in degrees:
            expansion_degree = st.num_vertices
        else:
            expansion_degree = (
                max(degrees) + 1
            )
        st.collapse_edges(num=num_collapses)
        if not expand_collapse:
            st.expansion(expansion_degree)  # edge collapse

        s = mp.Slicer(st, vineyard=vineyard)
    elif complex == "delaunay":
        s = mp.slicer.from_function_delaunay(x.detach().numpy(),codensity.detach().numpy())
        st = mp.slicer.to_simplextree(s)
        st.flagify(2)
        s = mp.Slicer(st, vineyard=vineyard)
    
    if None not in degrees:
        s = s.minpres(degrees=degrees)
    else:
        from joblib import Parallel, delayed
        s = tuple(Parallel(n_jobs = -1, backend="threading")(delayed(lambda d : s if d is None else s.minpres(degree=d))(d) for d in degrees))

    sms = tuple(
        sm 
        for slicer_of_degree in s
        for sm in mp.signed_measure(slicer_of_degree, grid_conversion=reduced_grid, **sm_kwargs)
    )  # computes the signed measure
    if plot:
        mp.plots.plot_signed_measures(
            tuple((sm.detach().numpy(), w.detach().numpy()) for sm, w in sms)
        )
    return sms
