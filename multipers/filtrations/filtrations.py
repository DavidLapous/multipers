from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from time import perf_counter
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from multipers.array_api import api_from_tensor, api_from_tensors, check_keops
import multipers.logs as _mp_logs
from multipers.filtrations.density import DTM, available_kernels
from multipers.grids import compute_grid, get_exact_grid

from multipers.simplex_tree_multi import SimplexTreeMulti_type


def _function_delaunay_presentation_to_slicer(
    slicer,
    point_cloud: np.ndarray,
    function_values: np.ndarray,
    clear: bool = True,
    verbose: bool = False,
    degree=-1,
    multi_chunk=False,
    recover_ids: bool = False,
):
    del clear
    import multipers._function_delaunay_interface as _function_delaunay_interface

    if not _function_delaunay_interface._is_available():
        raise RuntimeError(
            "function_delaunay interface is not available in this build. "
            "Rebuild multipers with function_delaunay support to enable this backend."
        )

    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    function_values = np.asarray(function_values, dtype=np.float64).reshape(-1)
    if point_cloud.ndim != 2:
        raise ValueError(f"point_cloud should be a 2d array. Got {point_cloud.shape=}")
    if function_values.ndim != 1:
        raise ValueError(
            f"function_values should be a 1d array. Got {function_values.shape=}"
        )
    if point_cloud.shape[0] != function_values.shape[0]:
        raise ValueError(
            f"point_cloud and function_values should have same number of points. "
            f"Got {point_cloud.shape[0]} and {function_values.shape[0]}."
        )
    if verbose:
        _mp_logs.log_verbose(
            f"[multipers.backends] backend=function_delaunay mode=cpp_interface degree={degree} multi_chunk={multi_chunk} recover_ids={recover_ids}",
            enabled=verbose,
        )
    return _function_delaunay_interface.function_delaunay_to_slicer(
        slicer,
        point_cloud,
        function_values,
        degree,
        multi_chunk,
        recover_ids,
        verbose,
    )


def _function_delaunay_presentation_to_simplextree(
    point_cloud: np.ndarray,
    function_values: np.ndarray,
    clear: bool = True,
    verbose: bool = False,
    dtype=np.float64,
    recover_ids: bool = False,
):
    del clear
    import multipers._function_delaunay_interface as _function_delaunay_interface
    from multipers.simplex_tree_multi import SimplexTreeMulti

    if not _function_delaunay_interface._is_available():
        raise RuntimeError(
            "function_delaunay interface is not available in this build. "
            "Rebuild multipers with function_delaunay support to enable this backend."
        )

    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    function_values = np.asarray(function_values, dtype=np.float64).reshape(-1)
    if point_cloud.ndim != 2:
        raise ValueError(f"point_cloud should be a 2d array. Got {point_cloud.shape=}")
    if function_values.ndim != 1:
        raise ValueError(
            f"function_values should be a 1d array. Got {function_values.shape=}"
        )
    if point_cloud.shape[0] != function_values.shape[0]:
        raise ValueError(
            f"point_cloud and function_values should have same number of points. "
            f"Got {point_cloud.shape[0]} and {function_values.shape[0]}."
        )
    st = SimplexTreeMulti(num_parameters=2, dtype=dtype)
    if verbose:
        _mp_logs.log_verbose(
            f"[multipers.backends] backend=function_delaunay mode=cpp_interface degree=-1 multi_chunk=False recover_ids={recover_ids}",
            enabled=verbose,
        )
    return _function_delaunay_interface.function_delaunay_to_simplextree(
        st,
        point_cloud,
        function_values,
        recover_ids,
        verbose,
    )


def _rhomboid_tiling_to_slicer(
    slicer,
    point_cloud: np.ndarray,
    k_max,
    degree=-1,
    clear: bool = True,
    verbose=False,
):
    del clear
    import multipers._rhomboid_tiling_interface as _rhomboid_tiling_interface

    if not _rhomboid_tiling_interface._is_available():
        raise RuntimeError(
            "rhomboid_tiling interface is not available in this build. "
            "Rebuild multipers with rhomboid_tiling support to enable this backend."
        )

    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    if point_cloud.ndim != 2 or point_cloud.shape[1] not in [2, 3]:
        raise ValueError(
            f"point_cloud should be a 2d array of shape (-,2) or (-,3). Got {point_cloud.shape=}"
        )

    with _mp_logs.timings(
        "rhomboid_tiling",
        enabled=verbose,
        details={"backend": "rhomboid_tiling", "mode": "cpp_interface", "degree": degree, "k_max": k_max},
    ) as timing:
        out = _rhomboid_tiling_interface.rhomboid_tiling_to_slicer(
            slicer,
            point_cloud,
            k_max,
            degree,
            verbose,
        )
        timing.substep("cpp_call")
        return out


def KDE(bandwidth, kernel, return_log):
    if find_spec("pykeops") is not None and check_keops():
        from multipers.filtrations.density import KDE as _KDE

        return _KDE(bandwidth=bandwidth, kernel=kernel, return_log=return_log)

    from sklearn.neighbors import KernelDensity

    _mp_logs.warn_fallback("pykeops not found. Falling back to sklearn.")
    if not return_log:
        raise ValueError("Sklearn returns log-density.")
    return KernelDensity(bandwidth=bandwidth, kernel=kernel)


def RipsLowerstar(
    *,
    points: Optional[ArrayLike] = None,
    distance_matrix: Optional[ArrayLike] = None,
    function: Optional[ArrayLike] = None,
    threshold_radius: Optional[float] = None,
    sparse: Optional[float] = None,
    verbose: bool = False,
):
    """
    Computes the Rips complex, with the usual rips filtration as a first parameter,
    and the lower star multi filtration as other parameter.

    Input:
     - points or distance_matrix: ArrayLike
     - function : ArrayLike of shape (num_data, num_parameters -1)
     - threshold_radius:  max edge length of the rips. Defaults at min(max(distance_matrix, axis=1)).
    """
    with _mp_logs.timings(
        "RipsLowerstar",
        enabled=verbose,
        details={
            "backend": "gudhi",
            "mode": "python",
            "input": "distance_matrix" if distance_matrix is not None else "points",
            "sparse": bool(sparse),
        },
    ) as timing:
        if points is None and distance_matrix is None:
            raise ValueError("`points` or `distance_matrix` has to be given.")

        import gudhi as gd
        from multipers.simplex_tree_multi import SimplexTreeMulti

        if distance_matrix is not None:
            api = api_from_tensor(distance_matrix)
            D = api.astensor(distance_matrix)
        else:
            api = api_from_tensor(points)
            points = api.astensor(points)
            D = api.cdist(points, points)  # this may be slow...
        timing.substep("resolved_distance_matrix")

        if threshold_radius is None:
            threshold_radius = api.min(api.maxvalues(D, axis=1))
        if sparse:
            _mp_logs.ExperimentalWarning("Sparse-RipsLowerstar has no known good property.")
            st = gd.RipsComplex(
                distance_matrix=api.asnumpy(D),
                max_edge_length=threshold_radius,
                sparse=sparse,
            ).create_simplex_tree()
        else:
            st = gd.SimplexTree.create_from_array(
                api.asnumpy(D), max_filtration=threshold_radius
            )
        timing.substep("built_rips")
        if function is None:
            return SimplexTreeMulti(st, num_parameters=1)

        function = api.astensor(function)
        if function.ndim == 1:
            function = function[:, None]
        if function.ndim != 2:
            raise ValueError(
                f"""
                `function.ndim` should be 0 or 1 . Got {function.ndim=}.{function=}
                """
            )
        num_parameters = function.shape[1] + 1
        st = SimplexTreeMulti(st, num_parameters=num_parameters)
        for i in range(function.shape[1]):
            st.fill_lowerstar(api.asnumpy(function[:, i]), parameter=1 + i)
        timing.substep("applied_lowerstar")
        if api.has_grad(D) or api.has_grad(function):
            from multipers.grids import compute_grid

            filtration_values = [D.ravel(), *[f for f in function.T]]
            grid = compute_grid(filtration_values)
            st = st.grid_squeeze(grid)
            st._clean_filtration_grid()
            timing.substep("squeezed_grad_grid")
        return st


def RipsCodensity(
    points: ArrayLike,
    bandwidth: Optional[float] = None,
    *,
    return_log: bool = True,
    dtm_mass: Optional[float] = None,
    kernel: available_kernels = "gaussian",
    threshold_radius: Optional[float] = None,
    sparse: Optional[float] = None,
):
    """
    Computes the Rips density filtration.
    """
    if bandwidth is not None and dtm_mass is not None:
        raise ValueError("Density estimation is either via kernels or dtm.")
    if bandwidth is not None:
        kde = KDE(bandwidth=bandwidth, kernel=kernel, return_log=return_log)
        f = -kde.fit(points).score_samples(points)
    elif dtm_mass is not None:
        f = DTM(masses=[dtm_mass]).fit(points).score_samples(points)[0]
    else:
        raise ValueError("Bandwidth or DTM mass has to be given.")
    return RipsLowerstar(
        points=points, function=f, threshold_radius=threshold_radius, sparse=sparse
    )


def DelaunayLowerstar(
    points: ArrayLike,
    function: ArrayLike,
    *,
    distance_matrix: Optional[ArrayLike] = None,
    threshold_radius: Optional[float] = None,
    reduce_degree: int = -1,
    vineyard: Optional[bool] = None,
    dtype=np.float64,
    verbose: bool = False,
    clear: bool = True,
    flagify: bool = False,
    recover_ids: bool = False,
):
    """
    Computes the Function Delaunay bifiltration. Similar to RipsLowerstar, but most suited for low-dimensional euclidean data.
    See [Delaunay bifiltrations of functions on point clouds, Alonso et al] https://doi.org/10.1137/1.9781611977912.173

    Input:
     - points or distance_matrix: ArrayLike
     - function : ArrayLike of shape (num_data, )
     - threshold_radius:  max edge length of the rips. Defaults at min(max(distance_matrix, axis=1)).
    """
    import multipers

    with _mp_logs.timings(
        "DelaunayLowerstar",
        enabled=verbose,
        details={
            "backend": "function_delaunay",
            "mode": "cpp_interface",
            "reduce_degree": reduce_degree,
            "flagify": flagify,
        },
    ) as timing:
        if distance_matrix is not None:
            raise ValueError("Delaunay cannot be built from distance matrices")
        if threshold_radius is not None:
            raise NotImplementedError("Delaunay with threshold not implemented yet.")
        api = api_from_tensors(points, function)
        timing.substep("resolved_backend")
        if not flagify and (api.has_grad(points) or api.has_grad(function)):
            _mp_logs.warn_autodiff(
                "Cannot keep points gradient unless using `flagify=True`."
            )
        points = api.astensor(points)
        function = api.astensor(function).squeeze()
        timing.substep("converted_inputs")
        if function.ndim != 1:
            raise ValueError(
                "Delaunay Lowerstar is only compatible with 1 additional parameter."
            )
        points_np = api.asnumpy(points)
        function_np = api.asnumpy(function)
        degree = -1 if flagify else reduce_degree
        if degree < 0:
            slicer = _function_delaunay_presentation_to_simplextree(
                points_np,
                function_np,
                verbose=verbose,
                clear=clear,
                dtype=dtype,
                recover_ids=recover_ids,
            )
        else:
            slicer = multipers.Slicer(None, backend=None, vineyard=vineyard, dtype=dtype)
            slicer = _function_delaunay_presentation_to_slicer(
                slicer,
                points_np,
                function_np,
                degree=degree,
                verbose=verbose,
                clear=clear,
                recover_ids=recover_ids,
            )
            slicer.minpres_degree = degree
        timing.substep("built_function_delaunay")
        if flagify:
            from multipers.simplex_tree_multi import is_simplextree_multi
            from multipers.slicer import to_simplextree

            max_dim = -1 if reduce_degree == -1 else reduce_degree + 1
            with timing.step("flagify") as flagify_timing:
                if is_simplextree_multi(slicer):
                    slicer = slicer.copy()
                    if max_dim >= 0:
                        slicer.prune_above_dimension(max_dim)
                    flagify_timing.substep("copied_pruned_simplextree")
                else:
                    slicer = to_simplextree(slicer, max_dim=max_dim)
                    flagify_timing.substep("converted_slicer_to_simplextree")
                slicer.flagify(2)
                flagify_timing.substep("flagify")

                if api.has_grad(points) or api.has_grad(function):
                    distances = api.pdist(points) / 2
                    zero = api.zeros(1, dtype=distances.dtype)
                    zero = api.to_device(zero, api.device(distances))
                    distance_values = api.cat([distances, zero])
                    grid = get_exact_grid([distance_values, function], api=api)
                    flagify_timing.substep("computed_exact_grid")
                    slicer = slicer.grid_squeeze(grid)
                    flagify_timing.substep("grid_squeezed")
                    slicer = slicer._clean_filtration_grid()
                    flagify_timing.substep("cleaned_grid")
                if reduce_degree >= 0:
                    from multipers import Slicer

                    slicer = Slicer(slicer)
                    flagify_timing.substep("converted_back_to_slicer")

        if reduce_degree >= 0:
            # Force resolution to avoid confusion with hilbert.
            slicer = slicer.minpres(degree=reduce_degree, force=True)
            timing.substep("computed_minpres")

        return slicer
def _AlphaLowerstar(
    points: ArrayLike,
    function: ArrayLike,
    *,
    threshold_radius: Optional[float] = None,
):
    """
    Computes the Alpha complex, with the usual alpha filtration as a first parameter,
    and the lower star multi filtration as other parameter.
    """

    _mp_logs.ExperimentalWarning("Alpha-Lowerstar has no known good property.")
    import gudhi as gd
    from multipers.simplex_tree_multi import SimplexTreeMulti

    api = api_from_tensor(points)
    points = api.astensor(points)
    function = api.astensor(function)
    if function.ndim == 1:
        function = function[:, None]

    if threshold_radius is None:
        threshold_radius = np.inf

    st = gd.AlphaComplex(points=api.asnumpy(points)).create_simplex_tree(
        max_alpha_square=threshold_radius
    )

    st = SimplexTreeMulti(st, num_parameters=function.shape[1] + 1)
    for i in range(function.shape[1]):
        st.fill_lowerstar(api.asnumpy(function[:, i]), parameter=1 + i)

    if api.has_grad(points) or api.has_grad(function):
        D = api.cdist(points, points) ** 2

        grid = compute_grid([D.ravel(), *filtrations.T])
        st = st.grid_squeeze(grid)
        st._clean_filtration_grid()
    return st


def _AlphaCodensity(
    points: ArrayLike,
    bandwidth: Optional[float] = None,
    *,
    return_log: bool = True,
    dtm_mass: Optional[float] = None,
    kernel: available_kernels = "gaussian",
    threshold_radius: Optional[float] = None,
):
    """
    Computes the Alpha density filtration:
        - complex is given by the delaunay,
        - first parameter is alpha
        - the second is given by a density estimation
    """
    if bandwidth is not None and dtm_mass is not None:
        raise ValueError("Density estimation is either via kernels or dtm.")
    if bandwidth is not None:
        kde = KDE(bandwidth=bandwidth, kernel=kernel, return_log=return_log)
        f = -kde.fit(points).score_samples(points)
    elif dtm_mass is not None:
        f = DTM(masses=[dtm_mass]).fit(points).score_samples(points)[0]
    else:
        raise ValueError("Bandwidth or DTM mass has to be given.")
    return _AlphaLowerstar(points=points, function=f, threshold_radius=threshold_radius)


def DelaunayCodensity(
    points: ArrayLike,
    bandwidth: Optional[float] = None,
    *,
    return_log: bool = True,
    dtm_mass: Optional[float] = None,
    kernel: available_kernels = "gaussian",
    threshold_radius: Optional[float] = None,
    reduce_degree: int = -1,
    vineyard: Optional[bool] = None,
    dtype=np.float64,
    verbose: bool = False,
    clear: bool = True,
    flagify: bool = False,
):
    """
    TODO
    """
    if bandwidth is not None and dtm_mass is not None:
        raise ValueError("Density estimation is either via kernels or dtm.")
    if bandwidth is not None:
        kde = KDE(bandwidth=bandwidth, kernel=kernel, return_log=return_log)
        f = -kde.fit(points).score_samples(points)
    elif dtm_mass is not None:
        f = DTM(masses=[dtm_mass]).fit(points).score_samples(points)[0]
    else:
        raise ValueError("Bandwidth or DTM mass has to be given.")
    return DelaunayLowerstar(
        points=points,
        function=f,
        threshold_radius=threshold_radius,
        reduce_degree=reduce_degree,
        vineyard=vineyard,
        dtype=dtype,
        verbose=verbose,
        clear=clear,
        flagify=flagify,
    )


def Cubical(image: ArrayLike, **slicer_kwargs):
    """
    Computes the cubical filtration of an image.
    The last axis dimention is interpreted as the number of parameters.

    Input:
     - image: ArrayLike of shape (*image_resolution, num_parameters)
     - ** args : specify non-default slicer parameters
    """
    from multipers._slicer_algorithms import from_bitmap

    api = api_from_tensor(image)
    image = api.astensor(image)
    if api.has_grad(image):
        img2 = image.reshape(-1, image.shape[-1]).T
        grid = compute_grid(img2)
        coord_img = np.empty(image.shape, dtype=np.int32)
        slice_shape = image.shape[:-1]
        for i in range(image.shape[-1]):
            coord_img[..., i] = np.searchsorted(
                api.asnumpy(grid[i]),
                api.asnumpy(image[..., i]).reshape(-1),
            ).reshape(slice_shape)
        slicer = from_bitmap(coord_img, **slicer_kwargs)
        slicer.filtration_grid = grid
        slicer._clean_filtration_grid()
        return slicer

    return from_bitmap(image, **slicer_kwargs)


def DegreeRips(
    *,
    simplex_tree=None,
    points=None,
    distance_matrix=None,
    ks=None,
    threshold_radius=None,
    num=None,
    squeeze_strategy="exact",
    squeeze_resolution=None,
    squeeze: bool = True,
    verbose: bool = False,
):
    """
    The DegreeRips filtration.
    """

    with _mp_logs.timings(
        "DegreeRips",
        enabled=verbose,
        details={
            "backend": "gudhi",
            "mode": "python",
            "input": "simplex_tree" if simplex_tree is not None else (
                "distance_matrix" if distance_matrix is not None else "points"
            ),
            "squeeze": squeeze,
        },
    ) as timing:
        import gudhi as gd
        from multipers.simplex_tree_multi import SimplexTreeMulti

        if simplex_tree is None:
            if distance_matrix is None:
                if points is None:
                    raise ValueError(
                        "A simplextree, a distance matrix or a point cloud has to be given."
                    )
                api = api_from_tensor(points)
                points = api.astensor(points)
                D = api.cdist(points, points)
            else:
                api = api_from_tensor(distance_matrix)
                D = api.astensor(distance_matrix)

            if threshold_radius is None:
                threshold_radius = api.min(api.maxvalues(D, axis=1))
            st = gd.SimplexTree.create_from_array(
                api.asnumpy(D), max_filtration=threshold_radius
            )
            rips_filtration = api.unique(D.ravel())
        else:
            import multipers.array_api.numpy as npapi

            if not isinstance(simplex_tree, gd.SimplexTree):
                raise ValueError(
                    f"`simplex_tree` has to be a gudhi SimplexTree. Got {simplex_tree=}."
                )
            st = simplex_tree
            api = npapi
            rips_filtration = None
        timing.substep("built_rips")

        if ks is None or rips_filtration is None:
            _mp_logs.warn_copy(
                "Had to copy the rips to infer the `degrees` or recover the 1st filtration parameter."
            )
            _temp_st = SimplexTreeMulti(
                st, num_parameters=1
            )  # Gudhi is missing some functionality
            if ks is None:
                max_degree = (
                    np.bincount(_temp_st.get_simplices_of_dimension(1).ravel()).max() // 2
                )
                ks = (
                    np.arange(1, max_degree + 1)
                    if num is None
                    else np.unique(np.linspace(1, max_degree, num, dtype=np.int32))
                )
                ks = api.copy(api.from_numpy(ks))
            if rips_filtration is None:
                rips_filtration = compute_grid(_temp_st)[0]
            timing.substep("recovered_degree_axis")

        ks_np = np.asarray(ks, dtype=np.int32)
        if ks_np.ndim != 1:
            raise ValueError(
                f"`ks` must be a 1D sequence of positive sorted integers. Got shape {ks_np.shape}."
            )
        if ks_np.size == 0:
            raise ValueError("`ks` must contain at least one value.")
        if np.any(ks_np <= 0):
            raise ValueError(
                "`ks` must contain strictly positive degree indices. "
                "DegreeRips uses 1-based degree indexing."
            )
        if np.any(ks_np[1:] < ks_np[:-1]):
            raise ValueError("`ks` must be sorted in nondecreasing order.")

        from multipers.function_rips import get_degree_rips

        st_multi = get_degree_rips(st, degrees=ks_np)
        timing.substep("built_degree_rips")
        if squeeze:
            ks_grid = api.copy(api.from_numpy(ks_np))
            F = [rips_filtration, api.astype(ks_grid, rips_filtration.dtype)]
            F = compute_grid(F, strategy=squeeze_strategy, resolution=squeeze_resolution)
            st_multi = st_multi.grid_squeeze(F)
            st_multi.filtration_grid = (F[0], F[1] - F[1][-1])  # degrees are negative
            timing.substep("squeezed_grid")
        return st_multi


def CoreDelaunay(
    points: ArrayLike,
    *,
    beta: float = 1.0,
    ks: Optional[Sequence[int]] = None,
    precision: str = "safe",
    verbose: bool = False,
    max_alpha_square: float = float("inf"),
    positive_degree: bool = False,
) -> SimplexTreeMulti_type:
    """
    Computes the Delaunay core bifiltration of a point cloud presented in the paper "Core Bifiltration" https://arxiv.org/abs/2405.01214, and returns the (multi-critical) bifiltration as a SimplexTreeMulti. The Delaunay core bifiltration is an alpha complex version of the core bifiltration which is smaller in size. Moreover, along the horizontal line k=1, the Delaunay core bifiltration is identical to the alpha complex.

    Input:
     - points: The point cloud as an ArrayLike of shape (n, d) where n is the number of points and d is the dimension of the points.
     - beta: The beta parameter for the Delaunay Core Bifiltration (default 1.0).
     - ks: The list of k-values to include in the bifiltration (default None). If None, the k-values are set to [1, 2, ..., n] where n is the number of points in the point cloud. For large point clouds, it is recommended to set ks to a smaller list of k-values to reduce computation time. The values in ks must all be integers, positive, and less than or equal to the number of points in the point cloud.
     - precision: The precision of the computation of the AlphaComplex, one of ['safe', 'exact', 'fast'] (default 'safe'). See the GUDHI documentation for more information.
     - verbose: Whether to print progress messages (default False).
     - max_alpha_square: The maximum squared alpha value to consider when createing the alpha complex (default inf). See the GUDHI documentation for more information.
    """
    import gudhi as gd
    from scipy.spatial import KDTree
    from multipers.simplex_tree_multi import SimplexTreeMulti

    points = np.asarray(points)
    if ks is None:
        ks = np.arange(1, len(points) + 1)
    else:
        ks = np.asarray(ks, dtype=int)
    ks: np.ndarray

    if len(ks) == 0:
        raise ValueError("The parameter ks must contain at least one value.")
    if not np.all(ks > 0):
        raise ValueError("All values in ks must be positive.")
    if not np.all(ks <= len(points)):
        raise ValueError(
            "All values in ks must be less than or equal to the number of points in the point cloud."
        )
    if len(points) == 0:
        raise ValueError("The point cloud must contain at least one point.")
    if points.ndim != 2:
        raise ValueError(f"The point cloud must be a 2D array, got {points.ndim}D.")
    if beta < 0:
        raise ValueError(f"The parameter beta must be positive, got {beta}.")
    if precision not in ["safe", "exact", "fast"]:
        raise ValueError(
            "The parameter precision must be one of ['safe', 'exact', 'fast'], "
            f"got {precision}."
        )

    if verbose:
        print(
            f"""Computing the Delaunay Core Bifiltration
            of {len(points)} points in dimension {points.shape[1]}
            with parameters:
            """
        )
        print(f"\tbeta = {beta}")
        print(f"\tks = {ks}")

    if verbose:
        print("Building the alpha complex...")
    alpha_complex = gd.AlphaComplex(
        points=points, precision=precision
    ).create_simplex_tree(max_alpha_square=max_alpha_square)

    if verbose:
        print("Computing the k-nearest neighbor distances...")
    knn_distances = KDTree(points).query(points, k=ks)[0]

    max_dim = alpha_complex.dimension()
    vertex_arrays_in_dimension = [[] for _ in range(max_dim + 1)]
    squared_alphas_in_dimension = [[] for _ in range(max_dim + 1)]
    for simplex, alpha_squared in alpha_complex.get_simplices():
        dim = len(simplex) - 1
        squared_alphas_in_dimension[dim].append(alpha_squared)
        vertex_arrays_in_dimension[dim].append(simplex)

    alphas_in_dimension = [
        np.sqrt(np.array(alpha_squared, dtype=np.float64))
        for alpha_squared in squared_alphas_in_dimension
    ]
    vertex_arrays_in_dimension = [
        np.array(vertex_array, dtype=np.int32)
        for vertex_array in vertex_arrays_in_dimension
    ]

    simplex_tree_multi = SimplexTreeMulti(
        num_parameters=2, kcritical=True, dtype=np.float64
    )

    for dim, (vertex_array, alphas) in enumerate(
        zip(vertex_arrays_in_dimension, alphas_in_dimension)
    ):
        num_simplices = len(vertex_array)
        if verbose:
            print(
                f"""
                Inserting {num_simplices} simplices of dimension {dim}
                ({num_simplices * len(ks)} birth values)...
                """
            )
        max_knn_distances = np.max(knn_distances[vertex_array], axis=1)
        critical_radii = np.maximum(alphas[:, None], beta * max_knn_distances)
        filtrations = np.stack(
            (
                critical_radii,
                (ks[-1] - ks if positive_degree else -ks)
                * np.ones_like(critical_radii),
            ),
            axis=-1,
        )
        simplex_tree_multi.insert_batch(vertex_array.T, filtrations)

    if verbose:
        print("Done computing the Delaunay Core Bifiltration.")

    return simplex_tree_multi


def RhomboidBifiltration(
    x,
    k_max: int,
    degree: int,
    verbose: bool = False,
):
    """
    Rhomboid Tiling bifiltration.
    This (1-critical) bifiltration is quasi-isomorphic to the (multi-critical) multicover bifiltration.
    From [Computing the Multicover Bifiltration](https://doi.org/10.1007/s00454-022-00476-8), whose code
    can be found here: https://github.com/geoo89/rhomboidtiling

    Parameters
     - x: 2d or 3d point cloud, of shape `(num_points,dimension)`.
     - k_max(int): maximum number of cover to consider
     - degree: dimension to consider
     - verbose:bool
    """
    from multipers import Slicer

    api = api_from_tensor(x)
    if api.has_grad(x):
        _mp_logs.warn_autodiff(
            "Found a gradient in input, which cannot be processed by RhomboidBifiltration."
        )
    x = api.asnumpy(x)
    if x.ndim not in (2, 3):
        raise ValueError("Only 2-3D dimensional point cloud are supported.")
    out = Slicer()
    out = _rhomboid_tiling_to_slicer(
        slicer=out,
        point_cloud=x,
        k_max=k_max,
        verbose=verbose,
        degree=degree,
    )

    return out
