from collections.abc import Sequence
from typing import Optional
from warnings import warn

import gudhi as gd
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from multipers.array_api import api_from_tensor, api_from_tensors
from multipers.filtrations.density import DTM, available_kernels
from multipers.grids import compute_grid
from multipers.simplex_tree_multi import SimplexTreeMulti, SimplexTreeMulti_type

try:
    import pykeops

    from multipers.filtrations.density import KDE
except ImportError:
    from sklearn.neighbors import KernelDensity

    warn("pykeops not found. Falling back to sklearn.")

    def KDE(bandwidth, kernel, return_log):
        assert return_log, "Sklearn returns log-density."
        return KernelDensity(bandwidth=bandwidth, kernel=kernel)


def RipsLowerstar(
    *,
    points: Optional[ArrayLike] = None,
    distance_matrix: Optional[ArrayLike] = None,
    function: Optional[ArrayLike] = None,
    threshold_radius: Optional[float] = None,
):
    """
    Computes the Rips complex, with the usual rips filtration as a first parameter,
    and the lower star multi filtration as other parameter.

    Input:
     - points or distance_matrix: ArrayLike
     - function : ArrayLike of shape (num_data, num_parameters -1)
     - threshold_radius:  max edge length of the rips. Defaults at min(max(distance_matrix, axis=1)).
    """
    assert (
        points is not None or distance_matrix is not None
    ), "`points` or `distance_matrix` has to be given."
    if distance_matrix is None:
        api = api_from_tensor(points)
        points = api.astensor(points)
        D = api.cdist(points, points)  # this may be slow...
    else:
        api = api_from_tensor(distance_matrix)
        D = api.astensor(distance_matrix)

    if threshold_radius is None:
        threshold_radius = api.min(api.maxvalues(D, axis=1))
    st = gd.SimplexTree.create_from_array(
        api.asnumpy(D), max_filtration=threshold_radius
    )
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
    if api.has_grad(D) or api.has_grad(function):
        from multipers.grids import compute_grid
        filtration_values = [D.ravel(), *[f for f in function.T]]
        grid = compute_grid(filtration_values)
        st = st.grid_squeeze(grid)
        st._clean_filtration_grid()
    return st


def RipsCodensity(
    points: ArrayLike,
    bandwidth: Optional[float] = None,
    *,
    return_log: bool = True,
    dtm_mass: Optional[float] = None,
    kernel: available_kernels = "gaussian",
    threshold_radius: Optional[float] = None,
):
    """
    Computes the Rips density filtration.
    """
    assert (
        bandwidth is None or dtm_mass is None
    ), "Density estimation is either via kernels or dtm."
    if bandwidth is not None:
        kde = KDE(bandwidth=bandwidth, kernel=kernel, return_log=return_log)
        f = -kde.fit(points).score_samples(points)
    elif dtm_mass is not None:
        f = DTM(masses=[dtm_mass]).fit(points).score_samples(points)[0]
    else:
        raise ValueError("Bandwidth or DTM mass has to be given.")
    return RipsLowerstar(points=points, function=f, threshold_radius=threshold_radius)


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
):
    """
    Computes the Function Delaunay bifiltration. Similar to RipsLowerstar, but most suited for low-dimensional euclidean data.
    See [Delaunay bifiltrations of functions on point clouds, Alonso et al] https://doi.org/10.1137/1.9781611977912.173

    Input:
     - points or distance_matrix: ArrayLike
     - function : ArrayLike of shape (num_data, )
     - threshold_radius:  max edge length of the rips. Defaults at min(max(distance_matrix, axis=1)).
    """
    from multipers.slicer import from_function_delaunay

    if flagify and reduce_degree >= 0:
        raise ValueError(
            "Got {reduce_degree=} and {flagify=}. Cannot flagify with reduce degree."
        )
    assert distance_matrix is None, "Delaunay cannot be built from distance matrices"
    if threshold_radius is not None:
        raise NotImplementedError("Delaunay with threshold not implemented yet.")
    api = api_from_tensors(points, function)
    if not flagify and (api.has_grad(points) or api.has_grad(function)):
        warn("Cannot keep points gradient unless using `flagify=True`.")
    points = api.astensor(points)
    function = api.astensor(function).squeeze()
    assert (
        function.ndim == 1
    ), "Delaunay Lowerstar is only compatible with 1 additional parameter."
    slicer = from_function_delaunay(
        api.asnumpy(points),
        api.asnumpy(function),
        degree=reduce_degree,
        vineyard=vineyard,
        dtype=dtype,
        verbose=verbose,
        clear=clear,
    )
    if reduce_degree >= 0:
        # Force resolution to avoid confusion with hilbert.
        slicer = slicer.minpres(degree=reduce_degree, force=True)
    if flagify:
        from multipers.slicer import to_simplextree

        slicer = to_simplextree(slicer)
        slicer.flagify(2)

        if api.has_grad(points) or api.has_grad(function):
            distances = api.cdist(points, points) / 2
            grid = compute_grid([distances.ravel(), function])
            slicer = slicer.grid_squeeze(grid)
            slicer = slicer._clean_filtration_grid()

    return slicer


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
    assert (
        bandwidth is None or dtm_mass is None
    ), "Density estimation is either via kernels or dtm."
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
    from multipers.slicer import from_bitmap

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


def DegreeRips(*, points=None, distance_matrix=None, ks=None, threshold_radius=None):
    """
    The DegreeRips filtration.
    """

    raise NotImplementedError("Use the default implentation ftm.")


def CoreDelaunay(
    points: ArrayLike,
    *,
    beta: float = 1.0,
    ks: Optional[Sequence[int]] = None,
    precision: str = "safe",
    verbose: bool = False,
    max_alpha_square: float = float("inf"),
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
    points = np.asarray(points)
    if ks is None:
        ks = np.arange(1, len(points) + 1)
    else:
        ks = np.asarray(ks, dtype=int)
    ks: np.ndarray

    assert len(ks) > 0, "The parameter ks must contain at least one value."
    assert np.all(ks > 0), "All values in ks must be positive."
    assert np.all(
        ks <= len(points)
    ), "All values in ks must be less than or equal to the number of points in the point cloud."
    assert len(points) > 0, "The point cloud must contain at least one point."
    assert points.ndim == 2, f"The point cloud must be a 2D array, got {points.ndim}D."
    assert beta >= 0, f"The parameter beta must be positive, got {beta}."
    assert precision in [
        "safe",
        "exact",
        "fast",
    ], f"""
    The parameter precision must be one of ['safe', 'exact', 'fast'],
    got {precision}.
    """

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
            (critical_radii, -ks * np.ones_like(critical_radii)), axis=-1
        )
        simplex_tree_multi.insert_batch(vertex_array.T, filtrations)

    if verbose:
        print("Done computing the Delaunay Core Bifiltration.")

    return simplex_tree_multi
