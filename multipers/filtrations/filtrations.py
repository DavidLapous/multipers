from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from multipers.array_api import api_from_tensor, api_from_tensors, check_keops
import multipers.logs as _mp_logs
from multipers.filtrations.density import DTM, available_kernels
from multipers.grids import compute_grid, get_exact_grid, push_to_grid

from multipers.simplex_tree_multi import SimplexTreeMulti_type


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

    _rhomboid_tiling_interface.require()

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


def _compute_codensity(
    points: ArrayLike,
    *,
    bandwidth: Optional[float] = None,
    dtm_mass: Optional[float] = None,
    knn: Optional[int] = None,
    kernel: available_kernels = "gaussian",
    return_log: bool = True,
):
    if sum(x is not None for x in (bandwidth, dtm_mass, knn)) != 1:
        raise ValueError("Density estimation is either via kernels, dtm, or knn.")
    if bandwidth is not None:
        if find_spec("pykeops") is not None and check_keops():
            from multipers.filtrations.density import KDE

            kde = KDE(bandwidth=bandwidth, kernel=kernel, return_log=return_log)
        else:
            from sklearn.neighbors import KernelDensity

            _mp_logs.warn_fallback("pykeops not found. Falling back to sklearn.")
            if not return_log:
                raise ValueError("Sklearn returns log-density.")
            kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        return -kde.fit(points).score_samples(points)
    if dtm_mass is not None:
        return DTM(masses=[dtm_mass]).fit(points).score_samples(points)[0]

    if knn < 1:
        raise ValueError("kNN parameter should be a positive integer.")
    if knn > len(points):
        raise ValueError("kNN parameter should be at most the sample size.")

    from multipers.filtrations.density import KNNmean

    return KNNmean(k=knn).fit(points).score_samples(points)


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
    Build Rips-lower-star bifiltration.

    First parameter is usual Rips scale. Remaining parameter(s) are lower-star
    coordinates induced by vertex values in `function`.

    For scalar `f : X -> R`, intended continuum model is sublevel-offset family

    ```
    O_{r,s}(f) = union_{x in X, f(x) <= s} B(x, r).
    ```

    Its discrete Rips analogue is

    ```
    R_{r,s}(f) = R_r(X_s),
    X_s = {x in X | f(x) <= s},
    ```

    where `R_r(X_s)` is the Rips complex at scale `r` on the sublevel vertex
    set. This constructor implements that Rips proxy and then applies lower-star
    extension along each extra function coordinate.

    Parameters
    ----------
    points, distance_matrix:
        Provide either point cloud or pairwise distance matrix.
    function:
        Vertex function values of shape `(n,)` or `(n, q)`. If omitted, this
        returns ordinary 1-parameter Rips complex as `SimplexTreeMulti`.
    threshold_radius:
        Maximal Rips edge length. Defaults to `min_i max_j D[i,j]`.
    sparse:
        Optional Gudhi sparse-Rips parameter.
    verbose:
        Emit timing logs.

    Notes
    -----
    If autodiff-tracked inputs are detected, output is grid-squeezed so the
    filtration grid stores the exact differentiable coordinate values.
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
    knn: Optional[int] = None,
    kernel: available_kernels = "gaussian",
    threshold_radius: Optional[float] = None,
    sparse: Optional[float] = None,
):
    """
    Build Rips-codensity bifiltration.

    This is `RipsLowerstar` where the extra coordinate is a codensity function
    estimated on the input points. First parameter is Rips scale; second is the
    estimated codensity value.

    Exactly one codensity model must be selected:

    - `bandwidth`: kernel density estimate, returned as negative log-density,
    - `dtm_mass`: DTM-based codensity,
    - `knn`: k-nearest-neighbor mean-distance score.

    Parameters
    ----------
    points:
        Point cloud.
    bandwidth, dtm_mass, knn:
        Mutually exclusive codensity estimators.
    return_log:
        For kernel density, keep log-density convention before negation.
    kernel:
        Kernel name for KDE.
    threshold_radius:
        Maximal Rips edge length.
    sparse:
        Optional Gudhi sparse-Rips parameter.

    Notes
    -----
    This is often used as a geometry+density bifiltration: horizontal coordinate
    controls scale, vertical coordinate favors dense regions before sparse ones.
    """
    f = _compute_codensity(
        points,
        bandwidth=bandwidth,
        dtm_mass=dtm_mass,
        knn=knn,
        kernel=kernel,
        return_log=return_log,
    )
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
    interlevel: bool = False,
    recover_ids: Optional[bool] = None,
):
    """
    Build Delaunay-lower-star bifiltration for low-dimensional Euclidean data.

    Reference: Alonso et al., "Delaunay bifiltrations of functions on point
    clouds", https://doi.org/10.1137/1.9781611977912.173
    Bifunction/interlevel support follows Alonso et al., "Bifunction and
    Interlevel Delaunay Trifiltrations", https://doi.org/10.4230/LIPIcs.SoCG.2026.5

    For scalar `phi : X -> R`, intended continuum model is sublevel-offset family

    ```
    O_{r,s}(phi) = O_r(X_s),
    X_s = {x in X | phi(x) <= s}.
    ```

    For bifunction `phi : X -> R^2`, grades are `(geometric_scale, f1, f2)`.
    For `interlevel=True`, scalar values `f` are encoded as bifunction
    `(-f, f)`, so intervals `[a,b]` correspond to coordinates `(-a,b)`.

    Instead of using full Rips complex, this constructor uses incremental
    Delaunay structure and assigns each simplex a grade of the form
    `(geometric_scale, max_vertex_value, ...)`. This is typically much smaller
    than a Rips-based model on low-dimensional Euclidean point clouds.

    Parameters
    ----------
    points:
        Euclidean point cloud of shape `(n, d)`.
    function:
        Vertex values of shape `(n,)`, `(n, 1)`, or `(n, 2)`. Two columns build
        the bifunction Delaunay-Cech trifiltration.
    distance_matrix:
        Not supported here; Delaunay construction requires coordinates.
    threshold_radius:
        Currently unsupported.
    reduce_degree:
        If nonnegative, compute minimal presentation in that homological degree.
    vineyard:
        Passed through when a `Slicer` output is built.
    dtype:
        Numeric dtype for returned simplicial object / slicer.
    verbose:
        Emit timing / backend logs.
    clear:
        Compatibility flag, currently unused.
    flagify:
        Convert output to a flag complex after native construction. Required to
        preserve point gradients through geometric scale values.
    interlevel:
        If `True`, require scalar vertex values and encode them as `(-f, f)`.
    recover_ids:
        Recover original vertex ordering in native output. Defaults to `True`
        when `reduce_degree >= 0`.

    Notes
    -----
    When `reduce_degree` is left at its default value `-1`, this usually returns
    a `SimplexTreeMulti`. When `reduce_degree >= 0`, it returns a
    minimal-presentation `Slicer`.
    """
    import multipers
    import multipers._function_delaunay_interface as _function_delaunay_interface

    if recover_ids is None:
        recover_ids = reduce_degree >= 0

    with _mp_logs.timings(
        "DelaunayLowerstar",
        enabled=verbose,
        details={
            "backend": "function_delaunay",
            "mode": "cpp_interface",
            "reduce_degree": reduce_degree,
            "flagify": flagify,
            "interlevel": interlevel,
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
        function = api.astensor(function)
        timing.substep("converted_inputs")
        if function.ndim == 1:
            scalar_function = api.reshape(function, (-1,))
            function_matrix = (
                api.cat([(-scalar_function)[:, None], scalar_function[:, None]], 1)
                if interlevel
                else scalar_function[:, None]
            )
        elif function.ndim == 2:
            if function.shape[1] == 1:
                scalar_function = api.reshape(function, (-1,))
                function_matrix = (
                    api.cat([(-scalar_function)[:, None], scalar_function[:, None]], 1)
                    if interlevel
                    else function
                )
            elif function.shape[1] == 2:
                if interlevel:
                    raise ValueError("interlevel=True expects scalar function values.")
                function_matrix = function
            else:
                raise ValueError(
                    "Delaunay Lowerstar supports one or two function parameters."
                )
        else:
            raise ValueError("function should be a 1d or 2d array.")
        num_function_parameters = int(function_matrix.shape[1])
        if reduce_degree >= 0 and num_function_parameters > 1:
            raise NotImplementedError(
                "DelaunayLowerstar minimal presentations currently support only "
                "scalar function values. Build the 3-parameter complex with "
                "reduce_degree=-1 first."
            )
        points_np = np.ascontiguousarray(api.asnumpy(points), dtype=np.float64)
        function_np = np.ascontiguousarray(api.asnumpy(function_matrix), dtype=np.float64)
        if points_np.ndim != 2:
            raise ValueError(f"point_cloud should be a 2d array. Got {points_np.shape=}")
        if function_np.ndim != 2:
            raise ValueError(f"function should be a 2d array after normalization. Got {function_np.shape=}")
        if points_np.shape[0] != function_np.shape[0]:
            raise ValueError(
                f"point_cloud and function_values should have same number of points. "
                f"Got {points_np.shape[0]} and {function_np.shape[0]}."
            )
        _function_delaunay_interface.require()
        degree = -1 if flagify else reduce_degree
        if degree < 0 or recover_ids:
            from multipers.simplex_tree_multi import SimplexTreeMulti

            if verbose:
                _mp_logs.log_verbose(
                    f"[multipers.backends] backend=function_delaunay mode=cpp_interface degree=-1 multi_chunk=False recover_ids={recover_ids} function_parameters={num_function_parameters} interlevel={interlevel}",
                    enabled=verbose,
                )
            slicer = _function_delaunay_interface.function_delaunay_to_simplextree(
                SimplexTreeMulti(num_parameters=1 + num_function_parameters, dtype=dtype),
                points_np,
                function_np,
                recover_ids,
                verbose,
            )
            if degree >= 0:
                slicer = multipers.Slicer(slicer, vineyard=vineyard, dtype=dtype)
        else:
            slicer = multipers.Slicer(None, backend=None, vineyard=vineyard, dtype=dtype)
            if verbose:
                _mp_logs.log_verbose(
                    f"[multipers.backends] backend=function_delaunay mode=cpp_interface degree={degree} multi_chunk=False recover_ids={recover_ids} function_parameters={num_function_parameters} interlevel={interlevel}",
                    enabled=verbose,
                )
            slicer = _function_delaunay_interface.function_delaunay_to_slicer(
                slicer,
                points_np,
                function_np,
                degree,
                False,
                recover_ids,
                verbose,
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
                    flagify_timing.substep("copy+pruned")
                else:
                    slicer = to_simplextree(slicer, max_dim=max_dim)
                    flagify_timing.substep("converted back to a simplextree")
                slicer.flagify(2)
                flagify_timing.substep("flagify")

                if api.has_grad(points) or api.has_grad(function):
                    distances = api.pdist(points) / 2
                    zero = api.zeros(1, dtype=distances.dtype)
                    zero = api.to_device(zero, api.device(distances))
                    distance_values = api.cat([distances, zero])
                    function_columns = [
                        function_matrix[:, parameter]
                        for parameter in range(num_function_parameters)
                    ]
                    grid = get_exact_grid([distance_values, *function_columns], api=api)
                    flagify_timing.substep("recomputed pairwise dists")
                    slicer = slicer.grid_squeeze(grid)
                    flagify_timing.substep("stored cdist gradients")
                    slicer = slicer._clean_filtration_grid()
                    flagify_timing.substep("cleaned slicer")
                if reduce_degree >= 0:
                    from multipers import Slicer

                    slicer = Slicer(slicer)
                    flagify_timing.substep("converted to slicer for minpres")

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
        D = api.pdist(points, points) ** 2

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
    knn: Optional[int] = None,
    kernel: available_kernels = "gaussian",
    threshold_radius: Optional[float] = None,
):
    """
    Computes the Alpha density filtration:
        - complex is given by the delaunay,
        - first parameter is alpha
        - the second is given by a density estimation
    """
    f = _compute_codensity(
        points,
        bandwidth=bandwidth,
        dtm_mass=dtm_mass,
        knn=knn,
        kernel=kernel,
        return_log=return_log,
    )
    return _AlphaLowerstar(points=points, function=f, threshold_radius=threshold_radius)


def DelaunayCodensity(
    points: ArrayLike,
    bandwidth: Optional[float] = None,
    *,
    return_log: bool = True,
    dtm_mass: Optional[float] = None,
    knn: Optional[int] = None,
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
    Build Delaunay-codensity bifiltration.

    This is `DelaunayLowerstar` where the scalar lower-star coordinate is a
    codensity value estimated on the vertices. First parameter is Delaunay-based
    geometric scale, second is codensity.

    Exactly one codensity model must be selected:

    - `bandwidth`: kernel density estimate, used through negative log-density,
    - `dtm_mass`: DTM-based codensity,
    - `knn`: k-nearest-neighbor mean-distance score.

    Parameters
    ----------
    points:
        Euclidean point cloud.
    bandwidth, dtm_mass, knn:
        Mutually exclusive codensity estimators.
    return_log:
        For kernel density, keep log-density convention before negation.
    kernel:
        Kernel name for KDE.
    threshold_radius:
        Currently unsupported by `DelaunayLowerstar` backend.
    reduce_degree, vineyard, dtype, verbose, clear, flagify:
        Forwarded to `DelaunayLowerstar`.

    Notes
    -----
    Prefer this over `RipsCodensity` for low-dimensional Euclidean data when a
    smaller geometric complex is desirable.
    """
    f = _compute_codensity(
        points,
        bandwidth=bandwidth,
        dtm_mass=dtm_mass,
        knn=knn,
        kernel=kernel,
        return_log=return_log,
    )
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
    Build lower-star cubical filtration from image / volume data.

    The last axis is interpreted as the number of filtration parameters. For a
    scalar image `phi`, vertices inherit their pixel/voxel values and each
    higher-dimensional cubical cell is born at the maximum value of its
    vertices:

    ```
    birth(sigma) = max_{v vertex of sigma} phi(v).
    ```

    The resulting sublevel filtration is

    ```
    K_a = {sigma | birth(sigma) <= a}.
    ```

    For multi-parameter input, same max-vertex rule is applied coordinatewise.

    Parameters
    ----------
    image:
        Array of shape `(*image_resolution, num_parameters)`.
    **slicer_kwargs:
        Extra keyword arguments forwarded to `multipers.Slicer(...)` when
        constructing returned object.

    Notes
    -----
    If autodiff-tracked input is detected, image values are first pushed to a
    discrete filtration grid and that grid is attached to the returned object.
    """
    from multipers import Slicer, _slicer_nanobind as mps
    from multipers.slicer import available_dtype

    verbose = bool(slicer_kwargs.pop("verbose", False))
    api = api_from_tensor(image)
    image = api.astensor(image)
    has_grad = api.has_grad(image)

    with _mp_logs.timings(
        "Cubical",
        enabled=verbose,
        details={
            "backend": "bitmap",
            "mode": "python",
            "has_grad": has_grad,
        },
    ) as timing:
        grid = None
        bitmap = image
        if has_grad:
            img2 = image.reshape(-1, image.shape[-1]).T
            grid = compute_grid(img2)
            timing.substep("computed_autodiff_grid")
            bitmap = push_to_grid(
                image.reshape(-1, image.shape[-1]),
                grid,
                return_coordinate=True,
            ).reshape(image.shape).astype(np.int32, copy=False)
            timing.substep("pushed_to_grid")

        bitmap = np.asarray(bitmap) if isinstance(bitmap, np.ndarray) else api.asnumpy(bitmap)
        dtype = slicer_kwargs.get("dtype", bitmap.dtype)
        slicer_kwargs["dtype"] = dtype
        if bitmap.dtype != dtype:
            raise ValueError(f"Invalid type matching. Got {dtype=} and {bitmap.dtype=}.")

        _Slicer = Slicer(return_type_only=True, **slicer_kwargs)
        builder_name = f"_build_bitmap_{np.dtype(dtype).name.replace('float64', 'f64').replace('int32', 'i32')}"
        if not hasattr(mps, builder_name):
            raise ValueError(
                f"Invalid dtype. Got {bitmap.dtype=}, was expecting {available_dtype=}."
            )
        timing.substep("resolved_builder")

        flattened = np.ascontiguousarray(bitmap.reshape(-1, bitmap.shape[-1]))
        shape = np.ascontiguousarray(bitmap.shape[:-1], dtype=np.uint32)
        timing.substep("prepared_bitmap")
        base = getattr(mps, builder_name)(flattened, shape)
        slicer = base if type(base) is _Slicer else _Slicer(base)
        timing.substep("built_bitmap")

        if grid is not None:
            slicer.filtration_grid = grid
            slicer._clean_filtration_grid()
            timing.substep("attached_grid")
        return slicer


def DegreeRips(
    *,
    simplex_tree=None,
    points=None,
    distance_matrix=None,
    ks=None,
    threshold_radius=None,
    backend: str = "multipers",
    collapse: Optional[int] = None,
    num=None,
    squeeze_strategy="exact",
    squeeze_resolution=None,
    squeeze: bool = True,
    normalize: bool = False,
    verbose: bool = False,
):
    """
    Build degree-Rips bifiltration of point cloud, distance matrix, or Rips graph.

    This is 2-parameter filtration with:

    - first coordinate = usual Rips scale / edge length threshold,
    - second coordinate = graph-degree threshold.

    We use convention

    ```
    D(T)_k = maximum subcomplex of T whose vertices have
             degree at least k - 1 in the 1-skeleton of T.
    ```

    Applied to Rips filtration `R(X)`, degree-Rips bifiltration is `DR^u(X) = D(R(X))`.
    In this API, `ks` uses 0-based degree indexing: `ks=0` keeps all vertices,
    `ks=1` requires one incident edge, etc. So Python value `j` corresponds to
    mathematical threshold `k = j + 1` above.

    Returned object is k-critical bifiltered `SimplexTreeMulti` whose degree axis
    is stored in opposite order internally, then exposed through a sorted
    `filtration_grid` from lower to upper degree threshold. When `normalize=True`,
    degree labels are divided by number of vertices.

    Parameters
    ----------
    simplex_tree:
        Optional 1-parameter Gudhi `SimplexTree` used as Rips input. Only
        supported by backend=`"multipers"`.
    points:
        Point cloud. Used to build pairwise distances when `distance_matrix` and
        `simplex_tree` are not provided.
    distance_matrix:
        Square pairwise distance matrix.
    ks:
        Sorted nonnegative degree thresholds. If omitted, thresholds are inferred
        from input Rips graph for backend=`"multipers"`, or when `num` requests a
        sampled subset.
    threshold_radius:
        Maximal Rips edge length. Defaults to `min_i max_j D[i,j]`, matching
        other Rips constructors in this module.
    backend:
        `"multipers"` uses existing degree-Rips construction from
        `multipers.function_rips`. `"deg_rips"` uses native deg_rips backend and
        currently accepts only `points` or `distance_matrix` inputs.
    collapse:
        If set, force backend=`"deg_rips"` and run that many whole-edge and
        edge-copy filtration-domination reduction passes before clique expansion.
        This follows filtration-domination pruning from Alonso, Kerber, and
        Pritam, "Filtration-Domination in Bifiltered Graphs" (ALENEX 2023,
        doi:10.1137/1.9781611977561.ch3).
    num:
        When `ks` is omitted, sample `num` thresholds between `0` and maximal
        graph degree instead of using full range.
    squeeze_strategy, squeeze_resolution, squeeze:
        Control optional grid squeezing of returned bifiltration.
    normalize:
        Rescale exposed degree-axis labels by number of vertices.
    verbose:
        Emit timing / backend logs.

    Notes
    -----
    Degree-Rips is often much smaller than subdivision-based density-sensitive
    bifiltrations. A fixed `m`-skeleton has size `O(|X|^{m + 2})`, and after
    coarsening to a constant-size grid this becomes `O(|X|^{m + 1})`.

    The `"deg_rips"` backend wraps upstream project:
    https://bitbucket.org/mkerber/deg_rips
    """

    if collapse is not None:
        if collapse < 0:
            raise ValueError("`collapse` must be nonnegative or None.")
        backend = "deg_rips"

    backend = str(backend).lower()
    if backend not in {"multipers", "deg_rips"}:
        raise ValueError(
            f"Invalid DegreeRips backend {backend!r}. Expected 'multipers' or 'deg_rips'."
        )
    if backend == "deg_rips" and simplex_tree is not None:
        raise ValueError("The 'deg_rips' DegreeRips backend only supports points= or distance_matrix= for now.")

    with _mp_logs.timings(
        "DegreeRips",
        enabled=verbose,
        details={
            "backend": backend,
            "mode": "python",
            "input": "simplex_tree" if simplex_tree is not None else (
                "distance_matrix" if distance_matrix is not None else "points"
            ),
            "squeeze": squeeze,
            "normalize": normalize,
            "collapse": collapse,
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
                api.asnumpy(D), max_filtration=float(threshold_radius)
            )
            rips_filtration = api.unique(D.ravel())
            num_vertices = D.shape[0]
        else:
            import multipers.array_api.numpy as npapi

            if not isinstance(simplex_tree, gd.SimplexTree):
                raise ValueError(
                    f"`simplex_tree` has to be a gudhi SimplexTree. Got {simplex_tree=}."
                )
            st = simplex_tree
            api = npapi
            rips_filtration = None
            num_vertices = st.num_vertices()
        timing.substep("built_rips")

        should_infer_ks = ks is None and (backend == "multipers" or num is not None)
        should_recover_rips_filtration = rips_filtration is None
        if should_infer_ks or should_recover_rips_filtration:
            _mp_logs.warn_copy(
                "Had to copy the rips to infer the `degrees` or recover the 1st filtration parameter."
            )
            _temp_st = SimplexTreeMulti(
                st, num_parameters=1
            )  # Gudhi is missing some functionality
            if should_infer_ks:
                max_degree = (
                    np.bincount(_temp_st.get_simplices_of_dimension(1).ravel()).max() // 2
                )
                ks = (
                    np.arange(0, max_degree + 1)
                    if num is None
                    else np.unique(np.linspace(0, max_degree, num, dtype=np.int32))
                )
                ks = api.copy(api.from_numpy(ks))
            if should_recover_rips_filtration:
                rips_filtration = compute_grid(_temp_st)[0]
            timing.substep("recovered_degree_axis")

        if ks is None:
            ks_np = None
        else:
            ks_np = np.asarray(ks, dtype=np.int32)
            if ks_np.ndim != 1:
                raise ValueError(
                    f"`ks` must be a 1D sequence of positive sorted integers. Got shape {ks_np.shape}."
                )
            if ks_np.size == 0:
                raise ValueError("`ks` must contain at least one value.")
            if np.any(ks_np < 0):
                raise ValueError(
                    "`ks` must contain nonnegative vertex degree thresholds. "
                    "DegreeRips uses 0-based degree indexing."
                )
            if np.any(ks_np[1:] < ks_np[:-1]):
                raise ValueError("`ks` must be sorted in nondecreasing order.")

        if backend == "multipers":
            from multipers.function_rips import get_degree_rips

            assert ks_np is not None
            st_multi = get_degree_rips(st, degrees=ks_np)
        else:
            import multipers._deg_rips_interface as _deg_rips_interface

            _deg_rips_interface.require()
            st_multi = SimplexTreeMulti(
                num_parameters=2,
                dtype=np.float64,
                kcritical=True,
                ftype="Contiguous",
            )
            _deg_rips_interface.degree_rips_to_simplextree(
                st_multi,
                np.ascontiguousarray(api.asnumpy(D), dtype=np.float64),
                None if ks_np is None else np.ascontiguousarray(ks_np, dtype=np.int32),
                threshold_radius=float(threshold_radius),
                vanilla=collapse is None,
                vertex_domination=collapse is not None,
                whole_edge_iterations=0 if collapse is None else collapse,
                edge_copy_iterations=0 if collapse is None else collapse,
                use_domination_for_whole_edge_removal=collapse is not None,
                use_domination_for_edge_copy_removal=collapse is not None,
                min_scale=0.0,
                verbose=verbose,
            )
        timing.substep("built_degree_rips")
        if squeeze:
            try:
                radius_resolution = None if squeeze_resolution is None else squeeze_resolution[0]
            except TypeError:
                radius_resolution = squeeze_resolution
            radius_grid = compute_grid(
                [rips_filtration],
                strategy=squeeze_strategy,
                resolution=radius_resolution,
            )[0]
            if backend == "multipers":
                # Degree_rips_bifiltration stores generators on a compact 0-based
                # degree axis; requested k-values are only external grid labels.
                degree_grid = np.arange(ks_np.size, dtype=np.int32)
                degree_labels = -ks_np[::-1] / num_vertices if normalize else -ks_np[::-1]
            elif ks_np is None:
                degree_values = []
                for _, filtration in st_multi.get_skeleton(1):
                    filtration = np.asarray(filtration, dtype=np.float64).reshape(-1, 2)
                    if filtration.size:
                        degree_values.append(filtration[:, 1])
                degree_grid = (
                    np.unique(np.concatenate(degree_values))
                    if degree_values
                    else np.asarray([], dtype=np.float64)
                )
                degree_labels = degree_grid / num_vertices if normalize else degree_grid
            else:
                degree_grid = -ks_np[::-1]
                degree_labels = degree_grid / num_vertices if normalize else degree_grid
            squeeze_degree_grid = api.astype(
                api.copy(api.from_numpy(degree_grid)),
                rips_filtration.dtype,
            )
            st_multi = st_multi.grid_squeeze((radius_grid, squeeze_degree_grid))
            # Degree threshold axis has opposite order. Keep the exposed grid
            # sorted from lower to upper, as all filtration grids assume.
            filtration_grid = (
                radius_grid,
                api.astype(
                    api.copy(api.from_numpy(degree_labels)),
                    rips_filtration.dtype,
                ),
            )
            st_multi.filtration_grid = filtration_grid
            if backend == "deg_rips":
                # Convert through native SimplexTreeMulti interface copy instead of
                # replaying every simplex in Python. After `grid_squeeze`, degree
                # axis already uses compact coordinates, so Flat storage matches.
                st_multi = SimplexTreeMulti(
                    st_multi,
                    num_parameters=2,
                    dtype=np.float64,
                    kcritical=True,
                    ftype="Flat",
                )
                st_multi.filtration_grid = filtration_grid
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
     - verbose: Whether to emit timing logs (default False).
     - max_alpha_square: The maximum squared alpha value to consider when createing the alpha complex (default inf). See the GUDHI documentation for more information.
    """
    from multipers._core_delaunay_nanobind import build_core_delaunay_simplextree
    from multipers.simplex_tree_multi import SimplexTreeMulti

    points = np.asarray(points)
    if len(points) == 0:
        return SimplexTreeMulti(num_parameters=2)
    if points.ndim != 2:
        raise ValueError(f"The point cloud must be a 2D array, got {points.ndim}D.")
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
    if beta < 0:
        raise ValueError(f"The parameter beta must be positive, got {beta}.")
    if precision not in ["safe", "exact", "fast"]:
        raise ValueError(
            "The parameter precision must be one of ['safe', 'exact', 'fast'], "
            f"got {precision}."
        )
    points = np.ascontiguousarray(points, dtype=np.float64)
    ks = np.ascontiguousarray(ks, dtype=np.int64)

    with _mp_logs.timings(
        "CoreDelaunay",
        enabled=verbose,
        details={
            "backend": "gudhi",
            "mode": "cpp_interface",
            "num_points": len(points),
            "dim": points.shape[1],
            "beta": beta,
            "ks_count": len(ks),
            "precision": precision,
            "positive_degree": positive_degree,
        },
    ) as timing:
        simplex_tree_multi = build_core_delaunay_simplextree(
            SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64),
            points,
            ks,
            float(beta),
            precision,
            float(max_alpha_square),
            positive_degree,
        )
        timing.substep("built_native_core_delaunay")
        return simplex_tree_multi


def RhomboidBifiltration(
    x,
    k_max: int,
    degree: int,
    verbose: bool = False,
):
    """
    Build rhomboid-tiling model for multicover bifiltration.

    Reference: Kerber, Lesnick, Corbet, and Osang, "Computing the Multicover
    Bifiltration" (Discrete & Computational Geometry, 2023),
    https://doi.org/10.1007/s00454-022-00476-8

    For a finite point cloud `X`, multicover bifiltration is indexed by scale
    `r` and cover parameter `k`, with

    ```
    Cov_{r,k}(X) = {z | z lies within distance r of at least k points of X}.
    ```

    This constructor returns a 1-critical bifiltered model based on rhomboid
    tilings, quasi-isomorphic to the multi-critical multicover bifiltration.
    It is intended for 2D or 3D Euclidean data.

    Parameters
    ----------
    x:
        2D or 3D point cloud of shape `(num_points, ambient_dimension)`.
    k_max:
        Largest cover parameter to include.
    degree:
        Homological degree / maximal simplex dimension requested from the native
        rhomboid backend.
    verbose:
        Emit timing / backend logs.

    Notes
    -----
    Backend code originates from the rhomboid tiling implementation at
    https://github.com/geoo89/rhomboidtiling
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
