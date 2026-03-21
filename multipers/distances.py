import importlib
import importlib.util
import re

import numpy as np
import ot
from joblib import Parallel, delayed

import multipers as mp
import multipers.logs as _mp_logs
from multipers.array_api import api_from_tensor


def _iter_arrays(obj):
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_arrays(item)
    elif hasattr(obj, "shape") or hasattr(obj, "dtype"):
        yield obj


def _infer_api(X, Y=None, api=None):
    if api is not None:
        return api
    for data in (X, [] if Y is None else Y):
        for entry in data:
            for arr in _iter_arrays(entry):
                return api_from_tensor(arr)
    return api_from_tensor(np.asarray(0.0))


def _extract_measure(entry, api):
    arrays = [api.astensor(arr) for arr in _iter_arrays(entry)]
    if len(arrays) == 2 and getattr(arrays[0], "ndim", 0) == 2:
        return arrays[0], arrays[1]
    for arr in arrays:
        if getattr(arr, "ndim", 0) == 2 and arr.shape[1] >= 2:
            return arr[:, :-1], arr[:, -1]
    raise TypeError(
        "Unsupported signed-measure shape passed to sliced-wasserstein projection."
    )


def _normalize_signed_measures(X, api=None):
    api = _infer_api(X, api=api)
    if len(X) == 0:
        return [], api, None, api.dtype_default(), None

    measures = []
    dimension = None
    coord_dtype = None
    device = None
    for entry in X:
        C, M = _extract_measure(entry, api)
        if device is None:
            device = api.device(C)
        C = api.to_device(api.astensor(C), device)
        M = api.to_device(api.astensor(M).reshape(-1), device)
        if C.ndim != 2:
            raise TypeError("Signed-measure coordinates must be 2D arrays.")
        if M.ndim != 1:
            raise TypeError("Signed-measure weights must be 1D arrays.")
        if C.shape[0] != M.shape[0]:
            raise ValueError(
                "Coordinate and weight arrays must have identical lengths."
            )
        if dimension is None:
            dimension = C.shape[1]
        elif C.shape[1] != dimension:
            raise ValueError(
                "All signed measures must share the same ambient dimension."
            )
        if coord_dtype is None:
            coord_dtype = (
                C.dtype if api.dtype_is_float(C.dtype) else api.dtype_default()
            )
        weight_dtype = M.dtype if api.dtype_is_float(M.dtype) else api.dtype_default()
        measures.append((api.astype(C, coord_dtype), api.astype(M, weight_dtype)))
    return measures, api, dimension, coord_dtype, device


def _repeat_signed_points(pts, weights, sign, api):
    idx = api.where(sign * weights > 0)[0]
    if api.size(idx) == 0:
        return pts[idx]
    repeats = api.astype(api.abs(weights[idx]).round(), api.int64)
    return api.repeat_interleave(pts[idx], repeats, 0)


def _compute_signed_measure_projections(
    X, num_directions, scales=None, api=None, seed: int = 42
):
    if len(X) == 0:
        return []

    measures, api, dimension, coord_dtype, device = _normalize_signed_measures(
        X, api=api
    )

    def _to_backend(value, dtype=None):
        tensor = api.astensor(value)
        if dtype is not None:
            tensor = api.astype(tensor, dtype)
        return api.to_device(tensor, device)

    lines = ot.sliced.get_random_projections(
        dimension,
        num_directions,
        seed=seed,
        type_as=measures[0][0],
    )
    lines = _to_backend(lines, dtype=coord_dtype)

    if scales is not None:
        scales_tensor = _to_backend(scales, dtype=coord_dtype).reshape(-1)
        if scales_tensor.shape[0] != dimension:
            raise ValueError("Scales must match the ambient dimension.")
        weights = api.norm(scales_tensor[:, None] * lines, axis=0)
    else:
        weights = _to_backend(np.ones(num_directions), dtype=coord_dtype)

    projections = []
    for C, M in measures:
        pos_pts = _repeat_signed_points(C, M, 1, api)
        neg_pts = _repeat_signed_points(C, M, -1, api)
        projections.append(
            [api.matmul(pos_pts, lines), api.matmul(neg_pts, lines), weights]
        )
    return projections


def _sliced_wasserstein_distance_on_projections(meas1, meas2):
    api = api_from_tensor(meas1[0])
    weights = meas1[2]
    meas1_plus, meas1_minus = meas1[0], meas1[1]
    meas2_plus, meas2_minus = meas2[0], meas2[1]
    A = api.sort(api.cat([meas1_plus, meas2_minus], 0), axis=0)
    B = api.sort(api.cat([meas2_plus, meas1_minus], 0), axis=0)
    L1 = api.sum(api.abs(A - B), axis=0)
    return api.mean(L1 * weights)


def pairwise_distances(items_X, items_Y=None, metric=None, n_jobs=None, api=None):
    if metric is None:
        raise ValueError("A metric callable is required.")
    if items_Y is not None:
        num_x = len(items_X)
        num_y = len(items_Y)
        if api is None:
            api = _infer_api(items_X, items_Y)
        if num_x == 0:
            return api.empty((0, num_y))
        if num_y == 0:
            return api.empty((num_x, 0))
        par = Parallel(n_jobs=n_jobs, prefer="threads")
        pairs = [(i, j) for i in range(num_x) for j in range(num_y)]
        dists = par(delayed(metric)(items_X[i], items_Y[j]) for i, j in pairs)
        if len(dists) == 0:
            return api.empty((num_x, num_y))
        return api.reshape(api.stack(dists), (num_x, num_y))

    num_items = len(items_X)
    if api is None:
        api = _infer_api(items_X)
    if num_items == 0:
        return api.empty((0, 0))
    if num_items == 1:
        zero = api.astensor(0.0, dtype=api.dtype_default())
        return api.reshape(api.stack([zero]), (1, 1))

    triu = np.triu_indices(num_items, k=1)
    par = Parallel(n_jobs=n_jobs, prefer="threads")
    dists = par(
        delayed(metric)(items_X[i], items_X[j]) for i, j in zip(triu[0], triu[1])
    )
    dists = api.stack(dists)
    matrix = api.zeros((num_items, num_items), dtype=dists.dtype)
    matrix = api.set_at(matrix, triu, dists)
    return api.set_at(matrix, (triu[1], triu[0]), dists)


def sm2diff(sm1, sm2, threshold=None, api=None):
    pts1, w1 = sm1
    pts2, w2 = sm2
    if api is None:
        api = api_from_tensor(pts1)
    pts1 = api.astensor(pts1)
    device = api.device(pts1)
    pts2 = api.to_device(api.astensor(pts2), device)
    w1 = api.to_device(api.astensor(w1).reshape(-1), device)
    w2 = api.to_device(api.astensor(w2).reshape(-1), device)

    x = api.cat(
        [
            _repeat_signed_points(pts1, w1, 1, api),
            _repeat_signed_points(pts2, w2, -1, api),
        ],
        0,
    )
    y = api.cat(
        [
            _repeat_signed_points(pts1, w1, -1, api),
            _repeat_signed_points(pts2, w2, 1, api),
        ],
        0,
    )
    if threshold is not None:
        threshold = api.to_device(api.astensor(threshold, dtype=pts1.dtype), device)
        x = api.where(x > threshold, threshold, x)
        y = api.where(y > threshold, threshold, y)
    return x, y


def sm_distance(
    sm1: tuple,
    sm2: tuple,
    sliced: bool = False,
    api=None,
    reg: float = 0,
    reg_m: float = 0,
    numItermax: int = 10000,
    p: float = 1,
    threshold=None,
    num_directions: int = 10,
    seed: int = 42,
):
    """
    Computes a distance between two signed measures,
    of the form
     - (pts,weights)
    with
     - pts : (num_pts, dim) float array
     - weights : (num_pts,) int array

    Regularisation:
      - sinkhorn if reg != 0
      - sinkhorn unbalanced if reg_m != 0

    """
    if api is None:
        api = api_from_tensor(sm1[0])
    x, y = sm2diff(sm1, sm2, threshold=threshold, api=api)
    if sliced:
        dist = ot.sliced.sliced_wasserstein_distance(
            x,
            y,
            n_projections=num_directions,
            p=1,
            seed=seed,
        )
        return dist * len(x)

    if p < 1:
        raise ValueError("Only lp metrics with p >= 1 are supported.")
    loss = api.cdist(x, y, p=p)
    empty_tensor = api.to_device(api.astensor([]), api.device(x))

    if reg == 0:
        return ot.lp.emd2(empty_tensor, empty_tensor, M=loss) * len(x)
    if reg_m == 0:
        return ot.sinkhorn2(
            a=empty_tensor, b=empty_tensor, M=loss, reg=reg, numItermax=numItermax
        )
    return ot.sinkhorn_unbalanced2(
        a=empty_tensor,
        b=empty_tensor,
        M=loss,
        reg=reg,
        reg_m=reg_m,
        numItermax=numItermax,
    )
    # return ot.sinkhorn2(a=onesx,b=onesy,M=loss,reg=reg, numItermax=numItermax)
    # return ot.bregman.empirical_sinkhorn2(x,y,reg=reg)


_MATCHING_DISTANCE_BOUND_STRATEGIES = {
    "bruteforce": 0,
    "local_dual_bound": 1,
    "local_dual_bound_refined": 2,
    "local_dual_bound_for_each_point": 3,
    "local_combined": 4,
}

_MATCHING_DISTANCE_TRAVERSE_STRATEGIES = {
    "depth_first": 0,
    "breadth_first": 1,
    "breadth_first_value": 2,
    "upper_bound": 3,
}



_MATCHING_DISTANCE_MONTE_CARLO_LP_REDUCTION = re.compile(r"l(\d+)")
_MATCHING_DISTANCE_MONTE_CARLO_SOFTMAX_REDUCTION = re.compile(r"softmax(\d+(?:\.\d+)?)")


def _require_hera_backend():
    if importlib.util.find_spec("multipers._hera_interface") is None:
        raise RuntimeError(
            "Hera in-memory interface is not available in this build. "
            "Rebuild multipers with Hera headers to enable this backend."
        )

    from multipers import _hera_interface

    if not _hera_interface._is_available():
        raise RuntimeError(
            "Hera in-memory interface is not available in this build. "
            "Rebuild multipers with Hera headers to enable this backend."
        )
    return _hera_interface


def _reduce_monte_carlo_line_distances(weighted_distances, *, line_reduction: str, api):
    if line_reduction == "max":
        return api.max(weighted_distances)
    if line_reduction == "mean":
        return api.mean(weighted_distances)
    if line_reduction == "softmax":
        shifted = weighted_distances - api.max(weighted_distances)
        softmax_weights = api.exp(shifted)
        softmax_weights = softmax_weights / api.sum(softmax_weights)
        return api.sum(softmax_weights * weighted_distances)

    match = _MATCHING_DISTANCE_MONTE_CARLO_SOFTMAX_REDUCTION.fullmatch(line_reduction)
    if match is not None:
        softmax_scale = float(match.group(1))
        shifted = softmax_scale * (weighted_distances - api.max(weighted_distances))
        softmax_weights = api.exp(shifted)
        softmax_weights = softmax_weights / api.sum(softmax_weights)
        return api.sum(softmax_weights * weighted_distances)

    match = _MATCHING_DISTANCE_MONTE_CARLO_LP_REDUCTION.fullmatch(line_reduction)
    if match is None:
        p = int(match.group(1))
        return api.mean(weighted_distances**p) ** (1 / p)

    raise ValueError(
        "Unknown Monte Carlo line reduction "
        f"{line_reduction!r}. Expected one of: max, mean, softmax, `l<p>` for an Lp mean, or `softmax<d>` for scaled softmax."
    )


def _sample_monte_carlo_lines(
    left,
    right,
    *,
    nlines: int,
    seed: int,
    oversampling: int,
    use_fpsample: bool,
    fpsample_bucket_height: int,
):
    import multipers.grids as mpg

    if nlines <= 0:
        raise ValueError("nlines must be positive.")
    if oversampling <= 0:
        raise ValueError("oversampling must be positive.")

    rng = np.random.default_rng(seed)
    dimension = int(left.num_parameters)
    left_box = mpg.compute_bounding_box(left)
    right_box = mpg.compute_bounding_box(right)
    low = np.minimum(left_box[0], right_box[0])
    high = np.maximum(left_box[1], right_box[1])

    fpsample_spec = importlib.util.find_spec("fpsample")
    fpsample = (
        importlib.import_module("fpsample") if fpsample_spec is not None else None
    )
    sampler = None
    if fpsample is None or not use_fpsample:
        _mp_logs.warn_fallback(
            "`fpsample` is unavailable; Monte Carlo matching distance falls back to random direction sampling.",
            stacklevel=3,
        )
        oversampling = 1
    else:
        sampler = fpsample.bucket_fps_kdline_sampling

    num_candidates = max(nlines, int(oversampling) * nlines)
    basepoints = rng.uniform(low=low, high=high, size=(num_candidates, dimension))
    directions = np.abs(rng.standard_normal(size=(num_candidates, dimension)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    if num_candidates > nlines:
        if sampler is not None:
            indices = np.asarray(
                sampler(directions, nlines, h=int(fpsample_bucket_height)),
                dtype=np.int64,
            )
        else:
            indices = rng.choice(num_candidates, size=nlines, replace=False)
        directions = directions[indices]
        basepoints = basepoints[indices]

    directions = np.ascontiguousarray(directions, dtype=np.float64)
    directions /= directions.max(axis=1, keepdims=True)
    return basepoints, directions


def _matching_distance_monte_carlo(
    left,
    right,
    degree,
    *,
    num_lines: int,
    seed: int,
    direction_oversampling: int,
    use_fpsample: bool,
    fpsample_bucket_height: int,
    bottleneck_delta: float,
    line_reduction: str,
    return_stats: bool,
    api=None,
):
    basepoints, directions = _sample_monte_carlo_lines(
        left,
        right,
        nlines=num_lines,
        seed=seed,
        oversampling=direction_oversampling,
        use_fpsample=use_fpsample,
        fpsample_bucket_height=fpsample_bucket_height,
    )
    api = _infer_api([basepoints, directions], api=api)
    basepoints = api.astensor(basepoints)
    directions = api.astensor(directions)

    left_diagrams = tuple(
        barcodes[degree]
        for barcodes in left.persistence_on_lines(basepoints, directions)
    )
    right_diagrams = tuple(
        barcodes[degree]
        for barcodes in right.persistence_on_lines(basepoints, directions)
    )
    raw_distances = hera_bottleneck_distances(
        left_diagrams, right_diagrams, delta=bottleneck_delta
    )
    raw_distances = api.astype(api.from_numpy(raw_distances), directions.dtype)
    weights = api.sort(directions, axis=1)[:, 0]
    weighted_distances = weights * raw_distances
    reduced_distance = _reduce_monte_carlo_line_distances(
        weighted_distances, line_reduction=line_reduction, api=api
    )
    weighted_distances_np = np.asarray(
        api.asnumpy(weighted_distances), dtype=np.float64
    )
    raw_distances_np = np.asarray(api.asnumpy(raw_distances), dtype=np.float64)
    weights_np = np.asarray(api.asnumpy(weights), dtype=np.float64)
    distance = float(np.asarray(api.asnumpy(reduced_distance), dtype=np.float64))
    if not return_stats:
        return distance
    best_index = (
        int(np.argmax(weighted_distances_np)) if len(weighted_distances_np) else -1
    )

    return distance, {
        "num_lines": int(num_lines),
        "line_reduction": line_reduction,
        "best_line_index": int(best_index),
        "best_raw_bottleneck": float(raw_distances_np[best_index])
        if best_index >= 0
        else 0.0,
        "best_weight": float(weights_np[best_index]) if best_index >= 0 else 0.0,
    }


def matching_distance(
    left,
    right,
    *,
    api=None,
    degree=None,
    strategy: str = "monte_carlo",
    mc_nlines: int = 128,
    mc_seed: int = 42,
    mc_oversampling: int = 10,
    mc_fpsample_bucket_height: int = 7,
    mc_bottleneck_delta: float = 0.0,
    mc_line_reduction: str = "max",
    hera_epsilon: float = 0.001,
    hera_delta: float = 0.1,
    hera_max_depth: int = 8,
    hera_initialization_depth: int = 2,
    hera_bound_strategy: str | int = "local_combined",
    hera_traverse_strategy: str | int = "breadth_first",
    hera_tolerate_max_iter_exceeded: bool = False,
    hera_stop_asap: bool = True,
    return_stats: bool = False,
):
    """
    Compute a 2-parameter matching-distance estimate between two minimal-presentation slicers.

    The input slicers must already encode minimal presentations. When they come
    from `mpfree`, any extra `degree + 2` syzygy block in the full resolution is
    ignored automatically; only the `degree -> degree + 1` presentation block is
    passed to Hera for the exact backend.

    Parameters
    ----------
    left, right : multipers.Slicer
        Two 1-critical minimal-presentation slicers in the same homological
        degree. The exact `"hera"` backend requires 2-parameter slicers. The
        Monte Carlo backend only requires both slicers to have the same number of
        parameters.
    api : multipers.array_api.available_interfaces, optional
        Array backend used for Monte Carlo post-processing of sampled directions
        and distances. Defaults to the backend inferred from the sampled arrays.
    strategy : {"monte_carlo", "hera"}, default="monte_carlo"
        Backend used to estimate the matching distance. `"monte_carlo"` samples
        lines, computes line barcodes, and aggregates
        `min(direction) * bottleneck_distance` across sampled lines.
        `"hera"` runs Hera's adaptive 2-parameter matching-distance algorithm
        on the minimal presentations.
    mc_nlines : int, default=128
        Number of sampled lines for the Monte Carlo backend.
    mc_seed : int, default=42
        Random seed used to sample Monte Carlo basepoints and directions.
        In 2 parameters, basepoints are sampled on the usual anti-diagonal
        segment of the common bounding box. In higher dimensions, basepoints are
        sampled uniformly in the common bounding box.
    mc_oversampling : int, default=10
        Monte Carlo direction oversampling factor before optional farthest-point
        subsampling with `fpsample`.
    mc_fpsample_bucket_height : int, default=7
        Bucket height passed to `fpsample` when direction resampling is enabled.
    mc_bottleneck_delta : float, default=0.0
        Relative error tolerance used by Hera's bottleneck solver on individual
        line diagrams in the Monte Carlo backend. Set to `0` for exact per-line
        bottleneck distances.
    mc_line_reduction : {"max", "mean", "softmax", "softmax<d>", "l<p>"}, default="max"
        Reduction applied to the sampled Monte Carlo line scores
        `min(direction) * bottleneck_distance`. `"max"` is the `Linf`
        reduction, `"mean"` is the `L1` mean across sampled line scores,
        `"l<p>"` applies the corresponding `Lp` mean (for example, `"l3"`),
        `"softmax"` uses unscaled softmax weights, and `"softmax<d>"`
        rescales sampled line scores by `d` before applying softmax weights.
        Ignored by the exact `"hera"` backend.
    hera_epsilon : float, default=0.001
        Relative error tolerance used by Hera when approximating bottleneck
        distances on individual slices in the exact `"hera"` backend. Smaller
        values are more accurate and can be slower.
    hera_delta : float, default=0.1
        Relative stopping tolerance for the adaptive search over slice directions
        in the exact `"hera"` backend. Smaller values request a tighter
        approximation of the matching distance.
    hera_max_depth : int, default=8
        Maximum quadtree refinement depth used by the exact `"hera"` backend.
        Larger values allow more refinement but increase runtime.
    hera_initialization_depth : int, default=2
        Initial uniform refinement depth before the exact `"hera"` backend
        switches to adaptive subdivision.
    hera_bound_strategy : {"bruteforce", "local_dual_bound", "local_dual_bound_refined", "local_dual_bound_for_each_point", "local_combined"} or int, default="local_combined"
        Strategy used by the exact `"hera"` backend to estimate upper and lower
        bounds on each dual cell. String values are mapped to Hera's
        `BoundStrategy` enum; integer enum values are also accepted.
    hera_traverse_strategy : {"depth_first", "breadth_first", "breadth_first_value", "upper_bound"} or int, default="breadth_first"
        Order in which the exact `"hera"` backend explores candidate dual cells
        during refinement. String values are mapped to Hera's
        `TraverseStrategy` enum; integer enum values are also accepted.
    hera_tolerate_max_iter_exceeded : bool, default=False
        Forwarded to Hera's internal bottleneck computations in the exact
        `"hera"` backend. If `True`, the current estimate is accepted when
        Hera's bottleneck solver hits its iteration limit instead of raising an
        error.
    hera_stop_asap : bool, default=True
        If `True`, the exact `"hera"` backend stops evaluating a slice as soon
        as the current point is already too far to improve the active bound.
        This is usually faster, at the cost of less informative intermediate
        bounds.
    return_stats : bool, default=False
        If `False`, return only the matching distance. If `True`, also return a
        backend-specific diagnostics. The exact `"hera"` backend returns
        `actual_error`, `actual_max_depth`, and `n_hera_calls`. The Monte Carlo
        backend returns the number of sampled lines, the reduction, and the best
        sampled line.

    Returns
    -------
    float or tuple[float, dict[str, float | int]]
        The matching distance, optionally paired with Hera diagnostics when
        `return_stats=True`.

    References
    ----------
    Hera project: https://github.com/anigmetov/hera
    """
    from multipers.simplex_tree_multi import is_simplextree_multi

    if is_simplextree_multi(left):
        left = mp.Slicer(left)
    if is_simplextree_multi(right):
        right = mp.Slicer(right)

    if api is None:
        if left.filtration_grid is not None:
            api = api_from_tensor(left.filtration_grid[0])
        else:
            api = api_from_tensor([])

    if degree is None:
        degree = left.minpres_degree
        if degree < 0:
            raise ValueError("`degree` has to be provided, unless inputs are minpres.")
        if right.minpres_degree != degree:
            raise ValueError(
                "Matching distance expects inputs to be in the same homological degree. "
                f"Got {left.minpres_degree=} and {right.minpres_degree=}."
            )

    if left.num_parameters != right.num_parameters:
        raise ValueError(
            "Matching distance expects slicers with the same number of parameters. "
            f"Got {left.num_parameters=} and {right.num_parameters=}."
        )

    if strategy == "monte_carlo":
        return _matching_distance_monte_carlo(
            left,
            right,
            degree,
            num_lines=int(mc_nlines),
            seed=int(mc_seed),
            direction_oversampling=int(mc_oversampling),
            use_fpsample=True,
            fpsample_bucket_height=int(mc_fpsample_bucket_height),
            bottleneck_delta=float(mc_bottleneck_delta),
            line_reduction=mc_line_reduction,
            return_stats=bool(return_stats),
            api=api,
        )

    if strategy != "hera":
        raise ValueError(
            "Unknown matching-distance strategy {!r}. Expected 'monte_carlo' or 'hera'.".format(
                strategy
            )
        )

    if not left.is_minpres or not right.is_minpres:
        mp.logs.warn_superfluous_computation(
            "Didn't get a presentation as an input, calling mpfree."
        )
        left = left.minpres(degree, full_resolution=False, force=False)
        right = right.minpres(degree, full_resolution=False, force=False)
    _hera_interface = _require_hera_backend()
    return _hera_interface.matching_distance(
        left,
        right,
        hera_epsilon=float(hera_epsilon),
        delta=float(hera_delta),
        max_depth=int(hera_max_depth),
        initialization_depth=int(hera_initialization_depth),
        bound_strategy=hera_bound_strategy,
        traverse_strategy=hera_traverse_strategy,
        tolerate_max_iter_exceeded=bool(hera_tolerate_max_iter_exceeded),
        stop_asap=bool(hera_stop_asap),
        return_stats=bool(return_stats),
    )
