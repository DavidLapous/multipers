import importlib
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
    target_rows = max(A.shape[0], B.shape[0])
    if target_rows == 0:
        return api.to_device(api.zeros((), dtype=weights.dtype), api.device(weights))
    if A.shape[0] != target_rows:
        padding = api.to_device(
            api.zeros((target_rows - A.shape[0], A.shape[1]), dtype=A.dtype),
            api.device(A),
        )
        A = api.cat(
            [A, padding], 0
        )
    if B.shape[0] != target_rows:
        padding = api.to_device(
            api.zeros((target_rows - B.shape[0], B.shape[1]), dtype=B.dtype),
            api.device(B),
        )
        B = api.cat(
            [B, padding], 0
        )
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


_MATCHING_DISTANCE_BOUND_STRATEGIES: dict[str, int] = {
    "bruteforce": 0,
    "local_dual_bound": 1,
    "local_dual_bound_refined": 2,
    "local_dual_bound_for_each_point": 3,
    "local_combined": 4,
}

_MATCHING_DISTANCE_TRAVERSE_STRATEGIES: dict[str, int] = {
    "depth_first": 0,
    "breadth_first": 1,
    "breadth_first_value": 2,
    "upper_bound": 3,
}


_MATCHING_DISTANCE_MONTE_CARLO_LP_REDUCTION = re.compile(r"l(\d+)")
_MATCHING_DISTANCE_MONTE_CARLO_SOFTMAX_REDUCTION = re.compile(r"softmax(\d+(?:\.\d+)?)")


def _normalize_hera_strategy(strategy, options, name: str) -> int:
    if isinstance(strategy, str):
        try:
            return options[strategy]
        except KeyError as exc:
            allowed = ", ".join(sorted(options))
            raise ValueError(
                f"Unknown {name} {strategy!r}. Expected one of: {allowed}."
            ) from exc

    if isinstance(strategy, (int, np.integer)):
        value = int(strategy)
        if value not in options.values():
            allowed = ", ".join(str(v) for v in sorted(set(options.values())))
            raise ValueError(
                f"Unknown {name} {strategy!r}. Expected one of: {allowed}."
            )
        return value

    raise TypeError(
        f"Invalid {name} type {type(strategy)!r}. Expected str or int-like value."
    )


def hera_bottleneck_distances(
    left_diagrams, right_diagrams, *, delta: float = 0.01, n_jobs: int = 0
):
    """
    Compute Hera bottleneck distances for aligned batches of persistence diagrams.

    The compiled Hera bridge drops diagonal points once, then evaluates the
    batch with a native TBB loop when available. `n_jobs <= 0` keeps backend
    default concurrency.
    """
    from multipers import _hera_interface

    _hera_interface.require()
    return _hera_interface.bottleneck_distances(
        left_diagrams,
        right_diagrams,
        delta=delta,
        n_jobs=int(n_jobs),
    )


def _reduce_monte_carlo_line_distances(weighted_distances, *, line_reduction: str, api):
    if line_reduction in ("max", "linf"):
        return api.max(weighted_distances)
    if line_reduction in ("l1", "mean"):
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
    if match is not None:
        p = int(match.group(1))
        return api.mean(weighted_distances**p) ** (1 / p)

    raise ValueError(
        "Unknown Monte Carlo line reduction "
        f"{line_reduction!r}. Expected one of: max, mean, softmax, `l<p>` for an Lp mean, or `softmax<d>` for scaled softmax."
    )


def _get_monte_carlo_line_reduction(line_reduction: str) -> None:
    if line_reduction in ("max", "linf", "l1", "mean", "softmax"):
        return
    if _MATCHING_DISTANCE_MONTE_CARLO_SOFTMAX_REDUCTION.fullmatch(line_reduction):
        return
    if _MATCHING_DISTANCE_MONTE_CARLO_LP_REDUCTION.fullmatch(line_reduction):
        return
    raise ValueError(
        "Unknown Monte Carlo line reduction "
        f"{line_reduction!r}. Expected one of: max, mean, softmax, `l<p>` for an Lp mean, or `softmax<d>` for scaled softmax."
    )


def _can_use_native_monte_carlo_matching_distance(left, right) -> bool:
    from multipers.slicer import is_slicer

    if not is_slicer(left) or not is_slicer(right):
        return False
    if left.is_squeezed or right.is_squeezed:
        return False
    if not np.issubdtype(np.dtype(left.dtype), np.floating):
        return False
    if not np.issubdtype(np.dtype(right.dtype), np.floating):
        return False
    return True


def _process_monte_carlo_raw_distances(
    raw_distances,
    *,
    api,
    directions,
    line_reduction: str,
    return_stats: bool,
):
    raw_distances = api.to_device(
        api.astype(api.from_numpy(raw_distances), directions.dtype),
        api.device(directions),
    )
    if directions.ndim == 1:
        directions = api.reshape(directions, (1, directions.shape[0]))
    weights = api.astype(api.minvalues(directions, axis=1), raw_distances.dtype)
    weighted_distances = weights * raw_distances
    reduced_distance = _reduce_monte_carlo_line_distances(
        weighted_distances, line_reduction=line_reduction, api=api
    )
    weighted_distances_np = np.asarray(
        api.asnumpy(weighted_distances), dtype=np.float64
    )
    raw_distances_np = np.asarray(api.asnumpy(raw_distances), dtype=np.float64)
    weights_np = np.asarray(api.asnumpy(weights), dtype=np.float64)
    num_lines = int(len(raw_distances_np))
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


def _sample_monte_carlo_lines(
    left,
    right,
    *,
    nlines: int,
    seed: int,
    oversampling: int,
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
    if fpsample is None:
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
    directions /= np.max(directions, axis=1, keepdims=True)

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
    return basepoints, directions


def _matching_distance_monte_carlo(
    left,
    right,
    degree,
    *,
    num_lines: int,
    seed: int,
    line_oversampling: int,
    fpsample_bucket_height: int,
    bottleneck_delta: float,
    n_jobs: int,
    line_reduction: str,
    return_stats: bool,
    api=None,
    basepoints=None,
    directions=None,
):
    _get_monte_carlo_line_reduction(line_reduction)

    if basepoints is None or directions is None:
        basepoints, directions = _sample_monte_carlo_lines(
            left,
            right,
            nlines=num_lines,
            seed=seed,
            oversampling=line_oversampling,
            fpsample_bucket_height=fpsample_bucket_height,
        )

    if _can_use_native_monte_carlo_matching_distance(left, right):
        from multipers import _hera_interface

        _hera_interface.require()
        basepoints_np = np.asarray(basepoints, dtype=np.float64)
        directions_np = np.asarray(directions, dtype=np.float64)
        if basepoints_np.ndim == 1:
            basepoints_np = basepoints_np[None]
        if directions_np.ndim == 1:
            directions_np = directions_np[None]
        if basepoints_np.ndim == 0 or basepoints_np.ndim > 2:
            raise ValueError(
                "Expected a basepoint shape of the form (num_parameters,). "
                f"Got {basepoints_np.shape=}"
            )
        if directions_np.ndim == 0 or directions_np.ndim > 2:
            raise ValueError(
                "Expected a direction shape of the form (num_parameters,). "
                f"Got {directions_np.shape=}"
            )
        if basepoints_np.shape != directions_np.shape:
            raise ValueError(
                "Monte Carlo matching distance expects basepoints and directions "
                f"with identical shapes. Got {basepoints_np.shape=} and {directions_np.shape=}."
            )
        basepoints_np = np.ascontiguousarray(basepoints_np, dtype=np.float64)
        directions_np = np.ascontiguousarray(directions_np, dtype=np.float64)
        raw_distances = _hera_interface.monte_carlo_bottleneck_distances(
            left,
            right,
            basepoints_np,
            directions_np,
            degree=degree,
            delta=bottleneck_delta,
            n_jobs=int(n_jobs),
        )
        api = _infer_api([basepoints, directions], api=api)
        directions = api.astensor(directions)
        return _process_monte_carlo_raw_distances(
            raw_distances,
            api=api,
            directions=directions,
            line_reduction=line_reduction,
            return_stats=return_stats,
        )

    # Python fallback slices lines directly on input slicers, so work on local
    # copies when callers reuse same slicer across threads.
    left = left.copy()
    right = right.copy()

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
        left_diagrams, right_diagrams, delta=bottleneck_delta, n_jobs=int(n_jobs)
    )
    return _process_monte_carlo_raw_distances(
        raw_distances,
        api=api,
        directions=directions,
        line_reduction=line_reduction,
        return_stats=return_stats,
    )


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
    mc_fpsample_bucket_height: int = 2,
    mc_bottleneck_delta: float = 0.0,
    mc_n_jobs: int = 0,
    mc_line_reduction: str = "max",
    hera_epsilon: float = 0.001,
    hera_delta: float = 0.1,
    hera_max_depth: int = 8,
    hera_initialization_depth: int = 2,
    hera_bound_strategy: str = "local_combined",
    hera_traverse_strategy: str = "breadth_first",
    hera_tolerate_max_iter_exceeded: bool = False,
    hera_stop_asap: bool = True,
    return_stats: bool = False,
    verbose: bool = False,
    mc_basepoints=None,
    mc_directions=None,
):
    """
    Compute a 2-parameter matching-distance estimate between two filtrations/presentations.

    The input should be a pair of simplextrees/slicers.
    The Hera strategy requires presentations, is usually slower,
    but guarantees the output precision to the specified parameters.

    Parameters
    ----------
    left, right : multipers.Slicer or multipers.SimplexTreeMulti
        Simplex trees are converted to slicers first. Both backends then require
        slicer inputs. The exact `"hera"` backend additionally requires
        2-parameter 1-critical slicers.
    api : multipers.array_api.available_interfaces, optional
        WIP for autodiff.
    strategy : {"monte_carlo", "hera"}, default="monte_carlo"
        Backend used to estimate the matching distance. `"monte_carlo"` samples
        lines, computes per-line bottleneck distances, and aggregates according
        to `mc_line_reduction` across sampled lines. When both inputs are native
        unsqueezed floating slicers, this path uses a fused compiled fast path;
        otherwise it falls back to the Python batch implementation.
        `"hera"` runs Hera's 2-parameter matching-distance algorithm
        on the presentations.
    mc_nlines : int, default=128
        Number of sampled lines for the Monte Carlo backend.
    mc_seed : int, default=42
        Random seed used to sample Monte Carlo basepoints and directions.
    mc_oversampling : int, default=10
        Monte Carlo direction oversampling factor before optional farthest-point
        subsampling with `fpsample`.
    mc_fpsample_bucket_height : int, default=7
        Bucket height passed to `fpsample` when direction resampling is enabled.
    mc_bottleneck_delta : float, default=0.0
        Relative error tolerance used by Hera's bottleneck solver on individual
        line diagrams in the Monte Carlo backend. Set to `0` for exact per-line
        bottleneck distances.
    mc_n_jobs : int, default=0
        Worker count used by the Monte Carlo backend for per-line bottleneck
        computations. `mc_n_jobs <= 0` keeps backend default concurrency.
    mc_line_reduction : {"max", "mean", "softmax", "softmax<d>", "l<p>"}, default="max"
        Reduction applied to the sampled Monte Carlo line scores
        `bottleneck_distance`. `"max"` is the `Linf` reduction, `"mean"`
        is the `L1` mean across sampled line scores, `"l<p>"` applies the
        corresponding `Lp` mean (for example, `"l3"`), `"softmax"` uses
        unscaled softmax weights, and `"softmax<d>"` rescales sampled line
        scores by `d` before applying softmax weights.
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
    verbose : bool, default=False
        If `True`, print timing scopes for validation and backend work.

    Returns
    -------
    float or tuple[float, dict[str, float | int]]
        The matching distance, optionally paired with Hera diagnostics when
        `return_stats=True`.

    References
    ----------
    Hera project: https://github.com/anigmetov/hera
    """
    with _mp_logs.timings(
        "matching_distance",
        enabled=verbose,
        details={"strategy": strategy, "degree": degree, "return_stats": return_stats},
    ) as timing:
        from multipers.simplex_tree_multi import is_simplextree_multi
        from multipers.slicer import is_slicer

        if is_simplextree_multi(left):
            left = mp.Slicer(left)
        if is_simplextree_multi(right):
            right = mp.Slicer(right)
        if not is_slicer(left) or not is_slicer(right):
            raise ValueError(
                "Invalid input. Expected slicers or simplex trees. "
                f"Got {type(left)=} and {type(right)=}."
            )
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
        timing.substep("validated_inputs")

        if strategy == "monte_carlo":
            with timing.step("monte_carlo", details={"nlines": int(mc_nlines)}) as step:
                out = _matching_distance_monte_carlo(
                    left,
                    right,
                    degree,
                    num_lines=int(mc_nlines),
                    seed=int(mc_seed),
                    line_oversampling=int(mc_oversampling),
                    fpsample_bucket_height=int(mc_fpsample_bucket_height),
                    bottleneck_delta=float(mc_bottleneck_delta),
                    n_jobs=int(mc_n_jobs),
                    line_reduction=mc_line_reduction,
                    return_stats=bool(return_stats),
                    api=api,
                    basepoints=mc_basepoints,
                    directions=mc_directions,
                )
                if return_stats:
                    step.add_stats(out[1])
                return out

        if strategy != "hera":
            raise ValueError(
                "Unknown matching-distance strategy {!r}. Expected 'monte_carlo' or 'hera'.".format(
                    strategy
                )
            )

        if left.num_parameters != 2 or right.num_parameters != 2:
            raise ValueError("Hera matching distance only supports 2-parameter slicers.")
        with timing.step("hera") as step:
            if not left.is_minpres or not right.is_minpres:
                mp.logs.warn_superfluous_computation(
                    "Didn't get a presentation as an input (hera strategy), calling mpfree."
                )
                left = left.minpres(degree, full_resolution=False, force=False)
                right = right.minpres(degree, full_resolution=False, force=False)
                step.substep("computed_minpres")
            from multipers import _hera_interface

            _hera_interface.require()
            out = _hera_interface.matching_distance(
                left,
                right,
                hera_epsilon=float(hera_epsilon),
                delta=float(hera_delta),
                max_depth=int(hera_max_depth),
                initialization_depth=int(hera_initialization_depth),
                bound_strategy=_normalize_hera_strategy(
                    hera_bound_strategy,
                    _MATCHING_DISTANCE_BOUND_STRATEGIES,
                    "Hera bound strategy",
                ),
                traverse_strategy=_normalize_hera_strategy(
                    hera_traverse_strategy,
                    _MATCHING_DISTANCE_TRAVERSE_STRATEGIES,
                    "Hera traverse strategy",
                ),
                tolerate_max_iter_exceeded=bool(hera_tolerate_max_iter_exceeded),
                stop_asap=bool(hera_stop_asap),
                return_stats=bool(return_stats),
            )
            if return_stats:
                step.add_stats(out[1])
            return out
