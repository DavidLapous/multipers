import numpy as np
import ot
from joblib import Parallel, delayed

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


# def estimate_matching(b1: PyMultiDiagrams_type, b2: PyMultiDiagrams_type):
#     assert len(b1) == len(b2)
#     from gudhi.bottleneck import bottleneck_distance

#     def get_bc(b: PyMultiDiagrams_type, i: int) -> np.ndarray:
#         temp = b[i].get_points()
#         out = (
#             np.array(temp)[:, :, 0] if len(temp) > 0 else np.empty((0, 2))
#         )  # GUDHI FIX
#         return out

#     return max(
#         (bottleneck_distance(get_bc(b1, i), get_bc(b2, i)) for i in range(len(b1)))
#     )


# # Functions to estimate precision
# def estimate_error(
#     st: SimplexTreeMulti_type,
#     module: PyModule_type,
#     degree: int,
#     nlines: int = 100,
#     verbose: bool = False,
# ):
#     """
#     Given an MMA SimplexTree and PyModule, estimates the bottleneck distance using barcodes given by gudhi.

#     Parameters
#     ----------
#      - st:SimplexTree
#         The simplextree representing the n-filtered complex. Used to define the gudhi simplextrees on different lines.
#      - module:PyModule
#         The module on which to estimate approximation error, w.r.t. the original simplextree st.
#      - degree:int
#         The homology degree to consider

#     Returns
#     -------
#      - float:The estimation of the matching distance, i.e., the maximum of the sampled bottleneck distances.

#     """
#     from time import perf_counter

#     parameter = 0

#     def _get_bc_ST(st, basepoint, degree: int):
#         """
#         Slices an mma simplextree to a gudhi simplextree, and compute its persistence on the diagonal line crossing the given basepoint.
#         """
#         gst = st.project_on_line(
#             basepoint=basepoint, parameter=parameter
#         )  # we consider only the 1rst coordinate (as )
#         gst.compute_persistence()
#         return gst.persistence_intervals_in_dimension(degree)

#     from gudhi.bottleneck import bottleneck_distance

#     low, high = module.get_box()
#     nfiltration = len(low)
#     basepoints = np.random.uniform(low=low, high=high, size=(nlines, nfiltration))
#     # barcodes from module
#     print("Computing mma barcodes...", flush=1, end="") if verbose else None
#     time = perf_counter()
#     bcs_from_mod = module.barcodes(degree=degree, basepoints=basepoints).get_points()
#     print(f"Done. {perf_counter() - time}s.") if verbose else None

#     def clean(dgm):
#         return np.array(
#             [
#                 [birth[parameter], death[parameter]]
#                 for birth, death in dgm
#                 if len(birth) > 0 and birth[parameter] != np.inf
#             ]
#         )

#     bcs_from_mod = [
#         clean(dgm) for dgm in bcs_from_mod
#     ]  # we only consider the 1st coordinate of the barcode
#     # Computes gudhi barcodes
#     from tqdm import tqdm

#     bcs_from_gudhi = [
#         _get_bc_ST(st, basepoint=basepoint, degree=degree)
#         for basepoint in tqdm(
#             basepoints, disable=not verbose, desc="Computing gudhi barcodes"
#         )
#     ]
#     return max(
#         (
#             bottleneck_distance(a, b)
#             for a, b in tqdm(
#                 zip(bcs_from_mod, bcs_from_gudhi),
#                 disable=not verbose,
#                 total=nlines,
#                 desc="Computing bottleneck distances",
#             )
#         )
#     )
