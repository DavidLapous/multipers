from contextlib import nullcontext
from functools import wraps
from importlib.util import find_spec

import numpy as _np
from scipy.special import logsumexp as _sp_logsumexp
import multipers.logs as _mp_logs

if find_spec("numba"):
    import numba as _numba
    from numba.core.errors import NumbaError as _NumbaError
else:
    _numba = None
    _NumbaError = None

backend = _np
name = "numpy"
_has_jit = _numba is not None
int64 = _np.int64
cat = _np.concatenate
det = _np.linalg.det
asnumpy = _np.asarray
tensor = _np.array
stack = _np.stack
empty = _np.empty
where = _np.where
no_grad = nullcontext
zeros = _np.zeros
min = _np.min
max = _np.max
reshape = _np.reshape
arange = _np.arange
moveaxis = _np.moveaxis
ones = _np.ones
repeat_interleave = _np.repeat
inf = _np.inf
searchsorted = _np.searchsorted
LazyTensor = None
abs = _np.abs
exp = _np.exp
log = _np.log
sin = _np.sin
cos = _np.cos
sinc = _np.sinc
sqrt = _np.sqrt
matmul = _np.matmul
einsum = _np.einsum


def jit(fn=None, **kwargs):
    if _numba is None:
        def decorator(func):
            @wraps(func)
            def wrapped(*args, **inner_kwargs):
                return func(*args, **inner_kwargs)

            return wrapped

        if fn is None:
            return decorator
        return decorator(fn)

    def decorator(func):
        compiled = _numba.njit(func)

        @wraps(func)
        def wrapped(*args, **inner_kwargs):
            try:
                return compiled(*args, **inner_kwargs)
            except _NumbaError:
                return func(*args, **inner_kwargs)

        return wrapped

    if fn is None:
        return decorator
    return decorator(fn)


def astensor(x, contiguous=False, dtype=None):
    if contiguous:
        return _np.ascontiguousarray(x, dtype=dtype)
    return _np.asarray(x, dtype=dtype)


def unique(x, assume_sorted=False, _mean=False):
    return _np.unique(x)


def cdist(x, y, p=2):
    from scipy.spatial.distance import cdist as _sp_cdist

    if p == 1:
        return _sp_cdist(x, y, metric="cityblock")
    if p == 2:
        return _sp_cdist(x, y, metric="euclidean")
    return _sp_cdist(x, y, metric="minkowski", p=p)


def pdist(x, p=2):
    from scipy.spatial.distance import pdist as _sp_pdist

    if p == 1:
        return _sp_pdist(x, metric="cityblock")
    if p == 2:
        return _sp_pdist(x, metric="euclidean")
    return _sp_pdist(x, metric="minkowski", p=p)


def sum(x, axis=None, dim=None, **kwargs):
    if axis is None:
        axis = dim
    return _np.sum(x, axis=axis, **kwargs)


def mean(x, axis=None, dim=None, **kwargs):
    if axis is None:
        axis = dim
    return _np.mean(x, axis=axis, **kwargs)


def any(x, axis=None, dim=None, **kwargs):
    if axis is None:
        axis = dim
    return _np.any(x, axis=axis, **kwargs)


def logsumexp(x, axis=None, dim=None, keepdims=False, keepdim=None):
    if axis is None:
        axis = dim
    if keepdim is None:
        keepdim = keepdims
    return _sp_logsumexp(x, axis=axis, keepdims=keepdim)


def norm(x, axis=None, dim=None, **kwargs):
    if axis is None:
        axis = dim
    return _np.linalg.norm(x, axis=axis, **kwargs)


def empty(*args, device=None, **kwargs):
    return _np.empty(*args, **kwargs)


def argsort(x, axis=-1):
    return _np.argsort(x, axis=axis)


def astype(x, dtype):
    return astensor(x).astype(dtype=dtype)


def clip(x, min=None, max=None):
    return _np.clip(x, a_min=min, a_max=max)


def relu(x):
    return _np.where(x >= 0, x, 0)


def split_with_sizes(arr, sizes):
    indices = _np.cumsum(sizes)[:-1]
    return _np.split(arr, indices)


# Test keops
_is_keops_available = None


def check_keops():
    global _is_keops_available, LazyTensor
    if _is_keops_available is not None:
        return _is_keops_available
    try:
        if _is_keops_available is not None:
            return _is_keops_available
        import pykeops.numpy as pknp
        from pykeops.numpy import LazyTensor as LT

        formula = "SqNorm2(x - y)"
        var = ["x = Vi(3)", "y = Vj(3)"]
        expected_res = _np.array([63.0, 90.0])
        x = _np.arange(1, 10).reshape(-1, 3).astype("float32")
        y = _np.arange(3, 9).reshape(-1, 3).astype("float32")

        my_conv = pknp.Genred(formula, var)
        _is_keops_available = _np.allclose(my_conv(x, y).flatten(), expected_res)
        LazyTensor = LT
    except:
        _mp_logs.warn_fallback("Could not initialize keops (numpy). using workarounds")
        _is_keops_available = False

    return _is_keops_available


def from_numpy(x):
    return _np.asarray(x)


def ascontiguous(x):
    return _np.ascontiguousarray(x)


def copy(x):
    return _np.copy(x)


def sort(x, axis=-1):
    return _np.sort(x, axis=axis)


def set_at(x, idx, y):
    x[idx] = y
    return x


def add_at(x, idx, y):
    x[idx] += y
    return x


def mul_at(x, idx, y):
    x[idx] *= y
    return x


def div_at(x, idx, y):
    x[idx] /= y
    return x


def min_at(x, idx, y):
    x[idx] = _np.minimum(x[idx], y)
    return x


def max_at(x, idx, y):
    x[idx] = _np.maximum(x[idx], y)
    return x


def device(x):
    return x.device


# type: ignore[no-unused-arg]
def linspace(low, high, r, device=None, dtype=None):
    return _np.linspace(low, high, r, dtype=dtype)


def cartesian_product(*arrays, dtype=None):
    mesh = _np.meshgrid(*arrays, indexing="ij")
    coordinates = _np.stack(mesh, axis=-1).reshape(-1, len(arrays)).astype(dtype)
    return coordinates


def quantile_closest(x, q, axis=None):
    return _np.quantile(x, q, axis=axis, method="closest_observation")


def minvalues(x: _np.ndarray, **kwargs):
    return _np.min(x, **kwargs)


def maxvalues(x: _np.ndarray, **kwargs):
    return _np.max(x, **kwargs)


def is_tensor(x):
    return isinstance(x, _np.ndarray)


def is_promotable(x):
    return isinstance(x, _np.ndarray | list | tuple)


def has_grad(_):
    return False


def to_device(x, device):
    return x.to_device(device)


def size(x):
    return int(_np.size(x))


def _dtype_like(x):
    return x.dtype if hasattr(x, "dtype") and not isinstance(x, type) else x


def is_float(x):
    try:
        return _np.issubdtype(_np.dtype(_dtype_like(x)), _np.floating)
    except TypeError:
        return False


def is_int(x):
    try:
        return _np.issubdtype(_np.dtype(_dtype_like(x)), _np.integer)
    except TypeError:
        return False


def dtype_is_float(dtype):
    return is_float(dtype)


def dtype_default():
    return _np.array(0.0).dtype


def inf_value(array):
    if isinstance(array, _np.ndarray):
        dtype = _np.dtype(array.dtype)
    else:
        dtype = _np.dtype(array)
    if _np.issubdtype(dtype, _np.inexact):
        return _np.asarray(_np.inf, dtype=dtype)
    if _np.issubdtype(dtype, _np.integer):
        return _np.iinfo(dtype).max
    raise ValueError(f"`dtype` must be integer or floating like (got {dtype=}).")
