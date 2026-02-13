from contextlib import nullcontext

import numpy as _np
from scipy.spatial.distance import cdist

backend = _np
cat = _np.concatenate
norm = _np.linalg.norm
astensor = _np.asarray
asnumpy = _np.asarray
tensor = _np.array
stack = _np.stack
empty = _np.empty
where = _np.where
no_grad = nullcontext
zeros = _np.zeros
min = _np.min
max = _np.max
repeat_interleave = _np.repeat
cdist = cdist  # type: ignore[no-redef]
unique = _np.unique
inf = _np.inf
searchsorted = _np.searchsorted
LazyTensor = None
abs = _np.abs
exp = _np.exp
log = _np.log
sin = _np.sin
cos = _np.cos
matmul = _np.matmul
einsum = _np.einsum


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
        from warnings import warn

        warn("Could not initialize keops (numpy). using workarounds")
        _is_keops_available = False

    return _is_keops_available


def from_numpy(x):
    return _np.asarray(x)


def ascontiguous(x):
    return _np.ascontiguousarray(x)


def sort(x, axis=-1):
    return _np.sort(x, axis=axis)


def device(x):  # type: ignore[no-unused-arg]
    return None


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
    if device is None or str(device) in {"None", "cpu"}:
        return x
    raise ValueError(
        f"NumPy backend only supports CPU tensors, requested device {device!r}."
    )


def size(x):
    return int(_np.size(x))


def dtype_is_float(dtype):
    try:
        return _np.issubdtype(_np.dtype(dtype), _np.floating)
    except TypeError:
        return False


def dtype_default():
    return _np.array(0.0).dtype
