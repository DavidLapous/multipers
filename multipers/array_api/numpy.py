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
