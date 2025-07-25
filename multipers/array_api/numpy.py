from contextlib import nullcontext

import numpy as _np
from pykeops.numpy import LazyTensor
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
LazyTensor = LazyTensor  # type: ignore[no-redef]


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
