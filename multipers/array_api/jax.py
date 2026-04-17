from contextlib import nullcontext
from functools import partial
from typing import Any, cast
import jax as _jax
import jax.numpy as _jnp
import jax.scipy.special as _jsp_special
import numpy as _np
import multipers.array_api as _mpapi
import sys

_mpapi.add_interface("jax")

backend = _jnp
name = "jax"
_has_jit = True
int64 = _jnp.int64
ones = _jnp.ones
reshape = _jnp.reshape
arange = _jnp.arange
cat = _jnp.concatenate
det = _jnp.linalg.det
tensor = _jnp.array
stack = _jnp.stack
empty = _jnp.empty
where = _jnp.where
no_grad = nullcontext
jit = _jax.jit
zeros = _jnp.zeros
min = _jnp.min
max = _jnp.max
repeat_interleave = _jnp.repeat
linspace = _jnp.linspace
inf = _jnp.inf
searchsorted = _jnp.searchsorted
LazyTensor = cast(Any, None)
relu = _jax.nn.relu
abs = _jnp.abs
exp = _jnp.exp
log = _jnp.log
sin = _jnp.sin
cos = _jnp.cos
sinc = _jnp.sinc
sqrt = _jnp.sqrt
matmul = _jnp.matmul
einsum = _jnp.einsum


def argsort(x, axis=-1):
    return _jnp.argsort(x, axis=axis)


def sum(x, axis=None, dim=None, **kwargs):
    if axis is None:
        axis = dim
    return _jnp.sum(x, axis=axis, **kwargs)


def mean(x, axis=None, dim=None, **kwargs):
    if axis is None:
        axis = dim
    return _jnp.mean(x, axis=axis, **kwargs)


def any(x, axis=None, dim=None, **kwargs):
    if axis is None:
        axis = dim
    return _jnp.any(x, axis=axis, **kwargs)


def logsumexp(x, axis=None, dim=None, keepdims=False, keepdim=None):
    if axis is None:
        axis = dim
    if keepdim is None:
        keepdim = keepdims
    return _jsp_special.logsumexp(x, axis=axis, keepdims=keepdim)


def norm(x, axis=None, dim=None, **kwargs):
    if axis is None:
        axis = dim
    return _jnp.linalg.norm(x, axis=axis, **kwargs)


def astype(x, dtype):
    return astensor(x).astype(dtype)


def astensor(x, contiguous=False, dtype=None):
    return _jnp.asarray(x, dtype=dtype)


def check_keops():
    return False


def from_numpy(x):
    return _jnp.asarray(x)


def ascontiguous(x):
    return _jnp.asarray(x)


def copy(x):
    return _jnp.array(x, copy=True)


def device(x):
    return getattr(x, 'device', None)

def sort(x, axis=-1):
    return _jnp.sort(x, axis=axis)


def set_at(x, idx, y):
    return x.at[idx].set(y)


def add_at(x, idx, y):
    return x.at[idx].add(y)


def mul_at(x, idx, y):
    return x.at[idx].multiply(y)


def div_at(x, idx, y):
    return x.at[idx].divide(y)


def min_at(x, idx, y):
    return x.at[idx].min(y)


def max_at(x, idx, y):
    return x.at[idx].max(y)


def unique(x, assume_sorted=False, _mean=False):
    if x.size == 0:
        return x
    if not assume_sorted:
        x = _jnp.sort(x)

    if _mean:
        # This part is tricky in JAX without boolean indexing/masking that changes shape
        # But for now, let's implement basic unique
        return _jnp.unique(x)

    return _jnp.unique(x)


def quantile_closest(x, q, axis=None):
    # JAX quantile doesn't have 'nearest' interpolation in the same way?
    # Actually it has 'method' in newer versions, but let's be safe.
    return _jnp.quantile(x, q, axis=axis)


def minvalues(x, **kwargs):
    return _jnp.min(x, **kwargs)


def maxvalues(x, **kwargs):
    return _jnp.max(x, **kwargs)


@partial(jit, static_argnames=("p",))
def cdist(x, y, p=2):
    diff = _jnp.abs(x[:, None, :] - y[None, :, :])
    if p == 1:
        return _jnp.sum(diff, axis=-1)
    if p == 2:
        return _jnp.sqrt(_jnp.sum(diff * diff, axis=-1))
    return _jnp.sum(diff**p, axis=-1) ** (1.0 / p)


@partial(jit, static_argnames=("p",))
def pdist(x, p=2):
    distances = cdist(x, x, p=p)
    row_idx, col_idx = _jnp.triu_indices(x.shape[0], k=1)
    return distances[row_idx, col_idx]


def asnumpy(x, dtype=None):
    return _np.asarray(_jax.lax.stop_gradient(x), dtype=dtype)


def is_tensor(x):
    return isinstance(x, _jax.Array)


def is_promotable(x):
    return isinstance(x, (_jax.Array, _np.ndarray, list, tuple))


def has_grad(x):
    # JAX arrays do not carry a `requires_grad` flag like torch tensors do.
    # The closest equivalent is whether the value is currently being traced by a
    # JAX transformation such as `grad`, `jit`, or `vmap`.
    return isinstance(x, _jax.core.Tracer)


def to_device(x, device):
    if device is None:
        return x
    return _jax.device_put(x, device)


def size(x):
    return x.size


def _dtype_like(x):
    return x.dtype if hasattr(x, "dtype") and not isinstance(x, type) else x


def is_float(x):
    dtype = _dtype_like(x)
    return _jnp.issubdtype(dtype, _jnp.floating)


def is_int(x):
    dtype = _dtype_like(x)
    return _jnp.issubdtype(dtype, _jnp.integer)


def dtype_is_float(dtype):
    return is_float(dtype)


def dtype_default():
    return _jnp.array(0.0).dtype


def inf_value(array):
    dtype = array.dtype if hasattr(array, "dtype") else _jnp.dtype(array)
    if _jnp.issubdtype(dtype, _jnp.inexact):
        return _jnp.asarray(_jnp.inf, dtype=dtype)
    if _jnp.issubdtype(dtype, _jnp.integer):
        return _jnp.iinfo(dtype).max
    raise ValueError(f"`dtype` must be integer or floating like (got {dtype=}).")


def moveaxis(x, source, destination):
    return _jnp.moveaxis(x, source, destination)


def cartesian_product(*arrays, dtype=None):
    # JAX doesn't have cartesian_prod like torch, but meshgrid works
    mesh = _jnp.meshgrid(*arrays, indexing="ij")
    result = _jnp.stack(mesh, axis=-1).reshape(-1, len(arrays))
    if dtype is not None:
        result = result.astype(dtype)
    return result
