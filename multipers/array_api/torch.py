from functools import wraps
from typing import Any, cast

import torch as _torch

import multipers.array_api as _mpapi
import multipers.logs as _mp_logs

_mpapi.add_interface("torch")

backend = _torch
_has_jit = False
LazyTensor = cast(Any, None)
_is_keops_available = None
int64 = _torch.int64
inf = float("inf")

ones = _torch.ones
reshape = _torch.reshape
arange = _torch.arange
cat = _torch.cat
det = _torch.linalg.det
tensor = _torch.tensor
stack = _torch.stack
empty = _torch.empty
where = _torch.where
no_grad = _torch.no_grad
cdist = _torch.cdist
pdist = _torch.pdist
zeros = _torch.zeros
min = _torch.min
max = _torch.max
repeat_interleave = _torch.repeat_interleave
linspace = _torch.linspace
cartesian_product = _torch.cartesian_prod
searchsorted = _torch.searchsorted
relu = _torch.relu
abs = _torch.abs
exp = _torch.exp
log = _torch.log
sin = _torch.sin
cos = _torch.cos
sinc = _torch.sinc
sqrt = _torch.sqrt
matmul = _torch.matmul
einsum = _torch.einsum
moveaxis = _torch.moveaxis
from_numpy = _torch.from_numpy


def jit(fn=None, **kwargs):
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **inner_kwargs):
            return func(*args, **inner_kwargs)

        return wrapped

    if fn is None:
        return decorator
    return decorator(fn)


def argsort(x, axis=-1):
    return _torch.argsort(x, dim=axis)


def sum(x, axis=None, dim=None, **kwargs):
    if dim is None:
        dim = axis
    return _torch.sum(x, dim=dim, **kwargs)


def mean(x, axis=None, dim=None, **kwargs):
    if dim is None:
        dim = axis
    return _torch.mean(x, dim=dim, **kwargs)


def any(x, axis=None, dim=None, **kwargs):
    if dim is None:
        dim = axis
    return _torch.any(x, dim=dim, **kwargs)


def logsumexp(x, axis=None, dim=None, keepdims=False, keepdim=None):
    if dim is None:
        dim = axis
    if keepdim is None:
        keepdim = keepdims
    if dim is None:
        if x.ndim == 0:
            return x
        dim = tuple(range(x.ndim))
    return _torch.logsumexp(x, dim=dim, keepdim=keepdim)


def norm(x, axis=None, dim=None, **kwargs):
    if dim is None:
        dim = axis
    return _torch.norm(x, dim=dim, **kwargs)


def astype(x, dtype):
    return astensor(x).to(dtype=dtype)


def astensor(x, contiguous=False, dtype=None):
    out = _torch.as_tensor(x, dtype=dtype)
    if contiguous:
        out = out.contiguous()
    return out


def clip(x, min=None, max=None):
    return _torch.clamp(x, min=min, max=max)


def split_with_sizes(arr, sizes):
    return arr.split_with_sizes(sizes)


def check_keops():
    global _is_keops_available, LazyTensor
    if _is_keops_available is not None:
        return _is_keops_available

    try:
        import pykeops.torch as pknp
        from pykeops.torch import LazyTensor as LT

        formula = "SqNorm2(x - y)"
        var = ["x = Vi(3)", "y = Vj(3)"]
        expected_res = _torch.tensor([63.0, 90.0])
        x = _torch.arange(1, 10, dtype=_torch.float32).view(-1, 3)
        y = _torch.arange(3, 9, dtype=_torch.float32).view(-1, 3)

        my_conv = pknp.Genred(formula, var)
        _is_keops_available = _torch.allclose(
            my_conv(x, y).view(-1), expected_res.type(_torch.float32)
        )
        LazyTensor = LT
    except Exception:
        _mp_logs.warn_fallback("Could not initialize keops (torch). using workarounds")
        _is_keops_available = False

    return _is_keops_available


def ascontiguous(x):
    return _torch.as_tensor(x).contiguous()


def copy(x):
    return x.clone()


def device(x):
    return x.device


def sort(x, axis=-1):
    return _torch.sort(x, dim=axis).values


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
    x[idx] = _torch.min(x[idx], y)
    return x


def max_at(x, idx, y):
    x[idx] = _torch.max(x[idx], y)
    return x


def unique(x, assume_sorted=False, _mean=False):
    if not x.requires_grad:
        return x.unique(sorted=assume_sorted)
    if x.ndim != 1:
        raise ValueError(f"Got ndim!=1. {x=}")
    if x.numel() == 0:
        return x
    if not assume_sorted:
        x = x.sort().values
    _, counts = _torch.unique_consecutive(x, return_counts=True)
    if _mean:
        return _torch.segment_reduce(
            data=x, reduce="mean", lengths=counts, unsafe=True, axis=0
        )
    starts = _torch.cumsum(counts, dim=0)
    starts.sub_(counts)
    return x[starts]


def quantile_closest(x, q, axis=None):
    return _torch.quantile(x, q, dim=axis, interpolation="nearest")


def minvalues(x, axis=None, dim=None, keepdims=False, keepdim=None):
    if dim is None:
        dim = axis
    if keepdim is None:
        keepdim = keepdims
    if dim is not None:
        keepdim = bool(keepdim)
        return _torch.min(x, dim, keepdim).values
    return _torch.min(x)


def maxvalues(x, axis=None, dim=None, keepdims=False, keepdim=None):
    if dim is None:
        dim = axis
    if keepdim is None:
        keepdim = keepdims
    if dim is not None:
        keepdim = bool(keepdim)
        return _torch.max(x, dim, keepdim).values
    return _torch.max(x)


def asnumpy(x, dtype=None):
    out = x.cpu().detach().numpy()
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def is_tensor(x):
    return isinstance(x, _torch.Tensor)


def is_promotable(x):
    return isinstance(x, (_torch.Tensor, list, tuple))


def has_grad(x):
    return getattr(x, "requires_grad", False)


def to_device(x, device):
    if device is None:
        return x
    return x.to(device)


def size(x):
    return x.numel()


def dtype_is_float(dtype):
    return False if dtype is None else dtype.is_floating_point


def dtype_default():
    return _torch.get_default_dtype()
