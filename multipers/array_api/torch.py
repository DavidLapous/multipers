import numpy as _np
import torch as _t

backend = _t
cat = _t.cat
norm = _t.norm
astensor = _t.as_tensor
tensor = _t.tensor
stack = _t.stack
empty = _t.empty
where = _t.where
no_grad = _t.no_grad
cdist = _t.cdist
zeros = _t.zeros
min = _t.min
max = _t.max
repeat_interleave = _t.repeat_interleave
linspace = _t.linspace
cartesian_product = _t.cartesian_prod
inf = _t.inf
searchsorted = _t.searchsorted
LazyTensor = None


_is_keops_available = None


def check_keops():
    global _is_keops_available, LazyTensor
    if _is_keops_available is not None:
        return _is_keops_available
    try:
        import pykeops.torch as pknp
        from pykeops.torch import LazyTensor as LT

        formula = "SqNorm2(x - y)"
        var = ["x = Vi(3)", "y = Vj(3)"]
        expected_res = _t.tensor([63.0, 90.0])
        x = _t.arange(1, 10, dtype=_t.float32).view(-1, 3)
        y = _t.arange(3, 9, dtype=_t.float32).view(-1, 3)

        my_conv = pknp.Genred(formula, var)
        _is_keops_available = _t.allclose(
            my_conv(x, y).view(-1), expected_res.type(_t.float32)
        )
        LazyTensor = LT

    except:
        from warnings import warn

        warn("Could not initialize keops (torch). using workarounds")

        _is_keops_available = False

    return _is_keops_available


check_keops()


def from_numpy(x):
    return _t.from_numpy(x)


def ascontiguous(x):
    return _t.as_tensor(x).contiguous()


def device(x):
    return x.device


def sort(x, axis=-1):
    return _t.sort(x, dim=axis).values


# in our context, this allows to get a correct gradient.
def unique(x, assume_sorted=False, _mean=True):
    if not x.requires_grad:
        return x.unique(sorted=assume_sorted)
    if x.ndim != 1:
        raise ValueError(f"Got ndim!=1. {x=}")
    if not assume_sorted:
        x = x.sort().values
    _, c = _t.unique(x, sorted=True, return_counts=True)
    if _mean:
        x = _t.segment_reduce(data=x, reduce="mean", lengths=c, unsafe=True, axis=0)
    else:
        c = _np.concatenate([[0], _np.cumsum(c[:-1])])
        x = x[c]
    return x


def quantile_closest(x, q, axis=None):
    return _t.quantile(x, q, dim=axis, interpolation="nearest")


def minvalues(x: _t.Tensor, **kwargs):
    return _t.min(x, **kwargs).values


def maxvalues(x: _t.Tensor, **kwargs):
    return _t.max(x, **kwargs).values


def asnumpy(x: _t.Tensor):
    return x.cpu().detach().numpy()


def is_tensor(x):
    return isinstance(x, _t.Tensor)


def is_promotable(x):
    return isinstance(x, _t.Tensor)


def has_grad(x):
    return x.requires_grad
