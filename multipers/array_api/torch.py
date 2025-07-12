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


def minvalues(x: _t.Tensor, **kwargs):
    return _t.min(x, **kwargs).values


def maxvalues(x: _t.Tensor, **kwargs):
    return _t.max(x, **kwargs).values


def asnumpy(x):
    return x.detach().numpy()


def is_promotable(x):
    return isinstance(x, _t.Tensor)


def has_grad(x):
    return x.requires_grad
