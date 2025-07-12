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


def minvalues(x: _np.ndarray, **kwargs):
    return _np.min(x, **kwargs)


def maxvalues(x: _np.ndarray, **kwargs):
    return _np.max(x, **kwargs)


def is_promotable(x):
    return isinstance(x, _np.ndarray | list | tuple)


def has_grad(_):
    return False
