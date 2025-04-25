import numpy as _np
backend=_np
cat = _np.concatenate
norm=_np.linalg.norm
astensor=_np.asarray
asnumpy=_np.asarray
tensor=_np.array
stack=_np.stack
empty=_np.empty
def minvalues(x:_np.ndarray,**kwargs):
    return _np.min(x, **kwargs)
def maxvalues(x:_np.ndarray,**kwargs):
    return _np.max(x, **kwargs)
zeros=_np.zeros
def is_promotable(x):
    return isinstance(x, _np.ndarray|list|tuple)
where = _np.where
