import torch as _t
backend=_t
cat=_t.cat
norm=_t.norm
astensor=_t.as_tensor
def asnumpy(x):
    return x.detach().numpy()
