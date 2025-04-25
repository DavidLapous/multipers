def api_from_tensor(x, verbose:bool = False):
    import numpy as np
    if isinstance(x, np.ndarray):
        import multipers.array_api.numpy as backend
        if verbose:
            print("using numpy backend")
        return backend
    import torch
    if isinstance(x,torch.Tensor):
        if verbose:
            print("using torch backend")
        import multipers.array_api.torch as backend
        return backend
    raise ValueError(f"Unsupported type {type(x)=}")

def api_from_tensors(*args):
    assert len(args) > 0, "no tensor given"
    import multipers.array_api.numpy as npapi
    is_numpy = True
    for x in args:
        if not npapi.is_promotable(x):
            is_numpy = False
            break
    if is_numpy:
        return npapi

    # only torch for now
    import multipers.array_api.torch as torchapi
    is_torch = True
    for x in args:
        if not torchapi.is_promotable(x):
            is_torch = False
            break
    if is_torch:
        return torchapi
    raise ValueError(f"Incompatible types got {[type(x) for x in args]=}.")
