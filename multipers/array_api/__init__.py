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


