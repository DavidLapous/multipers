def api_from_tensor(x, *, verbose: bool = False):
    import multipers.array_api.numpy as npapi

    if npapi.is_promotable(x):
        if verbose:
            print("using numpy backend")
        return npapi
    import multipers.array_api.torch as torchapi

    if torchapi.is_promotable(x):
        if verbose:
            print("using torch backend")
        return torchapi
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

def to_numpy(x):
    api = api_from_tensor(x)
    return api.asnumpy(x)
