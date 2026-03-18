from typing import Literal
import multipers.array_api.numpy as npapi

available_api = [npapi]
available_interfaces = Literal["numpy"]


def add_interface(interface: str):
    global available_interfaces
    available_api.append(interface)
    available_interfaces = Literal[*available_api]


def api_from_tensor(x, *, verbose: bool = False, strict=False):
    from importlib.util import find_spec

    if strict:
        if npapi.is_tensor(x):
            return npapi
        if find_spec("torch"):
            import multipers.array_api.torch as torchapi

            if torchapi.is_tensor(x):
                return torchapi
        if find_spec("jax"):
            try:
                import multipers.array_api.jax as jaxapi

                if jaxapi.is_tensor(x):
                    return jaxapi
            except ImportError:
                pass
        raise ValueError(f"Unsupported (strict) type {type(x)=}")
    if npapi.is_promotable(x):
        if verbose:
            print("using numpy backend")
        return npapi

    if find_spec("torch"):
        import multipers.array_api.torch as torchapi

        if torchapi.is_promotable(x):
            if verbose:
                print("using torch backend")
            return torchapi

    if find_spec("jax"):
        try:
            import multipers.array_api.jax as jaxapi

            if jaxapi.is_tensor(x):  # Check specifically for jax array before torch
                if verbose:
                    print("using jax backend")
                return jaxapi
        except ImportError:
            pass
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

    from importlib.util import find_spec

    if find_spec("torch"):
        import multipers.array_api.torch as torchapi

        is_torch = True
        for x in args:
            if not torchapi.is_promotable(x):
                is_torch = False
                break
        if is_torch:
            return torchapi

    if find_spec("jax"):
        import multipers.array_api.jax as jaxapi

        is_jax = True
        for x in args:
            if not jaxapi.is_promotable(x):
                is_jax = False
                break
        if is_jax:
            # Check if there is at least one jax array, else it might be torch
            has_jax = False
            for x in args:
                if jaxapi.is_tensor(x):
                    has_jax = True
                    break
            if has_jax:
                return jaxapi
    raise ValueError(f"Incompatible types got {[type(x) for x in args]=}.")


def to_numpy(x):
    api = api_from_tensor(x)
    return api.asnumpy(x)


def check_keops():
    import os

    if os.name == "nt":
        # see https://github.com/getkeops/keops/pull/421
        return False
    return npapi.check_keops()
