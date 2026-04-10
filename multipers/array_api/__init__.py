from typing import Any, Literal, cast
import multipers.array_api.numpy as npapi

available_api: list[Any] = [npapi]
available_interfaces = Literal["numpy"]


def add_interface(interface: str):
    global available_interfaces
    available_api.append(cast(Any, interface))
    available_interfaces = Literal.__getitem__(tuple(available_api))


def _module_name(x):
    return getattr(type(x), "__module__", "")


def _looks_like_torch(x):
    return _module_name(x).startswith("torch")


def _looks_like_jax(x):
    module = _module_name(x)
    return module.startswith("jax")


def _has_jit(api):
    return getattr(api, "_has_jit", False)


def api_from_tensor(x, *, verbose: bool = False, strict=False):
    from importlib.util import find_spec

    if strict:
        if npapi.is_tensor(x):
            return npapi
        if _looks_like_torch(x) and find_spec("torch"):
            import multipers.array_api.torch as torchapi

            if torchapi.is_tensor(x):
                return torchapi
        if _looks_like_jax(x) and find_spec("jax"):
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

    if _looks_like_torch(x) and find_spec("torch"):
        import multipers.array_api.torch as torchapi

        if torchapi.is_promotable(x):
            if verbose:
                print("using torch backend")
            return torchapi

    if _looks_like_jax(x) and find_spec("jax"):
        try:
            import multipers.array_api.jax as jaxapi

            if jaxapi.is_tensor(x):  # Check specifically for jax array before torch
                if verbose:
                    print("using jax backend")
                return jaxapi
        except ImportError:
            pass
    raise ValueError(f"Unsupported type {type(x)=}")


def api_from_tensors(*args, jit_promote: bool = False):
    if len(args) == 0:
        raise ValueError("no tensor given")
    import multipers.array_api.numpy as npapi

    is_numpy = True
    for x in args:
        if not npapi.is_promotable(x):
            is_numpy = False
            break
    if is_numpy:
        if jit_promote and not _has_jit(npapi):
            from importlib.util import find_spec

            if find_spec("jax"):
                import multipers.array_api.jax as jaxapi

                if _has_jit(jaxapi):
                    return jaxapi
        return npapi

    from importlib.util import find_spec

    has_torch = any(_looks_like_torch(x) for x in args)
    if has_torch and find_spec("torch"):
        import multipers.array_api.torch as torchapi

        is_torch = True
        for x in args:
            if not torchapi.is_promotable(x):
                is_torch = False
                break
        if is_torch:
            return torchapi

    has_jax = any(_looks_like_jax(x) for x in args)
    if has_jax and find_spec("jax"):
        import multipers.array_api.jax as jaxapi

        is_jax = True
        for x in args:
            if not jaxapi.is_promotable(x):
                is_jax = False
                break
        if is_jax:
            # Check if there is at least one jax array, else it might be torch
            has_jax_tensor = False
            for x in args:
                if jaxapi.is_tensor(x):
                    has_jax_tensor = True
                    break
            if has_jax_tensor:
                return jaxapi
    raise ValueError(f"Incompatible types got {[type(x) for x in args]=}.")


def to_numpy(x, dtype=None):
    api = api_from_tensor(x)
    return api.asnumpy(x, dtype=dtype)


def check_keops():
    import os

    if os.name == "nt":
        # see https://github.com/getkeops/keops/pull/421
        return False
    return npapi.check_keops()
