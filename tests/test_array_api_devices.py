import numpy as np
import pytest


def test_numpy_to_device_accepts_none_and_cpu():
    import multipers.array_api.numpy as numpy_api

    x = np.asarray([1.0])

    assert numpy_api.to_device(x, None) is x
    assert numpy_api.to_device(x, "cpu") is x


def test_jax_to_device_accepts_cpu_string():
    pytest.importorskip("jax")
    import multipers.array_api.jax as jax_api

    x = jax_api.astensor([1.0])
    y = jax_api.to_device(x, "cpu")

    assert jax_api.device(y).platform == "cpu"
