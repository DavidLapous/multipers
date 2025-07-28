import numpy as np
import pytest

import multipers.grids as mpg
import multipers as mp

try:
    import torch
except ImportError:
    torch = None


def test_regular():
    x = np.array([0, 1] + list(np.random.uniform(size=(100)))).astype(np.float32)
    (y,) = mpg.compute_grid([x], resolution=3, strategy="regular")
    assert y.dtype == np.float32
    assert np.isclose(y, [0, 0.5, 1]).all()
    if torch is not None:
        import multipers.array_api.torch as torchapi

        y_torch = torchapi.from_numpy(y)
        assert y_torch.dtype == torch.float32
        assert torch.allclose(y_torch, torch.tensor([0, 0.5, 1], dtype=torch.float32))
    else:
        pytest.skip("Skipping test as torch is not available.")


def test_regular_closest():
    x = np.asarray(
        [0.0, 0.08, 0.15, 0.3, 0.39, 0.5, 0.55, 0.7, 0.8, 0.899999, 0.9, 0.900001, 1.0]
    ).astype(np.float32)
    (y,) = mpg.compute_grid([x], strategy="regular_closest", resolution=11)
    assert y.dtype == x.dtype
    assert np.isclose(
        y, [0.0, 0.08, 0.15, 0.3, 0.39, 0.5, 0.55, 0.7, 0.8, 0.9, 1.0]
    ).all(), y
    if torch is not None:
        import multipers.array_api.torch as torchapi

        y_torch = torchapi.from_numpy(y)
        assert torch.allclose(
            y_torch,
            torch.tensor(
                [0.0, 0.08, 0.15, 0.3, 0.39, 0.5, 0.55, 0.7, 0.8, 0.9, 1.0],
                dtype=torch.float32,
            ),
        ), y_torch
    else:
        pytest.skip("Skipping test as torch is not available.")


def test_regular_left():
    x = np.asarray([0.0, 0.08, 0.1, 0.19, 0.21, 1.0]).astype(np.float32)
    (y,) = mpg.compute_grid([x], strategy="regular_left", resolution=11)
    assert y.dtype == x.dtype
    assert np.isclose(
        y,
        [
            0.0,
            .1,
            .21,
            1,
        ],
    ).all(), y
    if torch is not None and mp.array_api.check_keops():
        import multipers.array_api.torch as torchapi

        y_torch = torchapi.from_numpy(y)
        assert torch.allclose(
            y_torch,
            torch.tensor(
                [0.0, 0.1, 0.21, 1.0], dtype=torch.float32
            ),
        ), y_torch
    else:
        pytest.skip("Skipping test as torch is not available.")
