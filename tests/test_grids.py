import numpy as np
import pytest

import multipers as mp
import multipers.grids as mpg

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
            0.1,
            0.21,
            1,
        ],
    ).all(), y
    if torch is not None and mp.array_api.check_keops():
        import multipers.array_api.torch as torchapi

        y_torch = torchapi.from_numpy(y)
        assert torch.allclose(
            y_torch,
            torch.tensor([0.0, 0.1, 0.21, 1.0], dtype=torch.float32),
        ), y_torch
    else:
        pytest.skip("Skipping test as torch is not available.")


def test_sanity_numpy():
    for k in range(2, 4):
        img = np.random.random(size=(50, 50, k))
        s = mp.filtrations.Cubical(img)
        for strat in mp.grids.available_strategies:
            _s = s.grid_squeeze(
                strategy=strat, resolution=None if strat == "exact" else 10
            )
            f = _s.filtration_grid[0]
            if strat not in ["exact", "precomputed"]:
                assert len(f) <= 10, f"invalid resolution for {strat=}"


@pytest.mark.skipif(torch is None, reason="Torch not installed.")
def test_sanity_torch():
    import torch

    for k in range(2, 4):
        img = torch.rand(size=(50, 50, k)).requires_grad_()
        s = mp.filtrations.Cubical(img)
        for strat in mp.grids.available_strategies:
            _s = s.grid_squeeze(
                strategy=strat, resolution=None if strat == "exact" else 10
            )
            f = _s.filtration_grid[0]
            if strat not in ["regular", "partition"]:
                assert f.requires_grad, f"Grad not working for {strat=}"
            if strat not in ["exact", "precomputed"]:
                assert len(f) <= 10, f"invalid resolution for {strat=}"


def test_errors_and_edge_cases():
    # empty input should return an empty grid
    out = mpg.compute_grid([])
    assert out == []

    # providing a resolution for exact strategy is invalid
    with pytest.raises(ValueError):
        mpg.compute_grid([np.array([0.0, 1.0])], resolution=5, strategy="exact")

    # missing resolution for non-exact strategy should raise (assertion)
    with pytest.raises(ValueError):
        mpg.compute_grid([np.array([0.0, 1.0])], resolution=None, strategy="regular")

    # invalid strategy name should raise ValueError
    with pytest.raises(ValueError):
        mpg.compute_grid(
            [np.array([0.0, 1.0])], resolution=3, strategy="this_is_not_a_strategy"
        )

    # sanitize_grid should raise on empty grid
    with pytest.raises(ValueError):
        mpg.sanitize_grid((), numpyfy=True)


def test_quantile_and_partition_and_torch():
    # quantile strategy should return requested number of quantiles including endpoints
    x = np.linspace(0.0, 1.0, 101).astype(np.float32)
    (y,) = mpg.compute_grid([x], resolution=5, strategy="quantile")
    assert y.dtype == x.dtype
    assert len(y) == 5
    assert np.isclose(y[0], 0.0)
    assert np.isclose(y[-1], 1.0)

    # partition strategy should select values from the original array
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    (p,) = mpg.compute_grid([data], resolution=4, strategy="partition")
    assert p.dtype == data.dtype
    assert len(p) <= 4
    assert set(np.asarray(p)).issubset(set(np.asarray(data)))

    # torch backend: ensure dtype and values are preserved for regular strategy
    if torch is not None:
        import multipers.array_api.torch as torchapi

        t = torch.linspace(0.0, 1.0, 101, dtype=torch.float32)
        (rt,) = mpg.compute_grid([t], resolution=5, strategy="regular")
        # rt may be a backend tensor; convert to numpy for checks
        rt_np = torchapi.asnumpy(rt) if hasattr(torchapi, "asnumpy") else np.asarray(rt)
        assert np.isclose(rt_np, np.linspace(0.0, 1.0, 5)).all()
    else:
        pytest.skip("Skipping torch-specific checks as torch is not available.")
