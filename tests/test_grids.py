import numpy as np

import multipers.grids as mpg


def test_regular():
    x = np.array([0, 1] + list(np.random.uniform(size=(100)))
                 ).astype(np.float32)
    (y,) = mpg.compute_grid([x], resolution=3, strategy="regular")
    assert y.dtype == np.float32
    assert np.isclose(y, [0, 0.5, 1]).all()


def test_regular_closest():
    x = [0.0, 0.08, 0.15, 0.3, 0.39, 0.5, 0.55,
         0.7, 0.8, 0.899999, 0.9, 0.900001, 1.0]
    (y,) = mpg.compute_grid([x], strategy="regular_closest", resolution=11)
    assert np.isclose(
        y, [0.0, 0.08, 0.15, 0.3, 0.39, 0.5, 0.55, 0.7, 0.8, 0.9, 1.0]
    ).all(), y
