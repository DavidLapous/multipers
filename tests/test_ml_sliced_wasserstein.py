import numpy as np
import pytest

import multipers as mp
import multipers.ml.sliced_wasserstein as sliced_wasserstein


def test_sliced_wasserstein_distance():
    """Test SlicedWassersteinDistance transformer"""
    # Create simple persistence diagrams
    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0]])
    dgm2 = np.array([[0.2, 1.2], [0.7, 1.8]])
    
    transformer = sliced_wasserstein.SlicedWassersteinDistance()
    result = transformer.fit_transform([[dgm1], [dgm2]])
    assert result is not None
    assert len(result) == 2


def test_wasserstein_distance():
    """Test WassersteinDistance transformer"""
    # Create simple persistence diagrams
    dgm1 = np.array([[0.0, 1.0], [0.5, 2.0]])
    dgm2 = np.array([[0.2, 1.2], [0.7, 1.8]])
    
    transformer = sliced_wasserstein.WassersteinDistance()
    result = transformer.fit_transform([[dgm1], [dgm2]])
    assert result is not None
    assert len(result) == 2