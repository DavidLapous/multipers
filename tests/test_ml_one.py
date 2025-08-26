import numpy as np
import pytest
import gudhi as gd

import multipers as mp
import multipers.ml.one as one
from multipers.tests import random_st


def test_simplextree_to_dgm():
    """Test SimplexTree2Dgm transformer"""
    # Create a simple simplex tree
    st = gd.SimplexTree()
    st.insert([0], 0.0)
    st.insert([1], 0.0) 
    st.insert([0, 1], 1.0)
    st.persistence()
    
    transformer = one.SimplexTree2Dgm()
    result = transformer.fit_transform([st])
    assert result is not None
    assert len(result) == 1


def test_dgm_to_histogram():
    """Test Dgm2Histogram transformer"""
    # Create simple persistence diagram
    dgm = np.array([[0.0, 1.0], [0.5, 2.0]])
    
    transformer = one.Dgm2Histogram()
    result = transformer.fit_transform([dgm])
    assert result is not None
    assert len(result) == 1


def test_simplextree_to_histogram():
    """Test SimplexTree2Histogram transformer"""
    # Create a simple simplex tree
    st = gd.SimplexTree()
    st.insert([0], 0.0)
    st.insert([1], 0.0)
    st.insert([0, 1], 1.0)
    st.persistence()
    
    transformer = one.SimplexTree2Histogram()
    result = transformer.fit_transform([st])
    assert result is not None
    assert len(result) == 1


def test_filvec_getter():
    """Test FilvecGetter transformer"""
    # Create a simple simplex tree
    st = gd.SimplexTree()
    st.insert([0], 0.0)
    st.insert([1], 0.0)
    st.insert([0, 1], 1.0)
    
    transformer = one.FilvecGetter()
    result = transformer.fit_transform([st])
    assert result is not None
    assert len(result) == 1


def test_dgms_to_landscapes():
    """Test Dgms2Landscapes transformer"""
    # Create simple persistence diagrams
    dgms = [np.array([[0.0, 1.0], [0.5, 2.0]])]
    
    transformer = one.Dgms2Landscapes()
    result = transformer.fit_transform([dgms])
    assert result is not None
    assert len(result) == 1


def test_dgms_to_image():
    """Test Dgms2Image transformer"""
    # Create simple persistence diagrams  
    dgms = [np.array([[0.0, 1.0], [0.5, 2.0]])]
    
    transformer = one.Dgms2Image()
    result = transformer.fit_transform([dgms])
    assert result is not None
    assert len(result) == 1