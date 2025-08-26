import numpy as np
import pytest
import platform

import multipers as mp
import multipers.ml.mma as mma
import multipers.ml.signed_measures as signed_measures
import multipers.ml.tools as tools
from multipers.tests import random_st


def test_filtered_complex_to_mma():
    """Test FilteredComplex2MMA transformer"""
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    
    transformer = mma.FilteredComplex2MMA()
    result = transformer.fit_transform([[st]])
    assert result is not None
    assert len(result) == 1


def test_mma_formatter():
    """Test MMAFormatter transformer"""
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    
    # First get MMA
    mma_transformer = mma.FilteredComplex2MMA()
    mma_result = mma_transformer.fit_transform([[st]])
    
    # Then format
    formatter = mma.MMAFormatter()
    result = formatter.fit_transform(mma_result)
    assert result is not None


def test_filtered_complex_to_signed_measure():
    """Test FilteredComplex2SignedMeasure transformer"""
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    
    transformer = signed_measures.FilteredComplex2SignedMeasure()
    result = transformer.fit_transform([[st]])
    assert result is not None
    assert len(result) == 1


def test_signed_measure_formatter():
    """Test SignedMeasureFormatter transformer"""
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0, 1])
    st.insert([1], [1, 0])
    st.insert([0, 1], [1, 1])
    
    # First get signed measures
    sm_transformer = signed_measures.FilteredComplex2SignedMeasure()
    sm_result = sm_transformer.fit_transform([[st]])
    
    # Then format
    formatter = signed_measures.SignedMeasureFormatter()
    result = formatter.fit_transform(sm_result)
    assert result is not None


def test_simplex_tree_edge_collapser():
    """Test SimplexTreeEdgeCollapser from tools"""
    st = random_st(num_parameters=2)
    
    collapser = tools.SimplexTreeEdgeCollapser()
    result = collapser.fit_transform([st])
    assert result is not None
    assert len(result) == 1


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Detected windows. Pykeops is not compatible with windows yet. Skipping this ftm.",
)
def test_point_cloud_to_filtered_complex():
    """Test point cloud to filtered complex pipeline"""
    import multipers.ml.point_clouds as mmp
    
    pts = np.array([[1, 1], [2, 2]], dtype=np.float32)
    
    # Test basic functionality
    transformer = mmp.PointCloud2FilteredComplex(masses=[0.1])
    result = transformer.fit_transform([pts])
    assert result is not None
    assert len(result) == 1
    
    # Check result type
    st = result[0][0]
    assert isinstance(st, mp.simplex_tree_multi.SimplexTreeMulti_type)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Detected windows. Pykeops is not compatible with windows yet. Skipping this ftm.",
)
def test_point_cloud_alpha_complex():
    """Test point cloud with alpha complex"""
    import multipers.ml.point_clouds as mmp
    
    pts = np.array([[1, 1], [2, 2]], dtype=np.float32)
    
    transformer = mmp.PointCloud2FilteredComplex(
        bandwidths=[-0.1], complex="alpha"
    )
    result = transformer.fit_transform([pts])
    assert result is not None
    
    st = result[0][0]
    assert isinstance(st, mp.simplex_tree_multi.SimplexTreeMulti_type)