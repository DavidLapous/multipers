import numpy as np
import pytest

import multipers as mp
import multipers.ml.kernels as kernels


def test_distance_matrix_to_list():
    """Test DistanceMatrix2DistanceList transformer"""
    # Create a simple distance matrix
    dist_matrix = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
    
    transformer = kernels.DistanceMatrix2DistanceList()
    result = transformer.fit_transform([dist_matrix])
    assert result is not None
    assert len(result) == 1


def test_distance_list_to_matrix():
    """Test DistanceList2DistanceMatrix transformer"""
    # Create a simple distance list
    dist_list = np.array([1, 2, 1.5])
    
    transformer = kernels.DistanceList2DistanceMatrix()
    result = transformer.fit_transform([dist_list])
    assert result is not None
    assert len(result) == 1


def test_distance_matrices_to_lists():
    """Test DistanceMatrices2DistancesList transformer"""
    # Create distance matrices
    dist_matrices = [
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 2], [2, 0]])
    ]
    
    transformer = kernels.DistanceMatrices2DistancesList()
    result = transformer.fit_transform([dist_matrices])
    assert result is not None
    assert len(result) == 1


def test_distance_lists_to_matrices():
    """Test DistancesLists2DistanceMatrices transformer"""
    # Create distance lists
    dist_lists = [np.array([1]), np.array([2])]
    
    transformer = kernels.DistancesLists2DistanceMatrices()
    result = transformer.fit_transform([dist_lists])
    assert result is not None
    assert len(result) == 1


def test_distance_matrix_to_kernel():
    """Test DistanceMatrix2Kernel transformer"""
    # Create a simple distance matrix
    dist_matrix = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
    
    transformer = kernels.DistanceMatrix2Kernel()
    result = transformer.fit_transform([dist_matrix])
    assert result is not None
    assert len(result) == 1