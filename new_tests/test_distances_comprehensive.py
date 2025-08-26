"""
Comprehensive tests for multipers.distances module
"""
import numpy as np
import pytest
import multipers as mp


class TestSignedMeasureDistance:
    """Test signed measure distance computations"""
    
    def test_sm_distance_basic(self):
        """Test basic signed measure distance computation"""
        # Create simple test signed measures
        sm1 = (np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
        sm2 = (np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
        
        # Distance between identical measures should be 0
        distance = mp.distances.sm_distance(sm1, sm2)
        assert np.isclose(distance, 0.0, atol=1e-10)
    
    def test_sm_distance_different_measures(self):
        """Test distance between different signed measures"""
        sm1 = (np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
        sm2 = (np.array([[0, 1], [1, 2]]), np.array([0.5, -0.5]))
        
        distance = mp.distances.sm_distance(sm1, sm2)
        assert distance > 0
        assert np.isfinite(distance)
    
    def test_sm_distance_with_regularization(self):
        """Test signed measure distance with different regularization parameters"""
        sm1 = (np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
        sm2 = (np.array([[0, 1], [1, 2]]), np.array([0.5, -0.5]))
        
        # Test with different regularization values
        reg_values = [0.01, 0.1, 1.0]
        distances = []
        
        for reg in reg_values:
            dist = mp.distances.sm_distance(sm1, sm2, reg=reg)
            distances.append(dist)
            assert np.isfinite(dist)
            assert dist >= 0
        
        # Generally, higher regularization should give different results
        assert not np.allclose(distances)
    
    def test_sm_distance_symmetry(self):
        """Test that signed measure distance is symmetric"""
        sm1 = (np.array([[0, 1], [1, 2], [0.5, 1.5]]), np.array([1.0, -1.0, 0.5]))
        sm2 = (np.array([[0.2, 1.1], [1.1, 2.1]]), np.array([0.8, -0.8]))
        
        dist12 = mp.distances.sm_distance(sm1, sm2)
        dist21 = mp.distances.sm_distance(sm2, sm1)
        
        assert np.isclose(dist12, dist21, rtol=1e-10)
    
    def test_sm_distance_triangle_inequality(self):
        """Test triangle inequality for signed measure distance"""
        # Create three different signed measures
        sm1 = (np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
        sm2 = (np.array([[0.1, 1.1], [0.9, 1.9]]), np.array([0.9, -0.9]))
        sm3 = (np.array([[0.2, 1.2], [0.8, 1.8]]), np.array([0.8, -0.8]))
        
        d12 = mp.distances.sm_distance(sm1, sm2)
        d23 = mp.distances.sm_distance(sm2, sm3)
        d13 = mp.distances.sm_distance(sm1, sm3)
        
        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        assert d13 <= d12 + d23 + 1e-10  # Small tolerance for numerical errors
    
    def test_sm_distance_empty_measures(self):
        """Test distance computation with empty measures"""
        # Empty signed measures
        sm_empty = (np.array([]).reshape(0, 2), np.array([]))
        sm_non_empty = (np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
        
        # Distance between empty measure and itself should be 0
        dist_empty = mp.distances.sm_distance(sm_empty, sm_empty)
        assert np.isclose(dist_empty, 0.0, atol=1e-10)
        
        # Distance between empty and non-empty should be positive
        dist_mixed = mp.distances.sm_distance(sm_empty, sm_non_empty)
        assert dist_mixed > 0


class TestDistanceUtilities:
    """Test utility functions in distances module"""
    
    def test_distance_function_existence(self):
        """Test that expected distance functions exist"""
        assert hasattr(mp.distances, 'sm_distance')
        assert callable(mp.distances.sm_distance)
    
    def test_distance_input_validation(self):
        """Test input validation for distance functions"""
        # Test with invalid input formats
        invalid_sm = ([1, 2, 3], [1, -1])  # Not numpy arrays
        valid_sm = (np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
        
        # This might raise an error or handle gracefully - test both cases
        try:
            result = mp.distances.sm_distance(invalid_sm, valid_sm)
            # If it doesn't raise an error, result should still be valid
            assert np.isfinite(result)
        except (TypeError, ValueError):
            # It's also acceptable to raise an error for invalid input
            pass


class TestDistanceParameters:
    """Test distance functions with various parameter combinations"""
    
    def test_sm_distance_parameter_validation(self):
        """Test parameter validation for sm_distance"""
        sm1 = (np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
        sm2 = (np.array([[0, 1], [1, 2]]), np.array([0.5, -0.5]))
        
        # Test with negative regularization (should handle gracefully or error)
        try:
            dist = mp.distances.sm_distance(sm1, sm2, reg=-0.1)
            # If it doesn't error, result should still be meaningful
            assert np.isfinite(dist)
        except ValueError:
            # It's acceptable to raise an error for negative regularization
            pass
        
        # Test with zero regularization
        try:
            dist = mp.distances.sm_distance(sm1, sm2, reg=0.0)
            assert np.isfinite(dist)
        except (ValueError, ZeroDivisionError):
            # Zero regularization might cause numerical issues
            pass
    
    @pytest.mark.parametrize("reg", [0.001, 0.01, 0.1, 1.0, 10.0])
    def test_sm_distance_regularization_range(self, reg):
        """Test sm_distance with different regularization values"""
        sm1 = (np.array([[0, 1], [1, 2], [0.5, 1.5]]), np.array([1.0, -0.5, 0.3]))
        sm2 = (np.array([[0.1, 1.1], [0.9, 1.9]]), np.array([0.8, -0.4]))
        
        distance = mp.distances.sm_distance(sm1, sm2, reg=reg)
        assert np.isfinite(distance)
        assert distance >= 0


class TestDistanceCornerCases:
    """Test corner cases and edge conditions for distance functions"""
    
    def test_sm_distance_identical_points_different_weights(self):
        """Test distance when points are identical but weights differ"""
        points = np.array([[0, 1], [1, 2]])
        sm1 = (points, np.array([1.0, -1.0]))
        sm2 = (points, np.array([0.5, -0.5]))
        
        distance = mp.distances.sm_distance(sm1, sm2)
        assert distance > 0  # Should be positive since weights differ
    
    def test_sm_distance_different_points_same_weights(self):
        """Test distance when points differ but weights are the same"""
        sm1 = (np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
        sm2 = (np.array([[0.1, 1.1], [1.1, 2.1]]), np.array([1.0, -1.0]))
        
        distance = mp.distances.sm_distance(sm1, sm2)
        assert distance > 0  # Should be positive since points differ
    
    def test_sm_distance_single_point_measures(self):
        """Test distance computation with single-point measures"""
        sm1 = (np.array([[0.5, 1.5]]), np.array([1.0]))
        sm2 = (np.array([[0.6, 1.4]]), np.array([1.0]))
        
        distance = mp.distances.sm_distance(sm1, sm2)
        assert np.isfinite(distance)
        assert distance >= 0
    
    def test_sm_distance_different_dimensions(self):
        """Test behavior with different dimensional measures"""
        # This test checks what happens when measures have different structures
        sm_2d = (np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
        sm_3d = (np.array([[0, 1, 0.5], [1, 2, 1.5]]), np.array([1.0, -1.0]))
        
        try:
            # This might work or raise an error depending on implementation
            distance = mp.distances.sm_distance(sm_2d, sm_3d)
            assert np.isfinite(distance)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for dimensional mismatch
            pass


class TestDistanceNumericalStability:
    """Test numerical stability of distance functions"""
    
    def test_sm_distance_large_values(self):
        """Test distance computation with large coordinate values"""
        large_coords = np.array([[1e6, 2e6], [3e6, 4e6]])
        sm1 = (large_coords, np.array([1.0, -1.0]))
        sm2 = (large_coords * 1.01, np.array([1.0, -1.0]))
        
        distance = mp.distances.sm_distance(sm1, sm2)
        assert np.isfinite(distance)
        assert distance >= 0
    
    def test_sm_distance_small_values(self):
        """Test distance computation with very small coordinate values"""
        small_coords = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        sm1 = (small_coords, np.array([1.0, -1.0]))
        sm2 = (small_coords * 1.1, np.array([1.0, -1.0]))
        
        distance = mp.distances.sm_distance(sm1, sm2)
        assert np.isfinite(distance)
        assert distance >= 0
    
    def test_sm_distance_mixed_scale_weights(self):
        """Test distance with weights of very different scales"""
        points = np.array([[0, 1], [1, 2], [2, 3]])
        sm1 = (points, np.array([1e-6, 1e6, 1.0]))
        sm2 = (points, np.array([2e-6, 0.5e6, 2.0]))
        
        distance = mp.distances.sm_distance(sm1, sm2)
        assert np.isfinite(distance)
        assert distance >= 0


@pytest.mark.parametrize("n_points", [1, 5, 10, 50])
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_sm_distance_scalability(n_points, dim):
    """Test sm_distance performance with different problem sizes"""
    # Generate random signed measures
    np.random.seed(42)
    points1 = np.random.randn(n_points, dim)
    points2 = np.random.randn(n_points, dim)
    weights1 = np.random.randn(n_points)
    weights2 = np.random.randn(n_points)
    
    sm1 = (points1, weights1)
    sm2 = (points2, weights2)
    
    distance = mp.distances.sm_distance(sm1, sm2)
    assert np.isfinite(distance)
    assert distance >= 0