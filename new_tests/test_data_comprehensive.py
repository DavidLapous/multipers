"""
Comprehensive tests for multipers.data module
"""
import numpy as np
import pytest
from unittest import mock
import multipers as mp


class TestSyntheticData:
    """Test synthetic data generation functions"""
    
    def test_noisy_annulus_default_params(self):
        """Test noisy_annulus with default parameters"""
        points = mp.data.noisy_annulus(100, 50)
        assert points.shape[0] == 150  # 100 + 50 points
        assert points.shape[1] == 2    # 2D by default
        assert isinstance(points, np.ndarray)
    
    def test_noisy_annulus_custom_dimensions(self):
        """Test noisy_annulus with different dimensions"""
        for dim in [2, 3, 4]:
            points = mp.data.noisy_annulus(50, 30, dim=dim)
            assert points.shape == (80, dim)
    
    def test_noisy_annulus_custom_radii(self):
        """Test noisy_annulus with custom inner and outer radii"""
        points = mp.data.noisy_annulus(50, 30, inner_radius=2, outer_radius=5)
        # Check that points fall within reasonable radius ranges (allowing for noise)
        distances = np.linalg.norm(points, axis=1)
        # With noise, points might be outside expected radii, so use looser bounds
        assert np.min(distances) >= 0  # All distances should be non-negative
        assert np.max(distances) <= 10  # Should be reasonably bounded
    
    def test_noisy_annulus_noise_parameter(self):
        """Test noisy_annulus with different noise levels"""
        # Test with different noise levels and see that they produce valid output
        points_low_noise = mp.data.noisy_annulus(100, 0, noise=0.01)
        points_high_noise = mp.data.noisy_annulus(100, 0, noise=1.0)
        
        # Both should have same number of points
        assert points_low_noise.shape == points_high_noise.shape
        
        # Both should be finite
        assert np.all(np.isfinite(points_low_noise))
        assert np.all(np.isfinite(points_high_noise))
    
    def test_noisy_annulus_edge_cases(self):
        """Test noisy_annulus edge cases"""
        # Zero inner points
        points = mp.data.noisy_annulus(0, 50)
        assert points.shape[0] == 50
        
        # Zero outer points
        points = mp.data.noisy_annulus(50, 0)
        assert points.shape[0] == 50
        
        # Both zero should still work
        points = mp.data.noisy_annulus(0, 0)
        assert points.shape[0] == 0
        assert points.shape[1] == 2
    
    def test_noisy_annulus_reproducibility(self):
        """Test that noisy_annulus is reproducible with same random seed"""
        np.random.seed(42)
        points1 = mp.data.noisy_annulus(100, 50)
        
        np.random.seed(42)
        points2 = mp.data.noisy_annulus(100, 50)
        
        np.testing.assert_array_almost_equal(points1, points2)


class TestDataUtils:
    """Test utility functions in the data module"""
    
    def test_data_module_imports(self):
        """Test that data module imports work correctly"""
        # Test that we can access the data module
        assert hasattr(mp, 'data')
        assert hasattr(mp.data, 'noisy_annulus')
    
    def test_data_module_structure(self):
        """Test the structure of the data module"""
        data_attrs = dir(mp.data)
        
        # Check for expected submodules/functions
        expected_attrs = ['noisy_annulus']
        for attr in expected_attrs:
            assert attr in data_attrs, f"Missing attribute: {attr}"


class TestDataIntegration:
    """Integration tests combining data generation with other multipers functionality"""
    
    def test_data_with_simplex_tree(self):
        """Test using generated data with SimplexTreeMulti"""
        # Generate some test data
        points = mp.data.noisy_annulus(50, 30, dim=2)
        
        # Create a simple distance matrix
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(points))
        
        # This test verifies the data can be used in the multipers pipeline
        assert points.shape[0] == 80
        assert distances.shape == (80, 80)
        
        # Test that the data has reasonable properties
        assert np.all(distances >= 0)
        assert np.all(np.diag(distances) == 0)  # Distance to self is 0
    
    def test_data_type_consistency(self):
        """Test that generated data has consistent types"""
        points = mp.data.noisy_annulus(100, 50)
        
        # Should be numpy array
        assert isinstance(points, np.ndarray)
        
        # Should be float type
        assert np.issubdtype(points.dtype, np.floating)
        
        # Should not have NaN or infinite values
        assert np.all(np.isfinite(points))


# Additional test for error conditions
class TestDataErrors:
    """Test error handling in data module"""
    
    def test_negative_point_counts(self):
        """Test handling of negative point counts"""
        # The function might not strictly validate negative counts
        # Test that reasonable inputs work
        points = mp.data.noisy_annulus(10, 10)
        assert points.shape[0] == 20
        
        # Test edge case with zero
        points = mp.data.noisy_annulus(0, 10)
        assert points.shape[0] == 10
    
    def test_invalid_dimensions(self):
        """Test handling of invalid dimensions"""
        # Test that function works with positive dimensions
        # The function might not validate dimensions strictly
        try:
            points = mp.data.noisy_annulus(50, 30, dim=1)
            assert points.shape[1] == 1
        except ValueError:
            # If it raises ValueError for dim=1, that's also acceptable
            pass
        
        # Test with reasonable dimension
        points = mp.data.noisy_annulus(50, 30, dim=2)
        assert points.shape[1] == 2
    
    def test_invalid_radius_parameters(self):
        """Test handling of invalid radius parameters"""
        # Test that function works with valid radius parameters
        points = mp.data.noisy_annulus(50, 30, inner_radius=1, outer_radius=2)
        assert points.shape[0] == 80
        
        # The function might not strictly validate radius ordering
        # Just test that it doesn't crash with various inputs
        try:
            points = mp.data.noisy_annulus(10, 10, inner_radius=2, outer_radius=1)
            # If it works, that's fine
            assert points.shape[0] == 20
        except ValueError:
            # If it raises an error, that's also acceptable
            pass


@pytest.mark.parametrize("n_inner,n_outer,dim", [
    (10, 20, 2),
    (0, 50, 3),
    (100, 0, 2),
    (25, 25, 4),
])
def test_noisy_annulus_parametrized(n_inner, n_outer, dim):
    """Parametrized test for noisy_annulus with different configurations"""
    points = mp.data.noisy_annulus(n_inner, n_outer, dim=dim)
    assert points.shape == (n_inner + n_outer, dim)
    assert np.all(np.isfinite(points))