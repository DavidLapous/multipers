"""
Comprehensive tests for multipers.filtrations module
"""
import numpy as np
import pytest
import multipers as mp
import gudhi as gd


class TestFiltrationBasics:
    """Test basic filtration functionality"""
    
    def test_filtrations_module_exists(self):
        """Test that filtrations module is accessible"""
        assert hasattr(mp, 'filtrations')
        assert hasattr(mp.filtrations, 'flag_filtration')
        assert hasattr(mp.filtrations, 'rips_filtration')
    
    def test_flag_filtration_basic(self):
        """Test basic flag filtration construction"""
        # Create simple distance matrix
        distances = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]])
        
        # Test flag filtration
        result = mp.filtrations.flag_filtration(distances, max_dimension=1)
        
        # Should return some kind of filtration structure
        assert result is not None
        # The exact structure depends on implementation, but should be iterable
        assert hasattr(result, '__iter__') or hasattr(result, '__len__')
    
    def test_rips_filtration_basic(self):
        """Test basic Rips filtration construction"""
        # Generate simple point cloud
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        
        # Test Rips filtration
        result = mp.filtrations.rips_filtration(points, max_dimension=1, max_edge_length=2.0)
        
        assert result is not None
        assert hasattr(result, '__iter__') or hasattr(result, '__len__')


class TestFiltrationParameters:
    """Test filtration functions with various parameters"""
    
    @pytest.mark.parametrize("max_dim", [0, 1, 2])
    def test_flag_filtration_dimensions(self, max_dim):
        """Test flag filtration with different maximum dimensions"""
        distances = np.array([
            [0, 1, 2, 3],
            [1, 0, 1.5, 2.5],
            [2, 1.5, 0, 1],
            [3, 2.5, 1, 0]
        ])
        
        result = mp.filtrations.flag_filtration(distances, max_dimension=max_dim)
        assert result is not None
    
    @pytest.mark.parametrize("max_edge", [1.0, 2.0, 5.0])
    def test_rips_filtration_edge_lengths(self, max_edge):
        """Test Rips filtration with different maximum edge lengths"""
        points = np.random.randn(10, 2)
        
        result = mp.filtrations.rips_filtration(
            points, max_dimension=1, max_edge_length=max_edge
        )
        assert result is not None
    
    def test_rips_filtration_different_dimensions(self):
        """Test Rips filtration with different point cloud dimensions"""
        for dim in [2, 3, 4]:
            points = np.random.randn(8, dim)
            result = mp.filtrations.rips_filtration(
                points, max_dimension=1, max_edge_length=3.0
            )
            assert result is not None


class TestFiltrationEdgeCases:
    """Test edge cases for filtration functions"""
    
    def test_flag_filtration_single_point(self):
        """Test flag filtration with single point"""
        distances = np.array([[0]])
        
        result = mp.filtrations.flag_filtration(distances, max_dimension=0)
        assert result is not None
    
    def test_flag_filtration_two_points(self):
        """Test flag filtration with two points"""
        distances = np.array([[0, 1], [1, 0]])
        
        result = mp.filtrations.flag_filtration(distances, max_dimension=1)
        assert result is not None
    
    def test_rips_filtration_minimal_points(self):
        """Test Rips filtration with minimal point sets"""
        # Single point
        single_point = np.array([[0, 0]])
        result = mp.filtrations.rips_filtration(
            single_point, max_dimension=0, max_edge_length=1.0
        )
        assert result is not None
        
        # Two points
        two_points = np.array([[0, 0], [1, 0]])
        result = mp.filtrations.rips_filtration(
            two_points, max_dimension=1, max_edge_length=2.0
        )
        assert result is not None
    
    def test_filtrations_empty_input(self):
        """Test filtrations with empty input"""
        # Empty distance matrix
        try:
            empty_distances = np.array([]).reshape(0, 0)
            result = mp.filtrations.flag_filtration(empty_distances)
            # If it doesn't error, result should handle empty case
            assert result is not None
        except (ValueError, IndexError):
            # It's acceptable to error on empty input
            pass
        
        # Empty point cloud
        try:
            empty_points = np.array([]).reshape(0, 2)
            result = mp.filtrations.rips_filtration(empty_points)
            assert result is not None
        except (ValueError, IndexError):
            pass


class TestFiltrationInputValidation:
    """Test input validation for filtration functions"""
    
    def test_flag_filtration_non_symmetric_matrix(self):
        """Test flag filtration with non-symmetric distance matrix"""
        # Non-symmetric matrix
        distances = np.array([[0, 1, 2], [1.1, 0, 1.5], [2, 1.5, 0]])
        
        try:
            result = mp.filtrations.flag_filtration(distances)
            # Should either work or raise appropriate error
            assert result is not None
        except ValueError:
            # It's acceptable to reject non-symmetric matrices
            pass
    
    def test_flag_filtration_non_zero_diagonal(self):
        """Test flag filtration with non-zero diagonal"""
        distances = np.array([[1, 1, 2], [1, 1, 1.5], [2, 1.5, 1]])
        
        try:
            result = mp.filtrations.flag_filtration(distances)
            assert result is not None
        except ValueError:
            # Some implementations might require zero diagonal
            pass
    
    def test_rips_filtration_invalid_dimensions(self):
        """Test Rips filtration with invalid parameters"""
        points = np.random.randn(5, 2)
        
        # Negative max_dimension
        try:
            result = mp.filtrations.rips_filtration(
                points, max_dimension=-1, max_edge_length=1.0
            )
        except ValueError:
            pass  # Should raise error for negative dimension
        
        # Negative max_edge_length
        try:
            result = mp.filtrations.rips_filtration(
                points, max_dimension=1, max_edge_length=-1.0
            )
        except ValueError:
            pass  # Should raise error for negative edge length


class TestFiltrationIntegration:
    """Test integration of filtrations with other multipers components"""
    
    def test_filtration_to_simplextree(self):
        """Test converting filtration to SimplexTreeMulti"""
        # Generate test data
        points = mp.data.noisy_annulus(20, 10, dim=2)
        
        # Create Rips filtration
        filtration = mp.filtrations.rips_filtration(
            points, max_dimension=1, max_edge_length=2.0
        )
        
        # This test verifies the filtration can be used downstream
        assert filtration is not None
        
        # If filtration is iterable, check it has some content
        if hasattr(filtration, '__iter__'):
            try:
                first_item = next(iter(filtration))
                assert first_item is not None
            except StopIteration:
                pass  # Empty filtration is also valid
    
    def test_filtration_consistency(self):
        """Test that filtrations produce consistent results"""
        np.random.seed(42)
        points = np.random.randn(10, 2)
        
        # Create same filtration twice
        filt1 = mp.filtrations.rips_filtration(
            points, max_dimension=1, max_edge_length=3.0
        )
        filt2 = mp.filtrations.rips_filtration(
            points, max_dimension=1, max_edge_length=3.0
        )
        
        # Results should be identical
        if hasattr(filt1, '__eq__'):
            assert filt1 == filt2
        # If direct comparison isn't available, check structure similarity
        elif hasattr(filt1, '__len__') and hasattr(filt2, '__len__'):
            assert len(filt1) == len(filt2)


class TestFiltrationNumericalProperties:
    """Test numerical properties and stability of filtrations"""
    
    def test_flag_filtration_metric_properties(self):
        """Test that flag filtration respects metric properties when applicable"""
        # Create metric distance matrix
        points = np.random.randn(8, 3)
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(points))
        
        result = mp.filtrations.flag_filtration(distances, max_dimension=1)
        assert result is not None
        
        # The filtration should handle metric distances correctly
        assert np.all(distances >= 0)  # Non-negativity
        assert np.allclose(np.diag(distances), 0)  # Zero diagonal
    
    def test_rips_filtration_scale_invariance(self):
        """Test behavior of Rips filtration under scaling"""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        
        # Original scale
        filt1 = mp.filtrations.rips_filtration(
            points, max_dimension=1, max_edge_length=2.0
        )
        
        # Scaled points
        scaled_points = points * 2
        filt2 = mp.filtrations.rips_filtration(
            scaled_points, max_dimension=1, max_edge_length=4.0  # Scaled accordingly
        )
        
        # Both should be non-None and structurally similar
        assert filt1 is not None
        assert filt2 is not None
    
    def test_filtration_large_datasets(self):
        """Test filtration performance with larger datasets"""
        # Test with moderately large point cloud
        np.random.seed(123)
        points = np.random.randn(50, 2)
        
        # Should complete without error
        result = mp.filtrations.rips_filtration(
            points, max_dimension=1, max_edge_length=2.0
        )
        assert result is not None
        
        # Test with larger distance matrix for flag filtration
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(points[:30]))  # Smaller subset for flag
        
        result = mp.filtrations.flag_filtration(distances, max_dimension=1)
        assert result is not None


class TestFiltrationSpecialCases:
    """Test special geometric configurations"""
    
    def test_rips_filtration_collinear_points(self):
        """Test Rips filtration with collinear points"""
        # Points on a line
        points = np.array([[i, 0] for i in range(5)])
        
        result = mp.filtrations.rips_filtration(
            points, max_dimension=1, max_edge_length=5.0
        )
        assert result is not None
    
    def test_rips_filtration_identical_points(self):
        """Test Rips filtration with some identical points"""
        points = np.array([[0, 0], [0, 0], [1, 0], [1, 0]])
        
        result = mp.filtrations.rips_filtration(
            points, max_dimension=1, max_edge_length=2.0
        )
        assert result is not None
    
    def test_flag_filtration_complete_graph(self):
        """Test flag filtration on complete graph distances"""
        n = 6
        # All pairwise distances equal (complete graph)
        distances = np.ones((n, n))
        np.fill_diagonal(distances, 0)
        
        result = mp.filtrations.flag_filtration(distances, max_dimension=2)
        assert result is not None


@pytest.mark.parametrize("n_points", [5, 10, 20])
@pytest.mark.parametrize("dim", [2, 3])
def test_filtration_scalability(n_points, dim):
    """Test filtration functions with different problem sizes"""
    np.random.seed(42)
    points = np.random.randn(n_points, dim)
    
    # Test Rips filtration scalability
    rips_result = mp.filtrations.rips_filtration(
        points, max_dimension=min(2, dim), max_edge_length=3.0
    )
    assert rips_result is not None
    
    # Test flag filtration scalability (smaller sizes due to O(n^2) distance matrix)
    if n_points <= 15:  # Keep reasonable for flag filtrations
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(points))
        
        flag_result = mp.filtrations.flag_filtration(
            distances, max_dimension=min(2, dim)
        )
        assert flag_result is not None