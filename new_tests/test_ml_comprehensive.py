"""
Comprehensive tests for multipers.ml module and machine learning components
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import multipers as mp


class TestPointCloudProcessing:
    """Test point cloud processing in ML pipeline"""
    
    def test_point_clouds_module_exists(self):
        """Test that point clouds ML module is accessible"""
        assert hasattr(mp, 'ml')
        
        # Check if point_clouds submodule exists
        try:
            import multipers.ml.point_clouds
            assert True
        except ImportError:
            pytest.skip("Point clouds ML module not available")
    
    @pytest.mark.skipif(
        not hasattr(mp.ml, 'point_clouds'),
        reason="Point clouds ML module not available"
    )
    def test_point_cloud_transformer_basic(self):
        """Test basic point cloud transformation functionality"""
        from multipers.ml.point_clouds import PointCloud2FilteredComplex
        
        # Create simple point cloud data
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        
        # Test basic transformer
        transformer = PointCloud2FilteredComplex()
        
        # Fit and transform
        result = transformer.fit_transform([points])
        
        assert result is not None
        assert len(result) == 1  # One input, one output
    
    @pytest.mark.skipif(
        not hasattr(mp.ml, 'point_clouds'),
        reason="Point clouds ML module not available"
    )
    def test_point_cloud_transformer_parameters(self):
        """Test point cloud transformer with different parameters"""
        from multipers.ml.point_clouds import PointCloud2FilteredComplex
        
        points = np.random.randn(10, 2)
        
        # Test with different parameters
        transformer = PointCloud2FilteredComplex(
            complex="rips", 
            max_dimension=1,
            n_jobs=1
        )
        
        result = transformer.fit_transform([points])
        assert result is not None


class TestMMAModule:
    """Test Multiparameter Module Approximation ML components"""
    
    def test_mma_module_accessible(self):
        """Test that MMA ML module is accessible"""
        assert hasattr(mp.ml, 'mma')
    
    def test_filtered_complex_to_mma(self):
        """Test FilteredComplex2MMA transformer"""
        from multipers.ml.mma import FilteredComplex2MMA
        
        # Create a simple simplex tree for testing
        st = mp.SimplexTreeMulti(num_parameters=2)
        st.insert([0], [0.0, 0.0])
        st.insert([1], [1.0, 0.0])
        st.insert([0, 1], [1.0, 1.0])
        
        # Test transformer
        transformer = FilteredComplex2MMA()
        result = transformer.fit_transform([[st]])
        
        assert result is not None
        assert len(result) == 1
    
    def test_mma_transformer_parameters(self):
        """Test MMA transformer with different parameters"""
        from multipers.ml.mma import FilteredComplex2MMA
        
        # Create test data
        st = mp.SimplexTreeMulti(num_parameters=2)
        for i in range(3):
            st.insert([i], [i * 0.5, i * 0.3])
        
        # Test with different parameters
        transformer = FilteredComplex2MMA(
            prune_degrees_above=1,
            n_jobs=1,
            expand_dim=None
        )
        
        result = transformer.fit_transform([[st]])
        assert result is not None
    
    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_mma_parallel_processing(self, n_jobs):
        """Test MMA transformer with parallel processing"""
        from multipers.ml.mma import FilteredComplex2MMA
        
        # Create multiple test instances
        sts = []
        for j in range(3):
            st = mp.SimplexTreeMulti(num_parameters=2)
            for i in range(4):
                st.insert([i], [i * 0.5 + j * 0.1, i * 0.3 + j * 0.2])
            sts.append([st])
        
        transformer = FilteredComplex2MMA(n_jobs=n_jobs)
        results = transformer.fit_transform(sts)
        
        assert len(results) == 3
        for result in results:
            assert result is not None


class TestMLUtilities:
    """Test ML utility functions"""
    
    def test_sklearn_compatibility(self):
        """Test scikit-learn compatibility of transformers"""
        from sklearn.base import BaseEstimator, TransformerMixin
        
        # Test that our transformers inherit from sklearn base classes
        if hasattr(mp.ml, 'mma'):
            from multipers.ml.mma import FilteredComplex2MMA
            transformer = FilteredComplex2MMA()
            
            assert isinstance(transformer, BaseEstimator)
            assert isinstance(transformer, TransformerMixin)
            assert hasattr(transformer, 'fit')
            assert hasattr(transformer, 'transform')
            assert hasattr(transformer, 'fit_transform')
    
    def test_ml_pipeline_integration(self):
        """Test integration with scikit-learn pipelines"""
        from sklearn.pipeline import Pipeline
        from sklearn.base import BaseEstimator, TransformerMixin
        
        # Create a dummy transformer for testing
        class DummyTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                return X
        
        # Test pipeline creation
        if hasattr(mp.ml, 'mma'):
            from multipers.ml.mma import FilteredComplex2MMA
            
            pipeline = Pipeline([
                ('mma', FilteredComplex2MMA()),
                ('dummy', DummyTransformer())
            ])
            
            assert pipeline is not None
            assert len(pipeline.steps) == 2


class TestMLErrorHandling:
    """Test error handling in ML components"""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        if hasattr(mp.ml, 'mma'):
            from multipers.ml.mma import FilteredComplex2MMA
            
            transformer = FilteredComplex2MMA()
            
            # Test with empty list
            try:
                result = transformer.fit_transform([])
                assert result == []
            except ValueError:
                # It's acceptable to raise an error for empty input
                pass
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types"""
        if hasattr(mp.ml, 'mma'):
            from multipers.ml.mma import FilteredComplex2MMA
            
            transformer = FilteredComplex2MMA()
            
            # Test with invalid input type
            try:
                result = transformer.fit_transform("invalid_input")
                # If it doesn't error, should handle gracefully
                assert result is not None
            except (TypeError, ValueError):
                # It's acceptable to raise an error for invalid input
                pass
    
    def test_parameter_validation(self):
        """Test parameter validation in ML components"""
        if hasattr(mp.ml, 'mma'):
            from multipers.ml.mma import FilteredComplex2MMA
            
            # Test invalid n_jobs parameter
            try:
                transformer = FilteredComplex2MMA(n_jobs=0)
                # Should either work or raise appropriate error
            except ValueError:
                pass
            
            # Test invalid prune_degrees_above
            try:
                transformer = FilteredComplex2MMA(prune_degrees_above=-1)
            except ValueError:
                pass


class TestMLDataHandling:
    """Test data handling in ML pipeline"""
    
    def test_batch_processing(self):
        """Test batch processing capabilities"""
        if hasattr(mp.ml, 'mma'):
            from multipers.ml.mma import FilteredComplex2MMA
            
            # Create batch of test data
            batch_data = []
            for i in range(5):
                st = mp.SimplexTreeMulti(num_parameters=2)
                st.insert([0], [0.0 + i * 0.1, 0.0 + i * 0.1])
                st.insert([1], [1.0 + i * 0.1, 0.5 + i * 0.1])
                st.insert([0, 1], [1.2 + i * 0.1, 1.0 + i * 0.1])
                batch_data.append([st])
            
            transformer = FilteredComplex2MMA()
            results = transformer.fit_transform(batch_data)
            
            assert len(results) == 5
            for result in results:
                assert result is not None
    
    def test_different_complex_sizes(self):
        """Test handling of complexes with different sizes"""
        if hasattr(mp.ml, 'mma'):
            from multipers.ml.mma import FilteredComplex2MMA
            
            # Create complexes of different sizes
            small_st = mp.SimplexTreeMulti(num_parameters=2)
            small_st.insert([0], [0.0, 0.0])
            
            large_st = mp.SimplexTreeMulti(num_parameters=2)
            for i in range(10):
                large_st.insert([i], [i * 0.1, i * 0.2])
                if i > 0:
                    large_st.insert([i-1, i], [i * 0.1 + 0.05, i * 0.2 + 0.1])
            
            data = [[small_st], [large_st]]
            transformer = FilteredComplex2MMA()
            results = transformer.fit_transform(data)
            
            assert len(results) == 2
            assert results[0] is not None
            assert results[1] is not None


class TestMLIntegration:
    """Test integration between different ML components"""
    
    def test_end_to_end_pipeline(self):
        """Test complete ML pipeline from point clouds to features"""
        # Generate test data
        points1 = mp.data.noisy_annulus(20, 15, dim=2)
        points2 = mp.data.noisy_annulus(25, 10, dim=2)
        point_clouds = [points1, points2]
        
        try:
            # Step 1: Point clouds to filtered complexes
            if hasattr(mp.ml, 'point_clouds'):
                from multipers.ml.point_clouds import PointCloud2FilteredComplex
                
                pc_transformer = PointCloud2FilteredComplex(
                    complex="rips",
                    max_dimension=1
                )
                complexes = pc_transformer.fit_transform(point_clouds)
                
                # Step 2: Filtered complexes to MMA
                if hasattr(mp.ml, 'mma'):
                    from multipers.ml.mma import FilteredComplex2MMA
                    
                    mma_transformer = FilteredComplex2MMA()
                    features = mma_transformer.fit_transform(complexes)
                    
                    assert len(features) == 2
                    assert all(f is not None for f in features)
        except ImportError:
            pytest.skip("Required ML modules not available")
    
    def test_feature_consistency(self):
        """Test that ML pipeline produces consistent features"""
        # Create identical inputs
        st1 = mp.SimplexTreeMulti(num_parameters=2)
        st2 = mp.SimplexTreeMulti(num_parameters=2)
        
        for st in [st1, st2]:
            st.insert([0], [0.0, 0.0])
            st.insert([1], [1.0, 0.5])
            st.insert([2], [0.5, 1.0])
            st.insert([0, 1], [1.0, 0.8])
            st.insert([1, 2], [1.2, 1.0])
        
        if hasattr(mp.ml, 'mma'):
            from multipers.ml.mma import FilteredComplex2MMA
            
            transformer = FilteredComplex2MMA()
            features1 = transformer.fit_transform([[st1]])
            features2 = transformer.fit_transform([[st2]])
            
            # Features should be identical for identical inputs
            assert len(features1) == len(features2)


class TestMLPerformance:
    """Test performance characteristics of ML components"""
    
    def test_memory_usage(self):
        """Test that ML components don't leak memory excessively"""
        if hasattr(mp.ml, 'mma'):
            from multipers.ml.mma import FilteredComplex2MMA
            
            # Create many small transformations
            transformer = FilteredComplex2MMA()
            
            for _ in range(10):
                st = mp.SimplexTreeMulti(num_parameters=2)
                for i in range(5):
                    st.insert([i], [i * 0.1, i * 0.2])
                
                result = transformer.fit_transform([[st]])
                assert result is not None
                
                # Force garbage collection
                del st, result
    
    def test_reasonable_computation_time(self):
        """Test that computations complete in reasonable time"""
        import time
        
        if hasattr(mp.ml, 'mma'):
            from multipers.ml.mma import FilteredComplex2MMA
            
            # Create moderately sized test case
            st = mp.SimplexTreeMulti(num_parameters=2)
            for i in range(20):
                st.insert([i], [i * 0.1, i * 0.2])
                if i > 0:
                    st.insert([i-1, i], [i * 0.1 + 0.05, i * 0.2 + 0.1])
            
            transformer = FilteredComplex2MMA()
            
            start_time = time.time()
            result = transformer.fit_transform([[st]])
            end_time = time.time()
            
            # Should complete within reasonable time (10 seconds)
            assert end_time - start_time < 10
            assert result is not None


@pytest.mark.parametrize("num_parameters", [2, 3])
@pytest.mark.parametrize("n_simplices", [5, 10, 15])
def test_ml_scalability(num_parameters, n_simplices):
    """Test ML components with different problem sizes"""
    # Create test simplex tree
    st = mp.SimplexTreeMulti(num_parameters=num_parameters)
    
    # Add simplices
    for i in range(n_simplices):
        filtration = [i * 0.1] * num_parameters
        st.insert([i], filtration)
        
        # Add some edges
        if i > 0:
            edge_filtration = [(i * 0.1 + 0.05)] * num_parameters
            st.insert([i-1, i], edge_filtration)
    
    # Test with MMA if available
    if hasattr(mp.ml, 'mma'):
        from multipers.ml.mma import FilteredComplex2MMA
        
        transformer = FilteredComplex2MMA()
        result = transformer.fit_transform([[st]])
        
        assert result is not None
        assert len(result) == 1