"""
Unit tests for SynergyOptimizer (both Cython and Pure Python versions).

Run with: python -m pytest Tests/test_synergy_optimizer.py -v
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


def create_test_synergy_matrix():
    """Create a simple test synergy matrix."""
    return {
        'Mango': {'Papaya': 0.1, 'Orange': 0.1},
        'Papaya': {'Mango': 0.1, 'Orange': 0.1},
        'Orange': {'Mango': 0.1, 'Papaya': 0.1},
        'Rice': {'Wheat': 0.15, 'Maize': 0.15},
        'Wheat': {'Rice': 0.15, 'Maize': 0.15},
        'Maize': {'Rice': 0.15, 'Wheat': 0.15},
    }


def create_test_foods():
    """Create a simple test foods dict."""
    return {
        'Mango': {'food_group': 'Fruits'},
        'Papaya': {'food_group': 'Fruits'},
        'Orange': {'food_group': 'Fruits'},
        'Rice': {'food_group': 'Starchy staples'},
        'Wheat': {'food_group': 'Starchy staples'},
        'Maize': {'food_group': 'Starchy staples'},
    }


class TestSynergyOptimizerPurePython:
    """Test pure Python implementation."""
    
    @pytest.fixture
    def optimizer(self):
        from src.synergy_optimizer_pure import SynergyOptimizer
        synergy_matrix = create_test_synergy_matrix()
        foods = create_test_foods()
        return SynergyOptimizer(synergy_matrix, foods)
    
    def test_n_pairs(self, optimizer):
        """Test that correct number of pairs is counted."""
        # Fruits: 3 crops -> 3 pairs (Mango-Papaya, Mango-Orange, Papaya-Orange)
        # Staples: 3 crops -> 3 pairs (Rice-Wheat, Rice-Maize, Wheat-Maize)
        # Total: 6 pairs
        assert optimizer.get_n_pairs() == 6
    
    def test_iter_pairs(self, optimizer):
        """Test iteration through pairs."""
        pairs_list = list(optimizer.iter_pairs())
        assert len(pairs_list) == 6
        
        # Check structure
        for crop1_idx, crop2_idx, boost_value in pairs_list:
            assert isinstance(crop1_idx, (int, np.integer))
            assert isinstance(crop2_idx, (int, np.integer))
            assert isinstance(boost_value, (float, np.floating))
            assert crop1_idx < crop2_idx  # No double counting
    
    def test_iter_pairs_with_names(self, optimizer):
        """Test iteration with crop names."""
        pairs_list = list(optimizer.iter_pairs_with_names())
        assert len(pairs_list) == 6
        
        # Extract all pairs
        pair_names = {(c1, c2) for c1, c2, _ in pairs_list}
        
        # Check expected fruit pairs
        assert ('Mango', 'Papaya') in pair_names or ('Papaya', 'Mango') in pair_names
        assert ('Mango', 'Orange') in pair_names or ('Orange', 'Mango') in pair_names
        assert ('Papaya', 'Orange') in pair_names or ('Orange', 'Papaya') in pair_names
        
        # Check expected staple pairs
        assert ('Rice', 'Wheat') in pair_names or ('Wheat', 'Rice') in pair_names
        assert ('Rice', 'Maize') in pair_names or ('Maize', 'Rice') in pair_names
        assert ('Wheat', 'Maize') in pair_names or ('Maize', 'Wheat') in pair_names
    
    def test_build_synergy_pairs_list(self, optimizer):
        """Test building pairs list for PuLP."""
        farms = ['Farm1', 'Farm2']
        pairs_list = optimizer.build_synergy_pairs_list(farms)
        
        # Should have 6 pairs × 2 farms = 12 tuples
        assert len(pairs_list) == 12
        
        # Check structure
        for farm, crop1, crop2, boost in pairs_list:
            assert farm in farms
            assert isinstance(crop1, str)
            assert isinstance(crop2, str)
            assert isinstance(boost, (float, np.floating))
    
    def test_to_numpy_arrays(self, optimizer):
        """Test export to NumPy arrays."""
        arrays = optimizer.to_numpy_arrays()
        
        assert 'crop1_indices' in arrays
        assert 'crop2_indices' in arrays
        assert 'boost_values' in arrays
        assert 'crop_names' in arrays
        
        assert len(arrays['crop1_indices']) == 6
        assert len(arrays['crop2_indices']) == 6
        assert len(arrays['boost_values']) == 6
        assert len(arrays['crop_names']) == 6


class TestSynergyOptimizerCython:
    """Test Cython implementation (if available)."""
    
    @pytest.fixture
    def optimizer(self):
        try:
            from synergy_optimizer import SynergyOptimizer
        except ImportError:
            pytest.skip("Cython version not compiled")
        
        synergy_matrix = create_test_synergy_matrix()
        foods = create_test_foods()
        return SynergyOptimizer(synergy_matrix, foods)
    
    def test_n_pairs(self, optimizer):
        """Test that correct number of pairs is counted."""
        assert optimizer.get_n_pairs() == 6
    
    def test_iter_pairs(self, optimizer):
        """Test iteration through pairs."""
        pairs_list = list(optimizer.iter_pairs())
        assert len(pairs_list) == 6
    
    def test_build_synergy_pairs_list(self, optimizer):
        """Test building pairs list for PuLP."""
        farms = ['Farm1', 'Farm2']
        pairs_list = optimizer.build_synergy_pairs_list(farms)
        assert len(pairs_list) == 12


class TestConsistency:
    """Test that Cython and Pure Python versions produce identical results."""
    
    def test_same_pairs_count(self):
        """Both versions should count same number of pairs."""
        synergy_matrix = create_test_synergy_matrix()
        foods = create_test_foods()
        
        from src.synergy_optimizer_pure import SynergyOptimizer as PurePython
        pure = PurePython(synergy_matrix, foods)
        
        try:
            from synergy_optimizer import SynergyOptimizer as Cython
            cython = Cython(synergy_matrix, foods)
            
            assert pure.get_n_pairs() == cython.get_n_pairs()
        except ImportError:
            pytest.skip("Cython version not compiled")
    
    def test_same_pairs_list(self):
        """Both versions should produce identical pairs lists."""
        synergy_matrix = create_test_synergy_matrix()
        foods = create_test_foods()
        farms = ['Farm1', 'Farm2']
        
        from src.synergy_optimizer_pure import SynergyOptimizer as PurePython
        pure = PurePython(synergy_matrix, foods)
        pure_pairs = set(pure.build_synergy_pairs_list(farms))
        
        try:
            from synergy_optimizer import SynergyOptimizer as Cython
            cython = Cython(synergy_matrix, foods)
            cython_pairs = set(cython.build_synergy_pairs_list(farms))
            
            assert pure_pairs == cython_pairs
        except ImportError:
            pytest.skip("Cython version not compiled")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_synergy_matrix(self):
        """Test with empty synergy matrix."""
        from src.synergy_optimizer_pure import SynergyOptimizer
        
        optimizer = SynergyOptimizer({}, create_test_foods())
        assert optimizer.get_n_pairs() == 0
        assert list(optimizer.iter_pairs()) == []
    
    def test_no_matching_foods(self):
        """Test when synergy matrix has crops not in foods."""
        from src.synergy_optimizer_pure import SynergyOptimizer
        
        synergy_matrix = {
            'NonExistent1': {'NonExistent2': 0.1},
            'NonExistent2': {'NonExistent1': 0.1},
        }
        foods = create_test_foods()
        
        optimizer = SynergyOptimizer(synergy_matrix, foods)
        assert optimizer.get_n_pairs() == 0
    
    def test_partial_matching(self):
        """Test when only some crops from synergy matrix are in foods."""
        from src.synergy_optimizer_pure import SynergyOptimizer
        
        synergy_matrix = create_test_synergy_matrix()
        foods = {'Mango': {}, 'Papaya': {}}  # Only 2 out of 6 crops
        
        optimizer = SynergyOptimizer(synergy_matrix, foods)
        # Should only have Mango-Papaya pair
        assert optimizer.get_n_pairs() == 1
        
        pairs = list(optimizer.iter_pairs_with_names())
        pair_names = {(c1, c2) for c1, c2, _ in pairs}
        assert ('Mango', 'Papaya') in pair_names or ('Papaya', 'Mango') in pair_names


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
