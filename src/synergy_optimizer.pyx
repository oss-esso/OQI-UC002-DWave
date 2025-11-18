# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Cython-optimized synergy bonus computation for Linear-Quadratic formulation.

This module provides ~10-100x speedup for building quadratic synergy terms
by using precomputed arrays and fast C loops instead of nested Python dict iteration.
"""

cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp
import numpy as np
cimport numpy as np

# Data structure for synergy pairs
ctypedef struct SynergyPair:
    int crop1_idx
    int crop2_idx
    double boost_value

cdef class SynergyOptimizer:
    """
    Precomputes and stores synergy pairs for fast iteration during CQM/model building.
    
    Usage:
        optimizer = SynergyOptimizer(synergy_matrix, foods)
        for farm in farms:
            for crop1_idx, crop2_idx, boost in optimizer.iter_pairs():
                crop1 = optimizer.get_crop_name(crop1_idx)
                crop2 = optimizer.get_crop_name(crop2_idx)
                objective += synergy_bonus_weight * boost * Y[(farm, crop1)] * Y[(farm, crop2)]
    """
    cdef SynergyPair* pairs
    cdef int n_pairs
    cdef list crop_names
    cdef dict crop_to_idx
    
    def __cinit__(self):
        self.pairs = NULL
        self.n_pairs = 0
    
    def __dealloc__(self):
        if self.pairs != NULL:
            free(self.pairs)
    
    def __init__(self, synergy_matrix, foods):
        """
        Initialize optimizer by precomputing synergy pairs.
        
        Args:
            synergy_matrix: Dict[str, Dict[str, float]] - sparse matrix of synergy values
            foods: Dict[str, Dict] or List[str] - food data or list of food names
        """
        # Build crop index mapping
        if isinstance(foods, dict):
            self.crop_names = list(foods.keys())
        else:
            self.crop_names = list(foods)
        
        self.crop_to_idx = {crop: idx for idx, crop in enumerate(self.crop_names)}
        
        # Count pairs and allocate memory
        cdef int pair_count = 0
        for crop1, pairs_dict in synergy_matrix.items():
            if crop1 in self.crop_to_idx:
                for crop2, boost_value in pairs_dict.items():
                    if crop2 in self.crop_to_idx and crop1 < crop2:
                        pair_count += 1
        
        self.n_pairs = pair_count
        self.pairs = <SynergyPair*>malloc(pair_count * sizeof(SynergyPair))
        
        if self.pairs == NULL:
            raise MemoryError("Failed to allocate memory for synergy pairs")
        
        # Populate pairs array
        cdef int idx = 0
        cdef int crop1_idx, crop2_idx
        cdef double boost
        
        for crop1, pairs_dict in synergy_matrix.items():
            if crop1 not in self.crop_to_idx:
                continue
            crop1_idx = self.crop_to_idx[crop1]
            
            for crop2, boost_value in pairs_dict.items():
                if crop2 not in self.crop_to_idx:
                    continue
                if crop1 >= crop2:  # Skip to avoid double counting
                    continue
                    
                crop2_idx = self.crop_to_idx[crop2]
                boost = float(boost_value)
                
                self.pairs[idx].crop1_idx = crop1_idx
                self.pairs[idx].crop2_idx = crop2_idx
                self.pairs[idx].boost_value = boost
                idx += 1
    
    def get_n_pairs(self):
        """Return number of synergy pairs."""
        return self.n_pairs
    
    def get_crop_name(self, int idx):
        """Get crop name from index."""
        return self.crop_names[idx]
    
    def get_pair(self, int pair_idx):
        """
        Get a specific synergy pair by index.
        
        Returns:
            tuple: (crop1_name, crop2_name, boost_value)
        """
        if pair_idx < 0 or pair_idx >= self.n_pairs:
            raise IndexError(f"Pair index {pair_idx} out of range [0, {self.n_pairs})")
        
        cdef SynergyPair* pair = &self.pairs[pair_idx]
        return (
            self.crop_names[pair.crop1_idx],
            self.crop_names[pair.crop2_idx],
            pair.boost_value
        )
    
    def iter_pairs(self):
        """
        Iterate through all synergy pairs.
        
        Yields:
            tuple: (crop1_idx, crop2_idx, boost_value)
        """
        cdef int i
        cdef SynergyPair* pair
        
        for i in range(self.n_pairs):
            pair = &self.pairs[i]
            yield (pair.crop1_idx, pair.crop2_idx, pair.boost_value)
    
    def iter_pairs_with_names(self):
        """
        Iterate through all synergy pairs with crop names.
        
        Yields:
            tuple: (crop1_name, crop2_name, boost_value)
        """
        cdef int i
        cdef SynergyPair* pair
        
        for i in range(self.n_pairs):
            pair = &self.pairs[i]
            yield (
                self.crop_names[pair.crop1_idx],
                self.crop_names[pair.crop2_idx],
                pair.boost_value
            )
    
    def build_synergy_terms_dimod(self, farms, Y_vars, double synergy_bonus_weight):
        """
        Fast construction of synergy terms for dimod CQM objective.
        
        Args:
            farms: List of farm names
            Y_vars: Dict mapping (farm, crop) -> Binary variable
            synergy_bonus_weight: Weight for synergy bonus
            
        Returns:
            Quadratic expression that can be added to objective
        """
        cdef int i
        cdef SynergyPair* pair
        objective_terms = 0
        
        for farm in farms:
            for i in range(self.n_pairs):
                pair = &self.pairs[i]
                crop1 = self.crop_names[pair.crop1_idx]
                crop2 = self.crop_names[pair.crop2_idx]
                boost = pair.boost_value
                
                objective_terms += synergy_bonus_weight * boost * Y_vars[(farm, crop1)] * Y_vars[(farm, crop2)]
        
        return objective_terms
    
    def build_synergy_pairs_list(self, farms):
        """
        Build list of (farm, crop1, crop2, boost_value) tuples for PuLP linearization.
        
        This is useful for creating Z variables in McCormick relaxation.
        
        Args:
            farms: List of farm names
            
        Returns:
            List of tuples: [(farm, crop1, crop2, boost_value), ...]
        """
        cdef int i
        cdef SynergyPair* pair
        pairs_list = []
        
        for farm in farms:
            for i in range(self.n_pairs):
                pair = &self.pairs[i]
                crop1 = self.crop_names[pair.crop1_idx]
                crop2 = self.crop_names[pair.crop2_idx]
                boost = pair.boost_value
                
                pairs_list.append((farm, crop1, crop2, boost))
        
        return pairs_list
    
    def to_numpy_arrays(self):
        """
        Export synergy pairs to NumPy arrays for maximum performance.
        
        Returns:
            dict with:
                - 'crop1_indices': np.ndarray[int]
                - 'crop2_indices': np.ndarray[int]
                - 'boost_values': np.ndarray[float64]
        """
        cdef int i
        cdef np.ndarray[np.int32_t, ndim=1] crop1_indices = np.empty(self.n_pairs, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] crop2_indices = np.empty(self.n_pairs, dtype=np.int32)
        cdef np.ndarray[np.float64_t, ndim=1] boost_values = np.empty(self.n_pairs, dtype=np.float64)
        
        for i in range(self.n_pairs):
            crop1_indices[i] = self.pairs[i].crop1_idx
            crop2_indices[i] = self.pairs[i].crop2_idx
            boost_values[i] = self.pairs[i].boost_value
        
        return {
            'crop1_indices': crop1_indices,
            'crop2_indices': crop2_indices,
            'boost_values': boost_values,
            'crop_names': self.crop_names
        }


def precompute_synergy_pairs(synergy_matrix, foods):
    """
    Convenience function to create and return synergy pairs list.
    
    This is a pure Python fallback if Cython optimization isn't needed.
    
    Args:
        synergy_matrix: Dict[str, Dict[str, float]]
        foods: Dict[str, Dict] or List[str]
        
    Returns:
        List of (crop1, crop2, boost_value) tuples
    """
    if isinstance(foods, dict):
        food_names = set(foods.keys())
    else:
        food_names = set(foods)
    
    pairs = []
    for crop1, pairs_dict in synergy_matrix.items():
        if crop1 not in food_names:
            continue
        for crop2, boost_value in pairs_dict.items():
            if crop2 not in food_names and crop1 < crop2:
                pairs.append((crop1, crop2, boost_value))
    
    return pairs
