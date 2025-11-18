"""
Example demonstrating the synergy optimizer usage and performance comparison.

This shows how to integrate the Cython/NumPy optimized synergy computation
into solver_runner_LQ.py for significant speedup.
"""

import sys
import os
import time
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.scenarios import load_food_data

# Try to import Cython version first, fallback to pure Python
try:
    from synergy_optimizer import SynergyOptimizer
    print("✓ Using Cython-optimized synergy computation")
    USING_CYTHON = True
except ImportError:
    from src.synergy_optimizer_pure import SynergyOptimizer
    print("✓ Using pure Python synergy computation (compile Cython for ~10x speedup)")
    USING_CYTHON = False


def benchmark_synergy_computation():
    """Compare old nested loop vs optimized approach."""
    
    # Create test scenario
    from Benchmark_Scripts.benchmark_scalability_LQ import load_full_family_with_n_farms
    
    print("\n" + "="*70)
    print("SYNERGY COMPUTATION BENCHMARK")
    print("="*70)
    
    for n_farms in [10, 25, 50, 100]:
        print(f"\n{'='*70}")
        print(f"Testing with {n_farms} farms")
        print(f"{'='*70}")
        
        farms, foods, food_groups, config = load_full_family_with_n_farms(n_farms, seed=42)
        synergy_matrix = config['parameters'].get('synergy_matrix', {})
        synergy_bonus_weight = config['parameters']['weights'].get('synergy_bonus', 0.1)
        
        n_foods = len(foods)
        n_pairs_raw = sum(len(pairs) for pairs in synergy_matrix.values()) // 2
        
        print(f"  Foods: {n_foods}")
        print(f"  Synergy pairs: {n_pairs_raw}")
        print(f"  Total iterations: {n_farms * n_pairs_raw}")
        
        # Method 1: OLD - Nested dict iteration (original code)
        print("\n  [1] OLD: Nested dict iteration")
        start = time.perf_counter()
        
        count_old = 0
        for farm in farms:
            for crop1, pairs in synergy_matrix.items():
                if crop1 in foods:
                    for crop2, boost_value in pairs.items():
                        if crop2 in foods and crop1 < crop2:
                            # Simulate adding to objective
                            _ = synergy_bonus_weight * boost_value
                            count_old += 1
        
        time_old = time.perf_counter() - start
        print(f"      Time: {time_old*1000:.2f} ms")
        print(f"      Iterations: {count_old}")
        
        # Method 2: Precomputed list (simple optimization)
        print("\n  [2] SIMPLE: Precomputed pairs list")
        start = time.perf_counter()
        
        # Precompute pairs once
        synergy_pairs = []
        for crop1, pairs in synergy_matrix.items():
            if crop1 in foods:
                for crop2, boost_value in pairs.items():
                    if crop2 in foods and crop1 < crop2:
                        synergy_pairs.append((crop1, crop2, boost_value))
        
        count_simple = 0
        for farm in farms:
            for crop1, crop2, boost_value in synergy_pairs:
                # Simulate adding to objective
                _ = synergy_bonus_weight * boost_value
                count_simple += 1
        
        time_simple = time.perf_counter() - start
        speedup_simple = time_old / time_simple if time_simple > 0 else 0
        print(f"      Time: {time_simple*1000:.2f} ms")
        print(f"      Speedup: {speedup_simple:.2f}x")
        
        # Method 3: SynergyOptimizer (Cython or Pure Python)
        print(f"\n  [3] OPTIMIZED: SynergyOptimizer ({'Cython' if USING_CYTHON else 'NumPy'})")
        start = time.perf_counter()
        
        optimizer = SynergyOptimizer(synergy_matrix, foods)
        
        count_opt = 0
        for farm in farms:
            for crop1_idx, crop2_idx, boost_value in optimizer.iter_pairs():
                # Simulate adding to objective
                _ = synergy_bonus_weight * boost_value
                count_opt += 1
        
        time_opt = time.perf_counter() - start
        speedup_opt = time_old / time_opt if time_opt > 0 else 0
        print(f"      Time: {time_opt*1000:.2f} ms")
        print(f"      Speedup: {speedup_opt:.2f}x")
        
        # Method 4: Batch building (best for CQM)
        print(f"\n  [4] BATCH: build_synergy_pairs_list")
        start = time.perf_counter()
        
        optimizer2 = SynergyOptimizer(synergy_matrix, foods)
        synergy_pairs_batch = optimizer2.build_synergy_pairs_list(farms)
        
        count_batch = len(synergy_pairs_batch)
        
        time_batch = time.perf_counter() - start
        speedup_batch = time_old / time_batch if time_batch > 0 else 0
        print(f"      Time: {time_batch*1000:.2f} ms")
        print(f"      Speedup: {speedup_batch:.2f}x")
        print(f"      Output: {count_batch} tuples ready for use")
        
        # Verify all methods produce same count
        assert count_old == count_simple == count_opt
        print(f"\n  ✓ All methods verified ({count_old} iterations)")


def example_usage_in_create_cqm():
    """
    Example showing how to use SynergyOptimizer in create_cqm() function.
    
    Replace the old nested loop:
    
        for farm in farms:
            for crop1, pairs in synergy_matrix.items():
                if crop1 in foods:
                    for crop2, boost_value in pairs.items():
                        if crop2 in foods and crop1 < crop2:
                            objective += synergy_bonus_weight * boost_value * Y[(farm, crop1)] * Y[(farm, crop2)]
    
    With the optimized version:
    
        optimizer = SynergyOptimizer(synergy_matrix, foods)
        objective += optimizer.build_synergy_terms_dimod(farms, Y, synergy_bonus_weight)
    """
    
    print("\n" + "="*70)
    print("EXAMPLE: Optimized create_cqm() integration")
    print("="*70)
    
    print("""
    # OLD CODE (slow):
    for farm in farms:
        for crop1, pairs in synergy_matrix.items():
            if crop1 in foods:
                for crop2, boost_value in pairs.items():
                    if crop2 in foods and crop1 < crop2:
                        objective += synergy_bonus_weight * boost_value * Y[(farm, crop1)] * Y[(farm, crop2)]
                        pbar.update(1)
    
    # NEW CODE (fast):
    from src.synergy_optimizer_pure import SynergyOptimizer  # or compiled synergy_optimizer
    
    optimizer = SynergyOptimizer(synergy_matrix, foods)
    objective += optimizer.build_synergy_terms_dimod(farms, Y, synergy_bonus_weight)
    pbar.update(optimizer.get_n_pairs() * len(farms))
    
    # For PuLP (McCormick linearization):
    optimizer = SynergyOptimizer(synergy_matrix, foods)
    synergy_pairs = optimizer.build_synergy_pairs_list(farms)
    # Now synergy_pairs is a list of (farm, crop1, crop2, boost_value) tuples
    # Use this to create Z variables and add synergy terms
    """)


if __name__ == "__main__":
    benchmark_synergy_computation()
    example_usage_in_create_cqm()
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("""
    To get maximum performance:
    
    1. Install Cython:
       pip install cython
    
    2. Compile the optimizer:
       cd /path/to/OQI-UC002-DWave
       python setup_synergy.py build_ext --inplace
    
    3. Use in solver_runner_LQ.py:
       from synergy_optimizer import SynergyOptimizer  # Cython version
       # OR
       from src.synergy_optimizer_pure import SynergyOptimizer  # Pure Python
    
    Expected speedup:
    - Pure Python (NumPy): 2-5x faster than original
    - Cython compiled: 10-100x faster than original
    
    The speedup is most noticeable for large problems (100+ farms).
    """)
