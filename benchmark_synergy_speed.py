"""
Quick benchmark to compare synergy computation performance.
Shows the speedup from using SynergyOptimizer.
"""

import sys
import os
import time

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

from benchmark_scalability_LQ import load_full_family_with_n_farms

# Try to import optimizer
try:
    from synergy_optimizer import SynergyOptimizer
    OPTIMIZER_TYPE = "Cython"
except ImportError:
    try:
        from src.synergy_optimizer_pure import SynergyOptimizer
        OPTIMIZER_TYPE = "NumPy"
    except ImportError:
        SynergyOptimizer = None
        OPTIMIZER_TYPE = "None"

print("="*80)
print("SYNERGY COMPUTATION PERFORMANCE COMPARISON")
print("="*80)
print(f"Optimizer available: {OPTIMIZER_TYPE}")
print("="*80)

# Test with the actual benchmark size
n_farms = 1096
farms, foods, food_groups, config = load_full_family_with_n_farms(n_farms, seed=42, fixed_total_land=100.0)
synergy_matrix = config['parameters'].get('synergy_matrix', {})
synergy_bonus_weight = config['parameters']['weights'].get('synergy_bonus', 0.1)

n_foods = len(foods)
n_pairs_raw = sum(len(pairs) for pairs in synergy_matrix.values()) // 2

print(f"\nProblem Size:")
print(f"  Farms: {n_farms}")
print(f"  Foods: {n_foods}")
print(f"  Synergy pairs: {n_pairs_raw}")
print(f"  Total iterations: {n_farms * n_pairs_raw} = {n_farms:,} × {n_pairs_raw}")

# Simulate creating the Y variables (just use a dict for timing)
Y = {(farm, food): 1 for farm in farms for food in foods}

print("\n" + "="*80)
print("METHOD 1: ORIGINAL - Nested dictionary iteration")
print("="*80)

start = time.perf_counter()
count_old = 0
objective_old = 0
for farm in farms:
    for crop1, pairs in synergy_matrix.items():
        if crop1 in foods:
            for crop2, boost_value in pairs.items():
                if crop2 in foods and crop1 < crop2:
                    # Simulate objective computation
                    objective_old += synergy_bonus_weight * boost_value
                    count_old += 1

time_old = time.perf_counter() - start

print(f"  Iterations: {count_old:,}")
print(f"  Time: {time_old*1000:.2f} ms")

if SynergyOptimizer is not None:
    print("\n" + "="*80)
    print(f"METHOD 2: OPTIMIZED - SynergyOptimizer ({OPTIMIZER_TYPE})")
    print("="*80)
    
    start = time.perf_counter()
    
    optimizer = SynergyOptimizer(synergy_matrix, foods)
    precompute_time = time.perf_counter() - start
    
    start = time.perf_counter()
    count_opt = 0
    objective_opt = 0
    for farm in farms:
        for crop1, crop2, boost_value in optimizer.iter_pairs_with_names():
            # Simulate objective computation
            objective_opt += synergy_bonus_weight * boost_value
            count_opt += 1
    
    time_opt = time.perf_counter() - start
    total_time_opt = precompute_time + time_opt
    
    print(f"  Precompute time: {precompute_time*1000:.2f} ms")
    print(f"  Iteration time: {time_opt*1000:.2f} ms")
    print(f"  Total time: {total_time_opt*1000:.2f} ms")
    print(f"  Iterations: {count_opt:,}")
    
    speedup = time_old / total_time_opt
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"  Original time:   {time_old*1000:8.2f} ms")
    print(f"  Optimized time:  {total_time_opt*1000:8.2f} ms")
    print(f"  Speedup:         {speedup:8.2f}x")
    print(f"  Time saved:      {(time_old - total_time_opt)*1000:8.2f} ms ({(1 - total_time_opt/time_old)*100:.1f}%)")
    
    print("\n  Verification:")
    print(f"    Iterations match: {count_old == count_opt}")
    print(f"    Results match: {abs(objective_old - objective_opt) < 1e-10}")
    
    print("\n" + "="*80)
    print("IMPACT ON BENCHMARK")
    print("="*80)
    print(f"  For 1096 farms benchmark:")
    print(f"    Original would take: ~{time_old:.2f}s for synergy computation")
    print(f"    Optimized takes:     ~{total_time_opt:.2f}s for synergy computation")
    print(f"    Time saved per run:   {time_old - total_time_opt:.2f}s")
    
    if OPTIMIZER_TYPE == "NumPy":
        print(f"\n  💡 TIP: Install Cython for even better performance:")
        print(f"      pip install cython")
        print(f"      python setup_synergy.py build_ext --inplace")
        print(f"      Expected additional 5-10x speedup (total 50-100x vs original)")

else:
    print("\n⚠️  SynergyOptimizer not available")
    print("   Install with: pip install numpy")
    print("   Or place synergy_optimizer_pure.py in src/")

print("\n" + "="*80)
