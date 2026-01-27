#!/usr/bin/env python3
"""
Simulated Annealing Resource Estimation Benchmark

Runs small-scale SA benchmarks to gather empirical timing data
for the Resource Estimation section of the proposal.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

# Import problem creation utilities
from Utils.patch_sampler import generate_farms as generate_patches
from src.scenarios import load_food_data

# Import SA sampler
try:
    from dwave.samplers import SimulatedAnnealingSampler
    print("Using dwave.samplers.SimulatedAnnealingSampler")
except ImportError:
    from neal import SimulatedAnnealingSampler
    print("Using neal.SimulatedAnnealingSampler")

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, cqm_to_bqm


def create_crop_allocation_bqm(n_patches: int, n_crops: int = 27, seed: int = 42) -> BinaryQuadraticModel:
    """
    Create a Binary Quadratic Model for crop allocation problem.
    
    This mimics the Variant A (Binary Crop Allocation) formulation.
    """
    np.random.seed(seed)
    
    # Generate random benefit scores for crops
    benefits = {f"crop_{i}": np.random.uniform(0.3, 1.0) for i in range(n_crops)}
    
    # Generate patch areas (normalized to sum to 1)
    patch_areas = np.random.uniform(0.5, 1.5, n_patches)
    patch_areas = patch_areas / patch_areas.sum()
    
    # Create BQM
    bqm = BinaryQuadraticModel(vartype='BINARY')
    
    # Add linear terms (benefit from planting crop c on patch p)
    for p in range(n_patches):
        for c in range(n_crops):
            var_name = f"Y_{p}_{c}"
            benefit = benefits[f"crop_{c}"] * patch_areas[p]
            bqm.add_variable(var_name, -benefit)  # Negative for maximization
    
    # Add one-hot constraint penalty (each patch gets at most one crop)
    penalty_weight = 2.0  # Strong penalty for constraint violation
    for p in range(n_patches):
        patch_vars = [f"Y_{p}_{c}" for c in range(n_crops)]
        # (sum - 1)^2 penalty for one-hot
        for i, v1 in enumerate(patch_vars):
            bqm.add_variable(v1, penalty_weight)  # -2*1 coefficient
            for v2 in patch_vars[i+1:]:
                bqm.add_interaction(v1, v2, 2 * penalty_weight)  # 2*1*1 coefficient
    
    return bqm


def create_rotation_bqm(n_farms: int, n_crops: int = 6, n_periods: int = 3, seed: int = 42) -> BinaryQuadraticModel:
    """
    Create a Binary Quadratic Model for multi-period rotation problem (Variant B).
    
    This includes quadratic rotation synergy terms.
    """
    np.random.seed(seed)
    
    # Generate random benefit scores
    benefits = {f"crop_{i}": np.random.uniform(0.3, 1.0) for i in range(n_crops)}
    
    # Generate farm areas
    farm_areas = np.random.uniform(0.5, 1.5, n_farms)
    farm_areas = farm_areas / farm_areas.sum()
    
    # Generate rotation synergy matrix (some positive, some negative)
    rotation_matrix = np.random.uniform(-0.3, 0.3, (n_crops, n_crops))
    np.fill_diagonal(rotation_matrix, -0.5)  # Monoculture penalty
    
    # Create BQM
    bqm = BinaryQuadraticModel(vartype='BINARY')
    
    # Linear terms: base benefit
    for f in range(n_farms):
        for c in range(n_crops):
            for t in range(n_periods):
                var_name = f"Y_{f}_{c}_{t}"
                benefit = benefits[f"crop_{c}"] * farm_areas[f]
                bqm.add_variable(var_name, -benefit)
    
    # Quadratic terms: temporal rotation synergy
    rotation_weight = 0.2
    for f in range(n_farms):
        for t in range(n_periods - 1):
            for c1 in range(n_crops):
                for c2 in range(n_crops):
                    v1 = f"Y_{f}_{c1}_{t}"
                    v2 = f"Y_{f}_{c2}_{t+1}"
                    synergy = rotation_matrix[c1, c2] * farm_areas[f] * rotation_weight
                    bqm.add_interaction(v1, v2, -synergy)
    
    # One-hot constraints per farm-period
    penalty_weight = 2.0
    for f in range(n_farms):
        for t in range(n_periods):
            period_vars = [f"Y_{f}_{c}_{t}" for c in range(n_crops)]
            for i, v1 in enumerate(period_vars):
                bqm.add_variable(v1, penalty_weight)
                for v2 in period_vars[i+1:]:
                    bqm.add_interaction(v1, v2, 2 * penalty_weight)
    
    return bqm


def run_sa_benchmark(bqm: BinaryQuadraticModel, num_reads: int = 100, 
                     num_sweeps: int = 1000) -> dict:
    """Run SA and return timing and quality metrics."""
    sampler = SimulatedAnnealingSampler()
    
    start_time = time.perf_counter()
    sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)
    end_time = time.perf_counter()
    
    solve_time = end_time - start_time
    best_energy = sampleset.first.energy
    
    return {
        'solve_time_s': solve_time,
        'best_energy': best_energy,
        'num_reads': num_reads,
        'num_sweeps': num_sweeps,
        'num_variables': len(bqm.variables),
        'num_interactions': len(bqm.quadratic)
    }


def main():
    print("="*80)
    print("SIMULATED ANNEALING RESOURCE ESTIMATION BENCHMARK")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'variant_a': [],  # Binary Crop Allocation
        'variant_b': [],  # Multi-Period Rotation
    }
    
    # Variant A: Binary Crop Allocation
    print("\n" + "="*80)
    print("VARIANT A: Binary Crop Allocation (27 crops)")
    print("="*80)
    
    patch_sizes = [5, 10, 25, 50, 100, 200, 500]
    n_crops = 27
    
    for n_patches in patch_sizes:
        print(f"\n--- {n_patches} patches × {n_crops} crops = {n_patches * n_crops} variables ---")
        
        bqm = create_crop_allocation_bqm(n_patches, n_crops)
        
        # Run SA with different read counts
        for num_reads in [100, 500]:
            result = run_sa_benchmark(bqm, num_reads=num_reads, num_sweeps=1000)
            result['n_patches'] = n_patches
            result['n_crops'] = n_crops
            results['variant_a'].append(result)
            
            print(f"  num_reads={num_reads:4d}: {result['solve_time_s']:.3f}s, "
                  f"vars={result['num_variables']:5d}, "
                  f"interactions={result['num_interactions']:6d}, "
                  f"energy={result['best_energy']:.4f}")
    
    # Variant B: Multi-Period Rotation
    print("\n" + "="*80)
    print("VARIANT B: Multi-Period Rotation (6 crops × 3 periods)")
    print("="*80)
    
    farm_sizes = [5, 10, 25, 50, 100, 200]
    n_crops_b = 6
    n_periods = 3
    
    for n_farms in farm_sizes:
        total_vars = n_farms * n_crops_b * n_periods
        print(f"\n--- {n_farms} farms × {n_crops_b} crops × {n_periods} periods = {total_vars} variables ---")
        
        bqm = create_rotation_bqm(n_farms, n_crops_b, n_periods)
        
        for num_reads in [100, 500]:
            result = run_sa_benchmark(bqm, num_reads=num_reads, num_sweeps=1000)
            result['n_farms'] = n_farms
            result['n_crops'] = n_crops_b
            result['n_periods'] = n_periods
            results['variant_b'].append(result)
            
            print(f"  num_reads={num_reads:4d}: {result['solve_time_s']:.3f}s, "
                  f"vars={result['num_variables']:5d}, "
                  f"interactions={result['num_interactions']:6d}, "
                  f"energy={result['best_energy']:.4f}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: SA Runtime Scaling")
    print("="*80)
    
    print("\nVariant A (100 reads):")
    a_100 = [r for r in results['variant_a'] if r['num_reads'] == 100]
    for r in a_100:
        ms_per_var = (r['solve_time_s'] * 1000) / r['num_variables']
        print(f"  {r['num_variables']:5d} vars: {r['solve_time_s']*1000:8.1f} ms "
              f"({ms_per_var:.3f} ms/var)")
    
    print("\nVariant B (100 reads):")
    b_100 = [r for r in results['variant_b'] if r['num_reads'] == 100]
    for r in b_100:
        ms_per_var = (r['solve_time_s'] * 1000) / r['num_variables']
        print(f"  {r['num_variables']:5d} vars: {r['solve_time_s']*1000:8.1f} ms "
              f"({ms_per_var:.3f} ms/var)")
    
    # Fit scaling model
    print("\n" + "="*80)
    print("SCALING FIT (linear regression on log-log)")
    print("="*80)
    
    # Variant A fit
    vars_a = np.array([r['num_variables'] for r in a_100])
    times_a = np.array([r['solve_time_s'] for r in a_100])
    log_vars_a = np.log(vars_a)
    log_times_a = np.log(times_a)
    slope_a, intercept_a = np.polyfit(log_vars_a, log_times_a, 1)
    print(f"\nVariant A: T ∝ n^{slope_a:.2f}")
    print(f"  Fit: T = {np.exp(intercept_a):.6f} × n^{slope_a:.2f}")
    
    # Variant B fit
    vars_b = np.array([r['num_variables'] for r in b_100])
    times_b = np.array([r['solve_time_s'] for r in b_100])
    log_vars_b = np.log(vars_b)
    log_times_b = np.log(times_b)
    slope_b, intercept_b = np.polyfit(log_vars_b, log_times_b, 1)
    print(f"\nVariant B: T ∝ n^{slope_b:.2f}")
    print(f"  Fit: T = {np.exp(intercept_b):.6f} × n^{slope_b:.2f}")
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'sa_benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Print extrapolation for hybrid comparison
    print("\n" + "="*80)
    print("EXTRAPOLATION FOR HYBRID COMPARISON")
    print("="*80)
    print("\nBased on arXiv:2412.07460 (Hrga et al.):")
    print("  - Hybrid solver: ~3-5s constant for 100-10,000 variables")
    print("  - SA (quality): ~200-300x slower than Hybrid")
    print("  - SA (speed): ~1x Hybrid speed but worse quality")
    print("\nOur empirical SA results (100 reads, 1000 sweeps):")
    print("  - Scales approximately as T ∝ n^{:.2f}".format((slope_a + slope_b)/2))
    print("\nProjected comparison at 1000 variables:")
    projected_sa_1000 = np.exp(intercept_a) * (1000 ** slope_a)
    print(f"  - SA time (our benchmark): ~{projected_sa_1000:.2f}s")
    print(f"  - Hybrid time (literature): ~3-5s")
    print(f"  - Ratio SA/Hybrid: ~{projected_sa_1000/4:.1f}x")
    
    return results


if __name__ == "__main__":
    results = main()
