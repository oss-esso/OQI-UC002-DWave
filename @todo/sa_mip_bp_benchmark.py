#!/usr/bin/env python3
"""
Simulated Annealing Resource Estimation Benchmark - MIP vs BP Formulations

Tests SA performance on BOTH formulations to match gurobi_scaling_benchmark.py:

1. MIP (Farm-level formulation) - the ORIGINAL problem:
   - Variables: Afc (continuous area), Yfc (binary indicator), Uc (binary usage)
   - Cannot be directly solved by SA (continuous variables)
   - We test the QUBO-converted version with discretized areas

2. BP (Binary Patch formulation) - required for QUBO conversion:
   - Variables: Y[p,c] (binary), U[c] (binary) - NO continuous area variables
   - Y[p,c] = 1 if crop c is assigned to patch p
   - U[c] = 1 if crop c is used on any patch
   - Pure binary, directly solvable by SA/quantum annealing

This benchmark gathers empirical timing data for the Resource Estimation section.
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

# Import SA sampler
try:
    from dwave.samplers import SimulatedAnnealingSampler
    print("Using dwave.samplers.SimulatedAnnealingSampler")
except ImportError:
    from neal import SimulatedAnnealingSampler
    print("Using neal.SimulatedAnnealingSampler")

from dimod import BinaryQuadraticModel


# Configuration matching gurobi_scaling_benchmark.py
N_FOODS = 27  # Fixed number of foods


def generate_synthetic_food_data(n_foods: int, seed: int = 42) -> dict:
    """Generate synthetic food data for benchmarking."""
    np.random.seed(seed)
    
    foods = {}
    for i in range(n_foods):
        foods[f"food_{i:02d}"] = {
            'nutritional_value': np.random.uniform(0.3, 1.0),
            'nutrient_density': np.random.uniform(0.2, 0.9),
            'environmental_impact': np.random.uniform(0.1, 0.8),
            'affordability': np.random.uniform(0.3, 1.0),
            'sustainability': np.random.uniform(0.2, 0.9),
        }
    return foods


def compute_benefits(foods: dict, n_foods: int) -> np.ndarray:
    """Compute composite benefit scores for each food."""
    weights = {
        'nutritional_value': 0.25,
        'nutrient_density': 0.20,
        'environmental_impact': 0.15,  # Negative contribution
        'affordability': 0.20,
        'sustainability': 0.20,
    }
    
    food_names = list(foods.keys())
    benefits = np.zeros(n_foods)
    for c_idx, c in enumerate(food_names):
        food = foods[c]
        benefits[c_idx] = (
            weights['nutritional_value'] * food['nutritional_value'] +
            weights['nutrient_density'] * food['nutrient_density'] -
            weights['environmental_impact'] * food['environmental_impact'] +
            weights['affordability'] * food['affordability'] +
            weights['sustainability'] * food['sustainability']
        )
    return benefits, food_names


def create_bp_bqm(n_patches: int, n_foods: int = N_FOODS, seed: int = 42) -> BinaryQuadraticModel:
    """
    Create BQM for Binary Patch (BP) formulation.
    
    This is the QUBO-compatible formulation with:
    - Y[p,c] ∈ {0,1}: Binary variable for crop c on patch p
    - U[c] ∈ {0,1}: Binary indicator if crop c is used anywhere
    
    Constraints converted to penalties:
    - One-hot: Each patch gets exactly one crop
    - U-Y linking: U[c] = 1 iff any Y[p,c] = 1
    
    Total variables: n_patches * n_foods + n_foods
    """
    np.random.seed(seed)
    
    # Generate food data
    foods = generate_synthetic_food_data(n_foods, seed)
    benefits, food_names = compute_benefits(foods, n_foods)
    
    # Normalize benefits
    total_area = 100.0
    patch_area = total_area / n_patches
    normalized_benefits = benefits * (patch_area / total_area)
    
    # Create BQM
    bqm = BinaryQuadraticModel(vartype='BINARY')
    
    # Add Y[p,c] variables with linear bias (benefit)
    for p in range(n_patches):
        for c in range(n_foods):
            var_name = f"Y_{p}_{c}"
            # Negative because we maximize (BQM minimizes)
            bqm.add_variable(var_name, -normalized_benefits[c])
    
    # Add U[c] variables (global usage indicators)
    for c in range(n_foods):
        var_name = f"U_{c}"
        bqm.add_variable(var_name, 0.0)  # No direct benefit from U
    
    # One-hot constraint penalty: sum_c Y[p,c] = 1 for each patch
    # Penalty: lambda * (sum_c Y[p,c] - 1)^2
    lambda_onehot = 2.0
    for p in range(n_patches):
        patch_vars = [f"Y_{p}_{c}" for c in range(n_foods)]
        # Expand (sum - 1)^2 = sum^2 - 2*sum + 1
        # = sum_i Y_i^2 + 2*sum_{i<j} Y_i*Y_j - 2*sum_i Y_i + 1
        # Since Y^2 = Y for binary: = sum_i Y_i + 2*sum_{i<j} Y_i*Y_j - 2*sum_i Y_i + 1
        # = -sum_i Y_i + 2*sum_{i<j} Y_i*Y_j + 1
        for i, v1 in enumerate(patch_vars):
            bqm.add_variable(v1, -lambda_onehot)  # -1 coefficient from expansion
            for v2 in patch_vars[i+1:]:
                bqm.add_interaction(v1, v2, 2 * lambda_onehot)
    
    # U-Y linking constraint: U[c] >= Y[p,c] for all p
    # Equivalently: Y[p,c] <= U[c], i.e., Y[p,c] * (1 - U[c]) = 0
    # Penalty: lambda * Y[p,c] * (1 - U[c]) = lambda * (Y[p,c] - Y[p,c]*U[c])
    lambda_link = 1.5
    for c in range(n_foods):
        u_var = f"U_{c}"
        for p in range(n_patches):
            y_var = f"Y_{p}_{c}"
            bqm.add_variable(y_var, lambda_link)  # +lambda * Y
            bqm.add_interaction(y_var, u_var, -lambda_link)  # -lambda * Y*U
    
    return bqm


def create_mip_discretized_bqm(n_farms: int, n_foods: int = N_FOODS, 
                               n_levels: int = 4, seed: int = 42) -> BinaryQuadraticModel:
    """
    Create BQM for discretized MIP (Farm-level) formulation.
    
    The original MIP has continuous A[f,c] variables. To convert to QUBO,
    we discretize: A[f,c] = sum_k (L_f / n_levels) * a[f,c,k]
    where a[f,c,k] ∈ {0,1} are binary.
    
    This is more complex than BP and shows why BP is preferred for QUBO.
    
    Variables:
    - a[f,c,k]: Binary, k-th level of area allocation for crop c on farm f
    - Y[f,c]: Binary indicator if any area allocated to c on f
    - U[c]: Binary indicator if crop c used anywhere
    
    Total variables: n_farms * n_foods * n_levels + n_farms * n_foods + n_foods
    """
    np.random.seed(seed)
    
    # Generate food data
    foods = generate_synthetic_food_data(n_foods, seed)
    benefits, food_names = compute_benefits(foods, n_foods)
    
    # Farm areas (uneven distribution)
    farm_areas = np.random.uniform(0.5, 2.0, n_farms)
    farm_areas = farm_areas / farm_areas.sum() * 100.0  # Normalize to 100 ha total
    
    total_area = farm_areas.sum()
    
    # Create BQM
    bqm = BinaryQuadraticModel(vartype='BINARY')
    
    # Add a[f,c,k] variables with linear bias (benefit proportional to area level)
    for f in range(n_farms):
        level_area = farm_areas[f] / n_levels
        for c in range(n_foods):
            for k in range(n_levels):
                var_name = f"a_{f}_{c}_{k}"
                # Benefit = benefit_c * level_area / total_area
                bqm.add_variable(var_name, -benefits[c] * level_area / total_area)
    
    # Add Y[f,c] and U[c] variables
    for f in range(n_farms):
        for c in range(n_foods):
            bqm.add_variable(f"Y_{f}_{c}", 0.0)
    for c in range(n_foods):
        bqm.add_variable(f"U_{c}", 0.0)
    
    # Land constraint: sum_c sum_k a[f,c,k] * level_area <= L_f
    # Since level_area = L_f / n_levels, this becomes: sum_c sum_k a[f,c,k] <= n_levels
    # Penalty for exceeding: lambda * max(0, sum - n_levels)^2
    # Approximate with soft penalty: lambda * (sum - n_levels)^2 when sum > n_levels
    lambda_land = 2.0
    for f in range(n_farms):
        farm_vars = [f"a_{f}_{c}_{k}" for c in range(n_foods) for k in range(n_levels)]
        # We use a simpler penalty: encourage sum <= n_levels
        # Add quadratic penalty for large sums
        for i, v1 in enumerate(farm_vars):
            for v2 in farm_vars[i+1:]:
                bqm.add_interaction(v1, v2, lambda_land / (n_levels * n_levels))
    
    # Y-a linking: Y[f,c] >= a[f,c,k] for all k
    # If any a[f,c,k]=1, then Y[f,c]=1
    lambda_ya = 1.0
    for f in range(n_farms):
        for c in range(n_foods):
            y_var = f"Y_{f}_{c}"
            for k in range(n_levels):
                a_var = f"a_{f}_{c}_{k}"
                bqm.add_variable(a_var, lambda_ya)
                bqm.add_interaction(a_var, y_var, -lambda_ya)
    
    # U-Y linking: U[c] >= Y[f,c] for all f
    lambda_uy = 1.0
    for c in range(n_foods):
        u_var = f"U_{c}"
        for f in range(n_farms):
            y_var = f"Y_{f}_{c}"
            bqm.add_variable(y_var, lambda_uy)
            bqm.add_interaction(y_var, u_var, -lambda_uy)
    
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
    print("SIMULATED ANNEALING BENCHMARK: MIP vs BP FORMULATIONS")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'hardware': 'Intel i7-12700H, 16 GB RAM',
        'bp_formulation': [],  # Binary Patch (pure binary, QUBO-native)
        'mip_discretized': [], # MIP with discretized areas (complex QUBO)
    }
    
    # ========================================================================
    # BP Formulation (Binary Patch) - Native QUBO
    # ========================================================================
    print("\n" + "="*80)
    print("BP FORMULATION: Binary Patch (pure binary, QUBO-native)")
    print("Variables = n_patches × n_foods + n_foods")
    print("="*80)
    
    # Log sweep matching gurobi_scaling_benchmark.py
    # Target variables: 100, 316, 1000, 3162, 10000, 31623, 100000
    bp_configs = [
        (3, 108),       # ~100 vars: 3×27 + 27 = 108
        (10, 297),      # ~300 vars: 10×27 + 27 = 297  
        (36, 999),      # ~1000 vars: 36×27 + 27 = 999
        (115, 3132),    # ~3000 vars: 115×27 + 27 = 3132
        (370, 9_017),   # ~10000 vars: 370×27 + 27 = 10017
        (1170, 31_617), # ~30000 vars: 1170×27 + 27 = 31617
    ]
    
    for n_patches, expected_vars in bp_configs:
        actual_vars = n_patches * N_FOODS + N_FOODS
        print(f"\n--- BP: {n_patches} patches × {N_FOODS} foods = {actual_vars} variables ---")
        
        bqm = create_bp_bqm(n_patches, N_FOODS)
        
        for num_reads in [100, 500]:
            result = run_sa_benchmark(bqm, num_reads=num_reads, num_sweeps=1000)
            result['n_patches'] = n_patches
            result['n_foods'] = N_FOODS
            result['formulation'] = 'BP'
            results['bp_formulation'].append(result)
            
            print(f"  reads={num_reads:4d}: {result['solve_time_s']:.3f}s, "
                  f"vars={result['num_variables']:6d}, "
                  f"interactions={result['num_interactions']:7d}")
    
    # ========================================================================
    # MIP Discretized (for comparison - shows QUBO conversion complexity)
    # ========================================================================
    print("\n" + "="*80)
    print("MIP DISCRETIZED: Farm-level with area discretization")
    print("Variables = n_farms × n_foods × n_levels + n_farms × n_foods + n_foods")
    print("(Much larger than BP for same problem size)")
    print("="*80)
    
    # Fewer configs since MIP discretized is much larger
    mip_configs = [
        (3, 4),   # 3 farms, 4 levels: 3×27×4 + 3×27 + 27 = 324 + 81 + 27 = 432
        (10, 4),  # 10 farms: 10×27×4 + 10×27 + 27 = 1080 + 270 + 27 = 1377
        (25, 4),  # 25 farms: 25×27×4 + 25×27 + 27 = 2700 + 675 + 27 = 3402
        (50, 4),  # 50 farms: 50×27×4 + 50×27 + 27 = 5400 + 1350 + 27 = 6777
    ]
    
    for n_farms, n_levels in mip_configs:
        actual_vars = n_farms * N_FOODS * n_levels + n_farms * N_FOODS + N_FOODS
        print(f"\n--- MIP: {n_farms} farms × {N_FOODS} foods × {n_levels} levels = {actual_vars} variables ---")
        
        bqm = create_mip_discretized_bqm(n_farms, N_FOODS, n_levels)
        
        for num_reads in [100]:  # Just 100 reads for MIP (slower)
            result = run_sa_benchmark(bqm, num_reads=num_reads, num_sweeps=1000)
            result['n_farms'] = n_farms
            result['n_foods'] = N_FOODS
            result['n_levels'] = n_levels
            result['formulation'] = 'MIP_discretized'
            results['mip_discretized'].append(result)
            
            print(f"  reads={num_reads:4d}: {result['solve_time_s']:.3f}s, "
                  f"vars={result['num_variables']:6d}, "
                  f"interactions={result['num_interactions']:7d}")
    
    # ========================================================================
    # Summary and Scaling Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY: SA Runtime Scaling")
    print("="*80)
    
    # BP Formulation scaling
    print("\nBP Formulation (100 reads):")
    bp_100 = [r for r in results['bp_formulation'] if r['num_reads'] == 100]
    for r in bp_100:
        ms_per_var = (r['solve_time_s'] * 1000) / r['num_variables']
        print(f"  {r['num_variables']:6d} vars: {r['solve_time_s']*1000:8.1f} ms "
              f"({ms_per_var:.3f} ms/var)")
    
    # MIP Discretized scaling
    print("\nMIP Discretized (100 reads):")
    mip_100 = [r for r in results['mip_discretized'] if r['num_reads'] == 100]
    for r in mip_100:
        ms_per_var = (r['solve_time_s'] * 1000) / r['num_variables']
        print(f"  {r['num_variables']:6d} vars: {r['solve_time_s']*1000:8.1f} ms "
              f"({ms_per_var:.3f} ms/var)")
    
    # Fit scaling models
    print("\n" + "="*80)
    print("SCALING FIT (power law: T = α × n^β)")
    print("="*80)
    
    if len(bp_100) >= 3:
        vars_bp = np.array([r['num_variables'] for r in bp_100])
        times_bp = np.array([r['solve_time_s'] for r in bp_100])
        log_vars = np.log(vars_bp)
        log_times = np.log(times_bp)
        slope, intercept = np.polyfit(log_vars, log_times, 1)
        print(f"\nBP: T ≈ {np.exp(intercept):.6f} × n^{slope:.2f}")
        results['bp_scaling'] = {'alpha': np.exp(intercept), 'beta': slope}
    
    if len(mip_100) >= 3:
        vars_mip = np.array([r['num_variables'] for r in mip_100])
        times_mip = np.array([r['solve_time_s'] for r in mip_100])
        log_vars = np.log(vars_mip)
        log_times = np.log(times_mip)
        slope, intercept = np.polyfit(log_vars, log_times, 1)
        print(f"MIP (discretized): T ≈ {np.exp(intercept):.6f} × n^{slope:.2f}")
        results['mip_scaling'] = {'alpha': np.exp(intercept), 'beta': slope}
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'sa_mip_bp_benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Literature comparison
    print("\n" + "="*80)
    print("COMPARISON WITH LITERATURE")
    print("="*80)
    print("\nFrom arXiv:2412.07460 (Hrga et al., Max-Cut benchmark):")
    print("  - D-Wave Hybrid: ~3-5s constant for 100-10,000 variables")
    print("  - SA (quality-focused): ~672s for 800-10,000 nodes")
    print("  - SA (speed-focused): ~2.4-2.7s, similar to Hybrid")
    print("  - Hybrid provides 100-200× speedup over quality SA")
    print("\nFrom KTH thesis (Reinholdsson & Odelius, 2023):")
    print("  - Embedding overhead dominates: 95-99% of wall time")
    print("  - Pure QPU time scales nearly linearly with problem size")
    print("  - Hybrid solvers best for constrained formulations")
    
    return results


if __name__ == "__main__":
    results = main()
