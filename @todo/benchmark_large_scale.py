#!/usr/bin/env python3
"""
Large-Scale Benchmark: Hierarchical Solver on 50, 100, 200, 500 Farms

Matches the scale of previous non-rotation tests.
Uses SimulatedAnnealing (no QPU access).

Author: OQI-UC002-DWave
Date: 2025-12-12
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from hierarchical_quantum_solver import solve_hierarchical, DEFAULT_CONFIG, OUTPUT_DIR
from src.scenarios import load_food_data

# Test configurations
TEST_CONFIGS = [
    {'name': '50_farms', 'scenario': 'rotation_250farms_27foods', 'farms_limit': 50, 'farms_per_cluster': 10, 'reads': 20},
    {'name': '100_farms', 'scenario': 'rotation_250farms_27foods', 'farms_limit': 100, 'farms_per_cluster': 10, 'reads': 20},
    {'name': '200_farms', 'scenario': 'rotation_250farms_27foods', 'farms_limit': 200, 'farms_per_cluster': 15, 'reads': 20},
    {'name': '500_farms', 'scenario': 'rotation_500farms_27foods', 'farms_limit': None, 'farms_per_cluster': 20, 'reads': 20},
]

def run_benchmark(config: dict, use_qpu: bool = False):
    """Run benchmark for a specific configuration."""
    
    name = config['name']
    scenario = config['scenario']
    farms_limit = config['farms_limit']
    
    print("\n" + "="*80)
    print(f"BENCHMARK: {name}")
    print("="*80)
    
    # Load scenario
    print(f"Loading {scenario}...")
    farms, foods, food_groups, cfg = load_food_data(scenario)
    
    params = cfg.get('parameters', {})
    weights = params.get('weights', {})
    la = params.get('land_availability', {})
    
    # Limit farms if specified
    if farms_limit:
        farm_names = list(la.keys())[:farms_limit]
        la = {f: la[f] for f in farm_names}
    else:
        farm_names = list(la.keys())
    
    # Build data dict
    food_names = list(foods.keys())
    food_benefits = {}
    for food in food_names:
        food_data = foods.get(food, {})
        benefit = sum(food_data.get(attr, 0.5) * w for attr, w in weights.items())
        food_benefits[food] = benefit
    
    data = {
        'foods': foods,
        'food_names': food_names,
        'food_groups': food_groups,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': la,
        'farm_names': farm_names,
        'total_area': sum(la.values()),
        'n_farms': len(farm_names),
        'n_foods': len(food_names),
        'config': cfg,
    }
    
    n_farms = data['n_farms']
    n_foods = data['n_foods']
    n_vars = n_farms * n_foods * 3
    n_vars_agg = n_farms * 6 * 3  # After aggregation
    
    print(f"  Farms: {n_farms}")
    print(f"  Foods: {n_foods}")
    print(f"  Variables (original): {n_vars:,}")
    print(f"  Variables (aggregated): {n_vars_agg:,}")
    
    # Configure solver
    solver_config = DEFAULT_CONFIG.copy()
    solver_config['farms_per_cluster'] = config['farms_per_cluster']
    solver_config['num_iterations'] = 2  # Fast for SA
    solver_config['num_reads'] = config['reads']
    
    n_clusters = (n_farms + solver_config['farms_per_cluster'] - 1) // solver_config['farms_per_cluster']
    
    print(f"\nSolver config:")
    print(f"  Farms per cluster: {solver_config['farms_per_cluster']}")
    print(f"  Expected clusters: {n_clusters}")
    print(f"  Iterations: {solver_config['num_iterations']}")
    print(f"  Reads per cluster: {solver_config['num_reads']}")
    print(f"  Est. total solves: {n_clusters * solver_config['num_iterations']}")
    
    # Solve
    print("\n" + "-"*80)
    print("SOLVING...")
    print("-"*80)
    
    start_time = time.time()
    result = solve_hierarchical(data, solver_config, use_qpu=use_qpu, verbose=True)
    total_time = time.time() - start_time
    
    # Extract results
    success = result['success']
    objective = result['objective']
    violations = result['violations']
    diversity = result['diversity_stats']
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{name}_{timestamp}.json"
    output_file = OUTPUT_DIR / filename
    
    # Convert for JSON
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(x) for x in obj]
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_for_json(result), f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print(f"BENCHMARK RESULTS: {name}")
    print("="*80)
    print(f"  Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Level 1 (decomposition): {result['timings']['level1_decomposition']:.4f}s")
    print(f"  Level 2 (quantum): {result['timings']['level2_quantum']:.2f}s")
    print(f"  Level 3 (post-process): {result['timings']['level3_postprocessing']:.4f}s")
    print(f"\n  Objective: {objective:.4f}")
    print(f"  Violations: {violations}")
    print(f"  Unique crops: {diversity['total_unique_crops']}/{diversity['max_possible_crops']}")
    print(f"  Shannon diversity: {diversity['shannon_diversity']:.3f}/{diversity['max_shannon']:.3f}")
    print(f"  Coverage: {diversity['coverage_ratio']:.1%}")
    print(f"\n  Saved to: {output_file}")
    print("="*80)
    
    return {
        'name': name,
        'n_farms': n_farms,
        'n_foods': n_foods,
        'n_vars': n_vars,
        'n_vars_agg': n_vars_agg,
        'n_clusters': result['levels']['decomposition']['n_clusters'],
        'total_time': total_time,
        'level1_time': result['timings']['level1_decomposition'],
        'level2_time': result['timings']['level2_quantum'],
        'level3_time': result['timings']['level3_postprocessing'],
        'objective': objective,
        'violations': violations,
        'unique_crops': diversity['total_unique_crops'],
        'shannon': diversity['shannon_diversity'],
        'success': success,
    }


def main():
    """Run all benchmarks."""
    
    print("="*80)
    print("LARGE-SCALE HIERARCHICAL SOLVER BENCHMARK")
    print("="*80)
    print("\nTesting on: 50, 100, 200, 500 farms (matching previous tests)")
    print("Using SimulatedAnnealing (no QPU access)")
    print("="*80)
    
    results = []
    
    for config in TEST_CONFIGS:
        try:
            result = run_benchmark(config, use_qpu=False)
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Benchmark interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': config['name'],
                'success': False,
                'error': str(e)
            })
    
    # Final summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    print("\n{:<15s} {:>8s} {:>8s} {:>12s} {:>10s} {:>8s} {:>8s}".format(
        "Test", "Farms", "Vars", "Time (s)", "Objective", "Crops", "Status"
    ))
    print("-"*80)
    
    for r in results:
        if r['success']:
            print("{:<15s} {:>8d} {:>8,d} {:>12.1f} {:>10.4f} {:>8d} {:>8s}".format(
                r['name'],
                r['n_farms'],
                r['n_vars_agg'],  # Show aggregated vars
                r['total_time'],
                r['objective'],
                r['unique_crops'],
                '✅'
            ))
        else:
            print("{:<15s} {:>8s} {:>8s} {:>12s} {:>10s} {:>8s} {:>8s}".format(
                r['name'], '-', '-', '-', '-', '-', '❌'
            ))
    
    print("-"*80)
    
    # Success rate
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    print(f"\nSuccess rate: {successful}/{total} ({successful/total*100:.0f}%)")
    
    if successful > 0:
        # Timing analysis
        times = [r['total_time'] for r in results if r.get('success', False)]
        farms = [r['n_farms'] for r in results if r.get('success', False)]
        
        print("\nScaling analysis:")
        for i in range(len(times)):
            time_per_farm = times[i] / farms[i]
            print(f"  {farms[i]:>3d} farms: {times[i]:>6.1f}s ({time_per_farm:.3f}s/farm)")
        
        # Extrapolate to QPU
        print("\n⚡ Estimated QPU times (assuming 10-20× speedup):")
        for i in range(len(times)):
            qpu_low = times[i] / 20
            qpu_high = times[i] / 10
            print(f"  {farms[i]:>3d} farms: {qpu_low:.1f}s - {qpu_high:.1f}s ({qpu_low/60:.1f}-{qpu_high/60:.1f} min)")
    
    print("="*80)
    
    return successful == total


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Large-scale benchmark')
    parser.add_argument('--qpu', action='store_true', help='Use real QPU (NOT recommended without confirming)')
    
    args = parser.parse_args()
    
    if args.qpu:
        response = input("\n⚠️  This will use significant QPU access. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            sys.exit(0)
    
    success = main()
    sys.exit(0 if success else 1)
