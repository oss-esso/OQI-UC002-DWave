#!/usr/bin/env python3
"""
Gurobi QUBO Only Benchmark Script

This script runs ONLY the Gurobi QUBO solver on the patch scenarios
to update cached results after parameter changes (e.g., MIP gap relaxation).

This avoids wasting resources on other solvers that don't need re-running.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import generators and solvers
from Utils.patch_sampler import generate_farms as generate_patches_small
from src.scenarios import load_food_data
from solver_runner_PATCH import (
    create_cqm, solve_with_gurobi_qubo,
    calculate_original_objective, extract_solution_summary,
    validate_solution_constraints
)
from dimod import cqm_to_bqm

# Benchmark configurations matching comprehensive_benchmark.py
BENCHMARK_CONFIGS = [10, 15, 20, 25, 50, 100]

def create_fallback_foods():
    """Create fallback food data if Excel is not available."""
    foods = {
        'Wheat': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 'affordability': 0.9, 'sustainability': 0.7},
        'Corn': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.5, 'affordability': 0.8, 'sustainability': 0.6},
        'Rice': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.7, 'affordability': 0.7, 'sustainability': 0.8},
        'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.4, 'affordability': 0.6, 'sustainability': 0.9},
        'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.8, 'affordability': 0.9, 'sustainability': 0.6}
    }
    food_groups = {
        'grains': ['Wheat', 'Corn', 'Rice'],
        'proteins': ['Soybeans'],
        'vegetables': ['Potatoes']
    }
    return foods, food_groups

def create_food_config(land_data: Dict[str, float]):
    """Create food and configuration data compatible with solvers."""
    try:
        food_list, foods, food_groups, _ = load_food_data('simple')
    except Exception as e:
        print(f"    Warning: Food data loading failed ({e}), using fallback")
        foods, food_groups = create_fallback_foods()
    
    config = {
        'parameters': {
            'land_availability': land_data,
            'minimum_planting_area': {food: 0.0 for food in foods},
            'food_group_constraints': {},
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.2,
                'environmental_impact': 0.25,
                'affordability': 0.15,
                'sustainability': 0.15
            },
            'idle_penalty_lambda': 0.1
        }
    }
    
    return foods, food_groups, config

def run_gurobi_qubo_benchmark(config_values: List[int], force_rerun: bool = True):
    """
    Run Gurobi QUBO solver only for specified configurations.
    
    Args:
        config_values: List of patch counts to test
        force_rerun: If True, ignore cached results and re-run everything
    """
    print(f"\n{'='*80}")
    print(f"GUROBI QUBO ONLY BENCHMARK")
    print(f"Configurations: {config_values}")
    print(f"Force re-run: {force_rerun}")
    print(f"{'='*80}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "Benchmarks", "COMPREHENSIVE", "Patch_GurobiQUBO")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    total_start = time.time()
    
    for idx, n_patches in enumerate(config_values):
        print(f"\n{'='*80}")
        print(f"CONFIG {idx+1}/{len(config_values)}: {n_patches} patches")
        print(f"{'='*80}")
        
        # Generate patch sample (matching comprehensive_benchmark seed logic)
        seed = 42 + idx * 100 + 50  # Same seed as comprehensive_benchmark
        land_data = generate_patches_small(n_farms=n_patches, seed=seed)
        total_area = sum(land_data.values())
        
        print(f"Generated {n_patches} patches, {total_area:.1f} ha total area")
        
        # Create problem setup
        foods, food_groups, config = create_food_config(land_data)
        
        # Create CQM
        print("Creating CQM...")
        cqm_start = time.time()
        cqm, (X, Y), constraint_metadata = create_cqm(list(land_data.keys()), foods, food_groups, config)
        cqm_time = time.time() - cqm_start
        print(f"  CQM: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints ({cqm_time:.3f}s)")
        
        # Convert CQM to BQM
        print("Converting CQM to BQM...")
        bqm_start = time.time()
        
        # Calculate Lagrange multiplier (same as comprehensive_benchmark)
        max_obj_coeff = 0
        for plot in land_data:
            s_p = land_data[plot]
            for crop in foods:
                B_c = (
                    config['parameters']['weights'].get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                    config['parameters']['weights'].get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                    config['parameters']['weights'].get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                    config['parameters']['weights'].get('affordability', 0) * foods[crop].get('affordability', 0) +
                    config['parameters']['weights'].get('sustainability', 0) * foods[crop].get('sustainability', 0)
                )
                idle_penalty = config['parameters'].get('idle_penalty_lambda', 0.1)
                coeff = abs((B_c + idle_penalty) * s_p)
                max_obj_coeff = max(max_obj_coeff, coeff)
        
        lagrange_multiplier = 10000 * max_obj_coeff if max_obj_coeff > 0 else 10000.0
        
        print(f"  Max objective coefficient: {max_obj_coeff:.6f}")
        print(f"  Lagrange multiplier: {lagrange_multiplier:.2f} (10000x)")
        
        bqm, invert = cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
        bqm_conversion_time = time.time() - bqm_start
        print(f"  BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} interactions ({bqm_conversion_time:.3f}s)")
        
        # Check if cached result exists
        run_id = idx + 1
        filename = f"config_{n_patches}_run_{run_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath) and not force_rerun:
            print(f"  ✓ Using cached result from {filename}")
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                results.append(result)
                continue
            except Exception as e:
                print(f"  Warning: Failed to load cached result: {e}")
                print(f"  Re-running solver...")
        
        # Run Gurobi QUBO solver
        print("Running Gurobi QUBO solver...")
        try:
            gurobi_result = solve_with_gurobi_qubo(
                bqm,
                farms=list(land_data.keys()),
                foods=foods,
                food_groups=food_groups,
                land_availability=land_data,
                weights=config['parameters']['weights'],
                idle_penalty=config['parameters'].get('idle_penalty_lambda', 0.1),
                config=config
            )
            
            # Package result
            result = {
                'status': gurobi_result['status'],
                'objective_value': gurobi_result['objective_value'],
                'solve_time': gurobi_result['solve_time'],
                'bqm_energy': gurobi_result['bqm_energy'],
                'bqm_conversion_time': bqm_conversion_time,
                'success': gurobi_result['status'] == 'Optimal',
                'sample_id': idx,
                'n_units': n_patches,
                'total_area': total_area,
                'n_foods': len(foods),
                'n_variables': len(bqm.variables),
                'n_quadratic': len(bqm.quadratic),
                'note': 'objective_value is reconstructed from BQM solution; comparable to CQM',
                'timestamp': datetime.now().isoformat()
            }
            
            # Add solution summary if available
            if 'solution_summary' in gurobi_result:
                result['solution_summary'] = gurobi_result['solution_summary']
            
            # Add validation if available
            if 'validation' in gurobi_result:
                result['validation'] = {
                    'is_feasible': gurobi_result['validation']['is_feasible'],
                    'n_violations': gurobi_result['validation']['n_violations'],
                    'violations': gurobi_result['validation']['violations'],
                    'summary': gurobi_result['validation']['summary']
                }
            
            # Save result
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"  ✓ Saved result to {filename}")
            print(f"  Status: {result['status']}")
            obj_val = result.get('objective_value')
            if obj_val is not None:
                print(f"  Objective: {obj_val:.6f}")
            else:
                print(f"  Objective: None (no solution found)")
            print(f"  Solve time: {result['solve_time']:.3f}s")
            if result.get('validation'):
                print(f"  Feasible: {result['validation']['is_feasible']}")
                print(f"  Violations: {result['validation']['n_violations']}")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Gurobi QUBO failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error result
            error_result = {
                'status': 'Error',
                'error': str(e),
                'success': False,
                'sample_id': idx,
                'n_units': n_patches,
                'total_area': total_area,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(error_result, f, indent=2, default=str)
            
            results.append(error_result)
    
    total_time = time.time() - total_start
    
    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Configurations tested: {len(config_values)}")
    print(f"Results saved to: {output_dir}")
    
    # Print detailed summary
    print(f"\nResults summary:")
    for i, result in enumerate(results):
        status_icon = "✓" if result.get('success') else "✗"
        n_units = result.get('n_units', config_values[i])
        status = result.get('status', 'Unknown')
        obj_val = result.get('objective_value', 'N/A')
        solve_time = result.get('solve_time', 'N/A')
        
        if isinstance(obj_val, (int, float)):
            obj_str = f"{obj_val:.6f}"
        else:
            obj_str = str(obj_val)
        
        if isinstance(solve_time, (int, float)):
            time_str = f"{solve_time:.3f}s"
        else:
            time_str = str(solve_time)
        
        print(f"  {status_icon} Config {n_units}: {status} | Obj: {obj_str} | Time: {time_str}")
    
    return results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run Gurobi QUBO benchmark only (for updating cached results)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python benchmark_gurobi_qubo_only.py                 # Run all configs, force re-run
  python benchmark_gurobi_qubo_only.py --no-force      # Skip cached results
  python benchmark_gurobi_qubo_only.py --configs 10 25 50  # Run specific configs only
        '''
    )
    
    parser.add_argument('--configs', type=int, nargs='+',
                       help='Specific configurations to run (overrides BENCHMARK_CONFIGS)')
    parser.add_argument('--no-force', action='store_true',
                       help='Use cached results if available (default: force re-run)')
    
    args = parser.parse_args()
    
    # Determine configurations to run
    if args.configs:
        config_values = args.configs
        print(f"Using custom configurations: {config_values}")
    else:
        config_values = BENCHMARK_CONFIGS
        print(f"Using BENCHMARK_CONFIGS: {config_values}")
    
    force_rerun = not args.no_force
    
    # Run benchmark
    try:
        results = run_gurobi_qubo_benchmark(config_values, force_rerun=force_rerun)
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"   Updated {len(results)} cached result files")
        print(f"   Location: Benchmarks/COMPREHENSIVE/Patch_GurobiQUBO/")
        
    except KeyboardInterrupt:
        print(f"\n\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
