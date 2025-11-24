#!/usr/bin/env python3
"""
Quick test to verify objective calculations are consistent.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenarios import load_food_data
from Utils.farm_sampler import generate_farms
from Utils.patch_sampler import generate_grid
from solver_runner_DECOMPOSED import solve_with_pulp_farm, solve_with_pulp_plots
from solver_runner_CUSTOM_HYBRID import solve_with_pulp_farm as solve_farm_baseline
from solver_runner_CUSTOM_HYBRID import solve_with_pulp_plots as solve_plots_baseline

def test_objective_consistency():
    """Test that different solvers see the same objectives."""
    
    n_units = 25
    total_land = 100.0
    
    # Generate farm data
    farms_unscaled = generate_farms(n_farms=n_units, seed=42)
    total = sum(farms_unscaled.values())
    scale = total_land / total
    farms_scaled = {k: v * scale for k, v in farms_unscaled.items()}
    farms_list = list(farms_scaled.keys())
    
    # Generate patch data
    patches = generate_grid(n_farms=n_units, area=total_land, seed=42)
    patches_list = list(patches.keys())
    
    # Load food data
    _, foods, food_groups, base_config = load_food_data('full_family')
    
    # Create config
    max_percentage = base_config['parameters'].get('max_percentage_per_crop', {})
    maximum_planting_area = {crop: max_pct * total_land for crop, max_pct in max_percentage.items()}
    
    config = {
        'parameters': {
            'land_availability': farms_scaled,
            'minimum_planting_area': base_config['parameters'].get('minimum_planting_area', {}),
            'maximum_planting_area': maximum_planting_area,
            'food_group_constraints': base_config['parameters'].get('food_group_constraints', {}),
            'weights': base_config['parameters'].get('weights', {}),
        }
    }
    
    config_patch = {
        'parameters': {
            'land_availability': patches,
            'minimum_planting_area': base_config['parameters'].get('minimum_planting_area', {}),
            'maximum_planting_area': maximum_planting_area,
            'food_group_constraints': base_config['parameters'].get('food_group_constraints', {}),
            'weights': base_config['parameters'].get('weights', {}),
        }
    }
    
    print("="*80)
    print("OBJECTIVE CONSISTENCY TEST")
    print("="*80)
    
    # Test FARM scenario
    print("\n[FARM SCENARIO - 25 farms]")
    
    print("  Baseline (CUSTOM_HYBRID)...", end=" ", flush=True)
    _, result_baseline_farm = solve_farm_baseline(farms_list, foods, food_groups, config)
    print(f"✓ Obj: {result_baseline_farm['objective_value']:.10f}")
    
    print("  Decomposed...", end=" ", flush=True)
    _, result_decomposed_farm = solve_with_pulp_farm(farms_list, foods, food_groups, config)
    print(f"✓ Obj: {result_decomposed_farm['objective_value']:.10f}")
    
    diff_farm = abs(result_baseline_farm['objective_value'] - result_decomposed_farm['objective_value'])
    print(f"\n  Difference: {diff_farm:.2e}")
    
    if diff_farm < 1e-6:
        print("  ✅ FARM objectives MATCH")
    else:
        print(f"  ❌ FARM objectives DIFFER by {diff_farm:.10f}")
    
    # Test PATCH scenario
    print("\n[PATCH SCENARIO - 25 patches]")
    
    print("  Baseline (CUSTOM_HYBRID)...", end=" ", flush=True)
    _, result_baseline_patch = solve_plots_baseline(patches_list, foods, food_groups, config_patch)
    print(f"✓ Obj: {result_baseline_patch['objective_value']:.10f}")
    
    print("  Decomposed...", end=" ", flush=True)
    _, result_decomposed_patch = solve_with_pulp_plots(patches_list, foods, food_groups, config_patch)
    print(f"✓ Obj: {result_decomposed_patch['objective_value']:.10f}")
    
    diff_patch = abs(result_baseline_patch['objective_value'] - result_decomposed_patch['objective_value'])
    print(f"\n  Difference: {diff_patch:.2e}")
    
    if diff_patch < 1e-6:
        print("  ✅ PATCH objectives MATCH")
    else:
        print(f"  ❌ PATCH objectives DIFFER by {diff_patch:.10f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"FARM:  Baseline={result_baseline_farm['objective_value']:.6f}, Decomposed={result_decomposed_farm['objective_value']:.6f}")
    print(f"PATCH: Baseline={result_baseline_patch['objective_value']:.6f}, Decomposed={result_decomposed_patch['objective_value']:.6f}")
    
    if diff_farm < 1e-6 and diff_patch < 1e-6:
        print("\n✅ All solvers are consistent!")
    else:
        print("\n❌ Solvers have inconsistent objectives")


if __name__ == "__main__":
    test_objective_consistency()
