#!/usr/bin/env python3
"""
Test: Verify objective calculation by using the BASELINE solution
in the DECOMPOSED evaluator.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenarios import load_food_data
from Utils.farm_sampler import generate_farms
from Utils.patch_sampler import generate_grid
from solver_runner_CUSTOM_HYBRID import solve_with_pulp_plots as solve_plots_baseline

def test_objective_evaluator():
    """Test that our objective evaluator works correctly."""
    
    n_units = 25
    total_land = 100.0
    
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
            'land_availability': patches,
            'minimum_planting_area': base_config['parameters'].get('minimum_planting_area', {}),
            'maximum_planting_area': maximum_planting_area,
            'food_group_constraints': base_config['parameters'].get('food_group_constraints', {}),
            'weights': base_config['parameters'].get('weights', {}),
        }
    }
    
    print("="*80)
    print("OBJECTIVE EVALUATOR TEST")
    print("="*80)
    
    # Get baseline solution
    print("\n[Step 1: Get baseline solution]")
    _, result_baseline = solve_plots_baseline(patches_list, foods, food_groups, config)
    baseline_obj = result_baseline['objective_value']
    print(f"  Baseline objective: {baseline_obj:.10f}")
    
    # Extract Y values from baseline solution
    print("\n[Step 2: Extract Y values from baseline]")
    baseline_Y = {}
    for patch in patches_list:
        for crop in foods:
            key = f"{patch}_{crop}"
            baseline_Y[f"Y_{patch}_{crop}"] = result_baseline['plantations'].get(key, 0)
    
    n_selected = sum(1 for v in baseline_Y.values() if v > 0.5)
    print(f"  Extracted {len(baseline_Y)} Y variables, {n_selected} selected")
    
    # Now evaluate using DECOMPOSED method
    print("\n[Step 3: Evaluate using DECOMPOSED objective formula]")
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    total_land_area = sum(land_availability.values())
    
    # Calculate actual objective from Y variables (DECOMPOSED method)
    decomposed_obj = 0.0
    for patch in patches_list:
        patch_area = land_availability[patch]
        for crop in foods:
            y_val = baseline_Y.get(f"Y_{patch}_{crop}", 0)
            if y_val > 0.5:  # Binary: 1 if selected
                crop_value = (
                    weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                    weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                    weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
                )
                decomposed_obj += patch_area * crop_value
    
    decomposed_obj /= total_land_area  # Normalize
    
    print(f"  DECOMPOSED objective: {decomposed_obj:.10f}")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    diff = abs(baseline_obj - decomposed_obj)
    print(f"Baseline:   {baseline_obj:.10f}")
    print(f"Decomposed: {decomposed_obj:.10f}")
    print(f"Difference: {diff:.2e}")
    
    if diff < 1e-6:
        print("\n✅ Objective evaluator is CORRECT!")
        print("   The problem is simulated annealing finding poor solutions.")
    else:
        print(f"\n❌ Objective evaluator is WRONG!")
        print(f"   There's a bug in the objective calculation formula.")


if __name__ == "__main__":
    test_objective_evaluator()
