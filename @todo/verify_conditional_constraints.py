#!/usr/bin/env python3
"""
Verify that conditional min/max plot constraints work correctly.

This script verifies:
1. If a crop is NOT planted (sum of Y=0), no constraints violated
2. If a crop IS planted (sum of Y>0), it respects min_plots <= sum(Y) <= max_plots
"""

import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenarios import load_food_data
from Utils.patch_sampler import generate_grid
from solver_runner_CUSTOM_HYBRID import solve_with_pulp_plots

def verify_conditional_constraints():
    """Verify the conditional constraints work correctly."""
    
    # Generate small test case
    n_patches = 25
    total_land = 100.0
    patches = generate_grid(n_farms=n_patches, area=total_land, seed=42)
    patches_list = list(patches.keys())
    
    # Load food data
    _, foods, food_groups, base_config = load_food_data('full_family')
    
    # Create config
    params = base_config['parameters']
    max_percentage = params.get('max_percentage_per_crop', {})
    min_planting_area = params.get('minimum_planting_area', {})
    maximum_planting_area = {crop: max_pct * total_land for crop, max_pct in max_percentage.items()}
    food_group_constraints = params.get('food_group_constraints', {})
    weights = params.get('weights', {})
    
    config = {
        'parameters': {
            'land_availability': patches,
            'minimum_planting_area': min_planting_area,
            'maximum_planting_area': maximum_planting_area,
            'food_group_constraints': food_group_constraints,
            'weights': weights,
        }
    }
    
    plot_area = list(patches.values())[0]
    
    print("="*80)
    print("VERIFYING CONDITIONAL MIN/MAX CONSTRAINTS")
    print("="*80)
    print(f"Number of patches: {n_patches}")
    print(f"Plot area: {plot_area:.4f} ha")
    print(f"Number of crops: {len(foods)}")
    
    # Solve
    print("\nSolving...")
    model, results = solve_with_pulp_plots(patches_list, foods, food_groups, config)
    
    print(f"Status: {results['status']}")
    print(f"Objective: {results['objective_value']:.6f}")
    
    # Analyze solution
    print("\n" + "="*80)
    print("CONSTRAINT VERIFICATION")
    print("="*80)
    
    crop_assignments = {}
    for crop in foods:
        total = sum(results['plantations'].get(f"{patch}_{crop}", 0) for patch in patches_list)
        crop_assignments[crop] = total
    
    # Check conditional constraints
    violations = []
    planted_crops = []
    unplanted_crops = []
    
    for crop in foods:
        total_plots = crop_assignments[crop]
        
        if total_plots > 0:
            planted_crops.append(crop)
            
            # Check minimum
            if crop in min_planting_area and min_planting_area[crop] > 0:
                min_plots = math.ceil(min_planting_area[crop] / plot_area)
                if total_plots < min_plots:
                    violations.append(f"{crop}: {total_plots} plots < min {min_plots} plots")
            
            # Check maximum
            if crop in maximum_planting_area:
                max_plots = math.floor(maximum_planting_area[crop] / plot_area)
                if total_plots > max_plots:
                    violations.append(f"{crop}: {total_plots} plots > max {max_plots} plots")
        else:
            unplanted_crops.append(crop)
    
    print(f"\nPlanted crops: {len(planted_crops)}")
    for crop in planted_crops[:10]:  # Show first 10
        total_plots = crop_assignments[crop]
        min_plots = math.ceil(min_planting_area.get(crop, 0) / plot_area) if crop in min_planting_area else 0
        max_plots = math.floor(maximum_planting_area.get(crop, float('inf')) / plot_area) if crop in maximum_planting_area else 'inf'
        print(f"  {crop}: {int(total_plots)} plots (min={min_plots}, max={max_plots})")
    
    if len(planted_crops) > 10:
        print(f"  ... and {len(planted_crops) - 10} more")
    
    print(f"\nUnplanted crops: {len(unplanted_crops)}")
    if unplanted_crops:
        print(f"  {', '.join(unplanted_crops[:10])}")
        if len(unplanted_crops) > 10:
            print(f"  ... and {len(unplanted_crops) - 10} more")
    
    print("\n" + "="*80)
    if violations:
        print("❌ VIOLATIONS FOUND:")
        for v in violations:
            print(f"  {v}")
    else:
        print("✅ ALL CONSTRAINTS SATISFIED!")
        print("\nKey findings:")
        print(f"  - Unplanted crops correctly have 0 plots (no min constraint violation)")
        print(f"  - Planted crops respect min/max bounds")
        print(f"  - Conditional constraints working as expected")
    print("="*80)


if __name__ == "__main__":
    verify_conditional_constraints()
