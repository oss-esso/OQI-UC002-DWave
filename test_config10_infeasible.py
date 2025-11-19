#!/usr/bin/env python3
"""Test why config 10 is infeasible"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

from src.scenarios import load_food_data
from Utils import patch_sampler
import solver_runner_BINARY as srb

# Load foods
food_list, foods, food_groups, _ = load_food_data('full_family')
print(f"Foods: {len(foods)}")
print(f"Food groups: {len(food_groups)}")
for group, items in food_groups.items():
    print(f"  {group}: {len(items)} items")

# Generate 10 patches
n_patches = 10
seed = 42
patch_seed = seed + 50
land_unscaled = patch_sampler.generate_farms(n_farms=n_patches, seed=patch_seed)
total_land = 100.0
current_total = sum(land_unscaled.values())
scale_factor = total_land / current_total
land_availability = {k: v * scale_factor for k, v in land_unscaled.items()}

print(f"\nPatches: {n_patches}")
print(f"Total land: {sum(land_availability.values()):.2f} ha")

# Create config
food_group_config = {
    group: {'min_foods': 1, 'max_foods': len(food_list)}
    for group, food_list in food_groups.items()
}

config = {
    'parameters': {
        'land_availability': land_availability,
        'minimum_planting_area': {food: 0.0001 for food in foods},
        'food_group_constraints': food_group_config,
        'idle_penalty_lambda': 0.0,
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        },
    }
}

print(f"\nFood group constraints:")
for group, constraints in food_group_config.items():
    print(f"  {group}: min={constraints['min_foods']}, max={constraints['max_foods']}")

# Solve
patches = list(land_availability.keys())
print(f"\nSolving...")
model, results = srb.solve_with_pulp_plots(patches, foods, food_groups, config)

print(f"\nStatus: {results.get('status')}")
print(f"Objective: {results.get('objective_value')}")

# Write model to file for inspection
try:
    model.writeLP("test_config10.lp")
    print("Model written to test_config10.lp")
except Exception as e:
    print(f"Could not write model: {e}")

if results.get('status') == 'Infeasible':
    print("\n⚠️  INFEASIBLE!")
    print("Checking constraints...")
    print(f"  Min foods per group: 1")
    print(f"  Number of groups: {len(food_groups)}")
    print(f"  Min patches needed: {len(food_groups)} (one per group)")
    print(f"  Available patches: {n_patches}")
    print(f"  Should be feasible: {n_patches >= len(food_groups)}")
    
    # Try to diagnose
    print("\n  Trying to compute IIS (Irreducible Infeasible Subsystem)...")
    try:
        model.solverModel.computeIIS()
        model.solverModel.write("infeasible.ilp")
        print("  IIS written to infeasible.ilp")
    except Exception as e:
        print(f"  Could not compute IIS: {e}")
