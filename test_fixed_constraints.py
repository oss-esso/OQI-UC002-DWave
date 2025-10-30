#!/usr/bin/env python3
"""Test the fixed food group constraints."""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from patch_sampler import generate_farms as generate_patches_small
from src.scenarios import load_food_data
import solver_runner_BINARY as solver_runner

print("="*80)
print("TESTING FIXED FOOD GROUP CONSTRAINTS")
print("="*80)

# Generate patch data
n_plots = 10
fixed_total_land = 100.0
seed = 42

patches_unscaled = generate_patches_small(n_farms=n_plots, seed=seed)
patches_total = sum(patches_unscaled.values())
patch_scale_factor = fixed_total_land / patches_total
patches_scaled = {k: v * patch_scale_factor for k, v in patches_unscaled.items()}

print(f"\nGenerated {n_plots} plots, {fixed_total_land:.2f} ha total\n")

# Load food data
food_list, foods, food_groups, _ = load_food_data('full_family')
print(f"Loaded {len(foods)} foods in {len(food_groups)} groups:")
for group, foods_in_group in food_groups.items():
    print(f"  - {group}: {len(foods_in_group)} foods")

# Create configuration
config = {
    'parameters': {
        'land_availability': patches_scaled,
        'minimum_planting_area': {food: 0.0001 for food in foods},
        'food_group_constraints': {
            group: {'min_foods': 1, 'max_foods': len(food_list)}
            for group, food_list in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        },
        'idle_penalty_lambda': 0.0
    }
}

# Build CQM with FIXED constraints
print(f"\nBuilding CQM with FIXED food group constraints (global, not per-plot)...")
plots_list = list(patches_scaled.keys())

cqm, Y, constraint_metadata = solver_runner.create_cqm_plots(
    plots_list, foods, food_groups, config
)

print(f"\n✓ CQM built successfully!")
print(f"  Variables: {len(cqm.variables)} (all binary)")
print(f"  Constraints: {len(cqm.constraints)}")

# Analyze constraint breakdown
print(f"\nConstraint breakdown:")
constraint_types = {}
for label in cqm.constraints.keys():
    if label.startswith('Max_Assignment_'):
        constraint_types.setdefault('plot_assignment', []).append(label)
    elif label.startswith('Min_Plots_'):
        constraint_types.setdefault('min_plots', []).append(label)
    elif label.startswith('Max_Plots_'):
        constraint_types.setdefault('max_plots', []).append(label)
    elif label.startswith('Food_Group_'):
        constraint_types.setdefault('food_group', []).append(label)
    else:
        constraint_types.setdefault('other', []).append(label)

for ctype, labels in constraint_types.items():
    print(f"  - {ctype}: {len(labels)} constraints")
    if ctype == 'food_group':
        print(f"    Labels: {labels[:5]}")

# Check that food group constraints are now global
print(f"\nVerifying food group constraints are GLOBAL:")
food_group_constraints_list = [label for label in cqm.constraints.keys() if label.startswith('Food_Group_')]
print(f"  Total food group constraints: {len(food_group_constraints_list)}")
print(f"  Expected (2 per group, 5 groups): 10")
print(f"  Before fix would have: 100 (10 per group per plot)")

if len(food_group_constraints_list) == 10:
    print(f"\n  ✅ FIXED! Food group constraints are now global (not per-plot)")
else:
    print(f"\n  ❌ Still has per-plot constraints!")

# Show a sample constraint
if food_group_constraints_list:
    sample_label = food_group_constraints_list[0]
    sample_constraint = cqm.constraints[sample_label]
    print(f"\n  Sample constraint: {sample_label}")
    print(f"    Variables in constraint: {len(sample_constraint.lhs.linear)}")
    print(f"    Expected (27 foods × 10 plots ≈ subset): many")
    
    # Count variables
    n_vars = len(sample_constraint.lhs.linear)
    print(f"    Actual: {n_vars} variables")
    if n_vars > 50:
        print(f"    ✅ Global constraint (spans multiple plots)")
    else:
        print(f"    ❌ Per-plot constraint (too few variables)")

print(f"\n{'='*80}")
print(f"CONCLUSION:")
print(f"{'='*80}")

if len(food_group_constraints_list) == 10:
    print(f"✅ Fix successful! Food group constraints are now global.")
    print(f"   The problem should now be FEASIBLE.")
    print(f"   DWave CQM should find feasible solutions without violations.")
else:
    print(f"❌ Fix incomplete. Need to verify constraint implementation.")

