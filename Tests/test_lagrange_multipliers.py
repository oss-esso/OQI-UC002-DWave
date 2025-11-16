#!/usr/bin/env python3
"""
Test different Lagrange multipliers to find the sweet spot between:
- Constraint satisfaction (no violations)
- Objective optimization (high objective value)
- Land utilization (use all patches)
"""

import json
import sys
import os
# Add project root and Benchmark Scripts to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

from Utils.patch_sampler import generate_farms
from src.scenarios import load_food_data
import solver_runner_BINARY as solver_binary
from dimod import cqm_to_bqm
import time

# Test with 10 patches (faster)
print("Generating 10-patch sample...")
patches = generate_farms(n_farms=10, seed=42)
total_area = sum(patches.values())

# Load food data
food_list, foods, food_groups, _ = load_food_data('full_family')

# Create config
config = {
    'parameters': {
        'land_availability': patches,
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

# Create CQM
print("Creating CQM...")
cqm, Y, constraint_metadata = solver_binary.create_cqm_plots(patches, foods, food_groups, config)

# Test different Lagrange multipliers
multipliers = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0]

print(f"\n{'='*80}")
print("TESTING DIFFERENT LAGRANGE MULTIPLIERS")
print(f"{'='*80}")
print(f"\n{'Lambda':>8} | {'Violations':>10} | {'Objective':>10} | {'Utilization':>12} | {'Crops':>6}")
print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*6}")

for lagrange in multipliers:
    print(f"{lagrange:8.1f} | ", end='', flush=True)
    
    try:
        # Convert to BQM
        bqm, invert = cqm_to_bqm(cqm, lagrange_multiplier=lagrange)
        
        # Solve with Gurobi QUBO (shorter time limit for testing)
        result = solver_binary.solve_with_gurobi_qubo(
            bqm,
            farms=list(patches.keys()),
            foods=foods,
            food_groups=food_groups,
            land_availability=patches,
            weights=config['parameters']['weights'],
            idle_penalty=config['parameters'].get('idle_penalty_lambda', 0.0),
            config=config,
            time_limit=30  # 30 seconds for quick test
        )
        
        violations = result['validation']['n_violations']
        objective = result['objective_value']
        utilization = result['solution_summary']['utilization'] * 100
        n_crops = result['solution_summary']['n_crops']
        
        print(f"{violations:10d} | {objective:10.4f} | {utilization:11.1f}% | {n_crops:6d}")
        
    except Exception as e:
        print(f"ERROR: {str(e)[:40]}")

print(f"\n{'='*80}")
print("RECOMMENDATION:")
print("Choose the SMALLEST multiplier that gives 0 violations")
print(f"{'='*80}")
