#!/usr/bin/env python3
"""
Quick validation test - fewer iterations for faster results.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("QUICK VALIDATION TEST")
print("="*80)

from data_loader_utils import load_food_data_as_dict
from hierarchical_quantum_solver import solve_hierarchical

# Test 25 farms (the one that showed -18 objective)
print("\nLoading rotation_250farms_27foods...")
data = load_food_data_as_dict('rotation_250farms_27foods')

n_farms = 25
data['farm_names'] = data['farm_names'][:n_farms]
data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
data['total_area'] = sum(data['land_availability'].values())

print(f"Testing with {n_farms} farms, {len(data['food_names'])} foods")

# Use minimal config for speed
config = {
    'farms_per_cluster': 5,
    'num_reads': 10,  # Very few reads for speed
    'num_iterations': 1,  # Single iteration
}

print("\nRunning hierarchical solver (SA mode, minimal config)...")
result = solve_hierarchical(
    data=data,
    config=config,
    use_qpu=False,
    verbose=True
)

print("\n" + "="*80)
print("QUICK VALIDATION RESULTS")
print("="*80)

obj = result.get('objective', 0)
violations = result.get('violations', -1)
has_solution = 'solution' in result and result['solution'] is not None

print(f"\nObjective: {obj}")
print(f"Violations: {violations}")
print(f"Has 'solution' key: {has_solution}")

if obj > 0 and has_solution:
    print("\n✓ VALIDATION PASSED!")
    print(f"\nOriginal benchmark: obj = -18.30, violations = 999")
    print(f"After fix:          obj = {obj:.2f}, violations = {violations}")
else:
    print("\n✗ VALIDATION FAILED")

print("="*80)
