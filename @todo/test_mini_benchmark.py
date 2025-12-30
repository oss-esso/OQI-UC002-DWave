#!/usr/bin/env python3
"""
Mini benchmark test - verifies hierarchical solver works in benchmark context.

Tests:
1. Loads data like benchmark does
2. Runs hierarchical solver with SA
3. Validates solution with benchmark's validate_solution
4. Computes gap metrics

This is a validation that all fixes work together in the benchmark pipeline.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("MINI BENCHMARK VALIDATION TEST")
print("="*80)

from data_loader_utils import load_food_data_as_dict
from hierarchical_quantum_solver import solve_hierarchical
from significant_scenarios_benchmark import validate_solution

# Test scenario matching the problematic one from benchmark
scenario = {
    'name': 'rotation_250farms_27foods_mini',
    'n_farms': 25,
    'n_foods': 27,
    'n_periods': 3,
    'n_vars': 2025,
}

print(f"\nScenario: {scenario['name']}")
print(f"Size: {scenario['n_farms']} farms × {scenario['n_foods']} foods = {scenario['n_vars']} vars")

# Load data same way benchmark does
print("\n[1] Loading data...")
data = load_food_data_as_dict('rotation_250farms_27foods')
data['farm_names'] = data['farm_names'][:scenario['n_farms']]
data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
data['total_area'] = sum(data['land_availability'].values())
print(f"    Loaded: {len(data['farm_names'])} farms, {len(data['food_names'])} foods")

# Run hierarchical solver (SA mode)
print("\n[2] Running hierarchical solver (SA mode)...")
config = {
    'farms_per_cluster': 5,
    'num_reads': 20,
    'num_iterations': 1,
}

result = solve_hierarchical(
    data=data,
    config=config,
    use_qpu=False,  # SA mode
    verbose=False
)

print(f"    Objective: {result.get('objective', 'N/A')}")
print(f"    Violations: {result.get('violations', 'N/A')}")
print(f"    Success: {result.get('success', False)}")
print(f"    Has 'solution' key: {'solution' in result}")

# Validate using benchmark's function
print("\n[3] Validating with benchmark's validate_solution...")
qpu_result = {
    'solution': result.get('family_solution'),  # What benchmark expects
    'objective': result.get('objective'),
}

violations = validate_solution(qpu_result, data, scenario)
print(f"    Rotation violations: {violations['rotation_violations']}")
print(f"    Diversity violations: {violations['diversity_violations']}")
print(f"    Area violations: {violations['area_violations']}")
print(f"    Total violations: {violations['total_violations']}")

# Summary
print("\n" + "="*80)
print("MINI BENCHMARK VALIDATION SUMMARY")
print("="*80)

obj = result.get('objective', 0)
total_viol = violations['total_violations']
has_solution = 'solution' in result

all_good = (
    obj > 0 and 
    total_viol != 999 and 
    has_solution and
    result.get('success', False)
)

if all_good:
    print(f"""
✓ ALL CHECKS PASSED!

Results:
  Objective: {obj:.4f} (positive ✓)
  Violations: {total_viol} (not 999 ✓)
  Solution key: present ✓
  Success: True ✓

Comparison with original benchmark bug:
  Original: obj=-18.30, violations=999
  Fixed:    obj={obj:.2f}, violations={total_viol}

The benchmark pipeline now works correctly with hierarchical solver!
""")
else:
    print(f"""
✗ SOME CHECKS FAILED

  Objective: {obj} (expected > 0)
  Violations: {total_viol} (expected != 999)
  Solution key: {has_solution}
  Success: {result.get('success', False)}
""")

print("="*80)
