#!/usr/bin/env python3
"""
Test the fixes to hierarchical solver and benchmark validation.

Tests:
1. Hierarchical solver now returns 'solution' key
2. Benchmark validation handles tuple format correctly
3. SA produces positive objectives

Uses SA only (no QPU) to save resources.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("TESTING FIXES TO HIERARCHICAL SOLVER")
print("="*80)

# ============================================================================
# Test 1: Hierarchical solver returns 'solution' key
# ============================================================================
print("\n[TEST 1] Checking if hierarchical solver returns 'solution' key...")

from data_loader_utils import load_food_data_as_dict
from hierarchical_quantum_solver import solve_hierarchical

data = load_food_data_as_dict('rotation_250farms_27foods')
data['farm_names'] = data['farm_names'][:5]
data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
data['total_area'] = sum(data['land_availability'].values())

result = solve_hierarchical(
    data=data,
    config={'farms_per_cluster': 3, 'num_reads': 20, 'num_iterations': 1},
    use_qpu=False,
    verbose=False
)

if 'solution' in result and result['solution'] is not None:
    print("  ✓ PASS: 'solution' key exists and is not None")
    print(f"    Solution has {len(result['solution'])} entries")
else:
    print("  ✗ FAIL: 'solution' key missing or None")
    print(f"    Keys in result: {list(result.keys())}")

# ============================================================================
# Test 2: Validate solution handles tuple format
# ============================================================================
print("\n[TEST 2] Testing validate_solution with tuple format...")

from significant_scenarios_benchmark import validate_solution

# Create mock scenario
scenario = {
    'n_farms': 5,
    'n_foods': 27,  # Hierarchical aggregates to 6
    'n_periods': 3,
}

mock_result = {'solution': result['solution']}

violations = validate_solution(mock_result, data, scenario)

print(f"  Violations returned: {violations}")
if violations['total_violations'] != 999:
    print("  ✓ PASS: validate_solution didn't return 999 (catastrophic failure marker)")
else:
    print("  ✗ FAIL: validate_solution returned 999 violations")

# ============================================================================
# Test 3: Objective is positive
# ============================================================================
print("\n[TEST 3] Checking objective value...")

obj = result.get('objective')
print(f"  Objective: {obj}")

if obj is not None and obj > 0:
    print("  ✓ PASS: Objective is positive")
else:
    print("  ✗ FAIL: Objective is not positive")

# ============================================================================
# Test 4: Full benchmark mini-run (just hierarchical scenario)
# ============================================================================
print("\n[TEST 4] Mini benchmark run (SA mode, 5 farms)...")

# This simulates what the benchmark does
test_scenario = {
    'name': 'test_hierarchical_5farms',
    'n_farms': 5,
    'n_foods': 27,
    'n_periods': 3,
    'qpu_method': 'hierarchical',
}

# Simulate solve_qpu for hierarchical
print("  Simulating QPU solve (with SA)...")

hier_config = {
    'farms_per_cluster': 3,
    'num_reads': 20,
    'num_iterations': 1,
}

qpu_result = solve_hierarchical(
    data=data,
    config=hier_config,
    use_qpu=False,  # SA mode
    verbose=False
)

# What benchmark does
result_dict = {
    'objective': qpu_result.get('objective'),
    'solution': qpu_result.get('family_solution'),
    'status': 'success' if qpu_result.get('success', True) else 'failed',
    'runtime': qpu_result.get('wall_time', 0),
}

print(f"  Result: obj={result_dict['objective']:.4f}, status={result_dict['status']}")

# Validate
violations = validate_solution(result_dict, data, test_scenario)
print(f"  Violations: {violations}")

if violations['total_violations'] != 999 and result_dict['objective'] > 0:
    print("  ✓ PASS: Mini benchmark succeeded")
else:
    print("  ✗ FAIL: Mini benchmark had issues")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

all_pass = (
    'solution' in result and result['solution'] is not None and
    violations['total_violations'] != 999 and
    obj is not None and obj > 0
)

if all_pass:
    print("✓ All tests PASSED!")
    print("\nFixes applied:")
    print("  1. hierarchical_quantum_solver.py: Added 'solution' key")
    print("  2. significant_scenarios_benchmark.py: Fixed duplicate code")
    print("  3. significant_scenarios_benchmark.py: validate_solution handles tuple format")
    print("\nNext step: Run full benchmark with these fixes (SA mode first, then QPU)")
else:
    print("✗ Some tests FAILED - review output above")

print("="*80)
