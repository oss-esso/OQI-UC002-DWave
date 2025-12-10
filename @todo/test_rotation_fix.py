#!/usr/bin/env python3
"""
Quick test of rotation formulation in qpu_benchmark.py
"""
import sys
sys.path.insert(0, '..')

from qpu_benchmark import load_problem_data_from_scenario, solve_ground_truth_rotation

print("="*80)
print("Testing Rotation Formulation in qpu_benchmark.py")
print("="*80)

# Load rotation scenario
print("\n[1] Loading rotation_micro_25...")
data = load_problem_data_from_scenario('rotation_micro_25')

print(f"  Farms: {data['n_farms']}")
print(f"  Crop families: {data['n_foods']}")
print(f"  Config present: {'config' in data}")
print(f"  Scenario name: {data.get('scenario_name')}")

# Extract rotation parameters
config = data.get('config', {})
params = config.get('parameters', {})
print(f"\n[2] Rotation parameters:")
print(f"  rotation_gamma: {params.get('rotation_gamma')}")
print(f"  frustration_ratio: {params.get('frustration_ratio')}")
print(f"  one_hot_penalty: {params.get('one_hot_penalty')}")
print(f"  diversity_bonus: {params.get('diversity_bonus')}")

# Solve with Gurobi
print(f"\n[3] Solving with Gurobi rotation formulation...")
result = solve_ground_truth_rotation(data, timeout=10)

if result['success']:
    print(f"  Status: {result['status']}")
    print(f"  Objective: {result['objective']:.6f}")
    print(f"  Variables: {result['n_variables']}")
    print(f"  Constraints: {result['n_constraints']}")
    print(f"  Solve time: {result['solve_time']:.3f}s")
    print(f"  Violations: {result['violations']}")
    
    # Expected: ~90 variables (5 farms × 6 families × 3 periods)
    expected_vars = data['n_farms'] * data['n_foods'] * 3
    actual_vars = result['n_variables']
    
    print(f"\n[4] Verification:")
    print(f"  Expected variables: {expected_vars}")
    print(f"  Actual variables: {actual_vars}")
    
    if actual_vars == expected_vars:
        print(f"  ✓ CORRECT - Using 3-period rotation formulation!")
    else:
        print(f"  ✗ WRONG - Not using rotation formulation")
    
    # Check objective value against benchmark_rotation_gurobi.py result
    # Expected: ~4.08 (from benchmark_rotation_gurobi.py output)
    print(f"\n[5] Objective comparison:")
    print(f"  Current: {result['objective']:.4f}")
    print(f"  Expected (from benchmark_rotation_gurobi.py): ~4.08")
    print(f"  Old wrong value (single-period): 0.64")
    
    if result['objective'] > 3.0:
        print(f"  ✓ LOOKS CORRECT - Matches rotation benchmark!")
    else:
        print(f"  ✗ WRONG - Matches old single-period formulation")
else:
    print(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")

print("\n" + "="*80)
print("Test complete!")
print("="*80)
