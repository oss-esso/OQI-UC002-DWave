#!/usr/bin/env python3
"""
Quick test: Run smallest scenario (rotation_micro_25) with updated Gurobi test
to verify solution extraction integrity.

This tests:
1. Solution extraction (binary variables)
2. Area allocation computation
3. Constraint validation
4. JSON structure matches hierarchical output
"""

import os
import sys
import json
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("SINGLE SCENARIO TEST: Verify Solution Extraction")
print("="*80)
print()

# Import test functions
from test_gurobi_timeout import (
    solve_gurobi_test,
    load_scenario_data,
    GUROBI_CONFIG
)

# Test smallest scenario
scenario = {
    'name': 'rotation_micro_25',
    'n_farms': 5,
    'n_foods': 6,
    'n_periods': 3,
    'n_vars': 90
}

print(f"Testing scenario: {scenario['name']}")
print(f"Size: {scenario['n_farms']} farms × {scenario['n_foods']} foods = {scenario['n_vars']} vars")
print()

# Load data
print("Loading data...")
data = load_scenario_data(scenario)
print(f"✓ Loaded: {len(data['farm_names'])} farms, {len(data['food_names'])} foods")
print()

# Solve
print("Solving with Gurobi...")
result = solve_gurobi_test(data, scenario, GUROBI_CONFIG)
print()

# Verify structure
print("="*80)
print("VERIFICATION RESULTS")
print("="*80)
print()

# Check metadata
print("1. Metadata structure:")
if 'metadata' in result:
    print("   ✓ metadata present")
    required_metadata = ['benchmark_type', 'solver', 'scenario', 'n_farms', 'n_foods', 'timestamp']
    for key in required_metadata:
        if key in result['metadata']:
            print(f"   ✓ metadata.{key}: {result['metadata'][key]}")
        else:
            print(f"   ✗ metadata.{key}: MISSING")
else:
    print("   ✗ metadata MISSING")
print()

# Check result structure
print("2. Result structure:")
if 'result' in result:
    print("   ✓ result present")
    required_result = ['status', 'objective_value', 'solve_time', 'solution_selections', 
                       'solution_areas', 'validation', 'success']
    for key in required_result:
        if key in result['result']:
            if key in ['solution_selections', 'solution_areas']:
                count = len(result['result'][key])
                print(f"   ✓ result.{key}: {count} variables")
            elif key == 'validation':
                print(f"   ✓ result.{key}: {result['result'][key].get('summary', 'N/A')}")
            else:
                print(f"   ✓ result.{key}: {result['result'][key]}")
        else:
            print(f"   ✗ result.{key}: MISSING")
else:
    print("   ✗ result MISSING")
print()

# Check solution variables
print("3. Solution variables:")
if 'result' in result and 'solution_selections' in result['result']:
    n_selections = len(result['result']['solution_selections'])
    n_areas = len(result['result']['solution_areas'])
    expected = scenario['n_farms'] * scenario['n_foods'] * scenario['n_periods']
    
    print(f"   Expected variables: {expected}")
    print(f"   Binary selections:  {n_selections} {'✓' if n_selections == expected else '✗'}")
    print(f"   Area allocations:   {n_areas} {'✓' if n_areas == expected else '✗'}")
    
    # Show a few examples
    print(f"\n   Sample binary variables:")
    for i, (var, val) in enumerate(list(result['result']['solution_selections'].items())[:5]):
        area = result['result']['solution_areas'][var]
        print(f"      {var}: binary={val}, area={area:.4f}")
    
    # Check binary values are 0 or 1
    non_binary = [var for var, val in result['result']['solution_selections'].items() 
                  if val not in [0.0, 1.0]]
    if non_binary:
        print(f"\n   ⚠️  Warning: {len(non_binary)} non-binary values found")
        for var in non_binary[:3]:
            print(f"      {var}: {result['result']['solution_selections'][var]}")
    else:
        print(f"\n   ✓ All variables are binary (0 or 1)")
else:
    print("   ✗ No solution variables found")
print()

# Check validation
print("4. Constraint validation:")
if 'result' in result and 'validation' in result['result']:
    val = result['result']['validation']
    print(f"   Valid: {val.get('is_valid', 'N/A')}")
    print(f"   Violations: {val.get('n_violations', 'N/A')}")
    if val.get('violations'):
        print(f"\n   First violations:")
        for v in val['violations'][:3]:
            print(f"      {v}")
else:
    print("   ✗ No validation found")
print()

# Save result
output_file = Path(__file__).parent / 'test_single_scenario_output.json'
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f"✓ Full result saved to: {output_file}")
print()

# Compare with hierarchical format
print("5. Format comparison with hierarchical solver:")
hierarchical_file = project_root / 'Phase3Report' / 'Data' / 'hierarchical_30_farms.json'
if hierarchical_file.exists():
    with open(hierarchical_file) as f:
        hierarchical = json.load(f)
    
    print(f"   Hierarchical structure:")
    print(f"      Keys: {list(hierarchical.keys())}")
    print(f"   Gurobi structure:")
    print(f"      Keys: {list(result.keys())}")
    
    if set(hierarchical.keys()) == set(result.keys()):
        print(f"   ✓ Top-level keys match!")
    else:
        print(f"   ⚠️  Top-level keys differ")
else:
    print(f"   ⚠️  Hierarchical reference file not found")

print()
print("="*80)
print("TEST COMPLETE")
print("="*80)
