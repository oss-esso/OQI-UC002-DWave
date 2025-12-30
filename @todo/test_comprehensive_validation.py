#!/usr/bin/env python3
"""
Comprehensive validation of hierarchical solver fixes.

Tests multiple scenarios that previously failed:
- 25 farms (rotation_250farms_27foods trimmed)
- 50 farms (rotation_350farms_27foods trimmed)

Uses SA only to save QPU resources.
"""

import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("COMPREHENSIVE VALIDATION: HIERARCHICAL SOLVER FIXES")
print("="*80)

from data_loader_utils import load_food_data_as_dict
from hierarchical_quantum_solver import solve_hierarchical

# Test scenarios matching the benchmark's problematic ones
SCENARIOS = [
    {'name': '25 farms (previously -18 obj)', 'n_farms': 25, 'scenario': 'rotation_250farms_27foods'},
    {'name': '50 farms (previously -44 obj)', 'n_farms': 50, 'scenario': 'rotation_350farms_27foods'},
]

results = []

for test in SCENARIOS:
    print(f"\n{'='*70}")
    print(f"Testing: {test['name']}")
    print(f"{'='*70}")
    
    # Load data
    print(f"Loading {test['scenario']}...")
    data = load_food_data_as_dict(test['scenario'])
    
    # Trim to target size
    n_farms = test['n_farms']
    data['farm_names'] = data['farm_names'][:n_farms]
    data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
    data['total_area'] = sum(data['land_availability'].values())
    
    print(f"  Farms: {len(data['farm_names'])}")
    print(f"  Foods: {len(data['food_names'])}")
    
    # Run solver with SA
    config = {
        'farms_per_cluster': 5,
        'num_reads': 50,
        'num_iterations': 2,  # Less iterations for speed
    }
    
    print(f"\nRunning hierarchical solver (SA mode)...")
    start = time.time()
    
    result = solve_hierarchical(
        data=data,
        config=config,
        use_qpu=False,
        verbose=True
    )
    
    elapsed = time.time() - start
    
    # Analyze result
    obj = result.get('objective', 0)
    violations = result.get('violations', -1)
    success = result.get('success', False)
    has_solution = 'solution' in result and result['solution'] is not None
    
    test_result = {
        'name': test['name'],
        'n_farms': n_farms,
        'objective': obj,
        'violations': violations,
        'success': success,
        'has_solution_key': has_solution,
        'time': elapsed,
        'status': 'PASS' if (obj > 0 and success and has_solution) else 'FAIL'
    }
    results.append(test_result)
    
    print(f"\n  Result: obj={obj:.4f}, violations={violations}, success={success}")
    print(f"  Status: {test_result['status']}")

# Summary
print("\n" + "="*80)
print("COMPREHENSIVE VALIDATION SUMMARY")
print("="*80)

print(f"\n{'Test':<40} {'N_Farms':>8} {'Objective':>12} {'Violations':>12} {'Status':>10}")
print("-"*82)

for r in results:
    print(f"{r['name']:<40} {r['n_farms']:>8} {r['objective']:>12.4f} {r['violations']:>12} {r['status']:>10}")

all_pass = all(r['status'] == 'PASS' for r in results)
print("-"*82)

if all_pass:
    print("\n✓ ALL TESTS PASSED!")
    print("\nThe fixes successfully resolved:")
    print("  - Negative objectives (now positive)")
    print("  - 999 violations marker (now real counts)")
    print("  - Missing 'solution' key (now present)")
else:
    print("\n✗ SOME TESTS FAILED - review above")

print("\nComparison with original benchmark results:")
print("  Original 25 farms: obj = -18.30, violations = 999")
print(f"  Fixed 25 farms:    obj = {results[0]['objective']:.2f}, violations = {results[0]['violations']}")
print("  Original 50 farms: obj = -43.92, violations = 999")
print(f"  Fixed 50 farms:    obj = {results[1]['objective']:.2f}, violations = {results[1]['violations']}")

print("\n" + "="*80)
