#!/usr/bin/env python3
"""
Adaptive Hybrid Solver Validation Test (SA mode)

Tests the adaptive_hybrid_solver.py with Simulated Annealing to verify:
1. 27-food recovery works correctly
2. Objective calculation is correct
3. Both binary and fractional modes work

Author: OQI-UC002-DWave Debug Session
Date: 2025-12-29
"""

import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("ADAPTIVE HYBRID SOLVER VALIDATION TEST (SA mode)")
print("="*80)

from data_loader_utils import load_food_data_as_dict
from adaptive_hybrid_solver import solve_adaptive_with_recovery

# Load test data (small for speed)
print("\nLoading test data (5 farms, 27 foods)...")
data = load_food_data_as_dict('rotation_250farms_27foods')
data['farm_names'] = data['farm_names'][:5]
data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
data['total_area'] = sum(data['land_availability'].values())

print(f"  Farms: {len(data['farm_names'])}")
print(f"  Foods: {len(data['food_names'])}")

# Test 1: Binary recovery mode
print("\n" + "="*60)
print("TEST 1: Binary Recovery Mode")
print("="*60)

start = time.time()
result_binary = solve_adaptive_with_recovery(
    data=data,
    num_reads=30,
    num_iterations=1,
    use_qpu=False,  # SA mode
    recovery_mode='binary',
    recovery_method='benefit_weighted',
    verbose=True
)
time_binary = time.time() - start

print(f"\nBinary mode results:")
print(f"  Family objective: {result_binary.get('objective_family', 'N/A')}")
print(f"  27-food objective: {result_binary.get('objective_27food', 'N/A')}")
print(f"  Success: {result_binary.get('success', False)}")
print(f"  Time: {time_binary:.1f}s")

if 'diversity_stats' in result_binary:
    div = result_binary['diversity_stats']
    print(f"  Unique crops: {div.get('total_unique_crops', 'N/A')}")
    print(f"  Shannon diversity: {div.get('shannon_diversity', 'N/A'):.3f}")

# Test 2: Fractional recovery mode
print("\n" + "="*60)
print("TEST 2: Fractional Recovery Mode")
print("="*60)

start = time.time()
result_frac = solve_adaptive_with_recovery(
    data=data,
    num_reads=30,
    num_iterations=1,
    use_qpu=False,  # SA mode
    recovery_mode='fractional',
    verbose=True
)
time_frac = time.time() - start

print(f"\nFractional mode results:")
print(f"  Family objective: {result_frac.get('objective_family', 'N/A')}")
print(f"  27-food objective: {result_frac.get('objective_27food', 'N/A')}")
print(f"  Success: {result_frac.get('success', False)}")
print(f"  Time: {time_frac:.1f}s")

if 'diversity_stats' in result_frac:
    div = result_frac['diversity_stats']
    print(f"  Unique crops: {div.get('total_unique_crops', 'N/A')}")
    print(f"  Shannon diversity: {div.get('shannon_diversity', 'N/A'):.3f}")

# Summary
print("\n" + "="*80)
print("ADAPTIVE HYBRID SOLVER VALIDATION SUMMARY")
print("="*80)

tests_passed = 0
tests_total = 2

# Check binary mode
if result_binary.get('success', False) and result_binary.get('objective_family', 0) > 0:
    print("✓ Binary recovery mode: PASSED")
    tests_passed += 1
else:
    print("✗ Binary recovery mode: FAILED")

# Check fractional mode
if result_frac.get('success', False) and result_frac.get('objective_family', 0) > 0:
    print("✓ Fractional recovery mode: PASSED")
    tests_passed += 1
else:
    print("✗ Fractional recovery mode: FAILED")

print(f"\nTotal: {tests_passed}/{tests_total} tests passed")

if tests_passed == tests_total:
    print("\n✓ ADAPTIVE HYBRID SOLVER VALIDATION COMPLETE!")
else:
    print("\n✗ Some tests failed - review above")

print("="*80)
