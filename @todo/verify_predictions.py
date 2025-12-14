#!/usr/bin/env python3
"""
Quick verification test - just test_360 to confirm predictions
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import from comprehensive_scaling_test
from comprehensive_scaling_test import TEST_PLAN, load_data_for_test, solve_gurobi, N_PERIODS, GUROBI_TIMEOUT

print("="*80)
print("VERIFICATION TEST - test_360 only")
print("="*80)

test_variants = TEST_PLAN['test_360']

for variant_name, test_config in test_variants.items():
    formulation_name = test_config.get('formulation', variant_name)
    
    print(f"\n{'='*80}")
    print(f"Variant: {formulation_name}")
    print("="*80)
    
    # Load data
    data = load_data_for_test(test_config)
    n_vars = data['n_farms'] * data['n_foods'] * N_PERIODS
    
    print(f"Variables: {n_vars}")
    print(f"Running Gurobi (timeout={GUROBI_TIMEOUT}s)...")
    
    # Run Gurobi
    result = solve_gurobi(data, GUROBI_TIMEOUT)
    
    print(f"\nResults:")
    print(f"  Status: {result['status']}")
    print(f"  Time: {result['solve_time']:.1f}s")
    print(f"  Objective: {result['objective']:.4f}")
    print(f"  Gap: {result['gap']*100:.1f}%")
    
    # Check against prediction
    print(f"\nPrediction Check:")
    if variant_name == 'native_6':
        if result['status'] == 'timeout' and result['solve_time'] >= 299:
            print(f"  ✓ CORRECT: Native 6-Family timed out as predicted (300s)")
        else:
            print(f"  ✗ WRONG: Expected timeout, got {result['status']} in {result['solve_time']:.1f}s")
    elif variant_name == 'aggregated':
        if result['status'] == 'timeout' or result['solve_time'] < 10:
            print(f"  ? PARTIAL: Aggregated formulation completed (expected fast or timeout)")
        else:
            print(f"  Status: {result['status']} in {result['solve_time']:.1f}s")
    elif variant_name == 'hybrid_27':
        if result['status'] == 'timeout' and result['solve_time'] >= 299:
            print(f"  ✓ CORRECT: 27-Food Hybrid timed out as predicted (300s)")
        else:
            print(f"  ✗ WRONG: Expected timeout, got {result['status']} in {result['solve_time']:.1f}s")

print(f"\n{'='*80}")
print("VERIFICATION COMPLETE")
print("="*80)
