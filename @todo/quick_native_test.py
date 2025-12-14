#!/usr/bin/env python3
"""
Quick test of Native 6-Family with hard_scenarios
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from comprehensive_scaling_test import TEST_PLAN, load_data_for_test, solve_gurobi, GUROBI_TIMEOUT, N_PERIODS

print("="*80)
print("QUICK TEST: Native 6-Family with hard_scenarios.py")
print("="*80)

for test_name in ['test_360', 'test_900', 'test_1620', 'test_4050']:
    test_config = TEST_PLAN[test_name]['native_6']
    n_farms = test_config['n_farms']
    n_vars = n_farms * 6 * N_PERIODS
    
    print(f"\n{test_name}: {n_farms} farms = {n_vars} variables")
    print("-"*80)
    
    data = load_data_for_test(test_config)
    
    print(f"  Running Gurobi (timeout={GUROBI_TIMEOUT}s)...")
    result = solve_gurobi(data, GUROBI_TIMEOUT)
    
    print(f"  Status: {result['status']}")
    print(f"  Time: {result['solve_time']:.1f}s")
    
    if result['status'] == 'timeout':
        print(f"  >>> TIMEOUT ✓")
    else:
        print(f"  >>> SOLVED (unexpected)")

print(f"\n{'='*80}")
print("If all timeout, we have consistent hardness!")
print("="*80)
