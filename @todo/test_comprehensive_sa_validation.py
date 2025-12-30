#!/usr/bin/env python3
"""
Comprehensive Scaling Test with Simulated Annealing (SA mode)

This is a VALIDATION test that runs with SA instead of QPU to:
1. Verify fixes work across all problem scales
2. Save QPU resources while debugging
3. Get baseline timings for comparison

Tests scenarios from small (15 farms) to large (100 farms)
Uses fewer iterations for speed.

Author: OQI-UC002-DWave Debug Session
Date: 2025-12-29
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader_utils import load_food_data_as_dict

print("="*80)
print("COMPREHENSIVE SA VALIDATION TEST")
print("="*80)
print()

# Test scenarios (subset for validation - faster)
SCENARIOS = [
    {'name': 'rotation_15farms_6foods', 'n_farms': 15, 'n_foods': 6, 'n_vars': 270},
    {'name': 'rotation_25farms_6foods', 'n_farms': 25, 'n_foods': 6, 'n_vars': 450},
    {'name': 'rotation_25farms_27foods', 'n_farms': 25, 'n_foods': 27, 'n_vars': 2025},
    {'name': 'rotation_50farms_27foods', 'n_farms': 50, 'n_foods': 27, 'n_vars': 4050},
]

OUTPUT_DIR = Path(__file__).parent / 'sa_validation_results'
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Testing {len(SCENARIOS)} scenarios with SA (no QPU)")
print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_scenario_data(scenario):
    """Load data for scenario."""
    n_farms = scenario['n_farms']
    n_foods = scenario['n_foods']
    
    if n_foods == 6:
        if n_farms <= 20:
            scenario_name = 'rotation_medium_100'
        else:
            scenario_name = 'rotation_large_200'
    else:
        scenario_name = 'rotation_250farms_27foods'
    
    data = load_food_data_as_dict(scenario_name)
    
    # Adjust farm count
    if len(data['farm_names']) > n_farms:
        data['farm_names'] = data['farm_names'][:n_farms]
        data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
    elif len(data['farm_names']) < n_farms:
        original_farms = data['farm_names'].copy()
        while len(data['farm_names']) < n_farms:
            idx = len(data['farm_names']) - len(original_farms)
            farm = original_farms[idx % len(original_farms)]
            new_farm = f"{farm}_dup{idx}"
            data['farm_names'].append(new_farm)
            data['land_availability'][new_farm] = data['land_availability'][farm]
    
    # Adjust food count
    if len(data['food_names']) > n_foods:
        data['food_names'] = data['food_names'][:n_foods]
        data['food_benefits'] = {f: data['food_benefits'][f] for f in data['food_names'] if f in data['food_benefits']}
    
    data['total_area'] = sum(data['land_availability'].values())
    
    return data

# ============================================================================
# SA SOLVER
# ============================================================================

from hierarchical_quantum_solver import solve_hierarchical

def solve_sa_hierarchical(data, scenario):
    """Solve using hierarchical solver with SA (not QPU)."""
    config = {
        'farms_per_cluster': min(10, max(3, scenario['n_farms'] // 5)),
        'num_reads': 50,  # Reduced for speed
        'num_iterations': 2,  # Reduced for speed
    }
    
    start = time.time()
    result = solve_hierarchical(
        data=data,
        config=config,
        use_qpu=False,  # SA mode
        verbose=False
    )
    elapsed = time.time() - start
    
    return {
        'method': 'hierarchical_sa',
        'objective': result.get('objective', 0),
        'violations': result.get('violations', -1),
        'solve_time': elapsed,
        'success': result.get('success', False),
        'has_solution': 'solution' in result and result['solution'] is not None,
    }

# ============================================================================
# RUN TESTS
# ============================================================================

print("="*80)
print("RUNNING SA VALIDATION TESTS")
print("="*80)

results = []

for idx, scenario in enumerate(SCENARIOS, 1):
    print(f"\n[{idx}/{len(SCENARIOS)}] {scenario['name']}")
    print(f"  Size: {scenario['n_farms']} farms × {scenario['n_foods']} foods = {scenario['n_vars']} vars")
    
    # Load data
    data = load_scenario_data(scenario)
    print(f"  Data loaded: {len(data['farm_names'])} farms, {len(data['food_names'])} foods")
    
    # Solve with SA
    print(f"  Running hierarchical SA solver...")
    sa_result = solve_sa_hierarchical(data, scenario)
    
    result = {
        'scenario': scenario['name'],
        'n_farms': scenario['n_farms'],
        'n_foods': scenario['n_foods'],
        'n_vars': scenario['n_vars'],
        'sa_objective': sa_result['objective'],
        'sa_violations': sa_result['violations'],
        'sa_time': sa_result['solve_time'],
        'sa_success': sa_result['success'],
        'sa_has_solution': sa_result['has_solution'],
        'status': 'PASS' if (sa_result['objective'] > 0 and sa_result['success'] and sa_result['has_solution']) else 'FAIL'
    }
    results.append(result)
    
    status_emoji = "✓" if result['status'] == 'PASS' else "✗"
    print(f"  {status_emoji} Objective: {sa_result['objective']:.4f}, Violations: {sa_result['violations']}, Time: {sa_result['solve_time']:.1f}s")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
df = pd.DataFrame(results)
csv_file = OUTPUT_DIR / f'sa_validation_{timestamp}.csv'
df.to_csv(csv_file, index=False)

# Summary
print("\n" + "="*80)
print("SA VALIDATION SUMMARY")
print("="*80)

print(f"\n{'Scenario':<30} {'Vars':>6} {'Obj':>10} {'Viol':>6} {'Time':>8} {'Status':>8}")
print("-"*70)
for r in results:
    print(f"{r['scenario']:<30} {r['n_vars']:>6} {r['sa_objective']:>10.4f} {r['sa_violations']:>6} {r['sa_time']:>7.1f}s {r['status']:>8}")

all_pass = all(r['status'] == 'PASS' for r in results)
print("-"*70)
print(f"\nTotal: {sum(1 for r in results if r['status'] == 'PASS')}/{len(results)} PASSED")

if all_pass:
    print("\n✓ ALL SCENARIOS PASSED SA VALIDATION!")
    print("The hierarchical solver fixes are working correctly.")
else:
    print("\n✗ SOME SCENARIOS FAILED - review above")

print(f"\n✓ Results saved to: {csv_file}")
print("="*80)
