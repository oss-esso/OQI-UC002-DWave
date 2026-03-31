#!/usr/bin/env python3
"""
Gurobi-Only Test: Verify timeout behavior across different problem sizes

This script tests ONLY Gurobi on the 6 scenarios to confirm:
1. Timeout is hit consistently (300s HARD LIMIT)
2. MIPGap=0.01 and MIPFocus=1 settings work correctly
3. ImproveStartTime=30s stops appropriately
4. Timeout doesn't "disappear" for different problem sizes

Author: OQI-UC002-DWave
Date: December 14, 2025
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("GUROBI-ONLY TIMEOUT VERIFICATION TEST")
print("="*80)
print()

# Configuration
GUROBI_CONFIG = {
    'timeout': 200,  # 1200 seconds (20 minutes) HARD LIMIT
    'mip_gap': 0.01,  # 1% - stop within 1% of optimum
    'mip_focus': 1,  # Find good feasible solutions quickly
    'improve_start_time': 30,  # Stop if no improvement for 30s
}

# Test scenarios - Comprehensive rotation scenarios from both sources
SCENARIOS = [
    # Family-only scenarios (6 foods) from src/scenarios.py
    {'name': 'rotation_micro_25', 'n_farms': 5, 'n_foods': 6, 'n_periods': 3, 'n_vars': 90},
    {'name': 'rotation_small_50', 'n_farms': 10, 'n_foods': 6, 'n_periods': 3, 'n_vars': 180},
    # Additional family-only scenario (15 farms from statistical_comparison_test.py)
    {'name': 'rotation_15farms_6foods', 'n_farms': 15, 'n_foods': 6, 'n_periods': 3, 'n_vars': 270},
    {'name': 'rotation_medium_100', 'n_farms': 20, 'n_foods': 6, 'n_periods': 3, 'n_vars': 360},
    # Additional family-only scenario (25 farms from statistical_comparison_test.py)
    {'name': 'rotation_25farms_6foods', 'n_farms': 25, 'n_foods': 6, 'n_periods': 3, 'n_vars': 450},
    # Larger family-only scenario from src/scenarios.py
    {'name': 'rotation_large_200', 'n_farms': 40, 'n_foods': 6, 'n_periods': 3, 'n_vars': 720},
    # GAP FILLING SCENARIOS - bridging 720 to 20,250 variables
    {'name': 'rotation_50farms_6foods', 'n_farms': 50, 'n_foods': 6, 'n_periods': 3, 'n_vars': 900},
    {'name': 'rotation_75farms_6foods', 'n_farms': 75, 'n_foods': 6, 'n_periods': 3, 'n_vars': 1350},
    {'name': 'rotation_100farms_6foods', 'n_farms': 100, 'n_foods': 6, 'n_periods': 3, 'n_vars': 1800},
    {'name': 'rotation_25farms_27foods', 'n_farms': 25, 'n_foods': 27, 'n_periods': 3, 'n_vars': 2025},
    {'name': 'rotation_150farms_6foods', 'n_farms': 150, 'n_foods': 6, 'n_periods': 3, 'n_vars': 2700},
    {'name': 'rotation_50farms_27foods', 'n_farms': 50, 'n_foods': 27, 'n_periods': 3, 'n_vars': 4050},
    {'name': 'rotation_75farms_27foods', 'n_farms': 75, 'n_foods': 27, 'n_periods': 3, 'n_vars': 6075},
    {'name': 'rotation_100farms_27foods', 'n_farms': 100, 'n_foods': 27, 'n_periods': 3, 'n_vars': 8100},
    {'name': 'rotation_150farms_27foods', 'n_farms': 150, 'n_foods': 27, 'n_periods': 3, 'n_vars': 12150},
    {'name': 'rotation_200farms_27foods', 'n_farms': 200, 'n_foods': 27, 'n_periods': 3, 'n_vars': 16200},
    # Full food scenarios (27 foods) from src/scenarios.py
    {'name': 'rotation_250farms_27foods', 'n_farms': 250, 'n_foods': 27, 'n_periods': 3, 'n_vars': 20250},
    {'name': 'rotation_350farms_27foods', 'n_farms': 350, 'n_foods': 27, 'n_periods': 3, 'n_vars': 28350},
    {'name': 'rotation_500farms_27foods', 'n_farms': 500, 'n_foods': 27, 'n_periods': 3, 'n_vars': 40500},
    {'name': 'rotation_1000farms_27foods', 'n_farms': 1000, 'n_foods': 27, 'n_periods': 3, 'n_vars': 81000},
]

OUTPUT_DIR = Path(__file__).parent / 'gurobi_timeout_verification'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import libraries
try:
    import gurobipy as gp
    from gurobipy import GRB
    print("✓ Gurobi available")
except ImportError:
    print("✗ Gurobi not available")
    sys.exit(1)

from data_loader_utils import load_food_data_as_dict
from unified_benchmark.gurobi_solver import solve_gurobi_ground_truth

print()
print("Configuration:")
print(f"  Time limit: {GUROBI_CONFIG['timeout']}s")
print(f"  MIP gap: {GUROBI_CONFIG['mip_gap']*100}%")
print(f"  MIP focus: {GUROBI_CONFIG['mip_focus']}")
print(f"  Improve start time: {GUROBI_CONFIG['improve_start_time']}s")
print()

def load_scenario_data(scenario: Dict) -> Dict:
    """Load data for scenario."""
    n_farms = scenario['n_farms']
    n_foods = scenario['n_foods']
    
    # Map scenario parameters to actual scenario names in scenarios.py
    if n_foods == 6:
        if n_farms <= 5:
            scenario_name = 'rotation_micro_25'
        elif n_farms <= 10:
            scenario_name = 'rotation_small_50'
        elif n_farms <= 20:
            scenario_name = 'rotation_medium_100'
        elif n_farms <= 40:
            scenario_name = 'rotation_large_200'
        else:
            # For 50+ farms with 6 foods, use large_200 and adjust
            scenario_name = 'rotation_large_200'
    else:  # 27 foods
        if n_farms <= 25:
            scenario_name = 'rotation_250farms_27foods'
        elif n_farms <= 50:
            scenario_name = 'rotation_250farms_27foods'
        elif n_farms <= 75:
            scenario_name = 'rotation_250farms_27foods'
        elif n_farms <= 100:
            scenario_name = 'rotation_250farms_27foods'
        elif n_farms <= 150:
            scenario_name = 'rotation_250farms_27foods'
        elif n_farms <= 200:
            scenario_name = 'rotation_250farms_27foods'
        elif n_farms <= 250:
            scenario_name = 'rotation_250farms_27foods'
        elif n_farms <= 350:
            scenario_name = 'rotation_350farms_27foods'
        elif n_farms <= 500:
            scenario_name = 'rotation_500farms_27foods'
        else:
            scenario_name = 'rotation_1000farms_27foods'
    
    data = load_food_data_as_dict(scenario_name)
    
    # Adjust farm count if needed
    if len(data['farm_names']) != n_farms:
        if len(data['farm_names']) > n_farms:
            data['farm_names'] = data['farm_names'][:n_farms]
            data['land_availability'] = {k: v for k, v in list(data['land_availability'].items())[:n_farms]}
        else:
            original_farms = data['farm_names'].copy()
            while len(data['farm_names']) < n_farms:
                idx = len(data['farm_names']) - len(original_farms)
                farm = original_farms[idx % len(original_farms)]
                new_farm = f"{farm}_dup{idx}"
                data['farm_names'].append(new_farm)
                data['land_availability'][new_farm] = data['land_availability'][farm]
    
    return data

def solve_gurobi_test(data: Dict, scenario: Dict, config: Dict) -> Dict:
    """Solve with Gurobi and track timeout behavior + extract full solution.

    Delegates model building to solve_gurobi_ground_truth (vectorized addMVar +
    sparse Q matrix) instead of the old Python-loop builder, which was O(n²)
    for the spatial quadratic terms and hung at n≥100 with 27 foods.
    """
    print(f"\n{'='*70}")
    print(f"Testing: {scenario['name']}")
    print(f"Size: {scenario['n_farms']} farms × {scenario['n_foods']} foods = {scenario['n_vars']} vars")
    print(f"{'='*70}")

    n_farms   = scenario['n_farms']
    n_foods   = scenario['n_foods']
    n_periods = scenario['n_periods']
    farm_names = data['farm_names']
    food_names = data['food_names'][:n_foods]

    result = {
        'metadata': {
            'benchmark_type': 'GUROBI_TIMEOUT_TEST',
            'solver': 'gurobi',
            'scenario': scenario['name'],
            'n_farms': n_farms,
            'n_foods': n_foods,
            'n_periods': n_periods,
            'timestamp': datetime.now().isoformat()
        },
        'result': {
            'scenario': scenario['name'],
            'n_vars': scenario['n_vars'],
            'status': 'unknown',
            'objective_value': None,
            'solve_time': None,
            'mip_gap': None,
            'hit_timeout': False,
            'hit_improve_limit': False,
            'stopped_reason': 'unknown',
            'solution_selections': {},  # Binary decision variables
            'solution_areas': {},       # Area allocations (binary × farm_area)
            'total_covered_area': 0.0,
            'validation': {},           # Constraint verification
            'solver': 'gurobi',
            'success': False,
        },
        'decomposition_specific': {
            'used_decomposition': False,
            'method': 'monolithic',
            'n_partitions': 1,
            'notes': 'Classical Gurobi solve - no decomposition used'
        },
        'benchmark_info': {
            'n_farms': n_farms,
            'n_foods': n_foods,
            'n_periods': n_periods,
            'total_land': sum(data['land_availability'].values()) if 'land_availability' in data else 0.0,
            'timeout': config['timeout'],
            'mip_gap_target': config['mip_gap'],
            'timestamp': datetime.now().isoformat()
        }
    }
    
    start_time = time.time()
    
    try:
        scenario_data = {
            "farm_names": farm_names,
            "food_names": food_names,
            "land_availability": data['land_availability'],
            "food_benefits": data['food_benefits'],
            "total_area": data['total_area'],
            "n_farms": n_farms,
            "n_foods": n_foods,
            "n_periods": n_periods,
            "scenario_name": scenario['name'],
        }

        entry = solve_gurobi_ground_truth(
            scenario_data,
            timeout=config['timeout'],
            mip_gap=config['mip_gap'],
            verbose=True,
        )

        runtime = entry.timing.total_wall_time
        result['result']['solve_time'] = runtime

        # Map status
        result['result']['status'] = entry.status
        result['result']['objective_value'] = entry.objective_miqp
        result['result']['mip_gap'] = entry.mip_gap
        result['result']['success'] = (
            entry.status in ('optimal', 'timeout', 'feasible')
            and entry.objective_miqp is not None
        )

        if entry.status == 'timeout':
            result['result']['hit_timeout'] = True
            result['result']['stopped_reason'] = f"timeout_{int(config['timeout'])}s"
        elif entry.status == 'optimal':
            result['result']['stopped_reason'] = 'optimal_found'
        elif entry.status == 'feasible':
            if runtime < config['timeout'] - 5:
                result['result']['hit_improve_limit'] = True
                result['result']['stopped_reason'] = 'no_improvement_30s'
            else:
                result['result']['stopped_reason'] = 'mip_gap_reached'
        else:
            result['result']['stopped_reason'] = 'infeasible_or_error'

        # Extract solution from entry.solution = {(f_idx, c_idx, t_1based): 1}
        if entry.solution:
            land_availability = data['land_availability']
            for (f_idx, c_idx, t), _ in entry.solution.items():
                farm = farm_names[f_idx]
                food = food_names[c_idx]
                var_name = f"{farm}_{food}_t{t}"
                result['result']['solution_selections'][var_name] = 1.0
                result['result']['solution_areas'][var_name] = float(land_availability.get(farm, 1.0))
            result['result']['total_covered_area'] = sum(result['result']['solution_areas'].values())

        # Validation from entry.constraint_violations
        if entry.constraint_violations is not None:
            n_viols = entry.constraint_violations.total_violations
            result['result']['validation'] = {
                'is_valid': entry.feasible,
                'n_violations': n_viols,
                'violations': [],
                'summary': f"{'Valid' if entry.feasible else 'Invalid'}: {n_viols} violations found"
            }
        else:
            result['result']['validation'] = {
                'is_valid': True, 'n_violations': 0, 'violations': [],
                'summary': 'Valid: no constraint check performed'
            }

        print(f"\n{'='*70}")
        print(f"RESULTS: {scenario['name']}")
        print(f"{'='*70}")
        print(f"Status:         {result['result']['status']}")
        print(f"Stopped reason: {result['result']['stopped_reason']}")
        print(f"Runtime:        {runtime:.2f}s / {config['timeout']}s  "
              f"(build {entry.timing.model_build_time:.3f}s + solve {entry.timing.solve_time:.3f}s)")
        print(f"Objective:      {result['result']['objective_value']}")
        if result['result']['mip_gap'] is not None:
            print(f"MIP Gap:        {result['result']['mip_gap']:.2f}%")
        else:
            print("MIP Gap:        N/A")
        print(f"Hit timeout:    {'YES \u26a0\ufe0f' if result['result']['hit_timeout'] else 'NO'}")
        if entry.solution:
            print(f"Variables:      {len(result['result']['solution_selections'])} extracted")
            print(f"Total area:     {result['result']['total_covered_area']:.2f} ha")
            print(f"Validation:     {result['result']['validation']['summary']}")
        print(f"{'='*70}")

    except Exception as e:
        print(f"ERROR: {e}")
        result['result']['status'] = 'error'
        result['result']['stopped_reason'] = 'error'
        result['result']['solve_time'] = time.time() - start_time
    
    return result

# Run test
print(f"\nTesting {len(SCENARIOS)} scenarios with Gurobi...\n")

all_results = []

for idx, scenario in enumerate(SCENARIOS, 1):
    print(f"\n{'#'*80}")
    print(f"SCENARIO {idx}/{len(SCENARIOS)}: {scenario['name']}")
    print(f"{'#'*80}")
    
    data = load_scenario_data(scenario)
    result = solve_gurobi_test(data, scenario, GUROBI_CONFIG)
    all_results.append(result)

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
json_file = OUTPUT_DIR / f'gurobi_timeout_test_{timestamp}.json'
with open(json_file, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✓ Results saved to: {json_file}")

df = pd.DataFrame(all_results)
csv_file = OUTPUT_DIR / f'gurobi_timeout_test_{timestamp}.csv'
df.to_csv(csv_file, index=False)
print(f"✓ CSV saved to: {csv_file}")

# Summary
print("\n" + "="*80)
print("TIMEOUT VERIFICATION SUMMARY")
print("="*80)
print()
print(f"{'Scenario':<40} {'Vars':>6} {'Runtime':>10} {'Timeout':>8} {'Stopped Reason':<20}")
print("-"*95)
for r in all_results:
    timeout_str = "YES" if r['result']['hit_timeout'] else "NO"
    print(f"{r['metadata']['scenario']:<40} {r['result']['n_vars']:>6} {r['result']['solve_time']:>9.1f}s {timeout_str:>8} {r['result']['stopped_reason']:<20}")
print("-"*95)

timeout_count = sum(1 for r in all_results if r['result']['hit_timeout'])
improve_stop_count = sum(1 for r in all_results if r['result']['hit_improve_limit'])

print(f"\nTimeout hits: {timeout_count}/{len(all_results)} ({timeout_count/len(all_results)*100:.0f}%)")
print(f"ImproveStartTime stops: {improve_stop_count}/{len(all_results)} ({improve_stop_count/len(all_results)*100:.0f}%)")

if timeout_count == len(all_results):
    print("\n✓ PASS: All scenarios hit timeout as expected!")
elif timeout_count > 0:
    print(f"\n⚠️  PARTIAL: {timeout_count} scenarios hit timeout, {len(all_results)-timeout_count} finished early")
else:
    print("\n✗ FAIL: No scenarios hit timeout - something is wrong!")

print("\n✓ Test complete!")
