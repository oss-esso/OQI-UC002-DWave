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
    'timeout': 100,  # 100 seconds HARD LIMIT
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
    """Solve with Gurobi and track timeout behavior."""
    print(f"\n{'='*70}")
    print(f"Testing: {scenario['name']}")
    print(f"Size: {scenario['n_farms']} farms × {scenario['n_foods']} foods = {scenario['n_vars']} vars")
    print(f"{'='*70}")
    
    result = {
        'scenario': scenario['name'],
        'n_vars': scenario['n_vars'],
        'status': 'unknown',
        'objective': None,
        'runtime': None,
        'mip_gap': None,
        'hit_timeout': False,
        'hit_improve_limit': False,
        'stopped_reason': 'unknown',
    }
    
    start_time = time.time()
    
    try:
        model = gp.Model("rotation_test")
        model.setParam('OutputFlag', 1)
        model.setParam('TimeLimit', config['timeout'])
        model.setParam('MIPGap', config['mip_gap'])
        model.setParam('MIPFocus', config['mip_focus'])
        model.setParam('ImproveStartTime', config['improve_start_time'])
        
        n_farms = scenario['n_farms']
        n_foods = scenario['n_foods']
        n_periods = scenario['n_periods']
        
        farm_names = data['farm_names']
        food_names = data['food_names'][:n_foods]
        land_availability = data['land_availability']
        food_benefits = data['food_benefits']
        total_area = data['total_area']
        
        # CRITICAL: Add problem complexity parameters (from statistical test)
        rotation_gamma = 0.2
        one_hot_penalty = 3.0
        diversity_bonus = 0.15
        k_neighbors = 4
        frustration_ratio = 0.7
        negative_strength = -0.8
        
        # Create rotation synergy matrix (makes problem MIQP, not MIP)
        rng = np.random.RandomState(42)
        R = np.zeros((n_foods, n_foods))
        for i in range(n_foods):
            for j in range(n_foods):
                if i == j:
                    R[i, j] = negative_strength * 1.5  # Same crop = negative
                elif rng.random() < frustration_ratio:
                    R[i, j] = rng.uniform(negative_strength * 1.2, negative_strength * 0.3)
                else:
                    R[i, j] = rng.uniform(0.02, 0.20)
        
        # Create spatial neighbor graph
        side = int(np.ceil(np.sqrt(n_farms)))
        positions = {}
        for i, farm in enumerate(farm_names):
            row, col = i // side, i % side
            positions[farm] = (row, col)
        
        neighbor_edges = []
        for f1_idx, f1 in enumerate(farm_names):
            distances = []
            for f2_idx, f2 in enumerate(farm_names):
                if f1 != f2:
                    dist = np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                                 (positions[f1][1] - positions[f2][1])**2)
                    distances.append((dist, f2_idx, f2))
            distances.sort()
            for _, f2_idx, f2 in distances[:k_neighbors]:
                if (f2_idx, f1_idx) not in [(e[1], e[0]) for e in neighbor_edges]:
                    neighbor_edges.append((f1_idx, f2_idx))
        
        # Variables
        Y = {}
        for i, farm in enumerate(farm_names):
            for j, food in enumerate(food_names):
                for t in range(1, n_periods + 1):
                    Y[(i, j, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_f{i}_c{j}_t{t}")
        
        # Objective: maximize benefit + synergies - penalties (MIQP formulation)
        obj = 0
        
        # Part 1: Base benefit (linear)
        for i, farm in enumerate(farm_names):
            farm_area = land_availability[farm]
            for j, food in enumerate(food_names):
                benefit = food_benefits[food]
                for t in range(1, n_periods + 1):
                    obj += (benefit * farm_area * Y[(i, j, t)]) / total_area
        
        # Part 2: Rotation synergies (QUADRATIC - makes it hard!)
        for i in range(n_farms):
            farm_area = land_availability[farm_names[i]]
            for t in range(2, n_periods + 1):
                for j1 in range(n_foods):
                    for j2 in range(n_foods):
                        synergy = R[j1, j2]
                        if abs(synergy) > 1e-6:
                            obj += (rotation_gamma * synergy * farm_area * 
                                   Y[(i, j1, t-1)] * Y[(i, j2, t)]) / total_area
        
        # Part 3: Spatial interactions (QUADRATIC)
        spatial_gamma = rotation_gamma * 0.5
        for (f1_idx, f2_idx) in neighbor_edges:
            for t in range(1, n_periods + 1):
                for j1 in range(n_foods):
                    for j2 in range(n_foods):
                        spatial_synergy = R[j1, j2] * 0.3
                        if abs(spatial_synergy) > 1e-6:
                            obj += (spatial_gamma * spatial_synergy * 
                                   Y[(f1_idx, j1, t)] * Y[(f2_idx, j2, t)]) / total_area
        
        # Part 4: Soft one-hot penalty (QUADRATIC)
        for i in range(n_farms):
            for t in range(1, n_periods + 1):
                crop_count = gp.quicksum(Y[(i, j, t)] for j in range(n_foods))
                obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
        
        # Part 5: Diversity bonus (linear)
        for i in range(n_farms):
            for j in range(n_foods):
                crop_used = gp.quicksum(Y[(i, j, t)] for t in range(1, n_periods + 1))
                obj += diversity_bonus * crop_used
        
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Constraints: Soft one-hot (allow 1-2 crops per farm per period)
        for i in range(n_farms):
            for t in range(1, n_periods + 1):
                model.addConstr(
                    gp.quicksum(Y[(i, j, t)] for j in range(n_foods)) <= 2,
                    name=f"max_crops_f{i}_t{t}"
                )
                model.addConstr(
                    gp.quicksum(Y[(i, j, t)] for j in range(n_foods)) >= 1,
                    name=f"min_crops_f{i}_t{t}"
                )
        
        for i in range(n_farms):
            for j in range(n_foods):
                for t in range(1, n_periods):
                    model.addConstr(
                        Y[(i, j, t)] + Y[(i, j, t + 1)] <= 1,
                        name=f"rotation_f{i}_c{j}_t{t}"
                    )
        
        print(f"Model: {model.NumVars} vars, {model.NumConstrs} constraints")
        print("Solving...")
        
        # Solve
        model.optimize()
        
        runtime = time.time() - start_time
        result['runtime'] = runtime
        
        # Analyze stopping reason
        if model.Status == GRB.OPTIMAL:
            result['status'] = 'optimal'
            result['stopped_reason'] = 'optimal_found'
            result['objective'] = model.ObjVal
            result['mip_gap'] = 0.0
        elif model.Status == GRB.TIME_LIMIT:
            result['status'] = 'timeout'
            result['hit_timeout'] = True
            result['stopped_reason'] = 'timeout_300s'
            if model.SolCount > 0:
                result['objective'] = model.ObjVal
                result['mip_gap'] = model.MIPGap * 100
        elif model.SolCount > 0:
            result['status'] = 'feasible'
            result['objective'] = model.ObjVal
            result['mip_gap'] = model.MIPGap * 100
            # Check if stopped due to ImproveStartTime
            if runtime < config['timeout'] - 5:
                result['hit_improve_limit'] = True
                result['stopped_reason'] = 'no_improvement_30s'
            else:
                result['stopped_reason'] = 'mip_gap_reached'
        else:
            result['status'] = 'no_solution'
            result['stopped_reason'] = 'infeasible'
        
        # Display results
        print(f"\n{'='*70}")
        print(f"RESULTS: {scenario['name']}")
        print(f"{'='*70}")
        print(f"Status:         {result['status']}")
        print(f"Stopped reason: {result['stopped_reason']}")
        print(f"Runtime:        {runtime:.2f}s / {config['timeout']}s")
        print(f"Objective:      {result['objective']}")
        print(f"MIP Gap:        {result['mip_gap']:.2f}%" if result['mip_gap'] else "MIP Gap:        N/A")
        print(f"Hit timeout:    {'YES ⚠️' if result['hit_timeout'] else 'NO'}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        result['status'] = 'error'
        result['stopped_reason'] = 'error'
        result['runtime'] = time.time() - start_time
    
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
    timeout_str = "YES" if r['hit_timeout'] else "NO"
    print(f"{r['scenario']:<40} {r['n_vars']:>6} {r['runtime']:>9.1f}s {timeout_str:>8} {r['stopped_reason']:<20}")
print("-"*95)

timeout_count = sum(1 for r in all_results if r['hit_timeout'])
improve_stop_count = sum(1 for r in all_results if r['hit_improve_limit'])

print(f"\nTimeout hits: {timeout_count}/{len(all_results)} ({timeout_count/len(all_results)*100:.0f}%)")
print(f"ImproveStartTime stops: {improve_stop_count}/{len(all_results)} ({improve_stop_count/len(all_results)*100:.0f}%)")

if timeout_count == len(all_results):
    print("\n✓ PASS: All scenarios hit timeout as expected!")
elif timeout_count > 0:
    print(f"\n⚠️  PARTIAL: {timeout_count} scenarios hit timeout, {len(all_results)-timeout_count} finished early")
else:
    print("\n✗ FAIL: No scenarios hit timeout - something is wrong!")

print("\n✓ Test complete!")
