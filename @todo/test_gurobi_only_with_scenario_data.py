#!/usr/bin/env python3
"""
Comprehensive Scaling Test - GUROBI ONLY
Tests with EXACT same data loading as statistical_comparison_test.py
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import time
import json
from datetime import datetime
from src.scenarios import load_food_data
import gurobipy as gp
from gurobipy import GRB

N_PERIODS = 3
GUROBI_TIMEOUT = 300

# Test configurations - matching comprehensive_scaling_test.py
TEST_CONFIGS = {
    'test_360': {
        'native_6': {'n_farms': 20, 'n_foods': 6, 'scenario': 'rotation_medium_100', 'formulation': 'Native 6-Family'},
    },
    'test_900': {
        'native_6': {'n_farms': 50, 'n_foods': 6, 'scenario': 'rotation_large_200', 'formulation': 'Native 6-Family'},
    },
}

def load_data_for_test(test_config):
    """Load data EXACTLY like statistical_comparison_test.py"""
    n_farms = test_config['n_farms']
    n_foods_requested = test_config['n_foods']
    scenario = test_config['scenario']
    
    print(f"  Loading: {n_farms} farms x {n_foods_requested} foods from {scenario}")
    
    # Load scenario
    farms, foods, food_groups, config = load_food_data(scenario)
    
    # Land availability - Load from scenario params (EXACT same as statistical test)
    params = config.get('parameters', {})
    land_availability = params.get('land_availability', {})
    all_farm_names = list(land_availability.keys())
    
    # Extend if needed
    if len(all_farm_names) < n_farms:
        for i in range(len(all_farm_names), n_farms):
            land_availability[f'Farm_{i+1}'] = np.random.uniform(15, 35)
        all_farm_names = list(land_availability.keys())
    
    # Trim to exact count
    farm_names = all_farm_names[:n_farms]
    land_availability = {f: land_availability[f] for f in farm_names}
    total_area = sum(land_availability.values())
    
    # Food data - EXACT same as statistical test
    weights = params.get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    food_names = list(foods.keys())[:n_foods_requested]
    food_benefits = {}
    for food in food_names:
        benefit = (
            weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
            weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
            weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
            weights.get('affordability', 0) * foods[food].get('affordability', 0) +
            weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
        )
        food_benefits[food] = benefit
    n_foods = len(food_names)
    
    # Build rotation matrix (EXACT same as statistical test)
    np.random.seed(42)
    frustration_ratio = 0.7
    negative_strength = -0.8
    R = np.zeros((n_foods, n_foods))
    
    for i in range(n_foods):
        for j in range(n_foods):
            if i == j:
                R[i, j] = negative_strength * 1.5
            elif np.random.random() < frustration_ratio:
                R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                R[i, j] = np.random.uniform(0.02, 0.20)
    
    print(f"    -> Built {n_foods}x{n_foods} frustration matrix (frustration_ratio=0.7)")
    print(f"    -> Total area: {total_area:.2f} ha")
    print(f"  -> Final problem: {n_farms} farms x {n_foods} foods = {n_farms * n_foods * N_PERIODS} variables")
    
    return {
        'farm_names': farm_names,
        'food_names': food_names,
        'land_availability': land_availability,
        'food_benefits': food_benefits,
        'total_area': total_area,
        'n_farms': n_farms,
        'n_foods': n_foods,
        'rotation_matrix': R,
        'config': config,
    }

def solve_gurobi(data, timeout=GUROBI_TIMEOUT):
    """Solve with Gurobi - EXACT same formulation as statistical test"""
    start_time = time.time()
    
    farm_names = data['farm_names']
    food_names = data['food_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    R = data['rotation_matrix']
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    
    # Model with EXACT parameters from statistical test
    model = gp.Model("GurobiOnly")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.1)
    model.setParam('MIPFocus', 1)
    model.setParam('ImproveStartTime', 30)
    model.setParam('Threads', 0)
    model.setParam('Presolve', 2)
    model.setParam('Cuts', 2)
    
    # Variables
    Y = {}
    for f in farm_names:
        for c in food_names:
            for t in range(1, N_PERIODS + 1):
                Y[(f, c, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}_t{t}")
    
    model.update()
    
    # Objective (EXACT same as statistical test)
    obj = 0
    
    # Base benefit
    for f in farm_names:
        farm_area = land_availability[f]
        for c in food_names:
            benefit = food_benefits.get(c, 0.5)
            for t in range(1, N_PERIODS + 1):
                obj += (benefit * farm_area * Y[(f, c, t)]) / total_area
    
    # Rotation synergies
    rotation_gamma = 0.2
    for f in farm_names:
        farm_area = land_availability[f]
        for t in range(2, N_PERIODS + 1):
            for i, c1 in enumerate(food_names):
                for j, c2 in enumerate(food_names):
                    synergy = R[i, j]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(f, c1, t-1)] * Y[(f, c2, t)]) / total_area
    
    # Spatial neighbors
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {farm: (i // side, i % side) for i, farm in enumerate(farm_names)}
    
    neighbor_edges = []
    for f1 in farm_names:
        distances = [(np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2), f2)
                    for f2 in farm_names if f1 != f2]
        distances.sort()
        for _, f2 in distances[:4]:
            if (f2, f1) not in neighbor_edges:
                neighbor_edges.append((f1, f2))
    
    # Spatial interactions
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, N_PERIODS + 1):
            for i, c1 in enumerate(food_names):
                for j, c2 in enumerate(food_names):
                    spatial_synergy = R[i, j] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_area
    
    # Soft one-hot penalty
    one_hot_penalty = 3.0
    for f in farm_names:
        for t in range(1, N_PERIODS + 1):
            crop_count = gp.quicksum(Y[(f, c, t)] for c in food_names)
            obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    
    # Diversity bonus
    diversity_bonus = 0.15
    for f in farm_names:
        for c in food_names:
            crop_used = gp.quicksum(Y[(f, c, t)] for t in range(1, N_PERIODS + 1))
            obj += diversity_bonus * crop_used
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    for f in farm_names:
        for t in range(1, N_PERIODS + 1):
            model.addConstr(
                gp.quicksum(Y[(f, c, t)] for c in food_names) <= 2,
                name=f"max_crops_{f}_t{t}"
            )
    
    # Solve
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    total_time = time.time() - start_time
    
    result = {
        'method': 'gurobi',
        'success': False,
        'objective': 0,
        'solve_time': total_time,
        'violations': 0,
        'status': 'unknown',
        'gap': 0,
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        result['success'] = True
        result['objective'] = model.ObjVal
        result['status'] = 'optimal' if model.Status == GRB.OPTIMAL else ('timeout' if model.Status == GRB.TIME_LIMIT else 'suboptimal')
        result['gap'] = model.MIPGap * 100 if hasattr(model, 'MIPGap') else 0
    
    return result

if __name__ == '__main__':
    print("="*80)
    print("GUROBI-ONLY TEST WITH EXACT STATISTICAL TEST DATA")
    print("="*80)
    print()
    
    results = []
    
    for test_name, variants in TEST_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Test Point: {test_name}")
        print("="*80)
        
        for variant_name, test_config in variants.items():
            print(f"\n  Variant: {test_config['formulation']}")
            print("  " + "-"*71)
            
            # Load data
            data = load_data_for_test(test_config)
            
            # Run Gurobi
            print(f"    Running Gurobi (timeout={GUROBI_TIMEOUT}s)...")
            result = solve_gurobi(data, timeout=GUROBI_TIMEOUT)
            
            if result['success']:
                print(f"      OK obj={result['objective']:.4f}, time={result['solve_time']:.1f}s, gap={result['gap']:.1f}%, status={result['status']}")
            else:
                print(f"      FAILED: {result['status']}")
            
            results.append({
                'test_point': test_name,
                'formulation': test_config['formulation'],
                'n_farms': data['n_farms'],
                'n_foods': data['n_foods'],
                'n_vars': data['n_farms'] * data['n_foods'] * N_PERIODS,
                **result
            })
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Test':<12} {'Formulation':<20} {'Vars':<6} {'Time(s)':<10} {'Obj':<10} {'Gap%':<8} {'Status':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['test_point']:<12} {r['formulation']:<20} {r['n_vars']:<6} {r['solve_time']:<10.1f} {r['objective']:<10.4f} {r['gap']:<8.1f} {r['status']:<10}")
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS:")
    print("="*80)
    
    timeouts = [r for r in results if r['status'] == 'timeout']
    fast_solves = [r for r in results if r['solve_time'] < 10]
    
    if timeouts:
        print(f"  TIMEOUTS: {len(timeouts)} tests hit {GUROBI_TIMEOUT}s limit")
        for r in timeouts:
            print(f"    - {r['formulation']} ({r['n_vars']} vars): {r['gap']:.1f}% gap")
    
    if fast_solves:
        print(f"  FAST SOLVES: {len(fast_solves)} tests solved in < 10s")
        for r in fast_solves:
            print(f"    - {r['formulation']} ({r['n_vars']} vars): {r['solve_time']:.2f}s")
    
    print(f"\nExpected behavior with scenario data:")
    print(f"  - Native 6-Family (360 vars): Should TIMEOUT (like statistical test)")
    print(f"  - Native 6-Family (900 vars): Should TIMEOUT (like statistical test)")
    
    if all(r['status'] == 'timeout' for r in results):
        print(f"\n  CORRECT! All tests timeout as expected with scenario data.")
    elif any(r['solve_time'] < 10 for r in results):
        print(f"\n  WARNING! Some tests solved quickly - data may not match statistical test!")
