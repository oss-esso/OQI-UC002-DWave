#!/usr/bin/env python3
"""
Statistical test - Gurobi only (no QPU)
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import time
from src.scenarios import load_food_data
import gurobipy as gp
from gurobipy import GRB

def load_rotation_data(n_farms: int):
    """Load data from statistical_comparison_test.py"""
    scenario_map = {
        5: 'rotation_micro_25',
        10: 'rotation_small_50',
        15: 'rotation_medium_100',
        20: 'rotation_medium_100',
        25: 'rotation_large_200',
    }
    
    base_scenario = scenario_map.get(n_farms)
    if base_scenario is None:
        for size in sorted(scenario_map.keys()):
            if size >= n_farms:
                base_scenario = scenario_map[size]
                break
    
    farms, foods, food_groups, config = load_food_data(base_scenario)
    
    params = config.get('parameters', {})
    weights = params.get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    land_availability = params.get('land_availability', {})
    all_farm_names = list(land_availability.keys())
    
    if len(all_farm_names) < n_farms:
        for i in range(len(all_farm_names), n_farms):
            land_availability[f'Farm_{i+1}'] = np.random.uniform(15, 35)
        all_farm_names = list(land_availability.keys())
    
    farm_names = all_farm_names[:n_farms]
    land_availability = {f: land_availability[f] for f in farm_names}
    total_area = sum(land_availability.values())
    
    food_names = list(foods.keys())
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
    
    return {
        'farm_names': farm_names,
        'food_names': food_names,
        'food_benefits': food_benefits,
        'land_availability': land_availability,
        'total_area': total_area,
        'config': config,
        'scenario': base_scenario,
    }

def solve_gurobi(data, timeout=300):
    """Solve with Gurobi"""
    farm_names = data['farm_names']
    food_names = data['food_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    config = data.get('config', {})
    params = config.get('parameters', {})
    
    rotation_gamma = params.get('rotation_gamma', 0.2)
    k_neighbors = params.get('spatial_k_neighbors', 4)
    frustration_ratio = params.get('frustration_ratio', 0.7)
    negative_strength = params.get('negative_synergy_strength', -0.8)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    
    n_periods = 3
    n_farms = len(farm_names)
    n_families = len(food_names)
    families_list = list(food_names)
    
    print(f"  Parameters: frustration={frustration_ratio}, neg_strength={negative_strength}, gamma={rotation_gamma}")
    
    # Rotation matrix
    np.random.seed(42)
    R = np.zeros((n_families, n_families))
    for i in range(n_families):
        for j in range(n_families):
            if i == j:
                R[i, j] = negative_strength * 1.5
            elif np.random.random() < frustration_ratio:
                R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                R[i, j] = np.random.uniform(0.02, 0.20)
    
    # Spatial graph
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {farm: (i // side, i % side) for i, farm in enumerate(farm_names)}
    
    neighbor_edges = []
    for f1 in farm_names:
        distances = [(np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2), f2)
                    for f2 in farm_names if f1 != f2]
        distances.sort()
        for _, f2 in distances[:k_neighbors]:
            if (f2, f1) not in neighbor_edges:
                neighbor_edges.append((f1, f2))
    
    start_time = time.time()
    model = gp.Model("StatisticalTest")
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
        for c in families_list:
            for t in range(1, n_periods + 1):
                Y[(f, c, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}_t{t}")
    
    model.update()
    
    # Objective
    obj = 0
    for f in farm_names:
        farm_area = land_availability[f]
        for c in families_list:
            benefit = food_benefits.get(c, 0.5)
            for t in range(1, n_periods + 1):
                obj += (benefit * farm_area * Y[(f, c, t)]) / total_area
    
    for f in farm_names:
        farm_area = land_availability[f]
        for t in range(2, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(f, c1, t-1)] * Y[(f, c2, t)]) / total_area
    
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_area
    
    for f in farm_names:
        for t in range(1, n_periods + 1):
            crop_count = gp.quicksum(Y[(f, c, t)] for c in families_list)
            obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    
    for f in farm_names:
        for c in families_list:
            crop_used = gp.quicksum(Y[(f, c, t)] for t in range(1, n_periods + 1))
            obj += diversity_bonus * crop_used
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    for f in farm_names:
        for t in range(1, n_periods + 1):
            model.addConstr(
                gp.quicksum(Y[(f, c, t)] for c in families_list) <= 2,
                name=f"max_crops_{f}_t{t}"
            )
    
    # Solve
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    total_time = time.time() - start_time
    
    result = {
        'solve_time': total_time,
        'objective': model.ObjVal if model.SolCount > 0 else 0,
        'status': 'optimal' if model.Status == GRB.OPTIMAL else ('timeout' if model.Status == GRB.TIME_LIMIT else 'other'),
        'gap': model.MIPGap * 100 if hasattr(model, 'MIPGap') and model.SolCount > 0 else 0,
    }
    
    return result

print("="*80)
print("STATISTICAL TEST - GUROBI ONLY")
print("="*80)

for n_farms in [5, 10, 15, 20]:
    print(f"\n{'-'*80}")
    print(f"Testing: {n_farms} farms x 6 foods = {n_farms * 6 * 3} variables")
    print("-"*80)
    
    data = load_rotation_data(n_farms)
    print(f"  Scenario: {data['scenario']}")
    print(f"  Total area: {data['total_area']:.2f} ha")
    
    result = solve_gurobi(data, timeout=300)
    
    print(f"\n  Results:")
    print(f"    Status: {result['status']}")
    print(f"    Time: {result['solve_time']:.1f}s")
    print(f"    Objective: {result['objective']:.4f}")
    print(f"    Gap: {result['gap']:.1f}%")
    
    if result['status'] == 'timeout':
        print(f"    *** TIMEOUT at 300s ***")
    else:
        print(f"    >>> Solved quickly!")
