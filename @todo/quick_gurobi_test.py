#!/usr/bin/env python3
"""
Quick Gurobi test on key problem sizes (100s timeout)
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
from src.scenarios import load_food_data

TIMEOUT = 100
N_PERIODS = 3

# Key test points only
TEST_POINTS = [
    {'n_farms': 5, 'scenario': 'rotation_micro_25', 'name': '5farms_90vars'},
    {'n_farms': 20, 'scenario': 'rotation_medium_100', 'name': '20farms_360vars'},
    {'n_farms': 50, 'scenario': 'rotation_large_200', 'name': '50farms_900vars'},
    {'n_farms': 90, 'scenario': 'rotation_large_200', 'name': '90farms_1620vars'},
]

def solve_gurobi(land_availability, food_benefits, params, timeout=100):
    """Solve with Gurobi"""
    farm_names = list(land_availability.keys())
    n_farms = len(farm_names)
    families_list = list(food_benefits.keys())
    n_families = len(families_list)
    total_area = sum(land_availability.values())
    
    rotation_gamma = params.get('rotation_gamma', 0.2)
    frustration_ratio = params.get('frustration_ratio', 0.7)
    negative_strength = params.get('negative_synergy_strength', -0.8)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    k_neighbors = params.get('spatial_k_neighbors', 4)
    
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
        for _, f2 in distances[:min(k_neighbors, len(distances))]:
            if (f2, f1) not in neighbor_edges:
                neighbor_edges.append((f1, f2))
    
    # Build model
    model = gp.Model("Test")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.1)
    model.setParam('MIPFocus', 1)
    model.setParam('Threads', 0)
    model.setParam('Presolve', 2)
    model.setParam('Cuts', 2)
    
    # Variables
    Y = {}
    for f in farm_names:
        for c in families_list:
            for t in range(1, N_PERIODS + 1):
                Y[(f, c, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}_t{t}")
    
    model.update()
    
    # Objective
    obj = 0
    for f in farm_names:
        farm_area = land_availability[f]
        for c in families_list:
            benefit = food_benefits[c]
            for t in range(1, N_PERIODS + 1):
                obj += (benefit * farm_area * Y[(f, c, t)]) / total_area
    
    for f in farm_names:
        farm_area = land_availability[f]
        for t in range(2, N_PERIODS + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(f, c1, t-1)] * Y[(f, c2, t)]) / total_area
    
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, N_PERIODS + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_area
    
    for f in farm_names:
        for t in range(1, N_PERIODS + 1):
            crop_count = gp.quicksum(Y[(f, c, t)] for c in families_list)
            obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    
    for f in farm_names:
        for c in families_list:
            crop_used = gp.quicksum(Y[(f, c, t)] for t in range(1, N_PERIODS + 1))
            obj += diversity_bonus * crop_used
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    for f in farm_names:
        for t in range(1, N_PERIODS + 1):
            model.addConstr(
                gp.quicksum(Y[(f, c, t)] for c in families_list) <= 2,
                name=f"max_crops_{f}_t{t}"
            )
    
    # Solve
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    return {
        'time': solve_time,
        'objective': model.ObjVal if model.SolCount > 0 else 0,
        'status': 'optimal' if model.Status == GRB.OPTIMAL else ('timeout' if model.Status == GRB.TIME_LIMIT else 'other'),
        'gap': model.MIPGap * 100 if hasattr(model, 'MIPGap') and model.SolCount > 0 else 0,
    }

print("="*80)
print("QUICK GUROBI TEST (100s timeout)")
print("="*80)

for test in TEST_POINTS:
    n_farms = test['n_farms']
    scenario = test['scenario']
    name = test['name']
    n_vars = n_farms * 6 * N_PERIODS
    
    print(f"\n{name}: {n_farms} farms = {n_vars} vars")
    print("-"*80)
    
    farms, foods, food_groups, config = load_food_data(scenario)
    params = config.get('parameters', {})
    land_availability_full = params.get('land_availability', {})
    weights = params.get('weights', {})
    
    all_farms = list(land_availability_full.keys())[:n_farms]
    land_availability = {f: land_availability_full[f] for f in all_farms}
    total_area = sum(land_availability.values())
    
    food_benefits = {}
    for food, attrs in foods.items():
        benefit = (
            weights.get('nutritional_value', 0) * attrs.get('nutritional_value', 0) +
            weights.get('nutrient_density', 0) * attrs.get('nutrient_density', 0) -
            weights.get('environmental_impact', 0) * attrs.get('environmental_impact', 0) +
            weights.get('affordability', 0) * attrs.get('affordability', 0) +
            weights.get('sustainability', 0) * attrs.get('sustainability', 0)
        )
        food_benefits[food] = benefit
    
    print(f"  Total area: {total_area:.1f} ha")
    print(f"  Running Gurobi...")
    
    result = solve_gurobi(land_availability, food_benefits, params, timeout=TIMEOUT)
    
    print(f"  Status: {result['status']}")
    print(f"  Time: {result['time']:.1f}s")
    print(f"  Gap: {result['gap']:.1f}%")
    
    if result['status'] == 'timeout':
        print(f"  >>> TIMEOUT - QPU could help!")
    else:
        print(f"  >>> Solved")

print(f"\n{'='*80}")
print("Done!")
print("="*80)
