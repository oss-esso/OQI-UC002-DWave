#!/usr/bin/env python3
"""
Compare rotation_medium_100 vs rotation_large_200 with SAME number of farms
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

def solve_scenario(scenario, n_farms, timeout=300):
    """Solve a specific scenario"""
    farms, foods, food_groups, config = load_food_data(scenario)
    
    params = config.get('parameters', {})
    land_availability = params.get('land_availability', {})
    all_farm_names = list(land_availability.keys())[:n_farms]
    land_availability = {f: land_availability[f] for f in all_farm_names}
    total_area = sum(land_availability.values())
    
    weights = params.get('weights', {})
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
    
    rotation_gamma = params.get('rotation_gamma', 0.2)
    k_neighbors = params.get('spatial_k_neighbors', 4)
    frustration_ratio = params.get('frustration_ratio', 0.7)
    negative_strength = params.get('negative_synergy_strength', -0.8)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    
    print(f"    Params: frust={frustration_ratio:.2f}, neg={negative_strength:.2f}, gamma={rotation_gamma:.2f}, penalty={one_hot_penalty:.1f}")
    
    n_periods = 3
    n_families = len(food_names)
    
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
    positions = {farm: (i // side, i % side) for i, farm in enumerate(all_farm_names)}
    neighbor_edges = []
    for f1 in all_farm_names:
        distances = [(np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2), f2)
                    for f2 in all_farm_names if f1 != f2]
        distances.sort()
        for _, f2 in distances[:k_neighbors]:
            if (f2, f1) not in neighbor_edges:
                neighbor_edges.append((f1, f2))
    
    # Build model
    model = gp.Model("ScenarioTest")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.1)
    model.setParam('MIPFocus', 1)
    model.setParam('Threads', 0)
    model.setParam('Presolve', 2)
    model.setParam('Cuts', 2)
    
    Y = {}
    for f in all_farm_names:
        for c in food_names:
            for t in range(1, n_periods + 1):
                Y[(f, c, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}_t{t}")
    
    model.update()
    
    # Objective
    obj = 0
    for f in all_farm_names:
        farm_area = land_availability[f]
        for c in food_names:
            benefit = food_benefits.get(c, 0.5)
            for t in range(1, n_periods + 1):
                obj += (benefit * farm_area * Y[(f, c, t)]) / total_area
    
    for f in all_farm_names:
        farm_area = land_availability[f]
        for t in range(2, n_periods + 1):
            for c1_idx, c1 in enumerate(food_names):
                for c2_idx, c2 in enumerate(food_names):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(f, c1, t-1)] * Y[(f, c2, t)]) / total_area
    
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for c1_idx, c1 in enumerate(food_names):
                for c2_idx, c2 in enumerate(food_names):
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_area
    
    for f in all_farm_names:
        for t in range(1, n_periods + 1):
            crop_count = gp.quicksum(Y[(f, c, t)] for c in food_names)
            obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    
    for f in all_farm_names:
        for c in food_names:
            crop_used = gp.quicksum(Y[(f, c, t)] for t in range(1, n_periods + 1))
            obj += diversity_bonus * crop_used
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    for f in all_farm_names:
        for t in range(1, n_periods + 1):
            model.addConstr(
                gp.quicksum(Y[(f, c, t)] for c in food_names) <= 2,
                name=f"max_crops_{f}_t{t}"
            )
    
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    return {
        'time': solve_time,
        'objective': model.ObjVal if model.SolCount > 0 else 0,
        'status': 'optimal' if model.Status == GRB.OPTIMAL else ('timeout' if model.Status == GRB.TIME_LIMIT else 'other'),
        'gap': model.MIPGap * 100 if hasattr(model, 'MIPGap') and model.SolCount > 0 else 0,
        'area': total_area,
    }

print("="*80)
print("SCENARIO COMPARISON: SAME FARM COUNT, DIFFERENT SCENARIOS")
print("="*80)

for n_farms in [20, 50]:
    print(f"\n{'='*80}")
    print(f"{n_farms} farms x 6 foods = {n_farms * 6 * 3} variables")
    print("="*80)
    
    for scenario in ['rotation_medium_100', 'rotation_large_200']:
        print(f"\n  Scenario: {scenario}")
        result = solve_scenario(scenario, n_farms, timeout=300)
        print(f"    Area: {result['area']:.2f} ha")
        print(f"    Time: {result['time']:.1f}s")
        print(f"    Status: {result['status']}")
        print(f"    Objective: {result['objective']:.4f}")
        print(f"    Gap: {result['gap']:.1f}%")
        
        if result['status'] == 'timeout':
            print(f"    >>> TIMEOUT")
        else:
            print(f"    >>> SOLVED")

print(f"\n{'='*80}")
print("KEY INSIGHT:")
print("If rotation_medium_100 times out but rotation_large_200 doesn't,")
print("then the SCENARIO PARAMETERS (not farm count) determine hardness!")
print("="*80)
