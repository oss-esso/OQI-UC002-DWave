#!/usr/bin/env python3
"""
Test replicated hard scenarios with Gurobi
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
from hard_scenarios import get_scenario_config

def solve_scenario(n_farms, total_area, timeout=300):
    """Solve scenario with Gurobi"""
    farms, foods, food_groups, config = get_scenario_config(n_farms, total_area)
    
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    
    # Calculate food benefits
    food_benefits = {}
    for food, attrs in foods.items():
        benefit = (
            weights['nutritional_value'] * attrs['nutritional_value'] +
            weights['nutrient_density'] * attrs['nutrient_density'] -
            weights['environmental_impact'] * attrs['environmental_impact'] +
            weights['affordability'] * attrs['affordability'] +
            weights['sustainability'] * attrs['sustainability']
        )
        food_benefits[food] = benefit
    
    rotation_gamma = params['rotation_gamma']
    frustration_ratio = params['frustration_ratio']
    negative_strength = params['negative_synergy_strength']
    one_hot_penalty = params['one_hot_penalty']
    diversity_bonus = params['diversity_bonus']
    k_neighbors = params['spatial_k_neighbors']
    
    n_families = len(foods)
    families_list = list(foods.keys())
    n_periods = 3
    total_area = sum(land_availability.values())
    
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
    positions = {farm: (i // side, i % side) for i, farm in enumerate(farms)}
    neighbor_edges = []
    for f1 in farms:
        distances = [(np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2), f2)
                    for f2 in farms if f1 != f2]
        distances.sort()
        for _, f2 in distances[:k_neighbors]:
            if (f2, f1) not in neighbor_edges:
                neighbor_edges.append((f1, f2))
    
    # Build model
    model = gp.Model("HardScenario")
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
    for f in farms:
        for c in families_list:
            for t in range(1, n_periods + 1):
                Y[(f, c, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}_t{t}")
    
    model.update()
    
    # Objective
    obj = 0
    for f in farms:
        farm_area = land_availability[f]
        for c in families_list:
            benefit = food_benefits[c]
            for t in range(1, n_periods + 1):
                obj += (benefit * farm_area * Y[(f, c, t)]) / total_area
    
    for f in farms:
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
    
    for f in farms:
        for t in range(1, n_periods + 1):
            crop_count = gp.quicksum(Y[(f, c, t)] for c in families_list)
            obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    
    for f in farms:
        for c in families_list:
            crop_used = gp.quicksum(Y[(f, c, t)] for t in range(1, n_periods + 1))
            obj += diversity_bonus * crop_used
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    for f in farms:
        for t in range(1, n_periods + 1):
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
print("TESTING REPLICATED HARD SCENARIOS")
print("All use rotation_medium_100 distribution pattern (seed=10001)")
print("="*80)

results = []
for n_farms, total_area in [(20, 100), (50, 250), (90, 450), (225, 1125)]:
    n_vars = n_farms * 6 * 3
    
    print(f"\n{'-'*80}")
    print(f"{n_farms} farms x 6 foods = {n_vars} variables")
    print(f"  Total area: {total_area:.0f} ha")
    print(f"  Running Gurobi (timeout=300s)...")
    print("-"*80)
    
    result = solve_scenario(n_farms, total_area, timeout=300)
    
    print(f"  Status: {result['status']}")
    print(f"  Time: {result['time']:.1f}s")
    print(f"  Objective: {result['objective']:.4f}")
    print(f"  Gap: {result['gap']:.1f}%")
    
    if result['status'] == 'timeout':
        print(f"  >>> TIMEOUT (consistently hard!)")
    else:
        print(f"  >>> SOLVED (may need adjustment)")
    
    results.append({
        'n_farms': n_farms,
        'n_vars': n_vars,
        **result
    })

print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)
print(f"{'Farms':<8} {'Vars':<6} {'Time(s)':<10} {'Status':<10}")
print("-"*80)
for r in results:
    print(f"{r['n_farms']:<8} {r['n_vars']:<6} {r['time']:<10.1f} {r['status']:<10}")

timeouts = sum(1 for r in results if r['status'] == 'timeout')
print(f"\n{'='*80}")
if timeouts == len(results):
    print("SUCCESS! All scenarios timeout - consistent hardness maintained!")
elif timeouts >= len(results) // 2:
    print(f"PARTIAL SUCCESS: {timeouts}/{len(results)} scenarios timeout")
else:
    print(f"WARNING: Only {timeouts}/{len(results)} scenarios timeout")
print("="*80)
