#!/usr/bin/env python3
"""
Test using EXACT data loading from statistical_comparison_test.py
"""
import sys
import os
import numpy as np
import time

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.scenarios import load_food_data
import gurobipy as gp
from gurobipy import GRB

def load_rotation_data(n_farms: int):
    """EXACT copy from statistical_comparison_test.py"""
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
        if base_scenario is None:
            base_scenario = 'rotation_large_200'
    
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
        'foods': foods,
        'food_names': food_names,
        'food_groups': food_groups,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': land_availability,
        'farm_names': farm_names,
        'total_area': total_area,
        'n_farms': n_farms,
        'n_foods': len(food_names),
        'config': config,
    }

def solve_ground_truth(data, timeout=60):
    """EXACT copy from statistical_comparison_test.py (abbreviated for testing)"""
    start_time = time.time()
    
    food_names = data['food_names']
    farm_names = data['farm_names']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    config = data.get('config', {})
    params = config.get('parameters', {})
    
    rotation_gamma = params.get('rotation_gamma', 0.2)
    k_neighbors = params.get('spatial_k_neighbors', 4)
    frustration_ratio = params.get('frustration_ratio', 0.7)
    negative_strength = params.get('negative_synergy_strength', -0.8)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    use_soft_constraint = params.get('use_soft_one_hot', True)
    
    n_periods = 3
    n_farms = len(farm_names)
    n_families = len(food_names)
    families_list = list(food_names)
    
    print(f"Parameters from config:")
    print(f"  rotation_gamma: {rotation_gamma}")
    print(f"  frustration_ratio: {frustration_ratio}")
    print(f"  negative_strength: {negative_strength}")
    print(f"  one_hot_penalty: {one_hot_penalty}")
    print(f"  k_neighbors: {k_neighbors}")
    
    # Create rotation matrix
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
    
    print(f"Rotation matrix: {(R < 0).sum()}/{R.size} negative ({100*(R < 0).sum()/R.size:.1f}%)")
    
    # Create spatial graph
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {}
    for i, farm in enumerate(farm_names):
        row, col = i // side, i % side
        positions[farm] = (row, col)
    
    neighbor_edges = []
    for f1 in farm_names:
        distances = []
        for f2 in farm_names:
            if f1 != f2:
                dist = np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2)
                distances.append((dist, f2))
        distances.sort()
        for _, f2 in distances[:k_neighbors]:
            if (f2, f1) not in neighbor_edges:
                neighbor_edges.append((f1, f2))
    
    print(f"Spatial graph: {len(neighbor_edges)} edges")
    
    # Build Gurobi model
    model = gp.Model("StatisticalTestExact")
    model.setParam('OutputFlag', 1)
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
    
    # Part 1: Base benefit
    for f in farm_names:
        farm_area = land_availability[f]
        for c in families_list:
            benefit = data['food_benefits'].get(c, 0.5)
            for t in range(1, n_periods + 1):
                obj += (benefit * farm_area * Y[(f, c, t)]) / total_area
    
    # Part 2: Rotation synergies
    for f in farm_names:
        farm_area = land_availability[f]
        for t in range(2, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(f, c1, t-1)] * Y[(f, c2, t)]) / total_area
    
    # Part 3: Spatial interactions
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for c1_idx, c1 in enumerate(families_list):
                for c2_idx, c2 in enumerate(families_list):
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_area
    
    # Part 4: Soft one-hot penalty
    if use_soft_constraint:
        for f in farm_names:
            for t in range(1, n_periods + 1):
                crop_count = gp.quicksum(Y[(f, c, t)] for c in families_list)
                obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    
    # Part 5: Diversity bonus
    for f in farm_names:
        for c in families_list:
            crop_used = gp.quicksum(Y[(f, c, t)] for t in range(1, n_periods + 1))
            obj += diversity_bonus * crop_used
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    if use_soft_constraint:
        for f in farm_names:
            for t in range(1, n_periods + 1):
                model.addConstr(
                    gp.quicksum(Y[(f, c, t)] for c in families_list) <= 2,
                    name=f"max_crops_{f}_t{t}"
                )
    
    print(f"Model: {model.NumVars} vars, {model.NumConstrs} constraints")
    print(f"Solving with timeout={timeout}s...")
    print("="*80)
    
    # Solve
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    
    print("="*80)
    print(f"Status: {model.Status}")
    print(f"Solve time: {solve_time:.2f}s")
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        print(f"Objective: {model.ObjVal:.6f}")
        print(f"MIP gap: {model.MIPGap * 100:.2f}%")
    
    return solve_time

if __name__ == '__main__':
    print("="*80)
    print("TESTING WITH EXACT STATISTICAL TEST DATA LOADING")
    print("="*80)
    
    print("\nTest 1: 5 farms (90 vars) - from rotation_micro_25 scenario")
    print("-"*80)
    data = load_rotation_data(5)
    print(f"Loaded: {data['n_farms']} farms, {data['n_foods']} foods")
    print(f"Farm names: {data['farm_names']}")
    print(f"Food names: {data['food_names']}")
    solve_time = solve_ground_truth(data, timeout=60)
    
    if solve_time >= 59:
        print("\n⚠️ TIMEOUT REPRODUCED!")
    else:
        print(f"\n✅ Solved quickly in {solve_time:.2f}s")
