#!/usr/bin/env python3
"""
Investigate what makes problems hard for Gurobi
Analyze relationship between problem characteristics and solve time
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
N_FAMILIES = 6

# Test many configurations to find patterns
TEST_CONFIGS = [
    # Vary farm counts
    {'n_farms': 3, 'scenario': 'rotation_micro_25'},
    {'n_farms': 5, 'scenario': 'rotation_micro_25'},
    {'n_farms': 8, 'scenario': 'rotation_small_50'},
    {'n_farms': 10, 'scenario': 'rotation_small_50'},
    {'n_farms': 15, 'scenario': 'rotation_medium_100'},
    {'n_farms': 20, 'scenario': 'rotation_medium_100'},
    {'n_farms': 25, 'scenario': 'rotation_large_200'},
    {'n_farms': 30, 'scenario': 'rotation_large_200'},
    {'n_farms': 40, 'scenario': 'rotation_large_200'},
    {'n_farms': 50, 'scenario': 'rotation_large_200'},
    {'n_farms': 75, 'scenario': 'rotation_large_200'},
    {'n_farms': 100, 'scenario': 'rotation_large_200'},
]

def analyze_problem_structure(land_availability, n_farms, n_families, n_periods, params):
    """Analyze structural properties of the problem"""
    total_area = sum(land_availability.values())
    areas = list(land_availability.values())
    
    # Basic stats
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)
    cv_area = std_area / mean_area if mean_area > 0 else 0
    
    # Constraint counts
    n_vars = n_farms * n_families * n_periods
    
    # One-hot constraints: 1 per farm per period
    n_onehot = n_farms * n_periods
    
    # Rotation constraints (temporal): farms × periods-1 × families^2 quadratic terms
    n_rotation = n_farms * (n_periods - 1) * n_families * n_families
    
    # Spatial constraints: depends on k-neighbors
    k = params.get('spatial_k_neighbors', 4)
    side = int(np.ceil(np.sqrt(n_farms)))
    # Approximate: each farm has ~k neighbors
    n_spatial_edges = (n_farms * k) // 2
    n_spatial = n_spatial_edges * n_periods * n_families * n_families
    
    # Total quadratic terms
    n_quadratic = n_rotation + n_spatial
    
    # Ratios
    farms_per_food = n_farms / n_families
    vars_per_farm = n_families * n_periods
    quadratic_per_var = n_quadratic / n_vars if n_vars > 0 else 0
    
    # Constraint density
    onehot_per_var = n_onehot / n_vars
    rotation_per_var = n_rotation / n_vars
    spatial_per_var = n_spatial / n_vars
    
    # Area-based metrics
    area_per_farm = total_area / n_farms
    area_per_var = total_area / n_vars
    
    return {
        'n_vars': n_vars,
        'n_farms': n_farms,
        'n_families': n_families,
        'n_periods': n_periods,
        'total_area': total_area,
        'mean_area': mean_area,
        'std_area': std_area,
        'min_area': min_area,
        'max_area': max_area,
        'cv_area': cv_area,
        'farms_per_food': farms_per_food,
        'vars_per_farm': vars_per_farm,
        'area_per_farm': area_per_farm,
        'area_per_var': area_per_var,
        'n_onehot': n_onehot,
        'n_rotation': n_rotation,
        'n_spatial': n_spatial,
        'n_quadratic': n_quadratic,
        'onehot_per_var': onehot_per_var,
        'rotation_per_var': rotation_per_var,
        'spatial_per_var': spatial_per_var,
        'quadratic_per_var': quadratic_per_var,
    }

def solve_gurobi(land_availability, food_benefits, params, timeout=100):
    """Solve with Gurobi and return detailed stats"""
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
    model = gp.Model("Hardness")
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
        'nodes': model.NodeCount if hasattr(model, 'NodeCount') else 0,
        'n_solutions': model.SolCount if hasattr(model, 'SolCount') else 0,
    }

print("="*100)
print("HARDNESS INVESTIGATION: What Makes Problems Hard?")
print("="*100)

results = []

for test_config in TEST_CONFIGS:
    n_farms = test_config['n_farms']
    scenario = test_config['scenario']
    n_vars = n_farms * N_FAMILIES * N_PERIODS
    
    print(f"\n{'-'*100}")
    print(f"Testing: {n_farms} farms ({n_vars} vars) from {scenario}")
    print(f"{'-'*100}")
    
    try:
        farms, foods, food_groups, config = load_food_data(scenario)
    except:
        print(f"  Skipping - scenario not available")
        continue
    
    params = config.get('parameters', {})
    land_availability_full = params.get('land_availability', {})
    weights = params.get('weights', {})
    
    # Get first n_farms
    all_farms = list(land_availability_full.keys())[:n_farms]
    land_availability = {f: land_availability_full[f] for f in all_farms}
    
    # Calculate food benefits
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
    
    # Analyze structure
    structure = analyze_problem_structure(land_availability, n_farms, N_FAMILIES, N_PERIODS, params)
    
    print(f"  Area: {structure['total_area']:.1f} ha (mean={structure['mean_area']:.2f}, CV={structure['cv_area']:.3f})")
    print(f"  Vars: {structure['n_vars']} | Quadratic: {structure['n_quadratic']} ({structure['quadratic_per_var']:.1f} per var)")
    print(f"  Farms/Food ratio: {structure['farms_per_food']:.2f}")
    
    # Solve
    print(f"  Solving...")
    result = solve_gurobi(land_availability, food_benefits, params, timeout=TIMEOUT)
    
    print(f"  Result: {result['status']} in {result['time']:.1f}s (gap={result['gap']:.1f}%, nodes={result['nodes']:.0f})")
    
    # Combine
    combined = {**structure, **result, 'scenario': scenario}
    results.append(combined)

# Analysis
print(f"\n{'=' * 100}")
print("ANALYSIS: CORRELATIONS WITH SOLVE TIME")
print("=" * 100)

# Separate hard vs easy
hard = [r for r in results if r['status'] == 'timeout' or r['time'] > 50]
easy = [r for r in results if r['status'] == 'optimal' and r['time'] <= 10]

print(f"\nHard problems ({len(hard)}): timeout or time > 50s")
print(f"Easy problems ({len(easy)}): optimal in <= 10s")

if hard and easy:
    print(f"\n{'Metric':<25} {'Hard (avg)':<15} {'Easy (avg)':<15} {'Ratio':<10}")
    print("-"*100)
    
    metrics = [
        'n_vars', 'n_farms', 'total_area', 'mean_area', 'cv_area',
        'farms_per_food', 'area_per_farm', 'area_per_var',
        'n_quadratic', 'quadratic_per_var',
        'onehot_per_var', 'rotation_per_var', 'spatial_per_var'
    ]
    
    for metric in metrics:
        hard_avg = np.mean([r[metric] for r in hard])
        easy_avg = np.mean([r[metric] for r in easy])
        ratio = hard_avg / easy_avg if easy_avg != 0 else 0
        
        print(f"{metric:<25} {hard_avg:<15.3f} {easy_avg:<15.3f} {ratio:<10.2f}x")

# Detailed breakdown
print(f"\n{'=' * 100}")
print("DETAILED BREAKDOWN")
print("=" * 100)

print(f"\n{'Farms':<7} {'Vars':<7} {'Area':<8} {'CV':<8} {'Farm/Food':<10} {'Time':<8} {'Status':<10}")
print("-"*100)

for r in sorted(results, key=lambda x: x['time'], reverse=True):
    print(f"{r['n_farms']:<7} {r['n_vars']:<7} {r['total_area']:<8.1f} {r['cv_area']:<8.3f} "
          f"{r['farms_per_food']:<10.2f} {r['time']:<8.1f} {r['status']:<10}")

# Key insights
print(f"\n{'=' * 100}")
print("KEY INSIGHTS")
print("=" * 100)

if hard and easy:
    # Area analysis
    hard_areas = [r['total_area'] for r in hard]
    easy_areas = [r['total_area'] for r in easy]
    
    print(f"\n1. TOTAL AREA:")
    print(f"   Hard problems: mean={np.mean(hard_areas):.1f} ha, range=[{min(hard_areas):.1f}, {max(hard_areas):.1f}]")
    print(f"   Easy problems: mean={np.mean(easy_areas):.1f} ha, range=[{min(easy_areas):.1f}, {max(easy_areas):.1f}]")
    
    # CV analysis
    hard_cv = [r['cv_area'] for r in hard]
    easy_cv = [r['cv_area'] for r in easy]
    
    print(f"\n2. COEFFICIENT OF VARIATION (land area):")
    print(f"   Hard problems: mean CV={np.mean(hard_cv):.3f}")
    print(f"   Easy problems: mean CV={np.mean(easy_cv):.3f}")
    
    # Farms/food ratio
    hard_ratio = [r['farms_per_food'] for r in hard]
    easy_ratio = [r['farms_per_food'] for r in easy]
    
    print(f"\n3. FARMS PER FOOD RATIO:")
    print(f"   Hard problems: mean={np.mean(hard_ratio):.2f}, range=[{min(hard_ratio):.1f}, {max(hard_ratio):.1f}]")
    print(f"   Easy problems: mean={np.mean(easy_ratio):.2f}, range=[{min(easy_ratio):.1f}, {max(easy_ratio):.1f}]")
    
    # Area per var
    hard_apv = [r['area_per_var'] for r in hard]
    easy_apv = [r['area_per_var'] for r in easy]
    
    print(f"\n4. AREA PER VARIABLE:")
    print(f"   Hard problems: mean={np.mean(hard_apv):.3f} ha/var")
    print(f"   Easy problems: mean={np.mean(easy_apv):.3f} ha/var")
    
    # Quadratic density
    hard_quad = [r['quadratic_per_var'] for r in hard]
    easy_quad = [r['quadratic_per_var'] for r in easy]
    
    print(f"\n5. QUADRATIC TERMS PER VARIABLE:")
    print(f"   Hard problems: mean={np.mean(hard_quad):.1f}")
    print(f"   Easy problems: mean={np.mean(easy_quad):.1f}")

print(f"\n{'=' * 100}")
print("CONCLUSION")
print("=" * 100)

if hard and easy:
    hard_area_avg = np.mean([r['total_area'] for r in hard])
    easy_area_avg = np.mean([r['total_area'] for r in easy])
    
    if hard_area_avg < easy_area_avg * 0.7:
        print("\nPRIMARY FACTOR: SMALL TOTAL AREA makes problems HARD!")
        print(f"Hard problems have {hard_area_avg:.1f} ha average")
        print(f"Easy problems have {easy_area_avg:.1f} ha average")
        print(f"\nHypothesis: Small area creates tight constraints and poor LP relaxation")
    
    hard_cv_avg = np.mean([r['cv_area'] for r in hard])
    easy_cv_avg = np.mean([r['cv_area'] for r in easy])
    
    if hard_cv_avg > easy_cv_avg * 1.3:
        print("\nSECONDARY FACTOR: HIGH VARIABILITY in land areas makes problems HARD!")
        print(f"Hard problems have CV={hard_cv_avg:.3f}")
        print(f"Easy problems have CV={easy_cv_avg:.3f}")
        print(f"\nHypothesis: High variance creates asymmetric problem structure")

print("="*100)
