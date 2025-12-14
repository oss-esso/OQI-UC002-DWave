#!/usr/bin/env python3
"""
Quick hardness analysis - 30s timeout for faster results
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

TIMEOUT = 30  # Faster timeout
N_PERIODS = 3
N_FAMILIES = 6

# Key test points
TESTS = [
    {'n_farms': 5, 'scenario': 'rotation_micro_25', 'name': '5f'},
    {'n_farms': 10, 'scenario': 'rotation_small_50', 'name': '10f'},
    {'n_farms': 20, 'scenario': 'rotation_medium_100', 'name': '20f'},
    {'n_farms': 30, 'scenario': 'rotation_large_200', 'name': '30f'},
    {'n_farms': 50, 'scenario': 'rotation_large_200', 'name': '50f'},
    {'n_farms': 100, 'scenario': 'rotation_large_200', 'name': '100f'},
]

def solve_quick(land_avail, food_benefits, params, timeout=30):
    """Quick solve"""
    farms = list(land_avail.keys())
    n_farms = len(farms)
    foods = list(food_benefits.keys())
    n_foods = len(foods)
    total_area = sum(land_avail.values())
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.1)
    
    # Variables
    Y = {}
    for f in farms:
        for c in foods:
            for t in range(1, N_PERIODS + 1):
                Y[(f, c, t)] = model.addVar(vtype=GRB.BINARY)
    
    model.update()
    
    # Simple objective
    obj = 0
    for f in farms:
        area = land_avail[f]
        for c in foods:
            for t in range(1, N_PERIODS + 1):
                obj += food_benefits[c] * area * Y[(f, c, t)] / total_area
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    for f in farms:
        for t in range(1, N_PERIODS + 1):
            model.addConstr(gp.quicksum(Y[(f, c, t)] for c in foods) <= 2)
    
    start = time.time()
    model.optimize()
    elapsed = time.time() - start
    
    status = 'optimal' if model.Status == GRB.OPTIMAL else ('timeout' if model.Status == GRB.TIME_LIMIT else 'other')
    gap = model.MIPGap * 100 if hasattr(model, 'MIPGap') and model.SolCount > 0 else 0
    
    return {'time': elapsed, 'status': status, 'gap': gap}

print("="*90)
print("QUICK HARDNESS ANALYSIS")
print("="*90)

results = []
for test in TESTS:
    n_farms = test['n_farms']
    scenario = test['scenario']
    name = test['name']
    
    print(f"\n{name}: {n_farms} farms from {scenario}")
    
    farms, foods, food_groups, config = load_food_data(scenario)
    params = config.get('parameters', {})
    land_full = params.get('land_availability', {})
    weights = params.get('weights', {})
    
    # Get subset
    all_farms = list(land_full.keys())[:n_farms]
    land_avail = {f: land_full[f] for f in all_farms}
    
    # Calculate benefits
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
    
    # Analyze
    total_area = sum(land_avail.values())
    areas = list(land_avail.values())
    mean_area = np.mean(areas)
    cv_area = np.std(areas) / mean_area if mean_area > 0 else 0
    
    n_vars = n_farms * N_FAMILIES * N_PERIODS
    farms_per_food = n_farms / N_FAMILIES
    area_per_var = total_area / n_vars
    
    print(f"  Area: {total_area:.1f} ha (CV={cv_area:.3f}), Area/var={area_per_var:.3f}")
    print(f"  Farms/Food={farms_per_food:.2f}, Vars={n_vars}")
    
    # Solve
    result = solve_quick(land_avail, food_benefits, params, timeout=TIMEOUT)
    
    print(f"  Result: {result['status']} in {result['time']:.1f}s (gap={result['gap']:.1f}%)")
    
    results.append({
        'name': name,
        'n_farms': n_farms,
        'n_vars': n_vars,
        'total_area': total_area,
        'cv_area': cv_area,
        'farms_per_food': farms_per_food,
        'area_per_var': area_per_var,
        **result
    })

# Analysis
print(f"\n{'='*90}")
print("ANALYSIS")
print("="*90)

hard = [r for r in results if r['status'] != 'optimal' or r['time'] > 10]
easy = [r for r in results if r['status'] == 'optimal' and r['time'] <= 5]

print(f"\nHard ({len(hard)}): {[r['name'] for r in hard]}")
print(f"Easy ({len(easy)}): {[r['name'] for r in easy]}")

if hard and easy:
    print(f"\n{'Metric':<20} {'Hard':<12} {'Easy':<12} {'Ratio':<10}")
    print("-"*90)
    
    for metric in ['total_area', 'cv_area', 'farms_per_food', 'area_per_var']:
        hard_avg = np.mean([r[metric] for r in hard])
        easy_avg = np.mean([r[metric] for r in easy])
        ratio = hard_avg / easy_avg if easy_avg != 0 else 0
        print(f"{metric:<20} {hard_avg:<12.3f} {easy_avg:<12.3f} {ratio:<10.2f}x")

print(f"\n{'='*90}")
print("SUMMARY")
print("="*90)

print(f"\n{'Name':<10} {'Farms':<7} {'Area':<8} {'CV':<8} {'F/Food':<8} {'Time':<8} {'Status'}")
print("-"*90)
for r in sorted(results, key=lambda x: x['time'], reverse=True):
    print(f"{r['name']:<10} {r['n_farms']:<7} {r['total_area']:<8.1f} {r['cv_area']:<8.3f} "
          f"{r['farms_per_food']:<8.2f} {r['time']:<8.1f} {r['status']}")

print("="*90)
