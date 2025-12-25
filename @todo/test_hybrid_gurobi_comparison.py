#!/usr/bin/env python3
"""Quick comparison of Adaptive Hybrid vs Gurobi."""
import sys
sys.path.insert(0, '..')
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from data_loader_utils import load_food_data_as_dict
from hybrid_formulation import solve_hybrid_adaptive
from food_grouping import aggregate_foods_to_families, create_family_rotation_matrix

# Load data - 10 farms for quick comparison
data = load_food_data_as_dict('rotation_250farms_27foods')
data['farm_names'] = data['farm_names'][:10]
data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
data['total_area'] = sum(data['land_availability'].values())

n_farms = len(data['farm_names'])
n_foods = len(data['food_names'])
print(f'Problem: {n_farms} farms x {n_foods} foods')
print()

# Aggregate to 6 families for fair comparison
family_data = aggregate_foods_to_families(data)
family_names = family_data['food_names']
n_families = len(family_names)
n_periods = 3

# =============================================
# GUROBI (Ground Truth) - 6 families
# =============================================
print('Running Gurobi on 6-family formulation...')
gurobi_start = time.time()

farm_names = family_data['farm_names']
family_benefits = family_data['food_benefits']
land_availability = family_data['land_availability']
total_area = family_data['total_area']

config = data.get('config', {})
params = config.get('parameters', {})
rotation_gamma = params.get('rotation_gamma', 0.2)
one_hot_penalty = params.get('one_hot_penalty', 3.0)
diversity_bonus = params.get('diversity_bonus', 0.15)

R = create_family_rotation_matrix(seed=42)

model = gp.Model()
model.setParam('OutputFlag', 0)
model.setParam('TimeLimit', 60)

# Variables
Y = {}
for farm in farm_names:
    for fam in family_names:
        for t in range(1, n_periods + 1):
            Y[(farm, fam, t)] = model.addVar(vtype=GRB.BINARY, name=f'Y_{farm}_{fam}_{t}')

# Objective
obj = 0

# Benefits
for farm in farm_names:
    area = land_availability[farm]
    for fam in family_names:
        benefit = family_benefits.get(fam, 0.5)
        for t in range(1, n_periods + 1):
            obj += (benefit * area / total_area) * Y[(farm, fam, t)]

# Rotation synergies
for farm in farm_names:
    area = land_availability[farm]
    for t in range(2, n_periods + 1):
        for i, fam1 in enumerate(family_names):
            for j, fam2 in enumerate(family_names):
                obj += (rotation_gamma * R[i, j] * area / total_area) * Y[(farm, fam1, t-1)] * Y[(farm, fam2, t)]

# One-hot penalty
for farm in farm_names:
    for t in range(1, n_periods + 1):
        for i, fam1 in enumerate(family_names):
            for fam2 in family_names[i+1:]:
                obj -= one_hot_penalty * Y[(farm, fam1, t)] * Y[(farm, fam2, t)]

# Diversity bonus
for farm in farm_names:
    area = land_availability[farm]
    for fam in family_names:
        for t in range(1, n_periods + 1):
            obj += (diversity_bonus / n_periods * area / total_area) * Y[(farm, fam, t)]

model.setObjective(obj, GRB.MAXIMIZE)

# Constraints: exactly 1 family per farm per period
for farm in farm_names:
    for t in range(1, n_periods + 1):
        model.addConstr(sum(Y[(farm, fam, t)] for fam in family_names) == 1)

model.optimize()
gurobi_time = time.time() - gurobi_start

gurobi_obj = model.ObjVal
gurobi_assigned = sum(1 for v in Y.values() if v.X > 0.5)

print(f'  Gurobi: obj={gurobi_obj:.4f}, assigned={gurobi_assigned}, time={gurobi_time:.2f}s')

# =============================================
# Adaptive Hybrid QPU
# =============================================
print()
print('Running Adaptive Hybrid QPU...')
result = solve_hybrid_adaptive(data, num_reads=100, num_iterations=2, verbose=False)

print(f'  Hybrid: obj={result["objective"]:.4f}, assigned={result["n_assigned"]}, time={result["solve_time"]:.2f}s, QPU={result["qpu_time"]:.3f}s')

# Gap calculation
gap = (gurobi_obj - result['objective']) / abs(gurobi_obj) * 100 if gurobi_obj != 0 else 0

print()
print('='*70)
print('COMPARISON (10 farms, 6-family formulation)')
print('='*70)
print(f'  Gurobi:  obj={gurobi_obj:.4f}, time={gurobi_time:.2f}s')
print(f'  Hybrid:  obj={result["objective"]:.4f}, time={result["solve_time"]:.2f}s, QPU={result["qpu_time"]:.3f}s')
print(f'  Gap:     {gap:.1f}%')
print(f'  Speedup: {gurobi_time / result["solve_time"]:.1f}x (wall time)')
print('='*70)
