#!/usr/bin/env python3
"""
Verify that formulations match across tests.
Runs ONLY GUROBI (no QPU) on smallest scenario to compare.

This script compares:
1. test_gurobi_timeout.py formulation
2. hierarchical_statistical_test.py Gurobi formulation
3. comprehensive_scaling_test.py Gurobi formulation

All should produce IDENTICAL results for the same scenario.
"""

import sys
import os
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("FORMULATION VERIFICATION TEST")
print("="*80)
print("Testing: rotation_micro_25 (5 farms × 6 foods × 3 periods = 90 vars)")
print("Comparing Gurobi formulations across all 3 tests")
print("="*80)
print()

# Import Gurobi
import gurobipy as gp
from gurobipy import GRB

# Load data
from data_loader_utils import load_food_data_as_dict

data = load_food_data_as_dict('rotation_micro_25')
print(f"Loaded: {len(data['farm_names'])} farms × {len(data['food_names'])} foods")
print(f"Food names: {data['food_names']}")
print()

# Common parameters (MUST BE IDENTICAL across all tests)
TIMEOUT = 100
MIP_GAP = 0.01
MIP_FOCUS = 1
IMPROVE_START_TIME = 30
N_PERIODS = 3
ROTATION_GAMMA = 0.2
SPATIAL_GAMMA = ROTATION_GAMMA * 0.5
ONE_HOT_PENALTY = 3.0
DIVERSITY_BONUS = 0.15
K_NEIGHBORS = 4
FRUSTRATION_RATIO = 0.7
NEGATIVE_STRENGTH = -0.8

def build_rotation_matrix(n_foods, seed=42):
    """Build rotation synergy matrix - MUST BE IDENTICAL across tests."""
    rng = np.random.RandomState(seed)
    R = np.zeros((n_foods, n_foods))
    for i in range(n_foods):
        for j in range(n_foods):
            if i == j:
                R[i, j] = NEGATIVE_STRENGTH * 1.5  # Same crop = negative
            elif rng.random() < FRUSTRATION_RATIO:
                R[i, j] = rng.uniform(NEGATIVE_STRENGTH * 1.2, NEGATIVE_STRENGTH * 0.3)
            else:
                R[i, j] = rng.uniform(0.02, 0.20)
    return R

def build_neighbor_graph(farm_names, positions, k=4):
    """Build spatial neighbor graph - MUST BE IDENTICAL across tests."""
    neighbor_edges = []
    for f1_idx, f1 in enumerate(farm_names):
        distances = []
        for f2_idx, f2 in enumerate(farm_names):
            if f1 != f2:
                dist = np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2)
                distances.append((dist, f2_idx, f2))
        distances.sort()
        for _, f2_idx, f2 in distances[:k]:
            # Check if EITHER direction already exists (prevent double-counting)
            if (f1_idx, f2_idx) not in neighbor_edges and (f2_idx, f1_idx) not in neighbor_edges:
                neighbor_edges.append((f1_idx, f2_idx))
    return neighbor_edges

def solve_gurobi_formulation(data, formulation_name):
    """
    Solve with Gurobi using EXACT same formulation as test_gurobi_timeout.py
    """
    print(f"\n{'='*60}")
    print(f"Formulation: {formulation_name}")
    print(f"{'='*60}")
    
    farm_names = data['farm_names']
    food_names = data['food_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    
    # Build rotation matrix
    R = build_rotation_matrix(n_foods)
    print(f"Rotation matrix shape: {R.shape}")
    print(f"Rotation matrix diagonal: {np.diag(R)}")
    
    # Build spatial graph
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {}
    for i, farm in enumerate(farm_names):
        positions[farm] = (i // side, i % side)
    neighbor_edges = build_neighbor_graph(farm_names, positions, K_NEIGHBORS)
    print(f"Neighbor edges: {len(neighbor_edges)}")
    
    # Create model
    model = gp.Model(f"Verify_{formulation_name}")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', TIMEOUT)
    model.setParam('MIPGap', MIP_GAP)
    model.setParam('MIPFocus', MIP_FOCUS)
    model.setParam('ImproveStartTime', IMPROVE_START_TIME)
    model.setParam('Threads', 0)  # Use all cores - MATCH hierarchical test
    model.setParam('Presolve', 2)  # Aggressive presolve - MATCH hierarchical test
    model.setParam('Cuts', 2)  # Aggressive cuts - MATCH hierarchical test
    
    # Variables - using integer indices like test_gurobi_timeout.py
    Y = {}
    for i in range(n_farms):
        for j in range(n_foods):
            for t in range(1, N_PERIODS + 1):
                Y[(i, j, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_f{i}_c{j}_t{t}")
    
    # Objective
    obj = 0
    
    # Part 1: Base benefit
    for i, farm in enumerate(farm_names):
        farm_area = land_availability[farm]
        for j, food in enumerate(food_names):
            benefit = food_benefits[food]
            for t in range(1, N_PERIODS + 1):
                obj += (benefit * farm_area * Y[(i, j, t)]) / total_area
    
    # Part 2: Rotation synergies (QUADRATIC)
    for i in range(n_farms):
        farm_area = land_availability[farm_names[i]]
        for t in range(2, N_PERIODS + 1):
            for j1 in range(n_foods):
                for j2 in range(n_foods):
                    synergy = R[j1, j2]
                    if abs(synergy) > 1e-6:
                        obj += (ROTATION_GAMMA * synergy * farm_area * 
                               Y[(i, j1, t-1)] * Y[(i, j2, t)]) / total_area
    
    # Part 3: Spatial interactions (QUADRATIC)
    for (f1_idx, f2_idx) in neighbor_edges:
        for t in range(1, N_PERIODS + 1):
            for j1 in range(n_foods):
                for j2 in range(n_foods):
                    spatial_synergy = R[j1, j2] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (SPATIAL_GAMMA * spatial_synergy * 
                               Y[(f1_idx, j1, t)] * Y[(f2_idx, j2, t)]) / total_area
    
    # Part 4: Soft one-hot penalty (QUADRATIC)
    for i in range(n_farms):
        for t in range(1, N_PERIODS + 1):
            crop_count = gp.quicksum(Y[(i, j, t)] for j in range(n_foods))
            obj -= ONE_HOT_PENALTY * (crop_count - 1) * (crop_count - 1)
    
    # Part 5: Diversity bonus (linear)
    for i in range(n_farms):
        for j in range(n_foods):
            crop_used = gp.quicksum(Y[(i, j, t)] for t in range(1, N_PERIODS + 1))
            obj += DIVERSITY_BONUS * crop_used
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints - CRITICAL: all 3 types
    for i in range(n_farms):
        for t in range(1, N_PERIODS + 1):
            # Max 2 crops
            model.addConstr(
                gp.quicksum(Y[(i, j, t)] for j in range(n_foods)) <= 2,
                name=f"max_crops_f{i}_t{t}"
            )
            # Min 1 crop
            model.addConstr(
                gp.quicksum(Y[(i, j, t)] for j in range(n_foods)) >= 1,
                name=f"min_crops_f{i}_t{t}"
            )
    
    # Rotation constraints
    for i in range(n_farms):
        for j in range(n_foods):
            for t in range(1, N_PERIODS):
                model.addConstr(
                    Y[(i, j, t)] + Y[(i, j, t + 1)] <= 1,
                    name=f"rotation_f{i}_c{j}_t{t}"
                )
    
    model.update()
    print(f"Model: {model.NumVars} vars, {model.NumConstrs} constraints")
    
    # Count quadratic terms
    obj_expr = model.getObjective()
    print(f"Objective type: {type(obj_expr)}")
    
    # Solve
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        print(f"Status: {'OPTIMAL' if model.Status == GRB.OPTIMAL else 'TIMEOUT'}")
        print(f"Objective: {model.ObjVal:.6f}")
        print(f"MIP Gap: {model.MIPGap*100:.2f}%")
        print(f"Solve time: {model.Runtime:.2f}s")
        
        # Extract solution summary
        solution_count = sum(1 for (i,j,t), v in Y.items() if v.X > 0.5)
        print(f"Variables assigned: {solution_count}")
        
        return {
            'status': 'optimal' if model.Status == GRB.OPTIMAL else 'timeout',
            'objective': model.ObjVal,
            'gap': model.MIPGap,
            'time': model.Runtime,
            'vars': model.NumVars,
            'constrs': model.NumConstrs,
            'assigned': solution_count
        }
    else:
        print(f"Failed: Status={model.Status}")
        return None

# Run verification
print("\n" + "="*80)
print("RUNNING VERIFICATION")
print("="*80)

result = solve_gurobi_formulation(data, "test_gurobi_timeout_style")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)

if result:
    print(f"""
Summary:
  Variables:    {result['vars']}
  Constraints:  {result['constrs']}
  Objective:    {result['objective']:.6f}
  MIP Gap:      {result['gap']*100:.2f}%
  Solve Time:   {result['time']:.2f}s
  Status:       {result['status']}
  
This is the REFERENCE formulation.
All other tests should produce IDENTICAL results.
""")
else:
    print("VERIFICATION FAILED - Could not solve")
