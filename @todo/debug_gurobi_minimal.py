#!/usr/bin/env python3
"""
Debug script to create a minimal test case matching both test configurations.
This will help identify if there's a data-related issue.
"""
import sys
import os
import numpy as np
import time

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: Gurobi not available!")
    sys.exit(1)

def build_test_problem(n_farms=5, n_foods=6, seed=42):
    """Build a test problem matching both tests' specifications."""
    
    # Generate deterministic data
    np.random.seed(seed)
    
    # Farm names and land
    farm_names = [f'Farm{i+1}' for i in range(n_farms)]
    land_availability = {f: np.random.uniform(15, 35) for f in farm_names}
    total_area = sum(land_availability.values())
    
    # Food names and benefits
    food_names = ['Fruits', 'Grains', 'Legumes', 'Leafy_Vegetables', 'Root_Vegetables', 'Proteins'][:n_foods]
    food_benefits = {f: np.random.uniform(0.3, 0.7) for f in food_names}
    
    # Rotation matrix (matching both tests)
    np.random.seed(42)  # Reset seed for matrix generation
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
    
    print(f"Generated problem: {n_farms} farms × {n_foods} foods × 3 periods = {n_farms * n_foods * 3} variables")
    print(f"  Total area: {total_area:.2f}")
    print(f"  Rotation matrix: {(R < 0).sum()}/{R.size} negative ({100*(R < 0).sum()/R.size:.1f}%)")
    
    return {
        'farm_names': farm_names,
        'food_names': food_names,
        'land_availability': land_availability,
        'food_benefits': food_benefits,
        'total_area': total_area,
        'R': R,
    }

def solve_with_gurobi(data, timeout=300):
    """Solve using Gurobi with EXACT settings from both tests."""
    
    farm_names = data['farm_names']
    food_names = data['food_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    R = data['R']
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    N_PERIODS = 3
    
    print(f"\nBuilding Gurobi model...")
    start_time = time.time()
    
    # Model with exact parameters from both tests
    model = gp.Model("DebugTest")
    model.setParam('OutputFlag', 1)  # ENABLE OUTPUT TO SEE WHAT'S HAPPENING
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
    print(f"  Variables: {model.NumVars}")
    
    # Objective
    obj = 0
    
    # Base benefit
    for f in farm_names:
        farm_area = land_availability[f]
        for c in food_names:
            benefit = food_benefits.get(c, 0.5)
            for t in range(1, N_PERIODS + 1):
                obj += (benefit * farm_area * Y[(f, c, t)]) / total_area
    
    # Rotation synergies (QUADRATIC TERMS)
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
    
    # Spatial interactions (QUADRATIC TERMS)
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, N_PERIODS + 1):
            for i, c1 in enumerate(food_names):
                for j, c2 in enumerate(food_names):
                    spatial_synergy = R[i, j] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_area
    
    # Soft one-hot penalty (QUADRATIC TERMS)
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
    print(f"  Objective: Built with quadratic terms")
    
    # Constraints
    for f in farm_names:
        for t in range(1, N_PERIODS + 1):
            model.addConstr(
                gp.quicksum(Y[(f, c, t)] for c in food_names) <= 2,
                name=f"max_crops_{f}_t{t}"
            )
    
    print(f"  Constraints: {model.NumConstrs}")
    build_time = time.time() - start_time
    print(f"  Build time: {build_time:.3f}s")
    
    # Count quadratic terms
    print(f"\n  Quadratic terms breakdown:")
    n_rotation = n_farms * (N_PERIODS - 1) * n_foods * n_foods
    n_spatial = len(neighbor_edges) * N_PERIODS * n_foods * n_foods
    n_one_hot = n_farms * N_PERIODS
    print(f"    Rotation synergies: ~{n_rotation} potential terms")
    print(f"    Spatial interactions: ~{n_spatial} potential terms")
    print(f"    One-hot penalties: {n_one_hot} quadratic expressions")
    print(f"    TOTAL: ~{n_rotation + n_spatial} quadratic variable pairs")
    
    # Solve
    print(f"\nSolving with Gurobi...")
    print(f"  Timeout: {timeout}s")
    print(f"  Parameters: Threads=0, Presolve=2, Cuts=2, MIPGap=10%")
    print("="*80)
    
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    total_time = time.time() - start_time
    
    print("="*80)
    print(f"\nResults:")
    print(f"  Status: {model.Status} ({['LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED', 
                                        'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT', 'TIME_LIMIT', 
                                        'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL'][model.Status]})")
    print(f"  Solve time: {solve_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        print(f"  Objective: {model.ObjVal:.6f}")
        print(f"  MIP gap: {model.MIPGap * 100:.2f}%")
        print(f"  Solutions found: {model.SolCount}")
    else:
        print(f"  No solution found!")
    
    return {
        'status': model.Status,
        'solve_time': solve_time,
        'total_time': total_time,
        'objective': model.ObjVal if model.SolCount > 0 else 0,
    }

if __name__ == '__main__':
    print("="*80)
    print("MINIMAL GUROBI TEST - MATCHING BOTH TEST CONFIGURATIONS")
    print("="*80)
    
    # Test 1: 5 farms (90 vars) - Statistical test scale
    print("\n\n" + "="*80)
    print("TEST 1: 5 farms × 6 foods × 3 periods = 90 variables")
    print("="*80)
    data = build_test_problem(n_farms=5, n_foods=6, seed=42)
    result = solve_with_gurobi(data, timeout=300)
    
    # Test 2: 20 farms (360 vars) - Comprehensive test scale
    print("\n\n" + "="*80)
    print("TEST 2: 20 farms × 6 foods × 3 periods = 360 variables")
    print("="*80)
    data = build_test_problem(n_farms=20, n_foods=6, seed=42)
    result = solve_with_gurobi(data, timeout=300)
