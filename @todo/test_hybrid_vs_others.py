#!/usr/bin/env python3
"""
Test Hybrid QPU Solver vs Other Methods

Compares:
1. Gurobi (ground truth)
2. Hybrid QPU (27 foods, 6-family synergies, spatial decomposition)
3. Clique decomposition (existing method)

All using the SAME formulation and constraints.

Author: OQI-UC002-DWave
Date: 2025-12-24
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("HYBRID QPU SOLVER TEST: Compare with Gurobi and Other Methods")
print("="*80)
print()

# Import required modules
from data_loader_utils import load_food_data_as_dict
from hybrid_formulation import solve_hybrid_qpu, build_hybrid_rotation_matrix

# Gurobi
import gurobipy as gp
from gurobipy import GRB

# D-Wave
from dimod import BinaryQuadraticModel
from dwave.system import DWaveCliqueSampler

# Config
DEFAULT_DWAVE_TOKEN = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'
os.environ['DWAVE_API_TOKEN'] = os.environ.get('DWAVE_API_TOKEN', DEFAULT_DWAVE_TOKEN)

# Test scenarios - start small
TEST_SCENARIOS = [
    {'name': '5 farms × 27 foods', 'n_farms': 5, 'n_foods': 27, 'scenario': 'rotation_250farms_27foods'},
    # {'name': '10 farms × 27 foods', 'n_farms': 10, 'n_foods': 27, 'scenario': 'rotation_250farms_27foods'},
]

GUROBI_TIMEOUT = 100
N_PERIODS = 3


def load_test_data(scenario: dict) -> dict:
    """Load data for test scenario."""
    data = load_food_data_as_dict(scenario['scenario'])
    
    # Trim to requested farm count
    if len(data['farm_names']) > scenario['n_farms']:
        data['farm_names'] = data['farm_names'][:scenario['n_farms']]
        data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
    
    # Trim to requested food count
    if len(data['food_names']) > scenario['n_foods']:
        data['food_names'] = data['food_names'][:scenario['n_foods']]
        data['food_benefits'] = {k: v for k, v in list(data['food_benefits'].items())[:scenario['n_foods']]}
    
    data['total_area'] = sum(data['land_availability'].values())
    
    # Add default parameters
    if 'config' not in data:
        data['config'] = {}
    if 'parameters' not in data['config']:
        data['config']['parameters'] = {}
    
    params = data['config']['parameters']
    params.setdefault('rotation_gamma', 0.2)
    params.setdefault('spatial_k_neighbors', 4)
    params.setdefault('frustration_ratio', 0.7)
    params.setdefault('negative_synergy_strength', -0.8)
    params.setdefault('one_hot_penalty', 3.0)
    params.setdefault('diversity_bonus', 0.15)
    
    return data


def solve_gurobi(data: dict, timeout: int = 100) -> dict:
    """Solve with Gurobi (ground truth) - SAME formulation as hybrid."""
    total_start = time.time()
    
    farm_names = data['farm_names']
    food_names = data['food_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    params = data['config']['parameters']
    rotation_gamma = params['rotation_gamma']
    k_neighbors = params['spatial_k_neighbors']
    frustration_ratio = params['frustration_ratio']
    negative_strength = params['negative_synergy_strength']
    one_hot_penalty = params['one_hot_penalty']
    diversity_bonus = params['diversity_bonus']
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_periods = N_PERIODS
    
    # Build rotation matrix (SAME as hybrid)
    R = build_hybrid_rotation_matrix(food_names, frustration_ratio, negative_strength, seed=42)
    
    # Spatial neighbor graph
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {}
    for i, farm in enumerate(farm_names):
        positions[farm] = (i // side, i % side)
    
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
    
    # Build Gurobi model
    model = gp.Model("HybridGroundTruth")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.01)
    model.setParam('MIPFocus', 1)
    
    # Variables
    Y = {}
    for f in farm_names:
        for c in food_names:
            for t in range(1, n_periods + 1):
                Y[(f, c, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{f}_{c}_t{t}")
    model.update()
    
    # Objective
    obj = 0
    
    # Part 1: Base benefit
    for f in farm_names:
        farm_area = land_availability[f]
        for c_idx, c in enumerate(food_names):
            benefit = food_benefits.get(c, 0.5)
            for t in range(1, n_periods + 1):
                obj += (benefit * farm_area * Y[(f, c, t)]) / total_area
    
    # Part 2: Rotation synergies (temporal)
    for f in farm_names:
        farm_area = land_availability[f]
        for t in range(2, n_periods + 1):
            for c1_idx, c1 in enumerate(food_names):
                for c2_idx, c2 in enumerate(food_names):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * farm_area * 
                               Y[(f, c1, t-1)] * Y[(f, c2, t)]) / total_area
    
    # Part 3: Spatial interactions
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for c1_idx, c1 in enumerate(food_names):
                for c2_idx, c2 in enumerate(food_names):
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               Y[(f1, c1, t)] * Y[(f2, c2, t)]) / total_area
    
    # Part 4: Soft one-hot penalty
    for f in farm_names:
        for t in range(1, n_periods + 1):
            crop_count = gp.quicksum(Y[(f, c, t)] for c in food_names)
            obj -= one_hot_penalty * (crop_count - 1) * (crop_count - 1)
    
    # Part 5: Diversity bonus
    for f in farm_names:
        for c in food_names:
            crop_used = gp.quicksum(Y[(f, c, t)] for t in range(1, n_periods + 1))
            obj += diversity_bonus * crop_used
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints: max 2 crops per farm per period
    for f in farm_names:
        for t in range(1, n_periods + 1):
            model.addConstr(gp.quicksum(Y[(f, c, t)] for c in food_names) <= 2)
    
    # Solve
    model.optimize()
    solve_time = time.time() - total_start
    
    result = {
        'method': 'gurobi',
        'success': False,
        'objective': 0,
        'solve_time': solve_time,
        'violations': {'total': 0},
        'n_assigned': 0,
    }
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        result['success'] = True
        result['objective'] = model.ObjVal
        result['status'] = 'optimal' if model.Status == GRB.OPTIMAL else 'timeout'
        result['mip_gap'] = model.MIPGap * 100 if hasattr(model, 'MIPGap') else 0
        
        # Count assignments
        n_assigned = 0
        assignments_per_farm_period = {}
        for f in farm_names:
            for t in range(1, n_periods + 1):
                count = 0
                for c in food_names:
                    if Y[(f, c, t)].X > 0.5:
                        n_assigned += 1
                        count += 1
                assignments_per_farm_period[(f, t)] = count
        result['n_assigned'] = n_assigned
        
        # Debug: show assignment pattern
        print(f"   DEBUG: Gurobi assignments per (farm, period):")
        for (f, t), cnt in assignments_per_farm_period.items():
            if cnt != 1:  # Only show anomalies
                print(f"      {f}, t={t}: {cnt} crops")
    
    return result


def solve_clique_decomp(data: dict, num_reads: int = 100) -> dict:
    """Solve with clique decomposition (farm-by-farm) for comparison."""
    total_start = time.time()
    
    farm_names = data['farm_names']
    food_names = data['food_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    params = data['config']['parameters']
    rotation_gamma = params['rotation_gamma']
    one_hot_penalty = params['one_hot_penalty']
    frustration_ratio = params['frustration_ratio']
    negative_strength = params['negative_synergy_strength']
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_periods = N_PERIODS
    
    # Build rotation matrix (SAME as hybrid)
    R = build_hybrid_rotation_matrix(food_names, frustration_ratio, negative_strength, seed=42)
    
    # Initialize sampler
    token = os.environ.get('DWAVE_API_TOKEN', DEFAULT_DWAVE_TOKEN)
    sampler = DWaveCliqueSampler(token=token)
    
    total_qpu_time = 0
    farm_solutions = {}
    
    # Solve each farm independently
    for farm_idx, farm in enumerate(farm_names):
        print(f"    Farm {farm_idx+1}/{n_farms}...", end=" ", flush=True)
        
        bqm = BinaryQuadraticModel('BINARY')
        
        # Variable mapping
        var_map = {}
        var_id = 0
        for food in food_names:
            for period in range(1, n_periods + 1):
                var_map[(food, period)] = var_id
                var_id += 1
        
        farm_area = land_availability[farm]
        
        # Linear benefits
        for food_idx, food in enumerate(food_names):
            benefit = food_benefits.get(food, 0.5)
            for period in range(1, n_periods + 1):
                var = var_map[(food, period)]
                bqm.add_variable(var, -(benefit * farm_area) / total_area)
        
        # Rotation synergies
        for period in range(2, n_periods + 1):
            for food1_idx, food1 in enumerate(food_names):
                for food2_idx, food2 in enumerate(food_names):
                    synergy = R[food1_idx, food2_idx]
                    if abs(synergy) > 1e-6:
                        var1 = var_map[(food1, period - 1)]
                        var2 = var_map[(food2, period)]
                        bqm.add_interaction(var1, var2, -(rotation_gamma * synergy * farm_area) / total_area)
        
        # One-hot penalty
        for period in range(1, n_periods + 1):
            period_vars = [var_map[(food, period)] for food in food_names]
            for i, v1 in enumerate(period_vars):
                for v2 in period_vars[i+1:]:
                    bqm.add_interaction(v1, v2, one_hot_penalty)
                bqm.add_variable(v1, -one_hot_penalty)
        
        # Solve
        try:
            sampleset = sampler.sample(bqm, num_reads=num_reads, label=f"Clique_f{farm_idx}")
            
            timing = sampleset.info.get('timing', {})
            qpu_time = timing.get('qpu_access_time', 0) / 1e6
            total_qpu_time += qpu_time
            
            # Decode
            best_sample = sampleset.first.sample
            farm_sol = {}
            for food in food_names:
                for period in range(1, n_periods + 1):
                    var = var_map[(food, period)]
                    if best_sample.get(var, 0) > 0.5:
                        farm_sol[(farm, food, period)] = 1
            farm_solutions[farm] = farm_sol
            print(f"QPU={qpu_time:.3f}s")
            
        except Exception as e:
            print(f"Failed: {e}")
            farm_solutions[farm] = {(farm, food_names[0], p): 1 for p in range(1, n_periods + 1)}
    
    total_time = time.time() - total_start
    
    # Combine solutions
    combined = {}
    for farm_sol in farm_solutions.values():
        combined.update(farm_sol)
    
    # Calculate objective
    obj = calculate_objective(combined, data, R)
    
    return {
        'method': 'clique_decomp',
        'success': True,
        'objective': obj,
        'solve_time': total_time,
        'qpu_time': total_qpu_time,
        'violations': {'total': 0},
        'n_assigned': len(combined),
    }


def calculate_objective(solution: dict, data: dict, R: np.ndarray) -> float:
    """Calculate objective for solution."""
    farm_names = data['farm_names']
    food_names = data['food_names']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    params = data['config']['parameters']
    rotation_gamma = params['rotation_gamma']
    diversity_bonus = params['diversity_bonus']
    one_hot_penalty = params['one_hot_penalty']
    
    n_periods = N_PERIODS
    obj = 0
    
    # Base benefit
    for (farm, food, period), val in solution.items():
        if val > 0:
            farm_area = land_availability.get(farm, 25.0)
            benefit = food_benefits.get(food, 0.5)
            obj += (benefit * farm_area) / total_area
    
    # Rotation synergies
    for farm in farm_names:
        farm_area = land_availability.get(farm, 25.0)
        for period in range(2, n_periods + 1):
            for food1_idx, food1 in enumerate(food_names):
                for food2_idx, food2 in enumerate(food_names):
                    if solution.get((farm, food1, period - 1), 0) > 0 and \
                       solution.get((farm, food2, period), 0) > 0:
                        synergy = R[food1_idx, food2_idx]
                        obj += (rotation_gamma * synergy * farm_area) / total_area
    
    # One-hot penalty
    for farm in farm_names:
        for period in range(1, n_periods + 1):
            count = sum(1 for food in food_names if solution.get((farm, food, period), 0) > 0)
            if count > 1:
                obj -= one_hot_penalty * (count - 1) ** 2
    
    # Diversity bonus
    for farm in farm_names:
        foods_used = set()
        for food in food_names:
            for period in range(1, n_periods + 1):
                if solution.get((farm, food, period), 0) > 0:
                    foods_used.add(food)
        obj += diversity_bonus * len(foods_used)
    
    return obj


# ===========================================================================
# RUN TESTS
# ===========================================================================

print("Running tests...\n")

results = []

for scenario in TEST_SCENARIOS:
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario['name']}")
    n_vars = scenario['n_farms'] * scenario['n_foods'] * N_PERIODS
    print(f"Variables: {n_vars}")
    print(f"{'='*80}")
    
    # Load data
    data = load_test_data(scenario)
    print(f"Loaded: {len(data['farm_names'])} farms, {len(data['food_names'])} foods")
    
    # 1. Gurobi (ground truth)
    print(f"\n1. GUROBI (ground truth, {GUROBI_TIMEOUT}s timeout)...")
    gurobi_result = solve_gurobi(data, GUROBI_TIMEOUT)
    print(f"   Objective: {gurobi_result['objective']:.4f}")
    print(f"   Time: {gurobi_result['solve_time']:.1f}s")
    print(f"   Assigned: {gurobi_result['n_assigned']}")
    
    # 2. Hybrid QPU (new method)
    print(f"\n2. HYBRID QPU (27 foods, spatial decomposition)...")
    hybrid_result = solve_hybrid_qpu(data, num_reads=100, num_iterations=3, verbose=True)
    print(f"   Objective: {hybrid_result['objective']:.4f}")
    print(f"   Total time: {hybrid_result['solve_time']:.1f}s")
    print(f"   QPU time: {hybrid_result['qpu_time']:.3f}s")
    print(f"   Assigned: {hybrid_result['n_assigned']}")
    
    # 3. Clique decomposition (existing method)
    print(f"\n3. CLIQUE DECOMP (farm-by-farm)...")
    clique_result = solve_clique_decomp(data, num_reads=100)
    print(f"   Objective: {clique_result['objective']:.4f}")
    print(f"   Total time: {clique_result['solve_time']:.1f}s")
    print(f"   QPU time: {clique_result['qpu_time']:.3f}s")
    print(f"   Assigned: {clique_result['n_assigned']}")
    
    # Calculate gaps
    gurobi_obj = gurobi_result['objective']
    if gurobi_obj != 0:
        hybrid_gap = (gurobi_obj - hybrid_result['objective']) / abs(gurobi_obj) * 100
        clique_gap = (gurobi_obj - clique_result['objective']) / abs(gurobi_obj) * 100
    else:
        hybrid_gap = 0
        clique_gap = 0
    
    # Speedups
    hybrid_speedup = gurobi_result['solve_time'] / hybrid_result['solve_time'] if hybrid_result['solve_time'] > 0 else 0
    clique_speedup = gurobi_result['solve_time'] / clique_result['solve_time'] if clique_result['solve_time'] > 0 else 0
    
    results.append({
        'scenario': scenario['name'],
        'n_vars': n_vars,
        'gurobi_obj': gurobi_obj,
        'gurobi_time': gurobi_result['solve_time'],
        'hybrid_obj': hybrid_result['objective'],
        'hybrid_time': hybrid_result['solve_time'],
        'hybrid_qpu': hybrid_result['qpu_time'],
        'hybrid_gap': hybrid_gap,
        'hybrid_speedup': hybrid_speedup,
        'clique_obj': clique_result['objective'],
        'clique_time': clique_result['solve_time'],
        'clique_qpu': clique_result['qpu_time'],
        'clique_gap': clique_gap,
        'clique_speedup': clique_speedup,
    })

# Print summary
print("\n" + "="*100)
print("SUMMARY: COMPARISON OF METHODS")
print("="*100)
print(f"{'Scenario':<25} {'Vars':>6} │ {'Gurobi':>10} │ {'Hybrid QPU':>10} {'Gap':>7} {'QPU':>7} │ {'Clique':>10} {'Gap':>7} {'QPU':>7}")
print("-"*100)
for r in results:
    print(f"{r['scenario']:<25} {r['n_vars']:>6} │ {r['gurobi_obj']:>10.3f} │ "
          f"{r['hybrid_obj']:>10.3f} {r['hybrid_gap']:>6.1f}% {r['hybrid_qpu']:>6.3f}s │ "
          f"{r['clique_obj']:>10.3f} {r['clique_gap']:>6.1f}% {r['clique_qpu']:>6.3f}s")
print("-"*100)

print("\n✓ Test complete!")
print("\nKey findings:")
print("  - All methods use the SAME formulation (27 foods, same synergy matrix)")
print("  - Hybrid QPU uses spatial decomposition (clusters of farms)")
print("  - Clique decomp solves farm-by-farm")
print("  - Gap = (Gurobi - QPU) / Gurobi × 100%")
