#!/usr/bin/env python3
"""
Comprehensive Gurobi Performance Investigation
Tests many problem sizes with 100s timeout to find where Gurobi fails
Includes QPU feasibility markers
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

# Test configuration
TIMEOUT = 100  # 100 seconds
N_PERIODS = 3
N_FAMILIES = 6

# QPU constraints
MAX_QPU_QUBITS = 166  # Advantage_system4.1 clique size
QPU_EMBEDDING_OVERHEAD = 1.5  # Approximate embedding overhead

# Test points - comprehensive range
TEST_POINTS = [
    # Very small (QPU direct embedding)
    {'n_farms': 3, 'name': 'micro_3'},
    {'n_farms': 5, 'name': 'micro_5'},
    {'n_farms': 8, 'name': 'small_8'},
    {'n_farms': 10, 'name': 'small_10'},
    
    # Small-medium (QPU with decomposition)
    {'n_farms': 15, 'name': 'medium_15'},
    {'n_farms': 20, 'name': 'medium_20'},
    {'n_farms': 25, 'name': 'medium_25'},
    {'n_farms': 30, 'name': 'medium_30'},
    
    # Medium (QPU borderline)
    {'n_farms': 40, 'name': 'large_40'},
    {'n_farms': 50, 'name': 'large_50'},
    {'n_farms': 60, 'name': 'large_60'},
    
    # Large (QPU infeasible without heavy decomposition)
    {'n_farms': 75, 'name': 'xlarge_75'},
    {'n_farms': 90, 'name': 'xlarge_90'},
    {'n_farms': 100, 'name': 'xlarge_100'},
    
    # Very large (stress test)
    {'n_farms': 150, 'name': 'xxlarge_150'},
    {'n_farms': 200, 'name': 'xxlarge_200'},
    {'n_farms': 300, 'name': 'huge_300'},
]

def estimate_qpu_feasibility(n_vars):
    """Estimate if problem is QPU-feasible"""
    # Binary variables in QUBO
    n_qubits_needed = n_vars * QPU_EMBEDDING_OVERHEAD
    
    if n_qubits_needed <= MAX_QPU_QUBITS:
        return "DIRECT", n_qubits_needed
    elif n_vars <= 500:
        return "DECOMP", n_qubits_needed
    else:
        return "NO", n_qubits_needed

def solve_gurobi(land_availability, food_benefits, config_params, timeout=100):
    """Solve with Gurobi"""
    farm_names = list(land_availability.keys())
    n_farms = len(farm_names)
    families_list = list(food_benefits.keys())
    n_families = len(families_list)
    total_area = sum(land_availability.values())
    
    rotation_gamma = config_params.get('rotation_gamma', 0.2)
    frustration_ratio = config_params.get('frustration_ratio', 0.7)
    negative_strength = config_params.get('negative_synergy_strength', -0.8)
    one_hot_penalty = config_params.get('one_hot_penalty', 3.0)
    diversity_bonus = config_params.get('diversity_bonus', 0.15)
    k_neighbors = config_params.get('spatial_k_neighbors', 4)
    
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
    model = gp.Model("GurobiInvestigation")
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
    }

# Main test
print("="*100)
print("COMPREHENSIVE GUROBI PERFORMANCE INVESTIGATION")
print(f"Timeout: {TIMEOUT}s | Problem: 6 crop families, 3 periods")
print("="*100)

results = []

for test_point in TEST_POINTS:
    n_farms = test_point['n_farms']
    name = test_point['name']
    n_vars = n_farms * N_FAMILIES * N_PERIODS
    
    # Load scenario (use rotation_medium_100 pattern for consistency)
    if n_farms <= 20:
        scenario = 'rotation_medium_100'
    elif n_farms <= 50:
        scenario = 'rotation_large_200'
    else:
        scenario = 'rotation_large_200'
    
    try:
        farms, foods, food_groups, config = load_food_data(scenario)
    except:
        print(f"\nSkipping {name} ({n_farms} farms) - scenario not available")
        continue
    
    params = config.get('parameters', {})
    land_availability_full = params.get('land_availability', {})
    weights = params.get('weights', {})
    
    # Get first n_farms
    all_farms = list(land_availability_full.keys())[:n_farms]
    land_availability = {f: land_availability_full[f] for f in all_farms}
    total_area = sum(land_availability.values())
    
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
    
    # QPU feasibility
    qpu_status, qubits_needed = estimate_qpu_feasibility(n_vars)
    
    print(f"\n{'-'*100}")
    print(f"{name:15} | {n_farms:3} farms | {n_vars:4} vars | Area: {total_area:6.1f} ha | QPU: {qpu_status:6} ({qubits_needed:.0f} qubits)")
    print(f"{'-'*100}")
    
    # Solve
    result = solve_gurobi(land_availability, food_benefits, params, timeout=TIMEOUT)
    
    print(f"  Status: {result['status']:8} | Time: {result['time']:6.2f}s | Gap: {result['gap']:6.1f}% | Nodes: {result['nodes']:8.0f}")
    
    results.append({
        'name': name,
        'n_farms': n_farms,
        'n_vars': n_vars,
        'area': total_area,
        'qpu_status': qpu_status,
        'qubits': qubits_needed,
        **result
    })

# Summary
print(f"\n{'='*100}")
print("SUMMARY: WHERE DOES GUROBI FAIL?")
print("="*100)

print(f"\n{'Size':15} {'Farms':>6} {'Vars':>6} {'QPU':>8} {'Time':>8} {'Status':>10} {'Gap%':>8} {'Nodes':>10}")
print("-"*100)

for r in results:
    qpu_marker = "[OK]" if r['qpu_status'] == "DIRECT" else ("[DC]" if r['qpu_status'] == "DECOMP" else "[NO]")
    status_marker = "[OK]" if r['status'] == 'optimal' else "[TO]"
    
    print(f"{r['name']:15} {r['n_farms']:6} {r['n_vars']:6} {qpu_marker:>8} {r['time']:8.2f} {r['status']:>10} {r['gap']:8.1f} {r['nodes']:10.0f}")

# Analysis
print(f"\n{'='*100}")
print("ANALYSIS")
print("="*100)

timeouts = [r for r in results if r['status'] == 'timeout']
optimal = [r for r in results if r['status'] == 'optimal']

print(f"\nGurobi Performance:")
print(f"  Optimal: {len(optimal)}/{len(results)} ({100*len(optimal)/len(results):.1f}%)")
print(f"  Timeout: {len(timeouts)}/{len(results)} ({100*len(timeouts)/len(results):.1f}%)")

if timeouts:
    print(f"\n  Timeout starts at: {timeouts[0]['n_farms']} farms ({timeouts[0]['n_vars']} vars)")
    print(f"  Smallest timeout: {min(r['n_farms'] for r in timeouts)} farms")
    print(f"  All timeout above: {max(r['n_farms'] for r in optimal) if optimal else 0} farms")

qpu_direct = [r for r in results if r['qpu_status'] == 'DIRECT']
qpu_decomp = [r for r in results if r['qpu_status'] == 'DECOMP']
qpu_no = [r for r in results if r['qpu_status'] == 'NO']

print(f"\nQPU Feasibility:")
print(f"  Direct embedding: {len(qpu_direct)} problem sizes (≤ {MAX_QPU_QUBITS} qubits)")
print(f"  With decomposition: {len(qpu_decomp)} problem sizes")
print(f"  Not feasible: {len(qpu_no)} problem sizes")

if qpu_direct:
    print(f"    Direct range: {min(r['n_vars'] for r in qpu_direct)}-{max(r['n_vars'] for r in qpu_direct)} vars")
if qpu_decomp:
    print(f"    Decomp range: {min(r['n_vars'] for r in qpu_decomp)}-{max(r['n_vars'] for r in qpu_decomp)} vars")

# Quantum advantage zones
print(f"\n{'='*100}")
print("QUANTUM ADVANTAGE ZONES")
print("="*100)

print(f"\nZone 1: DIRECT QPU + GUROBI STRUGGLES")
zone1 = [r for r in results if r['qpu_status'] == 'DIRECT' and (r['status'] == 'timeout' or r['time'] > 10)]
if zone1:
    print(f"  Found {len(zone1)} problem sizes where QPU could help!")
    for r in zone1:
        print(f"    {r['name']:15} {r['n_vars']:4} vars - Gurobi: {r['time']:.1f}s ({r['status']})")
else:
    print(f"  No problems found (Gurobi solves all small problems quickly)")

print(f"\nZone 2: DECOMPOSITION QPU + GUROBI TIMEOUTS")
zone2 = [r for r in results if r['qpu_status'] == 'DECOMP' and r['status'] == 'timeout']
if zone2:
    print(f"  Found {len(zone2)} problem sizes where QPU with decomposition could help!")
    for r in zone2:
        print(f"    {r['name']:15} {r['n_vars']:4} vars - Gurobi timeout at {TIMEOUT}s")
else:
    print(f"  No problems found in this zone")

print(f"\nZone 3: TOO LARGE FOR QPU")
zone3 = [r for r in results if r['qpu_status'] == 'NO']
if zone3:
    print(f"  {len(zone3)} problem sizes too large for current QPU")
    timeouts_in_zone3 = [r for r in zone3 if r['status'] == 'timeout']
    print(f"  Gurobi times out on {len(timeouts_in_zone3)}/{len(zone3)} of these")

print(f"\n{'='*100}")
print("RECOMMENDATION")
print("="*100)

if zone1:
    print(f"[+] Test QPU on: {', '.join(r['name'] for r in zone1[:3])} (direct embedding)")
elif zone2:
    print(f"[+] Test QPU with decomposition on: {', '.join(r['name'] for r in zone2[:3])}")
else:
    print(f"[!] Gurobi performs well on all tested sizes with {TIMEOUT}s timeout")
    print(f"  Consider harder scenarios or longer timeout to find quantum advantage zones")

print("="*100)
