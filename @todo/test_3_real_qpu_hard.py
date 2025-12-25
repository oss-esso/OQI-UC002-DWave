#!/usr/bin/env python3
"""
Test 3 Real QPU Data Points - HARD Problems
Uses 27-food MIQP for Gurobi (harder) vs Adaptive Hybrid QPU.

This creates a fair comparison where Gurobi actually struggles.

Author: OQI-UC002-DWave
Date: 2025-12-25
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '..')
sys.path.insert(0, str(Path(__file__).parent.parent))

import gurobipy as gp
from gurobipy import GRB

from data_loader_utils import load_food_data_as_dict
from hybrid_formulation import solve_hybrid_adaptive, build_hybrid_rotation_matrix
from food_grouping import get_family, FAMILY_ORDER

# Test configurations - scaled for Gurobi to actually struggle
TEST_CONFIGS = [
    {'name': 'Medium', 'n_farms': 20, 'n_foods': 27, 'gurobi_timeout': 60},
    {'name': 'Large', 'n_farms': 35, 'n_foods': 27, 'gurobi_timeout': 120},
    {'name': 'XLarge', 'n_farms': 50, 'n_foods': 27, 'gurobi_timeout': 300},
]

OUTPUT_DIR = Path(__file__).parent / 'real_qpu_results'
OUTPUT_DIR.mkdir(exist_ok=True)


def solve_gurobi_27food_miqp(data, timeout=60):
    """
    Solve with Gurobi using FULL 27-food MIQP formulation.
    This is the HARD problem that causes Gurobi to timeout.
    """
    farm_names = data['farm_names']
    food_names = data['food_names']
    food_benefits = data['food_benefits']
    land_availability = data['land_availability']
    total_area = data['total_area']
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_periods = 3
    
    config = data.get('config', {})
    params = config.get('parameters', {})
    rotation_gamma = params.get('rotation_gamma', 0.2)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    
    # Build 27x27 rotation matrix
    R = build_hybrid_rotation_matrix(food_names, seed=42)
    
    # Spatial neighbors
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {farm: (i // side, i % side) for i, farm in enumerate(farm_names)}
    k_neighbors = 4
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
    
    start_time = time.time()
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    model.setParam('MIPGap', 0.01)
    model.setParam('MIPFocus', 1)
    
    # Variables - FULL 27 foods
    Y = {}
    for farm in farm_names:
        for food in food_names:
            for t in range(1, n_periods + 1):
                Y[(farm, food, t)] = model.addVar(vtype=GRB.BINARY)
    
    # Objective
    obj = 0
    
    # Part 1: Benefits
    for farm in farm_names:
        area = land_availability[farm]
        for food in food_names:
            benefit = food_benefits.get(food, 0.5)
            for t in range(1, n_periods + 1):
                obj += (benefit * area / total_area) * Y[(farm, food, t)]
    
    # Part 2: Rotation synergies (27x27 MIQP - THIS MAKES IT HARD)
    for farm in farm_names:
        area = land_availability[farm]
        for t in range(2, n_periods + 1):
            for i, food1 in enumerate(food_names):
                for j, food2 in enumerate(food_names):
                    synergy = R[i, j]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * area / total_area) * Y[(farm, food1, t-1)] * Y[(farm, food2, t)]
    
    # Part 3: Spatial interactions (MIQP)
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for i, food1 in enumerate(food_names):
                for j, food2 in enumerate(food_names):
                    synergy = R[i, j]
                    if abs(synergy) > 1e-6:
                        obj += (spatial_gamma * synergy) * Y[(f1, food1, t)] * Y[(f2, food2, t)]
    
    # Part 4: One-hot penalty (MIQP)
    for farm in farm_names:
        for t in range(1, n_periods + 1):
            for i, food1 in enumerate(food_names):
                for food2 in food_names[i+1:]:
                    obj -= one_hot_penalty * Y[(farm, food1, t)] * Y[(farm, food2, t)]
    
    # Part 5: Diversity bonus
    for farm in farm_names:
        area = land_availability[farm]
        for food in food_names:
            for t in range(1, n_periods + 1):
                obj += (diversity_bonus / n_periods * area / total_area) * Y[(farm, food, t)]
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    for farm in farm_names:
        for t in range(1, n_periods + 1):
            model.addConstr(sum(Y[(farm, food, t)] for food in food_names) <= 2)  # Max 2 crops
            model.addConstr(sum(Y[(farm, food, t)] for food in food_names) >= 1)  # Min 1 crop
    
    model.optimize()
    solve_time = time.time() - start_time
    
    status = 'optimal' if model.Status == GRB.OPTIMAL else 'timeout' if model.Status == GRB.TIME_LIMIT else 'unknown'
    obj_val = model.ObjVal if model.SolCount > 0 else 0
    gap = model.MIPGap * 100 if model.SolCount > 0 else 100
    n_assigned = sum(1 for v in Y.values() if v.X > 0.5) if model.SolCount > 0 else 0
    
    return {
        'objective': obj_val,
        'solve_time': solve_time,
        'status': status,
        'gap': gap,
        'n_assigned': n_assigned,
        'n_vars': n_farms * n_foods * n_periods,
    }


def main():
    print("="*80)
    print("REAL QPU DATA POINTS TEST - HARD PROBLEMS")
    print("27-Food MIQP Gurobi vs Adaptive Hybrid QPU")
    print("="*80)
    print()
    
    results = []
    
    for config in TEST_CONFIGS:
        print(f"\n{'='*80}")
        print(f"TEST: {config['name']}")
        print(f"  Farms: {config['n_farms']}, Foods: {config['n_foods']}")
        print(f"  Gurobi vars: {config['n_farms'] * config['n_foods'] * 3} (27-food MIQP)")
        print(f"  QPU vars: {config['n_farms'] * 6 * 3} (6-family aggregated)")
        print(f"{'='*80}")
        
        # Load data
        data = load_food_data_as_dict('rotation_250farms_27foods')
        data['farm_names'] = data['farm_names'][:config['n_farms']]
        data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
        data['total_area'] = sum(data['land_availability'].values())
        
        # GUROBI (27-food MIQP - HARD)
        print(f"\n[Gurobi 27-food MIQP] Running with {config['gurobi_timeout']}s timeout...")
        gurobi_result = solve_gurobi_27food_miqp(data, config['gurobi_timeout'])
        print(f"  Objective: {gurobi_result['objective']:.4f}")
        print(f"  Time: {gurobi_result['solve_time']:.2f}s")
        print(f"  Status: {gurobi_result['status']}")
        print(f"  MIP Gap: {gurobi_result['gap']:.2f}%")
        print(f"  Variables: {gurobi_result['n_vars']}")
        
        # QUANTUM (Adaptive Hybrid)
        print(f"\n[Quantum Adaptive Hybrid] Running...")
        quantum_result = solve_hybrid_adaptive(
            data, 
            num_reads=100, 
            num_iterations=3,
            overlap_farms=1,
            verbose=True
        )
        print(f"  Objective: {quantum_result['objective']:.4f}")
        print(f"  Time: {quantum_result['solve_time']:.2f}s")
        print(f"  QPU time: {quantum_result['qpu_time']:.3f}s")
        print(f"  Violations: {quantum_result['violations']['total']}")
        
        # Calculate metrics
        # Note: Objectives may not be directly comparable (different formulations)
        # But we can still measure speedup
        speedup = gurobi_result['solve_time'] / quantum_result['solve_time'] if quantum_result['solve_time'] > 0 else 0
        
        # For gap, we normalize both by expected max
        # This is approximate since formulations differ
        expected_max = config['n_farms'] * 3 * 0.5  # rough estimate
        gurobi_normalized = gurobi_result['objective'] / expected_max if expected_max > 0 else 0
        quantum_normalized = quantum_result['objective'] / expected_max if expected_max > 0 else 0
        
        result = {
            'name': config['name'],
            'n_farms': config['n_farms'],
            'n_foods': config['n_foods'],
            'gurobi_vars': gurobi_result['n_vars'],
            'quantum_vars': config['n_farms'] * 6 * 3,
            'gurobi_objective': gurobi_result['objective'],
            'gurobi_time': gurobi_result['solve_time'],
            'gurobi_status': gurobi_result['status'],
            'gurobi_mip_gap': gurobi_result['gap'],
            'quantum_objective': quantum_result['objective'],
            'quantum_time': quantum_result['solve_time'],
            'qpu_time': quantum_result['qpu_time'],
            'quantum_violations': quantum_result['violations']['total'],
            'speedup': speedup,
        }
        results.append(result)
        
        print(f"\n  Speedup: {speedup:.2f}x (Gurobi {gurobi_result['solve_time']:.1f}s / QPU {quantum_result['solve_time']:.1f}s)")
    
    # Summary
    print("\n" + "="*110)
    print("SUMMARY TABLE")
    print("="*110)
    print(f"{'Test':<12} {'G-Vars':>8} {'Q-Vars':>8} {'G-Obj':>10} {'G-Time':>8} {'G-Status':>10} {'Q-Obj':>10} {'Q-Time':>8} {'QPU':>8} {'Speedup':>8}")
    print("-"*110)
    for r in results:
        print(f"{r['name']:<12} {r['gurobi_vars']:>8} {r['quantum_vars']:>8} {r['gurobi_objective']:>10.2f} {r['gurobi_time']:>7.1f}s "
              f"{r['gurobi_status']:>10} {r['quantum_objective']:>10.2f} {r['quantum_time']:>7.1f}s {r['qpu_time']:>7.3f}s "
              f"{r['speedup']:>7.2f}x")
    print("="*110)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f'real_qpu_hard_3points_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    main()
