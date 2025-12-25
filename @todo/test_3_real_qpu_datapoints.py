#!/usr/bin/env python3
"""
Test 3 Real QPU Data Points with Adaptive Hybrid Solver
Uses real D-Wave QPU and compares with Gurobi timeout results.

Data points:
1. Small: 10 farms × 27 foods = 810 vars (after aggregation: 180 vars)
2. Medium: 25 farms × 27 foods = 2025 vars (after aggregation: 450 vars)  
3. Large: 50 farms × 27 foods = 4050 vars (after aggregation: 900 vars)

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
from hybrid_formulation import solve_hybrid_adaptive
from food_grouping import aggregate_foods_to_families, create_family_rotation_matrix

# Test configurations - 3 data points
TEST_CONFIGS = [
    {'name': 'Small (10f×27c)', 'n_farms': 10, 'n_foods': 27, 'gurobi_timeout': 60},
    {'name': 'Medium (25f×27c)', 'n_farms': 25, 'n_foods': 27, 'gurobi_timeout': 120},
    {'name': 'Large (50f×27c)', 'n_farms': 50, 'n_foods': 27, 'gurobi_timeout': 300},
]

OUTPUT_DIR = Path(__file__).parent / 'real_qpu_results'
OUTPUT_DIR.mkdir(exist_ok=True)

def solve_gurobi_6family(data, timeout=60):
    """Solve with Gurobi using 6-family formulation (same as hybrid QPU)."""
    family_data = aggregate_foods_to_families(data)
    family_names = family_data['food_names']
    farm_names = family_data['farm_names']
    family_benefits = family_data['food_benefits']
    land_availability = family_data['land_availability']
    total_area = family_data['total_area']
    
    n_periods = 3
    
    config = data.get('config', {})
    params = config.get('parameters', {})
    rotation_gamma = params.get('rotation_gamma', 0.2)
    one_hot_penalty = params.get('one_hot_penalty', 3.0)
    diversity_bonus = params.get('diversity_bonus', 0.15)
    
    R = create_family_rotation_matrix(seed=42)
    
    start_time = time.time()
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', timeout)
    
    # Variables
    Y = {}
    for farm in farm_names:
        for fam in family_names:
            for t in range(1, n_periods + 1):
                Y[(farm, fam, t)] = model.addVar(vtype=GRB.BINARY)
    
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
    }


def main():
    print("="*80)
    print("REAL QPU DATA POINTS TEST")
    print("Adaptive Hybrid Solver vs Gurobi (6-family formulation)")
    print("="*80)
    print()
    
    results = []
    
    for config in TEST_CONFIGS:
        print(f"\n{'='*80}")
        print(f"TEST: {config['name']}")
        print(f"  Farms: {config['n_farms']}, Foods: {config['n_foods']}")
        print(f"  Original vars: {config['n_farms'] * config['n_foods'] * 3}")
        print(f"  After aggregation: {config['n_farms'] * 6 * 3} vars (6 families)")
        print(f"{'='*80}")
        
        # Load data
        data = load_food_data_as_dict('rotation_250farms_27foods')
        data['farm_names'] = data['farm_names'][:config['n_farms']]
        data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
        data['total_area'] = sum(data['land_availability'].values())
        
        # GUROBI
        print(f"\n[Gurobi] Running with {config['gurobi_timeout']}s timeout...")
        gurobi_result = solve_gurobi_6family(data, config['gurobi_timeout'])
        print(f"  Objective: {gurobi_result['objective']:.4f}")
        print(f"  Time: {gurobi_result['solve_time']:.2f}s")
        print(f"  Status: {gurobi_result['status']}")
        print(f"  Gap: {gurobi_result['gap']:.2f}%")
        
        # QUANTUM (Adaptive Hybrid)
        print(f"\n[Quantum] Running Adaptive Hybrid QPU...")
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
        if gurobi_result['objective'] != 0:
            gap_percent = (gurobi_result['objective'] - quantum_result['objective']) / abs(gurobi_result['objective']) * 100
        else:
            gap_percent = 0
        
        speedup = gurobi_result['solve_time'] / quantum_result['solve_time'] if quantum_result['solve_time'] > 0 else 0
        
        result = {
            'name': config['name'],
            'n_farms': config['n_farms'],
            'n_foods': config['n_foods'],
            'n_vars_original': config['n_farms'] * config['n_foods'] * 3,
            'n_vars_aggregated': config['n_farms'] * 6 * 3,
            'gurobi_objective': gurobi_result['objective'],
            'gurobi_time': gurobi_result['solve_time'],
            'gurobi_status': gurobi_result['status'],
            'gurobi_gap': gurobi_result['gap'],
            'quantum_objective': quantum_result['objective'],
            'quantum_time': quantum_result['solve_time'],
            'qpu_time': quantum_result['qpu_time'],
            'quantum_violations': quantum_result['violations']['total'],
            'gap_to_gurobi': gap_percent,
            'speedup': speedup,
        }
        results.append(result)
        
        print(f"\n  Gap to Gurobi: {gap_percent:.1f}%")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Test':<20} {'Vars':>8} {'G-Obj':>10} {'G-Time':>8} {'Q-Obj':>10} {'Q-Time':>8} {'QPU':>8} {'Gap%':>8} {'Speedup':>8}")
    print("-"*100)
    for r in results:
        print(f"{r['name']:<20} {r['n_vars_aggregated']:>8} {r['gurobi_objective']:>10.2f} {r['gurobi_time']:>7.1f}s "
              f"{r['quantum_objective']:>10.2f} {r['quantum_time']:>7.1f}s {r['qpu_time']:>7.3f}s "
              f"{r['gap_to_gurobi']:>7.1f}% {r['speedup']:>7.2f}x")
    print("="*100)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f'real_qpu_3points_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    main()
