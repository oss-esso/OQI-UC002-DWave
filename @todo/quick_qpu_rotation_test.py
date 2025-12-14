#!/usr/bin/env python3
"""
Quick QPU Test for Rotation Scenarios
Tests 2 rotation scenarios with actual QPU calls (clique_decomp method)
"""

import sys
import os
import time
import json
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from scenarios import load_food_data
import numpy as np

# D-Wave token
DWAVE_TOKEN = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'
os.environ['DWAVE_API_TOKEN'] = DWAVE_TOKEN

print("="*80)
print("QUICK QPU TEST - ROTATION SCENARIOS")
print("="*80)
print("\nTesting 2 scenarios with clique_decomp QPU method")
print("Expected: ~1-2 minutes QPU time total\n")

# Test scenarios (carefully chosen to be manageable)
TEST_SCENARIOS = [
    {
        'name': 'rotation_micro_25',
        'description': '5 farms × 6 foods × 3 periods (90 vars)',
        'expected_time': '~20-30s',
        'expected_gap': '~8%',
    },
    {
        'name': 'rotation_small_50', 
        'description': '10 farms × 6 foods × 3 periods (180 vars)',
        'expected_time': '~35-45s',
        'expected_gap': '~10%',
    },
]

OUTPUT_DIR = Path(__file__).parent / 'qpu_rotation_test_results'
OUTPUT_DIR.mkdir(exist_ok=True)

results = []

# ============================================================================
# QPU SOLVER (clique_decomp method)
# ============================================================================

def solve_with_qpu_clique(scenario_name: str):
    """Solve using clique decomposition with QPU."""
    from dimod import BinaryQuadraticModel
    from dwave.system import LeapHybridCQMSampler, DWaveSampler, EmbeddingComposite
    from neal import SimulatedAnnealingSampler
    
    print(f"\n{'='*80}")
    print(f"Solving: {scenario_name}")
    print(f"{'='*80}")
    
    start_total = time.time()
    
    # Load scenario
    print("Loading scenario data...")
    farms, foods, food_groups, config = load_food_data(scenario_name)
    
    food_names = list(foods.keys())
    farm_names = farms if isinstance(farms, list) else list(farms.keys())
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_periods = 3
    n_vars = n_farms * n_foods * n_periods
    
    print(f"  Farms: {n_farms}")
    print(f"  Foods: {n_foods}")
    print(f"  Variables: {n_vars}")
    
    # Build simple BQM (rotation-aware)
    print("\nBuilding BQM with rotation constraints...")
    bqm_start = time.time()
    
    bqm = BinaryQuadraticModel('BINARY')
    
    # Variables: y_{f,c,t} for farm f, food c, period t
    var_map = {}
    idx = 0
    for farm in farm_names:
        for food in food_names:
            for t in range(n_periods):
                var_name = f"{farm}_{food}_t{t}"
                var_map[(farm, food, t)] = var_name
                idx += 1
    
    # Objective: maximize benefit (minimize negative)
    food_benefits = {f: np.mean([foods[f][k] for k in ['nutritional_value', 'sustainability']]) 
                     for f in food_names}
    
    for farm in farm_names:
        for food in food_names:
            benefit = food_benefits[food]
            for t in range(n_periods):
                var = var_map[(farm, food, t)]
                bqm.add_variable(var, -benefit)  # Negative for maximization
    
    # Rotation synergy (quadratic terms)
    rotation_gamma = 0.3
    for farm in farm_names:
        for food in food_names:
            for t in range(n_periods - 1):
                var1 = var_map[(farm, food, t)]
                var2 = var_map[(farm, food, t+1)]
                bqm.add_interaction(var1, var2, -rotation_gamma)  # Bonus for rotation
    
    # One-hot constraint (soft) per farm-period
    one_hot_penalty = 2.0
    for farm in farm_names:
        for t in range(n_periods):
            vars_ft = [var_map[(farm, food, t)] for food in food_names]
            # Penalize selecting multiple foods
            for i, v1 in enumerate(vars_ft):
                for v2 in vars_ft[i+1:]:
                    bqm.add_interaction(v1, v2, one_hot_penalty)
    
    bqm_time = time.time() - bqm_start
    print(f"  BQM built in {bqm_time:.2f}s")
    print(f"  BQM size: {len(bqm.variables)} vars, {len(bqm.quadratic)} interactions")
    
    # Decompose into cliques (spatial decomposition by farms)
    print("\nDecomposing into farm-based cliques...")
    decomp_start = time.time()
    
    # Simple spatial decomposition: each farm is a subproblem
    subproblems = []
    for farm in farm_names:
        sub_vars = [var_map[(farm, food, t)] for food in food_names for t in range(n_periods)]
        
        # Extract sub-BQM
        sub_bqm = BinaryQuadraticModel('BINARY')
        for v in sub_vars:
            if v in bqm.linear:
                sub_bqm.add_variable(v, bqm.linear[v])
        
        for (v1, v2), bias in bqm.quadratic.items():
            if v1 in sub_vars and v2 in sub_vars:
                sub_bqm.add_interaction(v1, v2, bias)
        
        subproblems.append({
            'farm': farm,
            'bqm': sub_bqm,
            'vars': sub_vars,
        })
    
    decomp_time = time.time() - decomp_start
    print(f"  Created {len(subproblems)} subproblems in {decomp_time:.2f}s")
    
    # Solve subproblems with QPU
    print(f"\nSolving {len(subproblems)} subproblems on QPU...")
    qpu_start = time.time()
    qpu_time_total = 0
    
    try:
        sampler = EmbeddingComposite(DWaveSampler())
        
        solutions = []
        for i, subprob in enumerate(subproblems):
            print(f"  Subproblem {i+1}/{len(subproblems)}: {subprob['farm']} ({len(subprob['bqm'].variables)} vars)...", end='')
            
            sub_start = time.time()
            sampleset = sampler.sample(subprob['bqm'], num_reads=100)
            sub_time = time.time() - sub_start
            qpu_time_total += sampleset.info.get('timing', {}).get('qpu_access_time', 0) / 1e6  # Convert to seconds
            
            best_sample = sampleset.first.sample
            solutions.append(best_sample)
            
            print(f" {sub_time:.2f}s")
        
        qpu_wall_time = time.time() - qpu_start
        
        # Combine solutions
        print("\nCombining solutions...")
        combined_solution = {}
        for sol in solutions:
            combined_solution.update(sol)
        
        # Calculate objective
        objective = 0
        for var, val in combined_solution.items():
            if val == 1:
                objective += bqm.linear.get(var, 0)
        
        for (v1, v2), bias in bqm.quadratic.items():
            if combined_solution.get(v1, 0) == 1 and combined_solution.get(v2, 0) == 1:
                objective += bias
        
        objective = -objective  # Convert back to maximization
        
        total_time = time.time() - start_total
        
        result = {
            'scenario': scenario_name,
            'method': 'clique_decomp_qpu',
            'n_farms': n_farms,
            'n_foods': n_foods,
            'n_vars': n_vars,
            'n_subproblems': len(subproblems),
            'success': True,
            'objective': objective,
            'total_time': total_time,
            'qpu_time': qpu_time_total,
            'qpu_wall_time': qpu_wall_time,
            'bqm_time': bqm_time,
            'decomp_time': decomp_time,
        }
        
        print(f"\n✓ SUCCESS")
        print(f"  Objective: {objective:.4f}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  QPU time: {qpu_time_total:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return {
            'scenario': scenario_name,
            'method': 'clique_decomp_qpu',
            'n_farms': n_farms,
            'n_foods': n_foods,
            'n_vars': n_vars,
            'success': False,
            'error': str(e),
            'total_time': time.time() - start_total,
        }

# ============================================================================
# RUN TESTS
# ============================================================================

for test_info in TEST_SCENARIOS:
    print(f"\n{'='*80}")
    print(f"TEST: {test_info['name']}")
    print(f"Description: {test_info['description']}")
    print(f"Expected: {test_info['expected_time']}, gap {test_info['expected_gap']}")
    print(f"{'='*80}")
    
    result = solve_with_qpu_clique(test_info['name'])
    results.append(result)
    
    # Save after each test
    output_file = OUTPUT_DIR / f"qpu_test_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Progress saved to: {output_file}")
    
    # Brief pause between tests
    if test_info != TEST_SCENARIOS[-1]:
        print("\nWaiting 5 seconds before next test...")
        time.sleep(5)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

successful = [r for r in results if r.get('success', False)]
failed = [r for r in results if not r.get('success', False)]

print(f"\nTests run: {len(results)}")
print(f"  Successful: {len(successful)}")
print(f"  Failed: {len(failed)}")

if successful:
    print(f"\nSuccessful results:")
    print(f"  Total QPU time used: {sum(r.get('qpu_time', 0) for r in successful):.2f}s")
    print(f"\n  Results table:")
    print(f"  {'Scenario':<25} {'Vars':>6} {'Time':>8} {'QPU':>8} {'Objective':>10}")
    print(f"  {'-'*70}")
    for r in successful:
        print(f"  {r['scenario']:<25} {r['n_vars']:>6} {r['total_time']:>7.1f}s {r['qpu_time']:>7.1f}s {r['objective']:>10.4f}")

if failed:
    print(f"\nFailed tests:")
    for r in failed:
        print(f"  {r['scenario']}: {r.get('error', 'Unknown error')}")

# Save final
final_output = OUTPUT_DIR / 'qpu_rotation_test_final.json'
with open(final_output, 'w') as f:
    json.dump({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'total_qpu_time': sum(r.get('qpu_time', 0) for r in successful),
        'results': results,
    }, f, indent=2)

print(f"\n✓ Final results saved to: {final_output}")
print("\n" + "="*80)
