#!/usr/bin/env python3
"""
Deep investigation of infeasibility causes.

Key question: Is the infeasibility due to:
1. 27→6 compression (matrix size)
2. Soft constraint penalties being too weak
3. SA getting stuck in local minima
"""

import sys
sys.path.insert(0, '.')

from unified_benchmark.scenarios import load_scenario, build_rotation_matrix
from unified_benchmark.miqp_scorer import MIQP_PARAMS, compute_miqp_objective, check_constraints
from unified_benchmark.quantum_solvers import solve
import numpy as np

print('=' * 70)
print('INFEASIBILITY INVESTIGATION')
print('=' * 70)

# Load scenario
data = load_scenario('rotation_micro_25')
n_farms = data['n_farms']
n_foods = data['n_foods']
n_periods = data['n_periods']

print(f'\nScenario: {n_farms} farms × {n_foods} foods × {n_periods} periods')
print(f'This is a 6-FAMILY scenario (not 27-food)')
print('So the "compression" from 27→6 is NOT the issue here.')

# Run multiple times to see if violations are consistent
print('\n' + '=' * 70)
print('RUNNING SA NATIVE 10 TIMES TO CHECK CONSISTENCY')
print('=' * 70)

violations_by_seed = []
for seed in range(10):
    result = solve(
        mode="qpu-native-6-family",
        scenario_data=data,
        use_qpu=False,
        num_reads=50,
        timeout=30,
        seed=seed,
        verbose=False,
    )
    cv = result.constraint_violations
    violations_by_seed.append({
        'seed': seed,
        'one_hot': cv.one_hot_violations,
        'rotation': cv.rotation_violations,
        'total': cv.total_violations,
        'obj': result.objective_miqp,
        'feasible': result.feasible
    })
    print(f'Seed {seed}: one_hot={cv.one_hot_violations}, rotation={cv.rotation_violations}, obj={result.objective_miqp:.4f}, feasible={result.feasible}')

# Statistics
feasible_count = sum(1 for v in violations_by_seed if v['feasible'])
avg_rotation_viol = np.mean([v['rotation'] for v in violations_by_seed])

print(f'\nStatistics over 10 runs:')
print(f'  Feasible solutions: {feasible_count}/10 ({feasible_count*10}%)')
print(f'  Average rotation violations: {avg_rotation_viol:.2f}')

# Now test with increased rotation penalty
print('\n' + '=' * 70)
print('TESTING WITH INCREASED ROTATION PENALTY')
print('=' * 70)

from unified_benchmark.quantum_solvers import solve_native_6family

# Modify the rotation_gamma to be stronger
# We'll need to patch the code or use a different approach

print('\nThe rotation constraint is implemented as a SOFT penalty in the BQM:')
print('  penalty = rotation_gamma * R[c,c] * area_frac')
print(f'  With rotation_gamma={MIQP_PARAMS["rotation_gamma"]} and R[c,c]=-1.2:')
print(f'  penalty ≈ {MIQP_PARAMS["rotation_gamma"] * -1.2:.3f} per violation')
print()
print('This is WEAKER than the one-hot penalty (3.0).')
print()
print('CONCLUSION: The infeasibility is NOT due to 27→6 compression.')
print('It is due to the SOFT rotation constraint being too weak.')
print()
print('Solutions:')
print('1. Add an explicit HARD rotation penalty in the BQM')
print('2. Increase rotation_gamma')
print('3. Use post-processing to fix violations')

# Let's test option 1 - add explicit rotation penalty
print('\n' + '=' * 70)
print('TESTING: EXPLICIT ROTATION CONSTRAINT PENALTY')
print('=' * 70)

from dimod import BinaryQuadraticModel

def build_bqm_with_rotation_constraint(data, rotation_penalty=5.0):
    """Build BQM with explicit hard rotation constraint."""
    farm_names = data['farm_names']
    food_names = data['food_names']
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_periods = data['n_periods']
    land_availability = data['land_availability']
    food_benefits = data['food_benefits']
    total_area = data['total_area']
    
    R = build_rotation_matrix(n_foods)
    
    bqm = BinaryQuadraticModel(vartype='BINARY')
    var_map = {}
    
    rotation_gamma = 0.2
    one_hot_penalty = 3.0
    diversity_bonus = 0.15
    
    # Variables with linear biases
    for f_idx in range(n_farms):
        area_frac = land_availability[farm_names[f_idx]] / total_area
        for c_idx, food in enumerate(food_names):
            benefit = food_benefits.get(food, 1.0)
            for t in range(1, n_periods + 1):
                var_name = f"Y_{f_idx}_{c_idx}_{t}"
                var_map[(f_idx, c_idx, t)] = var_name
                
                linear_bias = -benefit * area_frac
                linear_bias -= diversity_bonus / n_periods
                linear_bias -= one_hot_penalty
                
                bqm.add_variable(var_name, linear_bias)
    
    # Temporal synergies
    for f_idx in range(n_farms):
        area_frac = land_availability[farm_names[f_idx]] / total_area
        for t in range(2, n_periods + 1):
            for c1_idx in range(n_foods):
                for c2_idx in range(n_foods):
                    synergy = R[c1_idx, c2_idx]
                    var1 = var_map[(f_idx, c1_idx, t-1)]
                    var2 = var_map[(f_idx, c2_idx, t)]
                    bqm.add_quadratic(var1, var2, -rotation_gamma * synergy * area_frac)
    
    # One-hot penalty
    for f_idx in range(n_farms):
        for t in range(1, n_periods + 1):
            vars_this = [var_map[(f_idx, c, t)] for c in range(n_foods)]
            for i in range(len(vars_this)):
                for j in range(i + 1, len(vars_this)):
                    bqm.add_quadratic(vars_this[i], vars_this[j], 2 * one_hot_penalty)
    
    # EXPLICIT ROTATION CONSTRAINT: Y_{f,c,t} + Y_{f,c,t+1} <= 1
    # Penalty: rotation_penalty * Y_{f,c,t} * Y_{f,c,t+1}
    for f_idx in range(n_farms):
        for c_idx in range(n_foods):
            for t in range(1, n_periods):
                var1 = var_map[(f_idx, c_idx, t)]
                var2 = var_map[(f_idx, c_idx, t+1)]
                # Add penalty for SAME crop in consecutive periods
                bqm.add_quadratic(var1, var2, rotation_penalty)
    
    return bqm, var_map

# Test with explicit rotation constraint
from neal import SimulatedAnnealingSampler

bqm, var_map = build_bqm_with_rotation_constraint(data, rotation_penalty=5.0)
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)

best_sample = sampleset.first.sample

# Extract solution
reverse_map = {v: k for k, v in var_map.items()}
solution = {}
for var_name, value in best_sample.items():
    if var_name in reverse_map and value == 1:
        key = reverse_map[var_name]
        solution[key] = 1

# Check constraints
violations = check_constraints(solution, data)
print(f'With explicit rotation penalty (5.0):')
print(f'  One-hot violations: {violations.one_hot_violations}')
print(f'  Rotation violations: {violations.rotation_violations}')
print(f'  Feasible: {violations.total_violations == 0}')

# Run multiple times
print('\nRunning 10 times with explicit rotation penalty:')
feasible_with_penalty = 0
for seed in range(10):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100)
    best_sample = sampleset.first.sample
    
    solution = {}
    for var_name, value in best_sample.items():
        if var_name in reverse_map and value == 1:
            key = reverse_map[var_name]
            solution[key] = 1
    
    violations = check_constraints(solution, data)
    if violations.total_violations == 0:
        feasible_with_penalty += 1

print(f'  Feasible solutions: {feasible_with_penalty}/10 ({feasible_with_penalty*10}%)')
