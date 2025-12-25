#!/usr/bin/env python3
"""Quick test of the adaptive hybrid solver with 27-food recovery."""

import sys
sys.path.insert(0, '..')

from data_loader_utils import load_food_data_as_dict
from adaptive_hybrid_solver import solve_adaptive_with_recovery

print('Testing adaptive solver with 27-food recovery...')
print()

# Load 27-food scenario - small test
data = load_food_data_as_dict('rotation_250farms_27foods')
data['farm_names'] = data['farm_names'][:5]  # Small for fast test
data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
data['total_area'] = sum(data['land_availability'].values())

n_farms = len(data['farm_names'])
n_foods = len(data['food_names'])
print(f'Problem: {n_farms} farms x {n_foods} foods')
print()

# Test with SA (simulated)
print('='*70)
print('SIMULATED ANNEALING TEST')
print('='*70)
result_sa = solve_adaptive_with_recovery(
    data, 
    num_reads=20,  # Low for speed
    num_iterations=1, 
    use_qpu=False, 
    verbose=True
)

print()
print('SA Summary:')
print(f'  Family objective: {result_sa["objective_family"]:.4f}')
print(f'  27-food objective: {result_sa["objective_27food"]:.4f}')
print(f'  Unique crops: {result_sa["diversity_stats"]["total_unique_crops"]}/27')
print(f'  Family assignments: {result_sa["n_assigned_family"]}')
print(f'  Food assignments: {result_sa["n_assigned_food"]}')

# Test with QPU
print()
print('='*70)
print('QPU TEST')
print('='*70)
result_qpu = solve_adaptive_with_recovery(
    data, 
    num_reads=50,
    num_iterations=1, 
    use_qpu=True, 
    verbose=True
)

print()
print('QPU Summary:')
print(f'  Family objective: {result_qpu["objective_family"]:.4f}')
print(f'  27-food objective: {result_qpu["objective_27food"]:.4f}')
print(f'  Unique crops: {result_qpu["diversity_stats"]["total_unique_crops"]}/27')
print(f'  QPU time: {result_qpu["qpu_time"]:.3f}s')

# Comparison
print()
print('='*70)
print('COMPARISON: SA vs QPU')
print('='*70)
print(f'             SA           QPU')
print(f'  Family:    {result_sa["objective_family"]:.4f}       {result_qpu["objective_family"]:.4f}')
print(f'  27-food:   {result_sa["objective_27food"]:.4f}       {result_qpu["objective_27food"]:.4f}')
print(f'  Diversity: {result_sa["diversity_stats"]["total_unique_crops"]}/27         {result_qpu["diversity_stats"]["total_unique_crops"]}/27')
print('='*70)
