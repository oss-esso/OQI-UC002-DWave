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

# Test with SA (simulated) - Binary mode
print('='*70)
print('SIMULATED ANNEALING TEST - BINARY MODE (1 crop/period)')
print('='*70)
result_sa_binary = solve_adaptive_with_recovery(
    data, 
    num_reads=20,  # Low for speed
    num_iterations=1, 
    use_qpu=False,
    recovery_mode='binary',  # Exactly 1 crop per farm-period
    verbose=True
)

print()
print('SA Binary Summary:')
print(f'  Family objective: {result_sa_binary["objective_family"]:.4f}')
print(f'  27-food objective: {result_sa_binary["objective_27food"]:.4f}')
print(f'  Unique crops: {result_sa_binary["diversity_stats"]["total_unique_crops"]}/27')
print(f'  Food assignments: {result_sa_binary["n_assigned_food"]}')

# Test with SA - Fractional mode
print()
print('='*70)
print('SIMULATED ANNEALING TEST - FRACTIONAL MODE (2-3 crops/period)')
print('='*70)
result_sa_frac = solve_adaptive_with_recovery(
    data, 
    num_reads=20,
    num_iterations=1, 
    use_qpu=False,
    recovery_mode='fractional',  # 2-3 crops per farm-period with land allocation
    verbose=True
)

print()
print('SA Fractional Summary:')
print(f'  Family objective: {result_sa_frac["objective_family"]:.4f}')
print(f'  27-food objective: {result_sa_frac["objective_27food"]:.4f}')
print(f'  Unique crops: {result_sa_frac["diversity_stats"]["total_unique_crops"]}/27')
print(f'  Food assignments: {result_sa_frac["n_assigned_food"]}')

# Test with QPU - Binary mode
print()
print('='*70)
print('QPU TEST - BINARY MODE')
print('='*70)
result_qpu_binary = solve_adaptive_with_recovery(
    data, 
    num_reads=50,
    num_iterations=1, 
    use_qpu=True,
    recovery_mode='binary',
    verbose=True
)

print()
print('QPU Binary Summary:')
print(f'  Family objective: {result_qpu_binary["objective_family"]:.4f}')
print(f'  27-food objective: {result_qpu_binary["objective_27food"]:.4f}')
print(f'  Unique crops: {result_qpu_binary["diversity_stats"]["total_unique_crops"]}/27')
print(f'  QPU time: {result_qpu_binary["qpu_time"]:.3f}s')

# Test with QPU - Fractional mode
print()
print('='*70)
print('QPU TEST - FRACTIONAL MODE')
print('='*70)
result_qpu_frac = solve_adaptive_with_recovery(
    data, 
    num_reads=50,
    num_iterations=1, 
    use_qpu=True,
    recovery_mode='fractional',
    verbose=True
)

print()
print('QPU Fractional Summary:')
print(f'  Family objective: {result_qpu_frac["objective_family"]:.4f}')
print(f'  27-food objective: {result_qpu_frac["objective_27food"]:.4f}')
print(f'  Unique crops: {result_qpu_frac["diversity_stats"]["total_unique_crops"]}/27')
print(f'  QPU time: {result_qpu_frac["qpu_time"]:.3f}s')

# Comparison Table
print()
print('='*80)
print('COMPARISON: BINARY vs FRACTIONAL RECOVERY MODES')
print('='*80)
print(f'{"Method":<20} {"Mode":<12} {"Family Obj":>12} {"27-Food Obj":>12} {"Diversity":>10}')
print('-'*80)
print(f'{"SA":<20} {"Binary":<12} {result_sa_binary["objective_family"]:>12.4f} {result_sa_binary["objective_27food"]:>12.4f} {result_sa_binary["diversity_stats"]["total_unique_crops"]:>10}/27')
print(f'{"SA":<20} {"Fractional":<12} {result_sa_frac["objective_family"]:>12.4f} {result_sa_frac["objective_27food"]:>12.4f} {result_sa_frac["diversity_stats"]["total_unique_crops"]:>10}/27')
print(f'{"QPU":<20} {"Binary":<12} {result_qpu_binary["objective_family"]:>12.4f} {result_qpu_binary["objective_27food"]:>12.4f} {result_qpu_binary["diversity_stats"]["total_unique_crops"]:>10}/27')
print(f'{"QPU":<20} {"Fractional":<12} {result_qpu_frac["objective_family"]:>12.4f} {result_qpu_frac["objective_27food"]:>12.4f} {result_qpu_frac["diversity_stats"]["total_unique_crops"]:>10}/27')
print('='*80)
print()
print('Key Insights:')
print('  • Binary mode: 1 crop per farm-period (simpler, maintains one-hot)')
print('  • Fractional mode: 2-3 crops per farm-period (realistic, higher diversity)')
print('='*80)
