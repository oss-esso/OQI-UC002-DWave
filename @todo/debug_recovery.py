#!/usr/bin/env python3
import sys
sys.path.insert(0, '..')

from data_loader_utils import load_food_data_as_dict
from adaptive_hybrid_solver import solve_adaptive_with_recovery
from collections import defaultdict

# Small test
data = load_food_data_as_dict('rotation_250farms_27foods')
data['farm_names'] = data['farm_names'][:3]
data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
data['total_area'] = sum(data['land_availability'].values())

print('Testing with 3 farms...')
result = solve_adaptive_with_recovery(data, num_reads=20, num_iterations=1, use_qpu=False, verbose=False)

print('\nFamily solution:')
for key, val in sorted(result['family_solution'].items()):
    if val == 1:
        print(f'  {key}')

print(f'\nFamily assignments: {result["n_assigned_family"]}')
print(f'Family objective: {result["objective_family"]:.4f}')

print('\n27-food solution:')
for key, val in sorted(result['food_solution'].items()):
    if val == 1:
        print(f'  {key}')

print(f'\nFood assignments: {result["n_assigned_food"]}')
print(f'27-food objective: {result["objective_27food"]:.4f}')

# Count assignments per farm-period
counts = defaultdict(int)
for (farm, food, period), val in result['food_solution'].items():
    if val == 1:
        counts[(farm, period)] += 1

print('\nAssignments per farm-period:')
for (farm, period), count in sorted(counts.items()):
    status = "✓ OK" if count == 1 else f"✗ PROBLEM ({count} crops)"
    print(f'  {farm}, period {period}: {count} crops {status}')
