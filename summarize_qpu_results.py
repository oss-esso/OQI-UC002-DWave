#!/usr/bin/env python3
"""Summary of QPU benchmark results."""

import json

# Load results
results = json.load(open('qpu_hier_all_6family.json'))['runs']

print('=' * 90)
print('QPU HIERARCHICAL BENCHMARK RESULTS')
print('=' * 90)

print(f'\n{"Scenario":<30} {"Farms":>6} {"Vars":>6} {"Wall Time":>12} {"QPU Time":>12} {"MIQP Obj":>12}')
print('-' * 90)

for r in results:
    scenario = r['scenario_name']
    farms = r['n_farms']
    vars_ = r['n_vars']
    wall_time = r['timing']['total_wall_time']
    qpu_time = r['timing'].get('qpu_access_time', 0)
    obj = r['objective_miqp']
    
    obj_str = f'{obj:.2f}' if obj else 'N/A'
    qpu_str = f'{qpu_time:.4f}s' if qpu_time else 'N/A'
    
    print(f'{scenario:<30} {farms:>6} {vars_:>6} {wall_time:>11.2f}s {qpu_str:>12} {obj_str:>12}')

print()
print('=' * 90)
print('QPU TIMING ANALYSIS')
print('=' * 90)

total_qpu = sum(r['timing'].get('qpu_access_time', 0) for r in results)
total_wall = sum(r['timing']['total_wall_time'] for r in results)

print(f'\nTotal QPU access time: {total_qpu:.4f}s ({total_qpu*1000:.2f}ms)')
print(f'Total wall time: {total_wall:.2f}s')
print(f'QPU utilization: {total_qpu/total_wall*100:.2f}%')
print()
print('Note: Negative objectives indicate constraint violations.')
print('      Solution quality is poor without proper coefficient tuning.')
print('      The key result is QPU timing for speedup analysis.')
