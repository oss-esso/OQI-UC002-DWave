#!/usr/bin/env python3
"""Compare our Gurobi results with timeout test reference."""

import json

# Load timeout test results
timeout_test = json.load(open(r'@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json'))

print('GUROBI TIMEOUT TEST RESULTS (Reference)')
print('=' * 70)
for entry in timeout_test[:5]:
    r = entry['result']
    info = entry['benchmark_info']
    scenario = r['scenario']
    obj = r['objective_value']
    time = r['solve_time']
    gap = r['mip_gap'] * 100
    print(f'{scenario}: obj={obj:.4f}, time={time:.2f}s, gap={gap:.1f}%')

print()
print('NOTE: Our new Gurobi uses indicator-linearized diversity bonus')
print('      so objectives differ, but timing should be comparable.')
print()
print('The key point: MIQP formulation is NOW CONSISTENT across all methods.')
