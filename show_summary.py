#!/usr/bin/env python3
"""Summary comparison of all benchmark results."""

import json

# Load results
gurobi = json.load(open('test_gurobi_300s.json'))['runs'][0]
native = json.load(open('test_native_sa.json'))['runs'][0]
hier = json.load(open('test_hier_sa.json'))['runs'][0]
hybrid = json.load(open('test_hybrid_sa.json'))['runs'][0]

results = [
    ('Gurobi (300s)', gurobi),
    ('SA Native 6-family', native),
    ('SA Hierarchical', hier),
    ('SA Hybrid 27-food', hybrid),
]

print('=' * 85)
print('COMPREHENSIVE BENCHMARK RESULTS - rotation_micro_25 (5 farms x 6 foods)')
print('=' * 85)
header = f"{'Method':<25} {'Status':<10} {'MIQP Obj':>12} {'Solve Time':>12} {'Feasible':>10}"
print(header)
print('-' * 85)

gurobi_obj = gurobi['objective_miqp']
gurobi_time = gurobi['timing']['solve_time']

for name, r in results:
    obj = r['objective_miqp']
    obj_str = f'{obj:.4f}' if obj else 'N/A'
    time_str = f"{r['timing']['solve_time']:.2f}s"
    feas = 'Yes' if r['feasible'] else 'No'
    row = f"{name:<25} {r['status']:<10} {obj_str:>12} {time_str:>12} {feas:>10}"
    print(row)

print()
print('=' * 85)
print('SPEEDUP AND GAP ANALYSIS (vs Gurobi 300s)')
print('=' * 85)
header2 = f"{'Method':<25} {'Obj Gap %':>12} {'Wall Speedup':>15}"
print(header2)
print('-' * 52)

for name, r in results[1:]:
    obj = r['objective_miqp']
    if obj and gurobi_obj:
        gap = (gurobi_obj - obj) / gurobi_obj * 100
        speedup = gurobi_time / r['timing']['solve_time']
        row = f"{name:<25} {gap:>11.1f}% {speedup:>14.2f}x"
        print(row)

print()
print('=' * 85)
print('NOTES:')
print('=' * 85)
print('- Gurobi solves the FULL MIQP with indicator-linearized diversity bonus')
print('- SA methods solve SIMPLIFIED BQMs (6-family aggregation)')
print('- Objective gap is expected due to model simplification')
print('- Feasibility issues in SA methods are from soft one-hot constraint violations')
print('- Wall speedup = Gurobi_time / Method_time (higher = faster)')
