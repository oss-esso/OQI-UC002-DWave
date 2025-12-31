#!/usr/bin/env python3
"""Compare QPU vs Gurobi results."""

import json

# Load QPU result
qpu = json.load(open('test_qpu_native.json'))['runs'][0]
gurobi = json.load(open('test_gurobi_300s.json'))['runs'][0]

print('=' * 70)
print('QPU vs GUROBI COMPARISON')
print('=' * 70)

print('')
print('GUROBI (300s timeout):')
print(f'  MIQP Objective: {gurobi["objective_miqp"]:.4f}')
print(f'  Solve Time: {gurobi["timing"]["solve_time"]:.2f}s')
print(f'  Feasible: {gurobi["feasible"]}')

print('')
print('QPU Native 6-family (100 reads):')
print(f'  MIQP Objective: {qpu["objective_miqp"]:.4f}')
print(f'  Wall Time: {qpu["timing"]["solve_time"]:.2f}s')
print(f'  QPU Access Time: {qpu["timing"].get("qpu_access_time", "N/A")}s')
print(f'  QPU Sampling Time: {qpu["timing"].get("qpu_sampling_time", "N/A")}s')
print(f'  Feasible: {qpu["feasible"]}')

gurobi_time = gurobi['timing']['solve_time']
qpu_wall = qpu['timing']['solve_time']
qpu_access = qpu['timing'].get('qpu_access_time', qpu_wall)

print('')
print('SPEEDUP (time only, ignoring solution quality):')
print(f'  vs Wall Time: {gurobi_time / qpu_wall:.2f}x')
print(f'  vs QPU Access: {gurobi_time / qpu_access:.0f}x')
print('')
print('NOTE: QPU solution has NEGATIVE objective - poor solution quality!')
print('      The QPU access time (40ms) is ~7500x faster than Gurobi (300s)')
print('      But wall time includes ~100s of network/embedding overhead.')
