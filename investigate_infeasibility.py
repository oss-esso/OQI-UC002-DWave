#!/usr/bin/env python3
"""Investigate infeasibility causes in SA methods."""

import json
import sys
sys.path.insert(0, '.')

from unified_benchmark.scenarios import load_scenario
from unified_benchmark.miqp_scorer import check_constraints, compute_miqp_objective

# Load results
native = json.load(open('test_native_sa.json'))['runs'][0]
hier = json.load(open('test_hier_sa.json'))['runs'][0]
hybrid = json.load(open('test_hybrid_sa.json'))['runs'][0]

print('=' * 70)
print('CONSTRAINT VIOLATION ANALYSIS')
print('=' * 70)

for name, r in [('Native 6-family', native), ('Hierarchical', hier), ('Hybrid 27-food', hybrid)]:
    print(f'\n{name}:')
    cv = r.get('constraint_violations', {})
    print(f'  One-hot violations: {cv.get("one_hot_violations", "N/A")}')
    print(f'  Rotation violations: {cv.get("rotation_violations", "N/A")}')
    print(f'  Total violations: {cv.get("total_violations", "N/A")}')
    details = cv.get('details', [])
    if details:
        print(f'  Details:')
        for d in details[:10]:
            print(f'    - {d}')

# Now let's investigate the solution structure more deeply
print('\n' + '=' * 70)
print('SOLUTION STRUCTURE ANALYSIS')
print('=' * 70)

# Load scenario
data = load_scenario('rotation_micro_25')
n_farms = data['n_farms']
n_foods = data['n_foods']
n_periods = data['n_periods']

print(f'\nScenario: {n_farms} farms × {n_foods} foods × {n_periods} periods')

# Re-run constraint check with detailed output
from unified_benchmark.quantum_solvers import solve

print('\n' + '=' * 70)
print('DETAILED NATIVE 6-FAMILY ANALYSIS')
print('=' * 70)

result = solve(
    mode="qpu-native-6-family",
    scenario_data=data,
    use_qpu=False,
    num_reads=50,
    timeout=60,
    seed=42,
)

# Analyze the solution
solution = result.solution
if solution:
    print(f'\nSolution has {len(solution)} non-zero entries')
    
    # Count crops per farm per period
    crop_counts = {}
    for (f_idx, c_idx, t), val in solution.items():
        if val == 1:
            key = (f_idx, t)
            if key not in crop_counts:
                crop_counts[key] = []
            crop_counts[key].append(c_idx)
    
    print('\nCrops per farm per period:')
    for (f_idx, t), crops in sorted(crop_counts.items()):
        status = 'OK' if 1 <= len(crops) <= 2 else 'VIOLATION'
        print(f'  Farm {f_idx}, Period {t}: {len(crops)} crops {crops} [{status}]')
    
    # Check rotation violations
    print('\nRotation constraint check (same crop in consecutive periods):')
    rotation_violations = []
    for f_idx in range(n_farms):
        for c_idx in range(n_foods):
            for t in range(1, n_periods):
                y_t = solution.get((f_idx, c_idx, t), 0)
                y_t1 = solution.get((f_idx, c_idx, t+1), 0)
                if y_t == 1 and y_t1 == 1:
                    rotation_violations.append((f_idx, c_idx, t, t+1))
                    print(f'  VIOLATION: Farm {f_idx}, Crop {c_idx} in periods {t} and {t+1}')
    
    if not rotation_violations:
        print('  No rotation violations found')
