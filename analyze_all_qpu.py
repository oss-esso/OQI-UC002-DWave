#!/usr/bin/env python3
"""Analyze all QPU data files from run 1."""

import json

print('=' * 100)
print('NATIVE QPU RESULTS ANALYSIS')
print('=' * 100)

with open('qpu_native_6family.json') as f:
    native = json.load(f)

print(f"\nNative 6-Family runs: {len(native['runs'])}")
header = f"{'Scenario':<35} {'Vars':>8} {'Objective':>12} {'Total(s)':>10} {'QPU(s)':>10} {'Status':>12}"
print(header)
print('-' * 100)

for r in native['runs']:
    sc = r['scenario_name']
    n_vars = r.get('n_vars', 0)
    obj = r.get('objective_miqp')
    timing = r.get('timing', {})
    total = timing.get('total_wall_time', 0)
    qpu = timing.get('qpu_access_time', 0)
    status = r.get('status', 'N/A')
    
    obj_str = f'{obj:.2f}' if obj is not None else 'None'
    print(f"{sc:<35} {n_vars:>8} {obj_str:>12} {total:>10.2f} {qpu:>10.3f} {status:>12}")

# Check constraint violations
print('\n' + '-' * 100)
print('CONSTRAINT VIOLATIONS:')
for r in native['runs']:
    sc = r['scenario_name']
    viols = r.get('constraint_violations', {})
    total_viols = viols.get('total_violations', 0)
    if total_viols > 0:
        print(f"  {sc}: {total_viols} violations")
        details = viols.get('details', [])
        for d in details[:3]:
            print(f"    - {d}")

# Now compare ALL methods
print('\n' + '=' * 100)
print('COMPREHENSIVE COMPARISON: ALL QPU METHODS')
print('=' * 100)

files = {
    'qpu_native_6family.json': 'Native',
    'qpu_hier_all_6family.json': 'Hier(Orig)',
    'qpu_hier_repaired.json': 'Hier(Rep)',
}

data = {}
all_scenarios = set()
for fname, label in files.items():
    with open(fname) as f:
        d = json.load(f)
    data[label] = {r['scenario_name']: r for r in d['runs']}
    all_scenarios.update(data[label].keys())

# Filter to 6-family scenarios
scenarios_6fam = [s for s in all_scenarios if '27foods' not in s]

print(f"\n{'Scenario':<30} {'Native':>10} {'Hier(O)':>10} {'Hier(R)':>10} {'Best':>10}")
print('-' * 80)

for sc in sorted(scenarios_6fam):
    values = {}
    for label in files.values():
        if sc in data[label]:
            obj = data[label][sc].get('objective_miqp')
            values[label] = obj if obj is not None else float('nan')
        else:
            values[label] = float('nan')
    
    # Best is most negative (maximizing benefit = minimizing cost)
    valid = {k: v for k, v in values.items() if v == v}  # filter NaN
    best = min(valid.values()) if valid else float('nan')
    best_label = [k for k, v in valid.items() if v == best][0] if valid else 'N/A'
    
    native_str = f"{values['Native']:.2f}" if values['Native'] == values['Native'] else 'N/A'
    hier_o_str = f"{values['Hier(Orig)']:.2f}" if values['Hier(Orig)'] == values['Hier(Orig)'] else 'N/A'
    hier_r_str = f"{values['Hier(Rep)']:.2f}" if values['Hier(Rep)'] == values['Hier(Rep)'] else 'N/A'
    
    print(f"{sc:<30} {native_str:>10} {hier_o_str:>10} {hier_r_str:>10} {best_label:>10}")

# 27-food comparison
print('\n' + '=' * 100)
print('27-FOOD SCENARIOS')
print('=' * 100)

with open('qpu_hybrid_27food.json') as f:
    hybrid = json.load(f)

hybrid_by_name = {r['scenario_name']: r for r in hybrid['runs']}

scenarios_27food = [s for s in all_scenarios if '27foods' in s]
if scenarios_27food:
    print(f"\n{'Scenario':<35} {'Hier(Rep)':>12} {'Hybrid':>12}")
    print('-' * 65)
    for sc in sorted(scenarios_27food):
        hier_obj = data['Hier(Rep)'].get(sc, {}).get('objective_miqp', 'N/A')
        hybrid_obj = hybrid_by_name.get(sc, {}).get('objective_miqp', 'N/A')
        
        hier_str = f"{hier_obj:.2f}" if isinstance(hier_obj, (int, float)) else 'N/A'
        hybrid_str = f"{hybrid_obj:.2f}" if isinstance(hybrid_obj, (int, float)) else 'N/A'
        print(f"{sc:<35} {hier_str:>12} {hybrid_str:>12}")
