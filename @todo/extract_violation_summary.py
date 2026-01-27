#!/usr/bin/env python3
"""Extract violation summary from QPU benchmark results."""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent / 'qpu_benchmark_results'
files = {
    'small': BASE / 'qpu_benchmark_20251201_160444.json',
    'large': BASE / 'qpu_benchmark_20251201_200012.json',
    'hg_small': BASE / 'qpu_benchmark_20251203_121526.json',
    'hg_large': BASE / 'qpu_benchmark_20251203_133144.json'
}

print('=' * 100)
print('COMPREHENSIVE VIOLATION DATA SUMMARY')
print('=' * 100)

all_entries = []
for key, fpath in files.items():
    if not fpath.exists():
        continue
    data = json.load(open(fpath, encoding='utf-8'))
    scales = data.get('scales', [])
    methods = data.get('methods', [])
    
    print(f"\n--- {key}: {fpath.name} ---")
    print(f"Scales: {scales}, Methods: {len(methods)}")
    
    for res in data['results']:
        nf = res['n_farms']
        gt = res.get('ground_truth', {})
        gurobi = gt.get('objective', 0)
        
        for m, r in res.get('method_results', {}).items():
            if not r.get('success'):
                continue
            v = r.get('violations', 0)
            obj = r.get('objective', 0)
            feasible = r.get('feasible', True)
            viol_details = r.get('violation_details', {})
            
            entry = {
                'file': key,
                'n_farms': nf,
                'method': m,
                'violations': v,
                'objective': obj,
                'gurobi_obj': gurobi,
                'feasible': feasible,
                'gap_pct': (obj - gurobi) / gurobi * 100 if gurobi else 0,
                'one_hot_viols': viol_details.get('one_crop_per_farm', {}).get('violations', 0),
                'food_group_viols': viol_details.get('food_group_constraints', {}).get('violations', 0),
            }
            all_entries.append(entry)

# Print entries with violations
print("\n" + "=" * 100)
print("ENTRIES WITH VIOLATIONS")
print("=" * 100)
print(f"{'n_farms':>7} {'Method':<45} {'Viols':>5} {'Obj':>10} {'Gap%':>8} {'Type'}")
print("-" * 100)

for e in all_entries:
    if e['violations'] > 0:
        vtype = []
        if e['one_hot_viols'] > 0:
            vtype.append(f"one-hot:{e['one_hot_viols']}")
        if e['food_group_viols'] > 0:
            vtype.append(f"food-grp:{e['food_group_viols']}")
        vtype_str = ", ".join(vtype) if vtype else "unspecified"
        print(f"{e['n_farms']:>7} {e['method']:<45} {e['violations']:>5} {e['objective']:>10.4f} {e['gap_pct']:>+7.1f}% {vtype_str}")

# Summary by method
print("\n" + "=" * 100)
print("SUMMARY BY DECOMPOSITION STRATEGY")
print("=" * 100)

stats = defaultdict(lambda: {
    'total': 0, 'w_viols': 0, 'total_viols': 0,
    'scales': set(), 'avg_gap': []
})

for e in all_entries:
    m = e['method']
    stats[m]['total'] += 1
    stats[m]['scales'].add(e['n_farms'])
    stats[m]['avg_gap'].append(e['gap_pct'])
    if e['violations'] > 0:
        stats[m]['w_viols'] += 1
        stats[m]['total_viols'] += e['violations']

print(f"\n{'Method':<45} {'Total':>6} {'w/Viol':>6} {'TotV':>6} {'AvgGap%':>8} {'Scales'}")
print("-" * 110)
for m, s in sorted(stats.items()):
    avg = sum(s['avg_gap']) / len(s['avg_gap']) if s['avg_gap'] else 0
    scales = ",".join(map(str, sorted(s['scales'])))
    print(f"{m:<45} {s['total']:>6} {s['w_viols']:>6} {s['total_viols']:>6} {avg:>+7.1f}% {scales}")

# Violation-free methods
print("\n" + "=" * 100)
print("METHODS WITH ZERO VIOLATIONS")
print("=" * 100)
for m, s in sorted(stats.items()):
    if s['total_viols'] == 0:
        avg = sum(s['avg_gap']) / len(s['avg_gap']) if s['avg_gap'] else 0
        print(f"  {m}: {s['total']} entries, avg gap {avg:+.1f}%")

# Methods with violations
print("\n" + "=" * 100)
print("METHODS WITH VIOLATIONS - HEALING ANALYSIS")
print("=" * 100)
for m, s in sorted(stats.items()):
    if s['total_viols'] > 0:
        avg = sum(s['avg_gap']) / len(s['avg_gap']) if s['avg_gap'] else 0
        viol_rate = s['w_viols'] / s['total'] * 100
        print(f"  {m}:")
        print(f"    - {s['w_viols']}/{s['total']} entries with violations ({viol_rate:.0f}%)")
        print(f"    - Total violations: {s['total_viols']}")
        print(f"    - Average gap: {avg:+.1f}%")
