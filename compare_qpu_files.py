#!/usr/bin/env python3
"""Compare QPU result files from run 1."""

import json

# Compare the different files
print('COMPARISON: qpu_hier_all_6family vs qpu_hier_repaired')
print('=' * 100)

with open('qpu_hier_all_6family.json') as f:
    original = json.load(f)
with open('qpu_hier_repaired.json') as f:
    repaired = json.load(f)

print(f"Original: {len(original['runs'])} runs")
print(f"Repaired: {len(repaired['runs'])} runs")

# Get common scenarios
orig_scenarios = {r['scenario_name'] for r in original['runs']}
rep_scenarios = {r['scenario_name'] for r in repaired['runs']}

print(f"\nOriginal scenarios: {sorted(orig_scenarios)}")
print(f"\nRepaired scenarios: {sorted(rep_scenarios)}")
print(f"\nAdded in repaired: {sorted(rep_scenarios - orig_scenarios)}")

# Compare objectives for common scenarios
print('\n' + '-' * 100)
print('OBJECTIVE COMPARISON (common scenarios)')
print(f"{'Scenario':<35} {'Orig Obj':>12} {'Rep Obj':>12} {'Diff':>12} {'Diff%':>10}")
print('-' * 100)

orig_by_name = {r['scenario_name']: r for r in original['runs']}
rep_by_name = {r['scenario_name']: r for r in repaired['runs']}

for sc in sorted(orig_scenarios & rep_scenarios):
    orig_obj = orig_by_name[sc]['objective_miqp']
    rep_obj = rep_by_name[sc]['objective_miqp']
    diff = rep_obj - orig_obj
    diff_pct = (diff / abs(orig_obj)) * 100 if orig_obj != 0 else 0
    print(f"{sc:<35} {orig_obj:>12.2f} {rep_obj:>12.2f} {diff:>12.2f} {diff_pct:>9.1f}%")

# Now look at all native vs hierarchical
print('\n' + '=' * 100)
print('NATIVE vs HIERARCHICAL COMPARISON')
print('=' * 100)

with open('qpu_native_6family.json') as f:
    native = json.load(f)

print(f"\nNative 6-Family runs: {len(native['runs'])}")

native_by_name = {r['scenario_name']: r for r in native['runs']}

common = set(native_by_name.keys()) & set(orig_by_name.keys())
print(f"\nCommon scenarios: {sorted(common)}")

print(f"\n{'Scenario':<35} {'Native':>12} {'Hier(Orig)':>12} {'Hier(Rep)':>12}")
print('-' * 100)
for sc in sorted(common):
    nat_obj = native_by_name[sc].get('objective_miqp', 0) or 0
    orig_obj = orig_by_name[sc]['objective_miqp'] or 0
    rep_obj = rep_by_name.get(sc, {}).get('objective_miqp', 0) or 0
    print(f"{sc:<35} {nat_obj:>12.2f} {orig_obj:>12.2f} {rep_obj:>12.2f}")

# Check hybrid 27-food
print('\n' + '=' * 100)
print('27-FOOD HYBRID RESULTS')
print('=' * 100)

with open('qpu_hybrid_27food.json') as f:
    hybrid = json.load(f)

print(f"\nHybrid 27-Food runs: {len(hybrid['runs'])}")
for r in hybrid['runs']:
    sc = r['scenario_name']
    obj = r.get('objective_miqp', 'N/A')
    timing = r.get('timing', {})
    total = timing.get('total_wall_time', 0)
    qpu = timing.get('qpu_access_time', 0)
    print(f"  {sc}: obj={obj}, total={total:.1f}s, qpu={qpu:.3f}s")

# Summary statistics
print('\n' + '=' * 100)
print('SUMMARY: All Available QPU Data')
print('=' * 100)

all_files = {
    'qpu_hier_all_6family.json': 'Hierarchical 6-Family (original)',
    'qpu_hier_repaired.json': 'Hierarchical Repaired',
    'qpu_native_6family.json': 'Native 6-Family',
    'qpu_hybrid_27food.json': 'Hybrid 27-Food',
}

for fname, desc in all_files.items():
    with open(fname) as f:
        data = json.load(f)
    runs = data['runs']
    total_qpu_time = sum(r.get('timing', {}).get('qpu_access_time', 0) for r in runs)
    total_wall_time = sum(r.get('timing', {}).get('total_wall_time', 0) for r in runs)
    max_vars = max(r.get('n_vars', 0) for r in runs)
    print(f"\n{desc}:")
    print(f"  Runs: {len(runs)}")
    print(f"  Max vars: {max_vars}")
    print(f"  Total wall time: {total_wall_time:.1f}s")
    print(f"  Total pure QPU time: {total_qpu_time:.3f}s ({100*total_qpu_time/total_wall_time:.1f}%)")
