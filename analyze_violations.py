#!/usr/bin/env python3
"""Analyze constraint violations across all QPU methods."""

import json
import pandas as pd
from collections import defaultdict

files = {
    'qpu_hier_repaired.json': 'Hierarchical (Repaired)',
    'qpu_hier_all_6family.json': 'Hierarchical (Original)',
    'qpu_native_6family.json': 'Native',
    'qpu_hybrid_27food.json': 'Hybrid 27-Food',
}

print('=' * 120)
print('CONSTRAINT VIOLATION ANALYSIS')
print('=' * 120)

all_data = []

for fname, label in files.items():
    with open(fname) as f:
        data = json.load(f)
    
    print(f"\n{label} ({fname})")
    print('-' * 80)
    
    total_viols = 0
    scenarios_with_viols = 0
    
    for r in data['runs']:
        sc = r.get('scenario_name', 'N/A')
        status = r.get('status', 'N/A')
        obj = r.get('objective_miqp')
        viols = r.get('constraint_violations', {})
        
        one_hot = viols.get('one_hot_violations', 0)
        rotation = viols.get('rotation_violations', 0)
        total = viols.get('total_violations', 0)
        
        all_data.append({
            'method': label,
            'scenario': sc,
            'status': status,
            'objective': obj,
            'one_hot_violations': one_hot,
            'rotation_violations': rotation,
            'total_violations': total,
            'n_vars': r.get('n_vars', 0),
        })
        
        if total > 0:
            scenarios_with_viols += 1
            total_viols += total
            print(f"  {sc}: {total} violations (one_hot={one_hot}, rotation={rotation})")
            details = viols.get('details', [])
            for d in details[:3]:
                print(f"    - {d}")
            if len(details) > 3:
                print(f"    ... and {len(details) - 3} more")
        else:
            print(f"  {sc}: 0 violations")
    
    n_runs = len(data['runs'])
    print(f"\n  SUMMARY: {scenarios_with_viols}/{n_runs} scenarios with violations, {total_viols} total violations")

# Create comparison DataFrame
df = pd.DataFrame(all_data)

print("\n" + "=" * 120)
print("VIOLATION SUMMARY BY METHOD")
print("=" * 120)

summary = df.groupby('method').agg({
    'scenario': 'count',
    'total_violations': ['sum', 'mean', 'max'],
    'one_hot_violations': 'sum',
    'rotation_violations': 'sum',
}).round(2)

print(summary)

print("\n" + "=" * 120)
print("OBJECTIVE vs VIOLATIONS CORRELATION")
print("=" * 120)

# For each method, check if violations correlate with objective
for method in df['method'].unique():
    method_df = df[df['method'] == method]
    successful = method_df[method_df['status'] == 'feasible']
    
    if len(successful) > 1 and successful['objective'].notna().any():
        # Check correlation
        viols = successful['total_violations']
        objs = successful['objective'].abs()
        
        if viols.std() > 0 and objs.std() > 0:
            corr = viols.corr(objs)
            print(f"\n{method}:")
            print(f"  Scenarios with violations: {(successful['total_violations'] > 0).sum()}/{len(successful)}")
            print(f"  Correlation (violations vs |objective|): {corr:.3f}")
        else:
            print(f"\n{method}:")
            print(f"  All scenarios have same violation count or objective")

print("\n" + "=" * 120)
print("DETAILED COMPARISON: SCENARIOS WITH VIOLATIONS")
print("=" * 120)

# Show scenarios where violations exist
viols_df = df[df['total_violations'] > 0].sort_values(['scenario', 'method'])
if len(viols_df) > 0:
    print("\n" + viols_df[['method', 'scenario', 'n_vars', 'objective', 'total_violations', 'one_hot_violations', 'rotation_violations']].to_string(index=False))
else:
    print("\nNo scenarios with violations found!")

# Check if violations explain the gap
print("\n" + "=" * 120)
print("KEY QUESTION: DO VIOLATIONS EXPLAIN THE OBJECTIVE GAP?")
print("=" * 120)

# Compare Hier(Rep) which we use vs violations
hier_rep = df[df['method'] == 'Hierarchical (Repaired)']
print(f"\nHierarchical (Repaired) - Our primary method:")
print(f"  Total scenarios: {len(hier_rep)}")
print(f"  Scenarios with violations: {(hier_rep['total_violations'] > 0).sum()}")
print(f"  Total violations across all: {hier_rep['total_violations'].sum()}")

if hier_rep['total_violations'].sum() > 0:
    print("\n  Violations breakdown:")
    print(f"    One-hot (crop selection): {hier_rep['one_hot_violations'].sum()}")
    print(f"    Rotation (crop rotation): {hier_rep['rotation_violations'].sum()}")
    
    # Show which scenarios
    viol_scenarios = hier_rep[hier_rep['total_violations'] > 0]
    print(f"\n  Affected scenarios:")
    for _, row in viol_scenarios.iterrows():
        print(f"    {row['scenario']}: {row['total_violations']} violations, obj={row['objective']:.2f}")
else:
    print("\n  ✅ NO VIOLATIONS - Objective gap is NOT due to constraint violations!")
    print("  The gap reflects different optimization strategies, not infeasibility.")
