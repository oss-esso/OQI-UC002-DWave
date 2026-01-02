#!/usr/bin/env python3
"""
Comprehensive analysis of all QPU data from 'run 1' and subsequent runs.
Creates summary tables and identifies what data is available.
"""

import json
from pathlib import Path
from collections import defaultdict

print("=" * 120)
print("COMPREHENSIVE QPU DATA INVENTORY")
print("=" * 120)

# Categorize all JSON files
qpu_files = list(Path('.').glob('qpu_*.json'))
benchmark_files = list(Path('.').glob('benchmark_*.json'))
test_files = list(Path('.').glob('test_*.json'))

print(f"\nFound {len(qpu_files)} QPU files, {len(benchmark_files)} benchmark files, {len(test_files)} test files")

# Analyze QPU files in detail
print("\n" + "=" * 120)
print("QPU DATA FILES DETAILED ANALYSIS")
print("=" * 120)

all_qpu_data = {}
for f in sorted(qpu_files):
    try:
        with open(f) as fp:
            data = json.load(fp)
        runs = data.get('runs', [])
        
        all_qpu_data[f.name] = {
            'generated_at': data.get('generated_at', 'N/A'),
            'runs': runs,
            'n_runs': len(runs),
        }
        
        print(f"\n--- {f.name} ---")
        print(f"Generated: {data.get('generated_at', 'N/A')}")
        print(f"Runs: {len(runs)}")
        
        # Group by mode/method
        modes = defaultdict(list)
        for r in runs:
            mode = r.get('mode', r.get('method', 'unknown'))
            modes[mode].append(r)
        
        for mode, mode_runs in modes.items():
            print(f"\n  Mode: {mode}")
            print(f"  {'Scenario':<35} {'Vars':>8} {'Obj':>12} {'Time(s)':>10} {'QPU(s)':>10} {'Status':>10}")
            for r in mode_runs:
                sc = r.get('scenario_name', 'N/A')
                n_vars = r.get('n_vars', 0)
                obj = r.get('objective_miqp', r.get('objective'))
                obj_str = f"{obj:.2f}" if obj is not None else 'N/A'
                timing = r.get('timing', {})
                total = timing.get('total_wall_time', 0)
                qpu = timing.get('qpu_access_time', 0)
                status = r.get('status', 'N/A')
                print(f"  {sc:<35} {n_vars:>8} {obj_str:>12} {total:>10.2f} {qpu:>10.3f} {status:>10}")
                
    except Exception as e:
        print(f"\n--- {f.name} ---")
        print(f"Error: {e}")

# Summary by scenario
print("\n" + "=" * 120)
print("SCENARIO COVERAGE MATRIX")
print("=" * 120)

# Collect all scenarios and their results from different files
scenario_results = defaultdict(dict)
for fname, fdata in all_qpu_data.items():
    for r in fdata['runs']:
        sc = r.get('scenario_name', 'unknown')
        mode = r.get('mode', r.get('method', 'unknown'))
        key = f"{fname}:{mode}"
        obj = r.get('objective_miqp', r.get('objective'))
        scenario_results[sc][key] = obj

all_scenarios = sorted(scenario_results.keys())
print(f"\nTotal unique scenarios: {len(all_scenarios)}")

# Group scenarios by type
scenarios_6fam = [s for s in all_scenarios if '27foods' not in s and '27food' not in s]
scenarios_27food = [s for s in all_scenarios if '27foods' in s or '27food' in s]

print(f"\n6-Family scenarios ({len(scenarios_6fam)}):")
for sc in sorted(scenarios_6fam):
    sources = list(scenario_results[sc].keys())
    best_obj = min([v for v in scenario_results[sc].values() if v is not None], default=None)
    best_str = f"{best_obj:.2f}" if best_obj is not None else 'N/A'
    print(f"  {sc:<40} best={best_str:>10} sources={len(sources)}")

print(f"\n27-Food scenarios ({len(scenarios_27food)}):")
for sc in sorted(scenarios_27food):
    sources = list(scenario_results[sc].keys())
    best_obj = min([v for v in scenario_results[sc].values() if v is not None], default=None)
    best_str = f"{best_obj:.2f}" if best_obj is not None else 'N/A'
    print(f"  {sc:<40} best={best_str:>10} sources={len(sources)}")

# Check test files (likely Simulated Annealing)
print("\n" + "=" * 120)
print("TEST FILES (Simulated Annealing Baselines)")
print("=" * 120)

for f in sorted(test_files):
    try:
        with open(f) as fp:
            data = json.load(fp)
        runs = data.get('runs', [])
        print(f"\n{f.name}: {len(runs)} runs")
        for r in runs[:3]:
            mode = r.get('mode', r.get('method', 'unknown'))
            sc = r.get('scenario_name', 'N/A')
            sampler = r.get('sampler', r.get('backend', 'N/A'))
            print(f"  {sc} - {mode} - {sampler}")
    except Exception as e:
        print(f"{f.name}: Error - {e}")

# Final summary
print("\n" + "=" * 120)
print("RECOMMENDATIONS FOR ANALYSIS")
print("=" * 120)

print("""
Based on the data inventory:

1. PRIMARY QPU DATA:
   - qpu_hier_repaired.json: Most complete, 13 scenarios (9 6-Family + 4 27-Food)
   - This is the "repaired" version with corrected objective calculations

2. ALTERNATIVE QPU DATA:
   - qpu_hier_all_6family.json: Original 6-Family hierarchical (9 scenarios)
   - qpu_native_6family.json: Native embedding (only micro_25 worked, others errored)
   - qpu_hybrid_27food.json: Hybrid approach for 27-Food (4 scenarios, 2 with results)

3. GUROBI BASELINES:
   - gurobi_baseline_60s.json: 60s timeout
   - @todo/gurobi_timeout_verification/gurobi_timeout_test_*.json: 300s timeout

4. KEY OBSERVATIONS:
   - Native embedding failed for problems > 90 variables (embedding too large)
   - Hierarchical decomposition successfully scales to 16,200 variables
   - Hybrid 27-Food only completed 2 of 4 scenarios
   - "Repaired" version has better objective values than original

5. FOR THE PAPER:
   - Use qpu_hier_repaired.json as primary QPU results
   - Compare against 300s Gurobi timeout for fair comparison
   - Note that Native embedding is not viable beyond tiny problems
""")
