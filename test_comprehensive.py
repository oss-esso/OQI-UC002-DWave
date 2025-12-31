#!/usr/bin/env python3
"""
Comprehensive test of all unified benchmark modes on rotation_micro_25.

Tests:
1. Gurobi ground truth
2. SA native 6-family
3. SA hierarchical aggregated 
4. (Skip hybrid 27-food for 6-family scenario)

Compares objectives and validates consistency.
"""

import os
import sys
import json
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "unified_benchmark"))

from unified_benchmark.scenarios import load_scenario
from unified_benchmark.core import BenchmarkLogger, save_benchmark_results
from unified_benchmark.quantum_solvers import solve

def main():
    scenario = "rotation_micro_25"
    timeout = 60  # 60s per method
    num_reads = 50
    seed = 42
    
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK TEST")
    print(f"Scenario: {scenario}")
    print("=" * 70)
    
    # Load scenario
    data = load_scenario(scenario)
    print(f"Loaded: {data['n_farms']} farms × {data['n_foods']} foods = {data['n_vars']} vars")
    print(f"Total area: {data['total_area']:.2f}")
    print()
    
    results = []
    
    # 1. Gurobi ground truth
    print("\n" + "=" * 40)
    print("1. GUROBI GROUND TRUTH")
    print("=" * 40)
    r_gurobi = solve(
        mode="gurobi-true-ground-truth",
        scenario_data=data,
        timeout=timeout,
        seed=seed,
    )
    results.append(r_gurobi)
    print(f"Status: {r_gurobi.status}")
    print(f"Model obj: {r_gurobi.objective_model}")
    print(f"MIQP obj:  {r_gurobi.objective_miqp}")
    print(f"Time:      {r_gurobi.timing.solve_time:.2f}s")
    print(f"Feasible:  {r_gurobi.feasible}")
    
    gurobi_time = r_gurobi.timing.solve_time
    gurobi_obj = r_gurobi.objective_miqp
    
    # 2. SA native 6-family
    print("\n" + "=" * 40)
    print("2. SA NATIVE 6-FAMILY")
    print("=" * 40)
    r_native = solve(
        mode="qpu-native-6-family",
        scenario_data=data,
        use_qpu=False,
        num_reads=num_reads,
        timeout=timeout,
        seed=seed,
    )
    results.append(r_native)
    print(f"Status: {r_native.status}")
    print(f"Model obj: {r_native.objective_model}")
    print(f"MIQP obj:  {r_native.objective_miqp}")
    print(f"Time:      {r_native.timing.solve_time:.2f}s")
    print(f"Feasible:  {r_native.feasible}")
    
    # Add speedup info
    if gurobi_time and r_native.timing.solve_time:
        r_native.timing.gurobi_reference_time = gurobi_time
        r_native.timing.speedup_vs_wall = gurobi_time / r_native.timing.solve_time
        print(f"Speedup (vs Gurobi wall): {r_native.timing.speedup_vs_wall:.2f}x")
    
    # 3. SA hierarchical aggregated
    print("\n" + "=" * 40)
    print("3. SA HIERARCHICAL AGGREGATED")
    print("=" * 40)
    r_hier = solve(
        mode="qpu-hierarchical-aggregated",
        scenario_data=data,
        use_qpu=False,
        num_reads=num_reads,
        timeout=timeout,
        seed=seed,
        num_iterations=2,
    )
    results.append(r_hier)
    print(f"Status: {r_hier.status}")
    print(f"MIQP obj:  {r_hier.objective_miqp}")
    print(f"Time:      {r_hier.timing.solve_time:.2f}s")
    print(f"Feasible:  {r_hier.feasible}")
    
    if gurobi_time and r_hier.timing.solve_time:
        r_hier.timing.gurobi_reference_time = gurobi_time
        r_hier.timing.speedup_vs_wall = gurobi_time / r_hier.timing.solve_time
        print(f"Speedup (vs Gurobi wall): {r_hier.timing.speedup_vs_wall:.2f}x")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Mode':<35} {'Status':<10} {'MIQP Obj':>12} {'Time':>10} {'Speedup':>10}")
    print("-" * 77)
    for r in results:
        obj_str = f"{r.objective_miqp:.4f}" if r.objective_miqp else "N/A"
        time_str = f"{r.timing.solve_time:.2f}s" if r.timing.solve_time else "N/A"
        speedup = r.timing.speedup_vs_wall if r.timing.speedup_vs_wall else None
        speedup_str = f"{speedup:.2f}x" if speedup else "-"
        print(f"{r.mode:<35} {r.status:<10} {obj_str:>12} {time_str:>10} {speedup_str:>10}")
    
    print("\n" + "=" * 70)
    print("OBJECTIVE COMPARISON")
    print("=" * 70)
    if gurobi_obj:
        for r in results[1:]:
            if r.objective_miqp:
                gap = (gurobi_obj - r.objective_miqp) / gurobi_obj * 100
                print(f"{r.mode}: gap vs Gurobi = {gap:.1f}%")
    
    # Save results
    output_file = "comprehensive_test_results.json"
    save_benchmark_results(results, output_file)
    print(f"\nResults saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
