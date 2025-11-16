#!/usr/bin/env python3
"""
Check consistency across all benchmark results.
"""

import json
import os

def check_results():
    """Check all benchmark results for consistency."""
    
    benchmark_dir = "Benchmarks/COMPREHENSIVE"
    
    solvers = {
        'Farm_PuLP': 'Farm Gurobi (PuLP)',
        'Farm_DWave': 'Farm D-Wave CQM',
        'Patch_PuLP': 'Patch Gurobi (PuLP)',
        'Patch_DWave': 'Patch D-Wave CQM',
        'Patch_GurobiQUBO': 'Patch Gurobi QUBO',
        'Patch_DWaveBQM': 'Patch D-Wave BQM'
    }
    
    print("="*80)
    print("BENCHMARK RESULTS CONSISTENCY CHECK")
    print("="*80)
    
    for solver_dir, solver_name in solvers.items():
        filepath = os.path.join(benchmark_dir, solver_dir, "config_10_run_1.json")
        
        if not os.path.exists(filepath):
            print(f"\n❌ {solver_name}: FILE NOT FOUND")
            continue
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"{solver_name}")
        print(f"{'='*80}")
        
        # Basic info
        print(f"Status: {data.get('status', 'N/A')}")
        print(f"Objective: {data.get('objective_value', 'N/A'):.6f}")
        print(f"Solve time: {data.get('solve_time', 'N/A'):.3f}s")
        
        # Timing details
        if 'qpu_time' in data:
            qpu = data.get('qpu_time')
            print(f"QPU time: {qpu:.6f}s" if qpu is not None else "QPU time: None/Not available")
        
        if 'hybrid_time' in data:
            hybrid = data.get('hybrid_time')
            print(f"Hybrid time: {hybrid:.6f}s" if hybrid is not None else "Hybrid time: None")
        
        # BQM-specific
        if 'bqm_energy' in data:
            print(f"BQM energy: {data.get('bqm_energy', 'N/A'):.6f}")
        
        if 'bqm_conversion_time' in data:
            print(f"BQM conversion: {data.get('bqm_conversion_time', 'N/A'):.6f}s")
        
        # Validation
        if 'validation' in data:
            val = data['validation']
            feasible = val.get('is_feasible', 'N/A')
            n_violations = val.get('n_violations', 0)
            print(f"Feasible: {feasible}, Violations: {n_violations}")
        
        # Sample info
        print(f"N units: {data.get('n_units', 'N/A')}")
        print(f"Total area: {data.get('total_area', 'N/A')} ha")
        print(f"N variables: {data.get('n_variables', 'N/A')}")
        print(f"N constraints: {data.get('n_constraints', 'N/A')}")
    
    # Cross-solver comparison
    print(f"\n{'='*80}")
    print("CROSS-SOLVER COMPARISON")
    print(f"{'='*80}")
    
    # Compare Patch solvers
    print(f"\nPatch Scenario Objectives:")
    patch_solvers = ['Patch_PuLP', 'Patch_DWave', 'Patch_GurobiQUBO', 'Patch_DWaveBQM']
    objectives = {}
    
    for solver_dir in patch_solvers:
        filepath = os.path.join(benchmark_dir, solver_dir, "config_10_run_1.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            obj = data.get('objective_value')
            objectives[solver_dir] = obj
            print(f"  {solvers[solver_dir]:30s}: {obj:.6f}")
    
    # Check if objectives are close
    if objectives:
        obj_values = list(objectives.values())
        min_obj = min(obj_values)
        max_obj = max(obj_values)
        diff = max_obj - min_obj
        diff_pct = (diff / min_obj) * 100 if min_obj > 0 else 0
        
        print(f"\nObjective range: {min_obj:.6f} to {max_obj:.6f}")
        print(f"Difference: {diff:.6f} ({diff_pct:.2f}%)")
        
        if diff_pct < 1:
            print("✅ All solvers agree within 1%")
        elif diff_pct < 5:
            print("⚠️ Solvers differ by {:.2f}% (acceptable)".format(diff_pct))
        else:
            print("❌ Solvers differ by {:.2f}% (significant!)".format(diff_pct))
    
    # Compare Farm solvers
    print(f"\nFarm Scenario Objectives:")
    farm_solvers = ['Farm_PuLP', 'Farm_DWave']
    farm_objectives = {}
    
    for solver_dir in farm_solvers:
        filepath = os.path.join(benchmark_dir, solver_dir, "config_10_run_1.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            obj = data.get('objective_value')
            farm_objectives[solver_dir] = obj
            print(f"  {solvers[solver_dir]:30s}: {obj:.6f}")
    
    # Check if farm objectives are close
    if farm_objectives:
        obj_values = list(farm_objectives.values())
        min_obj = min(obj_values)
        max_obj = max(obj_values)
        diff = max_obj - min_obj
        diff_pct = (diff / min_obj) * 100 if min_obj > 0 else 0
        
        print(f"\nObjective range: {min_obj:.6f} to {max_obj:.6f}")
        print(f"Difference: {diff:.6f} ({diff_pct:.2f}%)")
        
        if diff_pct < 1:
            print("✅ All solvers agree within 1%")
        elif diff_pct < 5:
            print("⚠️ Solvers differ by {:.2f}% (acceptable)".format(diff_pct))
        else:
            print("❌ Solvers differ by {:.2f}% (significant!)".format(diff_pct))

if __name__ == "__main__":
    check_results()
