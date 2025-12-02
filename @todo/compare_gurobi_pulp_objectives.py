#!/usr/bin/env python3
"""
Compare Gurobi Ground Truth (QPU Benchmark) vs Patch PuLP Results

This script compares the objective values from:
1. QPU Benchmark's Gurobi ground_truth (solve_ground_truth)
2. Comprehensive Benchmark's Patch_PuLP results (solve_with_pulp_plots)

Key differences identified:
- QPU Benchmark: sum(Y) == 1 (EXACTLY one crop per farm)
- Patch PuLP: sum(Y) <= 1 (AT MOST one crop per farm)

Author: OQI-UC002-DWave
Date: 2025-12-01
"""

import json
import os
from pathlib import Path
from typing import Dict, List

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
QPU_RESULTS_DIR = PROJECT_ROOT / "@todo" / "qpu_benchmark_results"
PATCH_PULP_DIR = PROJECT_ROOT / "Benchmarks" / "COMPREHENSIVE" / "Patch_PuLP"


def load_qpu_benchmark_results() -> Dict:
    """Load the most recent QPU benchmark results."""
    json_files = sorted(QPU_RESULTS_DIR.glob("*.json"), reverse=True)
    if not json_files:
        print("No QPU benchmark results found!")
        return {}
    
    latest = json_files[0]
    print(f"Loading QPU benchmark: {latest.name}")
    with open(latest) as f:
        return json.load(f)


def load_patch_pulp_results() -> Dict[int, Dict]:
    """Load all Patch PuLP results keyed by config size."""
    results = {}
    for json_file in PATCH_PULP_DIR.glob("config_*_run_*.json"):
        # Parse config size from filename
        parts = json_file.stem.split("_")
        config_size = int(parts[1])
        
        with open(json_file) as f:
            data = json.load(f)
            results[config_size] = data
    
    return results


def compare_formulations():
    """Compare objective values between the two formulations."""
    print("=" * 80)
    print("COMPARISON: Gurobi Ground Truth (QPU Benchmark) vs Patch PuLP")
    print("=" * 80)
    print()
    
    # Load data
    qpu_results = load_qpu_benchmark_results()
    pulp_results = load_patch_pulp_results()
    
    if not qpu_results or not pulp_results:
        print("Missing results, cannot compare.")
        return
    
    print(f"QPU Benchmark timestamp: {qpu_results.get('timestamp', 'N/A')}")
    print(f"QPU Benchmark scales: {qpu_results.get('scales', [])}")
    print(f"Patch PuLP configs: {sorted(pulp_results.keys())}")
    print()
    
    # Detailed analysis
    print("-" * 80)
    print("FORMULATION STATUS (after alignment):")
    print("-" * 80)
    print("""
    ✅ QPU Benchmark NOW ALIGNED with comprehensive_benchmark.py:
    
    Both formulations now use:
    - Constraint: sum(Y[f,c]) <= 1  (AT MOST ONE crop per farm)
    - Farms CAN be idle (no crop assigned)
    - Linking: Y[f,c] <= U[c] and U[c] <= sum(Y[f,c])
    - Food groups: min_foods: 1 per group (allows 5 unique foods minimum)
    - max_plots_per_crop: DISABLED (no limit)
    
    NOTE: The cached QPU benchmark results shown below are from BEFORE alignment.
    Re-run qpu_benchmark.py to get updated results matching Patch PuLP.
    """)
    
    # Compare results
    print("-" * 80)
    print("RESULTS COMPARISON:")
    print("-" * 80)
    print()
    print(f"{'Size':>6} | {'QPU Gurobi':>15} | {'Patch PuLP':>15} | {'Difference':>12} | {'% Diff':>8} | {'Notes'}")
    print("-" * 80)
    
    for scale_result in qpu_results.get('results', []):
        n_farms = scale_result.get('n_farms', 0)
        gt = scale_result.get('ground_truth', {})
        
        qpu_obj = gt.get('objective', None)
        pulp_data = pulp_results.get(n_farms, {})
        pulp_obj = pulp_data.get('objective_value', None)
        
        if qpu_obj is not None and pulp_obj is not None:
            diff = pulp_obj - qpu_obj
            pct_diff = (diff / qpu_obj * 100) if qpu_obj != 0 else 0
            notes = "PuLP higher" if diff > 0 else "Gurobi higher"
            print(f"{n_farms:>6} | {qpu_obj:>15.10f} | {pulp_obj:>15.10f} | {diff:>12.10f} | {pct_diff:>7.2f}% | {notes}")
            
            # Compare metadata
            print(f"       | Variables: {gt.get('n_variables', 'N/A'):>5} | Variables: {pulp_data.get('n_variables', 'N/A'):>5}")
            print(f"       | Constraints: {gt.get('n_constraints', 'N/A'):>3} | Constraints: {pulp_data.get('n_constraints', 'N/A'):>3}")
            print()
        else:
            print(f"{n_farms:>6} | {'N/A':>15} | {pulp_obj if pulp_obj else 'N/A':>15} | {'N/A':>12} | {'N/A':>8}")
    
    # Also show all Patch PuLP results
    print("-" * 80)
    print("ALL PATCH PULP RESULTS:")
    print("-" * 80)
    print(f"{'Size':>6} | {'Objective':>15} | {'Variables':>10} | {'Constraints':>12} | {'Status'}")
    print("-" * 80)
    for size in sorted(pulp_results.keys()):
        data = pulp_results[size]
        print(f"{size:>6} | {data.get('objective_value', 0):>15.10f} | {data.get('n_variables', 0):>10} | {data.get('n_constraints', 0):>12} | {data.get('status', 'N/A')}")
    
    # Solution analysis
    print()
    print("-" * 80)
    print("SOLUTION ANALYSIS (for matching sizes):")
    print("-" * 80)
    
    for scale_result in qpu_results.get('results', []):
        n_farms = scale_result.get('n_farms', 0)
        gt = scale_result.get('ground_truth', {})
        pulp_data = pulp_results.get(n_farms, {})
        
        if 'solution' in gt and 'solution_plantations' in pulp_data:
            qpu_solution = gt['solution']
            pulp_solution = pulp_data['solution_plantations']
            
            # Count active plots
            qpu_active = sum(1 for f in qpu_solution.get('Y', {}).values() 
                           for v in f.values() if v == 1)
            pulp_active = sum(1 for v in pulp_solution.values() if v > 0.5)
            
            print(f"\n{n_farms} farms:")
            print(f"  QPU Gurobi: {qpu_active} active plots (should be {n_farms} due to == constraint)")
            print(f"  Patch PuLP: {pulp_active} active plots (can be <= {n_farms} due to <= constraint)")
            
            # Unique foods used
            qpu_foods = qpu_solution.get('summary', {}).get('foods_used', [])
            pulp_foods = set()
            for key, val in pulp_solution.items():
                if val > 0.5:
                    food = key.split('_', 1)[1] if '_' in key else key
                    pulp_foods.add(food)
            
            print(f"  QPU Gurobi unique foods: {len(qpu_foods)} - {sorted(qpu_foods)[:5]}...")
            print(f"  Patch PuLP unique foods: {len(pulp_foods)} - {sorted(pulp_foods)[:5]}...")


def analyze_constraint_differences():
    """Analyze which constraints are different between the formulations."""
    print()
    print("=" * 80)
    print("CONSTRAINT ANALYSIS")
    print("=" * 80)
    
    # From qpu_benchmark.py build_binary_cqm():
    print("""
    QPU Benchmark CQM Constraints (for 25 farms, 27 foods):
    -------------------------------------------------------
    1. OneCrop_{farm}: sum(Y[f,c]) == 1 for each farm (25 constraints)
    2. Link_{farm}_{food}: U[c] - Y[f,c] >= 0 for each f,c (25*27 = 675 constraints)
    3. FG_Min_{group}: sum(U[c]) >= 2 for each of 5 groups (5 constraints)
    4. FG_Max_{group}: sum(U[c]) <= 5 for each of 5 groups (5 constraints)
    5. MaxPlots_{food}: sum(Y[f,c]) <= 5 for each food (27 constraints)
    
    Total: 25 + 675 + 5 + 5 + 27 = 737 constraints
    
    Food group constraints (from qpu_benchmark.py):
    - food_group_constraints = {
        'Proteins': {'min': 2, 'max': 5},
        'Fruits': {'min': 2, 'max': 5},
        'Legumes': {'min': 2, 'max': 5},
        'Staples': {'min': 2, 'max': 5},
        'Vegetables': {'min': 2, 'max': 5}
      }
    - max_plots_per_crop = max(5, 25 // 5) = 5
    """)
    
    print("""
    Patch PuLP Constraints (from solver_runner_BINARY.py):
    -------------------------------------------------------
    1. Max_Assignment_{farm}: sum(X[f,c]) <= 1 for each farm (25 constraints)
    2. U_Link_{farm}_{food}: X[f,c] - U[c] <= 0 (25*27 = 675 constraints)
    3. U_Bound_{food}: U[c] - sum(X[f,c]) <= 0 (27 constraints)  
    4. MinFoodGroup_Unique_{group}: sum(U[c]) >= min_foods (5 constraints)
    5. MaxFoodGroup_Unique_{group}: sum(U[c]) <= max_foods (5 constraints)
    6. Max_Plots_{food}: sum(X[f,c]) <= max_plots (only if max_planting_area set)
    
    BUT the Patch PuLP JSON shows only 62 constraints!
    This means U_Link and U_Bound constraints are NOT being counted.
    
    62 = 25 (farms) + 27 (max_plots) + 10 (food groups) = 62
    """)
    
    # Check what config is being used
    pulp_results = load_patch_pulp_results()
    
    if 25 in pulp_results:
        print("\nPatch PuLP config_25 details:")
        data = pulp_results[25]
        print(f"  n_variables: {data.get('n_variables')}")
        print(f"  n_constraints: {data.get('n_constraints')}")
        
        # Check solution for food group compliance
        solution = data.get('solution_plantations', {})
        foods_used = set()
        for key, val in solution.items():
            if val > 0.5:
                food = key.split('_', 1)[1] if '_' in key else key
                foods_used.add(food)
        
        print(f"  Unique foods: {sorted(foods_used)}")
        print(f"  Count: {len(foods_used)}")
        
        # Map to food groups
        # From qpu_benchmark.py food_groups
        food_group_mapping = {
            'Animal-source foods': ['Beef', 'Chicken', 'Egg', 'Lamb', 'Pork'],
            'Fruits': ['Apple', 'Avocado', 'Banana', 'Durian', 'Guava', 'Mango', 'Orange', 'Papaya', 'Watermelon'],
            'Pulses, nuts, and seeds': ['Chickpeas', 'Peanuts', 'Tempeh', 'Tofu'],
            'Starchy staples': ['Corn', 'Potato'],
            'Vegetables': ['Cabbage', 'Cucumber', 'Eggplant', 'Long bean', 'Pumpkin', 'Spinach', 'Tomatoes']
        }
        
        print("\n  Food group analysis:")
        for group, foods in food_group_mapping.items():
            used = [f for f in foods if f in foods_used]
            print(f"    {group}: {len(used)} foods ({used})")


def print_root_cause_summary():
    """Print the root cause analysis summary."""
    print()
    print("=" * 80)
    print("ALIGNMENT COMPLETE - FORMULATIONS NOW MATCH")
    print("=" * 80)
    print("""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ CHANGES MADE TO qpu_benchmark.py                                            │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ 1. food_group_constraints: min: 2 → min_foods: 1 per group                  │
    │ 2. One-crop constraint: == 1 → <= 1 (allows idle plots)                     │
    │ 3. max_plots_per_crop: hardcoded(5) → None (disabled)                       │
    │ 4. Constraint keys: min/max → min_foods/max_foods                           │
    │ 5. Removed reverse_mapping (use food_groups directly)                       │
    │ 6. U-Y linking: U >= Y → Y <= U and U <= sum(Y)                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ EXPECTED RESULT AFTER RE-RUNNING                                            │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ QPU Benchmark Gurobi ground_truth should now match Patch PuLP objectives    │
    │                                                                             │
    │ To verify, run: python @todo/qpu_benchmark.py --scales 25 --token YOUR_TOKEN│
    │                                                                             │
    │ Expected: ~0.388 objective (same as Patch PuLP config_25)                   │
    └─────────────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    compare_formulations()
    analyze_constraint_differences()
    print_root_cause_summary()
