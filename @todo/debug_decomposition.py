#!/usr/bin/env python3
"""Debug the decomposition to understand why objectives are wrong"""

import sys
sys.path.insert(0, 'D:/Projects/OQI-UC002-DWave')
sys.path.insert(0, 'D:/Projects/OQI-UC002-DWave/@todo')

from comprehensive_embedding_and_solving_benchmark import (
    load_real_data, generate_land_data, create_real_config,
    decompose_louvain, decompose_plot_based, extract_sub_bqm,
    solve_bqm_with_gurobi, calculate_actual_objective_from_bqm_solution,
    N_FOODS
)
from dimod import cqm_to_bqm

# Import create_cqm_plots
sys.path.insert(0, 'D:/Projects/OQI-UC002-DWave/Benchmark Scripts')
from solver_runner_BINARY import create_cqm_plots

def analyze_decomposition():
    n_farms = 25
    
    # Build CQM
    print("Building CQM...")
    foods, food_groups, base_config, weights = load_real_data()
    land_data = generate_land_data(n_farms, total_land=100.0)
    config = create_real_config(land_data, base_config)
    
    farms = list(land_data.keys())
    food_names = list(foods.keys())[:N_FOODS]
    
    cqm, Y, constraint_metadata = create_cqm_plots(farms, foods, food_groups, config)
    
    print(f"\nCQM: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints")
    
    # Convert to BQM
    print("\nConverting to BQM...")
    bqm, _ = cqm_to_bqm(cqm, lagrange_multiplier=10.0)
    print(f"BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} quadratic terms")
    
    meta = {
        "unit_names": farms,
        "food_names": food_names,
        "weights": weights,
        "total_land": 100.0
    }
    
    # Test Louvain decomposition
    print("\n" + "="*60)
    print("LOUVAIN DECOMPOSITION")
    print("="*60)
    
    partitions = decompose_louvain(bqm)
    print(f"Created {len(partitions)} partitions")
    
    # Check if same-patch variables are in same partition
    patch_partition_map = {}
    for p_idx, partition in enumerate(partitions):
        for var in partition:
            if var.startswith("Y_"):
                parts = var.split("_", 2)
                if len(parts) >= 2:
                    patch_name = parts[1]
                    if patch_name not in patch_partition_map:
                        patch_partition_map[patch_name] = set()
                    patch_partition_map[patch_name].add(p_idx)
    
    print(f"\nPatch distribution across partitions:")
    split_patches = 0
    for patch, partitions_containing in sorted(patch_partition_map.items()):
        if len(partitions_containing) > 1:
            split_patches += 1
            if split_patches <= 5:  # Show first 5
                print(f"  {patch}: split across partitions {sorted(partitions_containing)}")
    
    print(f"\n{split_patches}/{len(patch_partition_map)} patches are SPLIT across partitions!")
    if split_patches > 0:
        print("⚠️  This breaks the 'at most one food per patch' constraint!")
    
    # Test PlotBased decomposition
    print("\n" + "="*60)
    print("PLOT-BASED DECOMPOSITION")
    print("="*60)
    
    partitions_plot = decompose_plot_based(bqm, plots_per_partition=3)
    print(f"Created {len(partitions_plot)} partitions")
    
    # Check if same-patch variables are in same partition
    patch_partition_map_plot = {}
    for p_idx, partition in enumerate(partitions_plot):
        for var in partition:
            if var.startswith("Y_"):
                parts = var.split("_", 2)
                if len(parts) >= 2:
                    patch_name = parts[1]
                    if patch_name not in patch_partition_map_plot:
                        patch_partition_map_plot[patch_name] = set()
                    patch_partition_map_plot[patch_name].add(p_idx)
    
    split_patches_plot = sum(1 for v in patch_partition_map_plot.values() if len(v) > 1)
    print(f"{split_patches_plot}/{len(patch_partition_map_plot)} patches are split")
    if split_patches_plot == 0:
        print("✅ All patch variables stay together!")
    
    # Solve with PlotBased (correct) decomposition
    print("\n" + "="*60)
    print("SOLVING WITH PLOT-BASED DECOMPOSITION")
    print("="*60)
    
    merged_solution = {}
    total_energy = 0
    
    for i, partition in enumerate(partitions_plot):
        sub_bqm = extract_sub_bqm(bqm, partition)
        result = solve_bqm_with_gurobi(sub_bqm, timeout=60)
        if result.get("has_solution"):
            merged_solution.update(result.get("solution", {}))
            total_energy += result.get("bqm_energy", 0)
    
    # Calculate actual objective
    actual_obj = calculate_actual_objective_from_bqm_solution(merged_solution, meta, n_farms)
    
    print(f"\nTotal BQM energy (sum of partitions): {total_energy:.4f}")
    print(f"Actual objective (recalculated): {actual_obj:.4f}")
    print(f"Reference (PuLP): ~0.38")
    
    # Count selected foods per patch
    selections_per_patch = {}
    for var, val in merged_solution.items():
        if val > 0.5 and var.startswith("Y_"):
            parts = var.split("_", 2)
            if len(parts) >= 2:
                patch = parts[1]
                if patch not in selections_per_patch:
                    selections_per_patch[patch] = 0
                selections_per_patch[patch] += 1
    
    violations = sum(1 for count in selections_per_patch.values() if count > 1)
    print(f"\nConstraint violations: {violations} patches have >1 food selected")
    
    if violations > 0:
        print("Sample violations:")
        for patch, count in list(selections_per_patch.items())[:5]:
            if count > 1:
                selected_foods = [v for v, val in merged_solution.items() 
                                  if val > 0.5 and v.startswith(f"Y_{patch}_")]
                print(f"  {patch}: {count} foods selected: {selected_foods}")

if __name__ == "__main__":
    analyze_decomposition()
