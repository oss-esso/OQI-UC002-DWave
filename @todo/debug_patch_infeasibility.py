#!/usr/bin/env python3
"""
Diagnostic script to identify PATCH scenario infeasibility.

This script will:
1. Create a minimal PATCH scenario
2. Test with progressively relaxed constraints
3. Identify which constraint(s) cause infeasibility
4. Compute IIS (Irreducible Inconsistent Subsystem) if available
"""

import os
import sys
import math

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scenarios import load_food_data
from Utils.patch_sampler import generate_grid
import pulp as pl

def create_minimal_patch_model(n_patches=25, total_land=100.0):
    """Create a minimal patch model to test for infeasibility."""
    
    # Generate patches
    patches = generate_grid(n_farms=n_patches, area=total_land, seed=42)
    patches_list = list(patches.keys())
    
    # Load food data
    _, foods, food_groups, base_config = load_food_data('full_family')
    
    # Get parameters
    params = base_config['parameters']
    max_percentage = params.get('max_percentage_per_crop', {})
    min_planting_area = params.get('minimum_planting_area', {})
    maximum_planting_area = {crop: max_pct * total_land for crop, max_pct in max_percentage.items()}
    food_group_constraints = params.get('food_group_constraints', {})
    weights = params.get('weights', {})
    
    # Calculate plot area (all equal in even grid)
    plot_area = list(patches.values())[0]
    
    print(f"\n{'='*80}")
    print(f"PATCH SCENARIO DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"Number of patches: {n_patches}")
    print(f"Total land: {total_land:.2f} ha")
    print(f"Plot area: {plot_area:.4f} ha")
    print(f"Number of crops: {len(foods)}")
    print(f"Number of food groups: {len(food_groups)}")
    
    # Check constraint feasibility
    print(f"\n{'='*80}")
    print("CONSTRAINT ANALYSIS")
    print(f"{'='*80}")
    
    # 1. Minimum plot requirements
    print("\n1. MINIMUM PLOT REQUIREMENTS:")
    total_min_plots_required = 0
    for crop in foods:
        if crop in min_planting_area and min_planting_area[crop] > 0:
            min_plots = math.ceil(min_planting_area[crop] / plot_area)
            total_min_plots_required += min_plots
            print(f"   {crop}: min_area={min_planting_area[crop]:.2f} ha → {min_plots} plots")
    
    print(f"\n   TOTAL MIN PLOTS REQUIRED: {total_min_plots_required}")
    print(f"   AVAILABLE PLOTS: {n_patches}")
    
    if total_min_plots_required > n_patches:
        print(f"   ⚠️  INFEASIBLE: Need {total_min_plots_required} plots but only have {n_patches}")
    else:
        print(f"   ✓ FEASIBLE: {n_patches - total_min_plots_required} plots remaining")
    
    # 2. Maximum plot constraints
    print("\n2. MAXIMUM PLOT CONSTRAINTS:")
    for crop in foods:
        if crop in maximum_planting_area:
            max_plots = math.floor(maximum_planting_area[crop] / plot_area)
            min_plots = math.ceil(min_planting_area.get(crop, 0) / plot_area) if crop in min_planting_area else 0
            
            if min_plots > max_plots:
                print(f"   {crop}: min={min_plots} plots > max={max_plots} plots ⚠️  CONFLICT")
            else:
                print(f"   {crop}: min={min_plots}, max={max_plots} plots ✓")
    
    # 3. Food group constraints
    print("\n3. FOOD GROUP CONSTRAINTS:")
    for group, constraints in food_group_constraints.items():
        foods_in_group = food_groups.get(group, [])
        min_foods = constraints.get('min_foods', 0)
        max_foods = constraints.get('max_foods', float('inf'))
        
        print(f"   {group}:")
        print(f"      Foods in group: {len(foods_in_group)}")
        print(f"      Min foods required: {min_foods}")
        print(f"      Max foods allowed: {max_foods}")
        
        # Check if constraint is satisfiable
        if min_foods > len(foods_in_group):
            print(f"      ⚠️  INFEASIBLE: Need {min_foods} but only {len(foods_in_group)} foods in group")
        
        # Check against available plots
        if min_foods > n_patches:
            print(f"      ⚠️  INFEASIBLE: Need {min_foods} assignments but only {n_patches} plots")
    
    # 4. Each plot can have at most 1 crop
    print("\n4. PLOT ASSIGNMENT CONSTRAINT:")
    print(f"   Each of {n_patches} plots can have at most 1 crop")
    print(f"   Maximum possible assignments: {n_patches}")
    
    # Now test the actual model
    print(f"\n{'='*80}")
    print("TESTING PULP MODEL")
    print(f"{'='*80}")
    
    # Create binary variables
    X = pl.LpVariable.dicts("X", [(p, c) for p in patches_list for c in foods], cat='Binary')
    
    # Objective
    total_land_area = sum(patches.values())
    goal = 0
    for p in patches_list:
        patch_area = patches[p]
        for c in foods:
            area_weighted_value = patch_area * (
                weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[c].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[c].get('sustainability', 0)
            )
            goal += area_weighted_value * X[(p, c)]
    goal = goal / total_land_area
    
    model = pl.LpProblem("Patch_Diagnostic", pl.LpMaximize)
    model += goal, "Objective"
    
    # Test different constraint combinations
    test_scenarios = [
        ("Base (no constraints)", []),
        ("Plot assignment only", ['plot_assignment']),
        ("Plot + minimum plots", ['plot_assignment', 'min_plots']),
        ("Plot + maximum plots", ['plot_assignment', 'max_plots']),
        ("Plot + min + max plots", ['plot_assignment', 'min_plots', 'max_plots']),
        ("All constraints", ['plot_assignment', 'min_plots', 'max_plots', 'food_groups'])
    ]
    
    for scenario_name, constraint_types in test_scenarios:
        # Create fresh model
        test_model = pl.LpProblem(f"Test_{scenario_name.replace(' ', '_')}", pl.LpMaximize)
        test_model += goal, "Objective"
        
        # Add constraints based on scenario
        if 'plot_assignment' in constraint_types:
            for p in patches_list:
                test_model += pl.lpSum([X[(p, c)] for c in foods]) <= 1, f"Plot_{p}"
        
        if 'min_plots' in constraint_types:
            for crop in foods:
                if crop in min_planting_area and min_planting_area[crop] > 0:
                    min_plots = math.ceil(min_planting_area[crop] / plot_area)
                    test_model += pl.lpSum([X[(p, crop)] for p in patches_list]) >= min_plots, f"Min_{crop}"
        
        if 'max_plots' in constraint_types:
            for crop in foods:
                if crop in maximum_planting_area:
                    max_plots = math.floor(maximum_planting_area[crop] / plot_area)
                    test_model += pl.lpSum([X[(p, crop)] for p in patches_list]) <= max_plots, f"Max_{crop}"
        
        if 'food_groups' in constraint_types:
            for g, constraints in food_group_constraints.items():
                foods_in_group = food_groups.get(g, [])
                if foods_in_group:
                    group_label = g.replace(' ', '_').replace(',', '').replace('-', '_')
                    if 'min_foods' in constraints:
                        test_model += pl.lpSum([X[(p, c)] for p in patches_list for c in foods_in_group]) >= constraints['min_foods'], f"MinGroup_{group_label}"
                    if 'max_foods' in constraints:
                        test_model += pl.lpSum([X[(p, c)] for p in patches_list for c in foods_in_group]) <= constraints['max_foods'], f"MaxGroup_{group_label}"
        
        # Solve
        solver = pl.GUROBI(msg=0)
        test_model.solve(solver)
        
        status = pl.LpStatus[test_model.status]
        print(f"\n{scenario_name}: {status}")
        
        if status == "Infeasible":
            print(f"   ⚠️  First infeasibility at: {scenario_name}")
            
            # Try to get IIS (Irreducible Inconsistent Subsystem)
            try:
                print("\n   Computing IIS (Irreducible Inconsistent Subsystem)...")
                # Write model to file
                test_model.writeLP("debug_patch_infeasible.lp")
                print(f"   Model written to: debug_patch_infeasible.lp")
                
                # Try to compute IIS using Gurobi directly
                import gurobipy as gp
                gp_model = gp.read("debug_patch_infeasible.lp")
                gp_model.computeIIS()
                gp_model.write("debug_patch_infeasible.ilp")
                print(f"   IIS written to: debug_patch_infeasible.ilp")
                
                # Print IIS constraints
                print("\n   Conflicting constraints:")
                for constr in gp_model.getConstrs():
                    if constr.IISConstr:
                        print(f"      - {constr.ConstrName}")
                
            except Exception as e:
                print(f"   Could not compute IIS: {e}")
            
            break  # Stop at first infeasibility
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    create_minimal_patch_model(n_patches=25, total_land=100.0)
