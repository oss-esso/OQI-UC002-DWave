"""
Grid Refinement Analysis Script

This script tests how different grid refinements affect the solution quality.
It compares:
1. The continuous formulation (original solver_runner.py format with continuous area variables)
2. Discretized formulations with varying numbers of patches (grid refinement levels)

The script:
- Creates the same scenario with different grid refinements
- Solves each discretized version with PuLP (MILP)
- Solves the continuous version with PuLP (reference solution)
- Compares objective values to understand approximation quality
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

# Standard imports
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Import only what we need from the solvers
# We'll import the solve functions directly to avoid dimod dependency issues
import importlib.util

def import_solve_functions():
    """Import solve functions from solver modules."""
    # Import patch solver (handles its own dimod imports internally)
    spec_patch = importlib.util.spec_from_file_location(
        "solver_runner_patch_module", 
        os.path.join(os.path.dirname(__file__), "solver_runner_PATCH.py")
    )
    solver_patch = importlib.util.module_from_spec(spec_patch)
    
    spec_cont = importlib.util.spec_from_file_location(
        "solver_runner_cont_module",
        os.path.join(os.path.dirname(__file__), "solver_runner.py")
    )
    solver_cont = importlib.util.module_from_spec(spec_cont)
    
    # Execute modules
    spec_patch.loader.exec_module(solver_patch)
    spec_cont.loader.exec_module(solver_cont)
    
    return solver_patch.solve_with_pulp, solver_cont.solve_with_pulp

# Try importing, fall back to simpler approach if needed
try:
    solve_with_pulp_patch, solve_with_pulp_continuous = import_solve_functions()
except Exception as e:
    print(f"Warning: Could not import solver functions: {e}")
    print("Falling back to direct imports...")
    # Direct import (will fail if dimod not available, but that's handled in the modules)
    from solver_runner_PATCH import solve_with_pulp as solve_with_pulp_patch
    from solver_runner import solve_with_pulp as solve_with_pulp_continuous

# Import patch sampler
from patch_sampler import generate_farms as generate_patches

# Grid refinement configurations
# These represent different levels of discretization
GRID_REFINEMENTS = [
    5,      # Very coarse (5 patches)
    10,     # Coarse (10 patches)
    25,     # Medium (25 patches)
    50,     # Fine (50 patches)
    100,    # Very fine (100 patches)
    200,    # Extra fine (200 patches)
    1000,
    10000
]

# Seed for reproducibility
SEED = 42


def load_foods_from_excel(n_foods_per_group: int = None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    """
    Load food data from Excel file.
    
    Args:
        n_foods_per_group: Number of foods to sample per group (None = all foods)
        
    Returns:
        Tuple of (foods dict, food_groups dict)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, "Inputs", "Combined_Food_Data.xlsx")
    
    if not os.path.exists(excel_path):
        print("⚠️  Excel file not found, using fallback foods")
        # Fallback: use simple food data
        foods = {
            'Wheat': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.3, 
                      'affordability': 0.8, 'sustainability': 0.7},
            'Corn': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.4, 
                     'affordability': 0.9, 'sustainability': 0.6},
            'Rice': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 
                     'affordability': 0.7, 'sustainability': 0.5},
            'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.2, 
                         'affordability': 0.6, 'sustainability': 0.8},
            'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.3, 
                         'affordability': 0.9, 'sustainability': 0.7},
            'Apples': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.2, 
                       'affordability': 0.5, 'sustainability': 0.8},
        }
        food_groups = {
            'Grains': ['Wheat', 'Corn', 'Rice'],
            'Legumes': ['Soybeans'],
            'Vegetables': ['Potatoes'],
            'Fruits': ['Apples'],
        }
        return foods, food_groups
    
    # Load from Excel
    df = pd.read_excel(excel_path)
    
    if n_foods_per_group is not None:
        # Sample n foods per group
        df_shuffled = df.sample(frac=1, random_state=SEED)
        sampled = df_shuffled.groupby('food_group').head(n_foods_per_group).reset_index(drop=True)
        foods_list = sampled['Food_Name'].tolist()
    else:
        # Use ALL foods
        foods_list = df['Food_Name'].tolist()
    
    filt = df[df['Food_Name'].isin(foods_list)][['Food_Name', 'food_group',
                                                   'nutritional_value', 'nutrient_density',
                                                   'environmental_impact', 'affordability',
                                                   'sustainability']].copy()
    filt.rename(columns={'food_group': 'Food_Group'}, inplace=True)
    
    objectives = ['nutritional_value', 'nutrient_density', 'environmental_impact', 'affordability', 'sustainability']
    for obj in objectives:
        filt[obj] = filt[obj].fillna(0.5).clip(0, 1)
    
    # Build foods dict
    foods = {}
    for _, row in filt.iterrows():
        fname = row['Food_Name']
        foods[fname] = {
            'nutritional_value': float(row['nutritional_value']),
            'nutrient_density': float(row['nutrient_density']),
            'environmental_impact': float(row['environmental_impact']),
            'affordability': float(row['affordability']),
            'sustainability': float(row['sustainability'])
        }
    
    # Build food groups
    food_groups = {}
    for _, row in filt.iterrows():
        g = row['Food_Group']
        fname = row['Food_Name']
        if g not in food_groups:
            food_groups[g] = []
        food_groups[g].append(fname)
    
    return foods, food_groups


def create_continuous_scenario(total_land: float, n_farms: int = 10) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Create a continuous scenario (original format with continuous area variables).
    This represents the "true" optimal solution without discretization.
    
    Args:
        total_land: Total land to distribute across farms
        n_farms: Number of farms (continuous areas)
        
    Returns:
        Tuple of (farms, land_availability, foods, food_groups, config)
    """
    # Generate farm areas (continuous values)
    np.random.seed(SEED)
    farm_areas = np.random.dirichlet(np.ones(n_farms)) * total_land
    
    farms = [f"Farm_{i}" for i in range(n_farms)]
    land_availability = {farm: float(area) for farm, area in zip(farms, farm_areas)}
    
    # Load foods (use a subset for manageable problem size)
    foods, food_groups = load_foods_from_excel(n_foods_per_group=2)
    
    # Create config
    n_food_groups = len(food_groups)
    food_group_config = {
        g: {'min_foods': 1, 'max_foods': len(lst)}
        for g, lst in food_groups.items()
    }
    
    parameters = {
        'land_availability': land_availability,
        'minimum_planting_area': {food: 0.01 for food in foods},
        'food_group_constraints': food_group_config,
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        },
    }
    
    config = {'parameters': parameters}
    
    return farms, land_availability, foods, food_groups, config


def create_discretized_scenario(n_patches: int, total_land: float) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Create a discretized scenario with n_patches patches (grid refinement).
    
    Args:
        n_patches: Number of patches (discretization level)
        total_land: Total land to distribute (should match continuous scenario)
        
    Returns:
        Tuple of (patches, land_availability, foods, food_groups, config)
    """
    # Generate patches with the same total land
    # We need to scale the patches to match the total land
    patches_dict = generate_patches(n_farms=n_patches, seed=SEED)
    
    # Scale patches to match total land
    current_total = sum(patches_dict.values())
    scale_factor = total_land / current_total
    land_availability = {k: v * scale_factor for k, v in patches_dict.items()}
    
    patches = list(land_availability.keys())
    
    # Load same foods as continuous scenario
    foods, food_groups = load_foods_from_excel(n_foods_per_group=2)
    
    # Create config (same as continuous)
    n_food_groups = len(food_groups)
    if n_patches >= n_food_groups:
        food_group_config = {
            g: {'min_foods': 1, 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        }
    else:
        # Not enough patches - relax constraints
        food_group_config = {
            g: {'min_foods': 0, 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        }
    
    parameters = {
        'land_availability': land_availability,
        'minimum_planting_area': {food: 0.0001 for food in foods},
        'food_group_constraints': food_group_config,
        'idle_penalty_lambda': 0.0,  # NO IDLE PENALTY to match continuous formulation
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        },
    }
    
    config = {'parameters': parameters}
    
    return patches, land_availability, foods, food_groups, config


def run_grid_refinement_analysis(total_land: float = 100.0):
    """
    Run grid refinement analysis comparing continuous and discretized formulations.
    
    Args:
        total_land: Total land area to use for all scenarios
    """
    print("="*80)
    print("GRID REFINEMENT ANALYSIS")
    print("="*80)
    print(f"Total Land: {total_land} hectares")
    print(f"Grid Refinements: {GRID_REFINEMENTS}")
    print("="*80)
    
    results = []
    
    # Step 1: Solve continuous scenario (reference/ground truth)
    print("\n" + "="*80)
    print("STEP 1: Solving CONTINUOUS scenario (ground truth)")
    print("="*80)
    
    farms, land_avail_cont, foods, food_groups, config_cont = create_continuous_scenario(
        total_land=total_land, n_farms=10
    )
    
    print(f"  Farms: {len(farms)}")
    print(f"  Foods: {len(foods)}")
    print(f"  Food Groups: {len(food_groups)}")
    print(f"  Total Land: {sum(land_avail_cont.values()):.2f} ha")
    print(f"  Formulation: Continuous (A_{{f,c}} ∈ [0, land_f])")
    
    print("\n  Solving with PuLP (continuous)...")
    start = time.time()
    model_cont, results_cont = solve_with_pulp_continuous(farms, foods, food_groups, config_cont)
    time_cont = time.time() - start
    
    obj_continuous = results_cont.get('objective_value', 0)
    status_cont = results_cont.get('status', 'Unknown')
    
    print(f"    Status: {status_cont}")
    print(f"    Objective: {obj_continuous:.6f}")
    print(f"    Time: {time_cont:.3f}s")
    
    if status_cont != 'Optimal':
        print(f"\n⚠️  WARNING: Continuous solution is not optimal!")
        return
    
    # Store continuous result
    results.append({
        'refinement': 'Continuous',
        'n_patches': len(farms),
        'objective': obj_continuous,
        'time': time_cont,
        'gap_percent': 0.0,  # Reference solution
        'status': status_cont
    })
    
    # Step 2: Solve discretized scenarios with different grid refinements
    print("\n" + "="*80)
    print("STEP 2: Solving DISCRETIZED scenarios (varying grid refinements)")
    print("="*80)
    
    for n_patches in GRID_REFINEMENTS:
        print(f"\n{'-'*80}")
        print(f"Grid Refinement: {n_patches} patches")
        print(f"{'-'*80}")
        
        # Create discretized scenario
        patches, land_avail_disc, foods_disc, food_groups_disc, config_disc = create_discretized_scenario(
            n_patches=n_patches, total_land=total_land
        )
        
        print(f"  Patches: {len(patches)}")
        print(f"  Foods: {len(foods_disc)}")
        print(f"  Total Land: {sum(land_avail_disc.values()):.2f} ha")
        print(f"  Formulation: Discretized (X_{{p,c}} ∈ {{0,1}})")
        
        # Solve with PuLP (discretized/MILP)
        print(f"\n  Solving with PuLP (MILP)...")
        start = time.time()
        model_disc, results_disc = solve_with_pulp_patch(patches, foods_disc, food_groups_disc, config_disc)
        time_disc = time.time() - start
        
        obj_discrete_raw = results_disc.get('objective_value', 0)
        status_disc = results_disc.get('status', 'Unknown')
        
        # NORMALIZE: The patch formulation doesn't divide by total_area, but continuous does
        # To make them comparable, divide discrete objective by total_area
        obj_discrete = obj_discrete_raw / total_land
        
        # Calculate gap
        if obj_continuous > 0:
            gap_percent = ((obj_continuous - obj_discrete) / obj_continuous) * 100
        else:
            gap_percent = 0.0
        
        print(f"    Status: {status_disc}")
        print(f"    Objective (raw): {obj_discrete_raw:.6f}")
        print(f"    Objective (normalized): {obj_discrete:.6f}")
        print(f"    Time: {time_disc:.3f}s")
        print(f"    Gap from continuous: {gap_percent:.2f}%")
        
        # Store result
        results.append({
            'refinement': f'{n_patches} patches',
            'n_patches': n_patches,
            'objective': obj_discrete,
            'time': time_disc,
            'gap_percent': gap_percent,
            'status': status_disc
        })
    
    # Step 3: Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Refinement':<20} {'Patches':<10} {'Objective':<15} {'Time (s)':<10} {'Gap (%)':<10} {'Status':<10}")
    print("-" * 85)
    
    for r in results:
        print(f"{r['refinement']:<20} {r['n_patches']:<10} {r['objective']:<15.6f} {r['time']:<10.3f} {r['gap_percent']:<10.2f} {r['status']:<10}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    print(f"\n  Reference (Continuous): {obj_continuous:.6f}")
    
    if len(results) > 1:
        best_discrete = max(results[1:], key=lambda x: x['objective'])
        print(f"  Best Discrete: {best_discrete['objective']:.6f} ({best_discrete['refinement']})")
        print(f"  Best Gap: {best_discrete['gap_percent']:.2f}%")
        
        # Convergence analysis
        print(f"\n  Convergence Analysis:")
        print(f"  {'Patches':<10} {'Objective':<15} {'Gap (%)':<10}")
        print("  " + "-" * 35)
        for r in results[1:]:
            print(f"  {r['n_patches']:<10} {r['objective']:<15.6f} {r['gap_percent']:<10.2f}")
        
        # Check convergence behavior
        print(f"\n  Interpretation:")
        if abs(best_discrete['gap_percent']) < 0.1:
            print(f"  ✓ Excellent convergence: gap < 0.1%")
            print(f"  ✓ Discretization at {best_discrete['n_patches']} patches closely approximates continuous solution")
        elif abs(best_discrete['gap_percent']) < 1.0:
            print(f"  ✓ Good convergence: gap < 1%")
            print(f"  ✓ Discretization provides reasonable approximation")
        elif abs(best_discrete['gap_percent']) < 5.0:
            print(f"  ⚠ Moderate convergence: gap < 5%")
            print(f"  ⚠ Some quality loss due to discretization")
        else:
            print(f"  ✗ Poor convergence: gap ≥ 5%")
            print(f"  ✗ Significant quality loss due to discretization")
        
        # Note about negative gaps
        if best_discrete['gap_percent'] < 0:
            print(f"\n  Note: Negative gap means discrete > continuous")
            print(f"        This may indicate slight numerical differences or")
            print(f"        different constraint handling between formulations")
    
    print("\n" + "="*80)
    
    return results


def main():
    """Main entry point."""
    results = run_grid_refinement_analysis(total_land=100.0)


if __name__ == "__main__":
    main()

