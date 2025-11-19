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

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard imports
from typing import Dict, List, Tuple

# Import only what we need from the solvers
# We'll import the solve functions directly to avoid dimod dependency issues
import importlib.util

def import_solve_functions():
    """Import solve functions from solver_runner_BINARY module.
    
    Uses the most updated binary solvers:
    - solve_with_pulp_plots: Discretized patch formulation
    - solve_with_pulp_farm: Continuous farm formulation
    """
    # Import BINARY solver module
    candidates_binary = [
        os.path.join(os.path.dirname(__file__), "solver_runner_BINARY.py"),
        os.path.join(project_root, "solver_runner_BINARY.py"),
        os.path.join(project_root, "Benchmark Scripts", "solver_runner_BINARY.py"),
    ]

    def _load_first(candidates, module_name):
        for p in candidates:
            if p and os.path.exists(p):
                spec = importlib.util.spec_from_file_location(module_name, p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        return None

    solver_binary = _load_first(candidates_binary, "solver_runner_binary_module")

    if solver_binary is None:
        raise ImportError(
            f"Could not locate solver_runner_BINARY.py. Tried: {candidates_binary}"
        )

    return solver_binary.solve_with_pulp_plots, solver_binary.solve_with_pulp_farm

# Try importing, fall back to simpler approach if needed
try:
    solve_with_pulp_patch, solve_with_pulp_continuous = import_solve_functions()
except Exception as e:
    print(f"Warning: Could not import solver functions using file loader: {e}")
    print("Falling back to module imports...")
    # Try importing from solver_runner_BINARY by name (sys.path was adjusted above)
    try:
        import solver_runner_BINARY as _srb
        solve_with_pulp_patch = _srb.solve_with_pulp_plots
        solve_with_pulp_continuous = _srb.solve_with_pulp_farm
    except Exception as e1:
        # Try loading explicitly from Benchmark Scripts path
        try:
            spec = importlib.util.spec_from_file_location(
                "solver_runner_binary_module",
                os.path.join(project_root, "Benchmark Scripts", "solver_runner_BINARY.py"),
            )
            _srb = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_srb)
            solve_with_pulp_patch = _srb.solve_with_pulp_plots
            solve_with_pulp_continuous = _srb.solve_with_pulp_farm
        except Exception:
            raise ImportError("Could not import solver_runner_BINARY module") from e1

# Import patch sampler (for both farms and patches) — be flexible with relative/absolute imports
try:
    from . import patch_sampler
    from .farm_sampler import generate_farms as generate_farms_continuous
except Exception:
    try:
        # Try importing as top-level modules (sys.path was adjusted above)
        import patch_sampler
        from farm_sampler import generate_farms as generate_farms_continuous
    except Exception:
        # Try absolute package import (when running from project root)
        from Utils import patch_sampler
        from Utils.farm_sampler import generate_farms as generate_farms_continuous

# Grid refinement configurations
# These represent different levels of discretization
# We'll test each with both continuous (farms) and discretized (patches) formulations
GRID_REFINEMENTS = [
    10,     # Coarse
    25,     # Medium
    50,     # Fine
    1096    # Very fine
]

# Seed for reproducibility
SEED = 42


def load_foods_from_excel(seed=42, fixed_total_land=None):
    """
    Load full_family scenario matching comprehensive benchmark.
    Uses src.scenarios.load_food_data to ensure consistency.
    
    Args:
        seed: Random seed for reproducibility (unused, kept for compatibility)
        fixed_total_land: Total land area (unused, kept for compatibility)
        
    Returns:
        Tuple of (foods, food_groups)
    """
    try:
        from src.scenarios import load_food_data
        food_list, foods, food_groups, _ = load_food_data('full_family')
        return foods, food_groups
    except Exception as e:
        print(f"Warning: Could not load food data from scenarios ({e}), using fallback")
        # Fallback: load from Excel directly
        try:
            import pandas as pd
        except Exception:
            pd = None
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        excel_path = os.path.join(project_root, "Inputs", "Combined_Food_Data.xlsx")
        
        if not os.path.exists(excel_path) or pd is None:
            print("Excel file not found, using fallback foods")
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
                'Tomatoes': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.2, 
                             'affordability': 0.7, 'sustainability': 0.9},
                'Carrots': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.2, 
                            'affordability': 0.8, 'sustainability': 0.8},
                'Lentils': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.2, 
                            'affordability': 0.7, 'sustainability': 0.8},
                'Spinach': {'nutritional_value': 0.8, 'nutrient_density': 0.9, 'environmental_impact': 0.1, 
                            'affordability': 0.6, 'sustainability': 0.9},
            }
            food_groups = {
                'Grains': ['Wheat', 'Corn', 'Rice'],
                'Legumes': ['Soybeans', 'Lentils'],
                'Vegetables': ['Potatoes', 'Tomatoes', 'Carrots', 'Spinach'],
                'Fruits': ['Apples'],
            }
        else:
            # Load from Excel - USE ALL FOODS, not just 2 per group
            df = pd.read_excel(excel_path)
            
            # Use ALL foods from the dataset
            foods_list = df['Food_Name'].tolist()
            
            filt = df[df['Food_Name'].isin(foods_list)][['Food_Name', 'food_group',
                                                           'nutritional_value', 'nutrient_density',
                                                           'environmental_impact', 'affordability',
                                                           'sustainability']].copy()
            filt.rename(columns={'Food_Name': 'Food_Name', 'food_group': 'Food_Group'}, inplace=True)
            
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


def create_continuous_scenario(n_farms: int, total_land: float = None, seed: int = 42) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Create a continuous scenario (original format with continuous area variables).
    Uses farm_sampler to generate farms with realistic size distribution.
    
    Args:
        n_farms: Number of farms (continuous areas)
        total_land: Total land to distribute (if None, use natural total from Utils.farm_sampler)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (farms, land_availability, foods, food_groups, config)
    """
    # Generate farms with realistic size distribution
    land_availability = generate_farms_continuous(n_farms=n_farms, seed=seed)
    farms = list(land_availability.keys())
    
    # Scale to match total_land if specified
    if total_land is not None:
        current_total = sum(land_availability.values())
        scale_factor = total_land / current_total
        land_availability = {k: v * scale_factor for k, v in land_availability.items()}
    
    # Load foods (use a subset for manageable problem size)
    foods, food_groups = load_foods_from_excel()
    
    # Create config
    n_food_groups = len(food_groups)
    food_group_config = {
        g: {'min_foods': 1, 'max_foods': len(lst)}
        for g, lst in food_groups.items()
    }
    
    parameters = {
        'land_availability': land_availability,
        'minimum_planting_area': {food: 0.0001 for food in foods},  # Match comprehensive benchmark
        'food_group_constraints': food_group_config,  # Restored
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


def create_discretized_scenario(n_patches: int, total_land: float, seed: int = 42) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Create a discretized scenario with n_patches patches (grid refinement).
    
    Args:
        n_patches: Number of patches (discretization level)
        total_land: Total land to distribute (should match continuous scenario)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (patches, land_availability, foods, food_groups, config)
    """
    # Generate patches using same method as comprehensive benchmark
    # Use patch_sampler.generate_farms (even grid) and scale to match total land
    patch_seed = seed + 50  # Offset seed like comprehensive benchmark
    land_availability_unscaled = patch_sampler.generate_farms(n_farms=n_patches, seed=patch_seed)
    
    # Scale to match total_land
    current_total = sum(land_availability_unscaled.values())
    scale_factor = total_land / current_total if current_total > 0 else 1.0
    land_availability = {k: v * scale_factor for k, v in land_availability_unscaled.items()}
    
    patches = list(land_availability.keys())
    
    # Load same foods as continuous scenario
    foods, food_groups = load_foods_from_excel()
    
    # Create config matching comprehensive benchmark exactly
    food_group_config = {
        group: {'min_foods': 1, 'max_foods': len(food_list)}
        for group, food_list in food_groups.items()
    }
    
    parameters = {
        'land_availability': land_availability,
        'minimum_planting_area': {},  # No minimum for patches - avoids infeasibility with small grids
        'food_group_constraints': food_group_config,
        'idle_penalty_lambda': 0.0,  # Match comprehensive benchmark
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
    For each refinement level, we create:
    1. A continuous scenario with n_farms using UNEVEN farm distribution (farm_sampler)
    2. A discretized scenario with n_patches using EVEN grid distribution (patch_sampler.generate_grid)
    
    Both scenarios use the same total land area for fair comparison.
    The key difference: 
    - Uneven grid: farms have different sizes based on realistic distribution
    - Even grid: patches all have equal size (total_land / n_patches)
    
    Args:
        total_land: Total land area to use for all scenarios
    """
    print("="*80)
    print("GRID REFINEMENT ANALYSIS")
    print("="*80)
    print(f"Total Land: {total_land} hectares")
    print(f"Grid Refinements: {GRID_REFINEMENTS}")
    print(f"\nFor each refinement level N:")
    print(f"  - Continuous: N farms with continuous area variables (farm_sampler)")
    print(f"  - Discretized: N patches with binary assignment (patch_sampler)")
    print("="*80)
    
    results = []
    
    # Test each refinement level
    for i, n in enumerate(GRID_REFINEMENTS):
        print(f"\n{'='*80}")
        print(f"REFINEMENT LEVEL: {n} units")
        print(f"{'='*80}")
        
        # Use varying seeds like comprehensive benchmark
        seed_for_config = SEED + i * 100
        
        # Step 1: Solve continuous scenario (n farms)
        print(f"\n{'-'*80}")
        print(f"CONTINUOUS: {n} farms")
        print(f"{'-'*80}")
        
        farms, land_avail_cont, foods, food_groups, config_cont = create_continuous_scenario(
            n_farms=n, total_land=total_land, seed=seed_for_config
        )
        
        print(f"  Farms: {len(farms)}")
        print(f"  Foods: {len(foods)}")
        print(f"  Food Groups: {len(food_groups)}")
        print(f"  Total Land: {sum(land_avail_cont.values()):.2f} ha")
        print(f"  Formulation: Continuous (A_{{f,c}} ∈ [0, land_f])")
        
        print(f"\n  Solving with PuLP (continuous)...")
        start = time.time()
        model_cont, results_cont = solve_with_pulp_continuous(farms, foods, food_groups, config_cont)
        time_cont = time.time() - start
        
        obj_continuous = results_cont.get('objective_value', None)
        status_cont = results_cont.get('status', 'Unknown')
        
        print(f"    Status: {status_cont}")
        if obj_continuous is not None:
            print(f"    Objective: {obj_continuous:.6f}")
        else:
            print(f"    Objective: N/A (infeasible)")
        print(f"    Time: {time_cont:.3f}s")
        
        # Skip discretized if continuous is not optimal
        if status_cont != 'Optimal' or obj_continuous is None:
            print(f"\n  ⚠️  Skipping discretized version (continuous not optimal)")
            continue
        
        # Step 2: Solve discretized scenario (n patches)
        print(f"\n{'-'*80}")
        print(f"DISCRETIZED: {n} patches")
        print(f"{'-'*80}")
        
        patches, land_avail_disc, foods_disc, food_groups_disc, config_disc = create_discretized_scenario(
            n_patches=n, total_land=total_land, seed=seed_for_config
        )
        
        print(f"  Patches: {len(patches)}")
        print(f"  Foods: {len(foods_disc)}")
        print(f"  Total Land: {sum(land_avail_disc.values()):.2f} ha")
        print(f"  Formulation: Discretized (X_{{p,c}} ∈ {{0,1}})")
        
        print(f"\n  Solving with PuLP (MILP)...")
        start = time.time()
        model_disc, results_disc = solve_with_pulp_patch(patches, foods_disc, food_groups_disc, config_disc)
        time_disc = time.time() - start
        
        obj_discrete_raw = results_disc.get('objective_value', None)
        status_disc = results_disc.get('status', 'Unknown')
        
        print(f"    Status: {status_disc}")
        
        # Handle infeasible discretized solutions
        if status_disc != 'Optimal' or obj_discrete_raw is None:
            print(f"    Objective: N/A (infeasible)")
            print(f"    Time: {time_disc:.3f}s")
            print(f"\n  ⚠️  Warning: Discretized formulation infeasible")
            print(f"\n{'-'*80}")
            print(f"COMPARISON")
            print(f"{'-'*80}")
            print(f"  Continuous Objective:  {obj_continuous:.6f}")
            print(f"  Discretized Objective: N/A (infeasible)")
            print(f"  Gap: N/A")
            
            # Still store results but mark discretized as failed
            results.append({
                'n_units': n,
                'continuous_obj': obj_continuous,
                'continuous_time': time_cont,
                'continuous_status': status_cont,
                'discretized_obj': None,
                'discretized_time': time_disc,
                'discretized_status': status_disc,
                'gap_percent': None,
                'time_ratio': time_disc / time_cont if time_cont > 0 else 0
            })
            continue
        
        # NORMALIZE: The patch formulation doesn't divide by total_area, but continuous does
        # To make them comparable, divide discrete objective by total_area
        obj_discrete = obj_discrete_raw 
        
        # Calculate gap
        if obj_continuous > 0:
            gap_percent = ((obj_continuous - obj_discrete) / obj_continuous) * 100
        else:
            gap_percent = 0.0
        
        print(f"    Objective (raw): {obj_discrete_raw:.6f}")
        print(f"    Objective (normalized): {obj_discrete:.6f}")
        print(f"    Time: {time_disc:.3f}s")
        
        # Step 3: Compare
        print(f"\n{'-'*80}")
        print(f"COMPARISON")
        print(f"{'-'*80}")
        print(f"  Continuous Objective:  {obj_continuous:.6f}")
        print(f"  Discretized Objective: {obj_discrete:.6f}")
        print(f"  Gap: {gap_percent:.2f}%")
        print(f"  Time Ratio (Discrete/Continuous): {time_disc/time_cont:.2f}x")
        
        if status_cont != 'Optimal' or status_disc != 'Optimal':
            print(f"  ⚠️  Warning: One or both solutions not optimal!")
        
        # Store results
        results.append({
            'n_units': n,
            'continuous_obj': obj_continuous,
            'continuous_time': time_cont,
            'continuous_status': status_cont,
            'discretized_obj': obj_discrete,
            'discretized_time': time_disc,
            'discretized_status': status_disc,
            'gap_percent': gap_percent,
            'time_ratio': time_disc / time_cont if time_cont > 0 else 0
        })
    
    # Step 4: Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'N':<8} {'Continuous':<15} {'Discretized':<15} {'Gap (%)':<10} {'Time Cont':<12} {'Time Disc':<12} {'Ratio':<8}")
    print("-" * 90)
    
    for r in results:
        disc_obj_str = f"{r['discretized_obj']:.6f}" if r['discretized_obj'] is not None else "INFEASIBLE"
        gap_str = f"{r['gap_percent']:.2f}" if r['gap_percent'] is not None else "N/A"
        print(f"{r['n_units']:<8} {r['continuous_obj']:<15.6f} {disc_obj_str:<15} "
              f"{gap_str:<10} {r['continuous_time']:<12.3f} {r['discretized_time']:<12.3f} "
              f"{r['time_ratio']:<8.2f}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if len(results) > 0:
        # Convergence analysis
        print(f"\n  Convergence of Discretized to Continuous:")
        print(f"  {'N':<10} {'Gap (%)':<10} {'Trend':<20}")
        print("  " + "-" * 40)
        for r in results:
            if r['gap_percent'] is None:
                print(f"  {r['n_units']:<10} {'N/A':<10} {'✗ Infeasible':<20}")
                continue
            trend = ""
            if abs(r['gap_percent']) < 0.1:
                trend = "✓ Excellent"
            elif abs(r['gap_percent']) < 1.0:
                trend = "✓ Good"
            elif abs(r['gap_percent']) < 5.0:
                trend = "⚠ Moderate"
            else:
                trend = "✗ Poor"
            print(f"  {r['n_units']:<10} {r['gap_percent']:<10.2f} {trend:<20}")
        
        # Time complexity analysis
        print(f"\n  Computational Time Analysis:")
        print(f"  {'N':<10} {'Continuous (s)':<15} {'Discretized (s)':<15} {'Ratio':<10}")
        print("  " + "-" * 50)
        for r in results:
            print(f"  {r['n_units']:<10} {r['continuous_time']:<15.3f} {r['discretized_time']:<15.3f} {r['time_ratio']:<10.2f}")
        
        # Best and worst gaps
        valid_results = [r for r in results if r['gap_percent'] is not None]
        if valid_results:
            best_result = min(valid_results, key=lambda x: abs(x['gap_percent']))
            worst_result = max(valid_results, key=lambda x: abs(x['gap_percent']))
            
            print(f"\n  Best Approximation: N={best_result['n_units']} with {best_result['gap_percent']:.2f}% gap")
            print(f"  Worst Approximation: N={worst_result['n_units']} with {worst_result['gap_percent']:.2f}% gap")
        
        # Overall interpretation
        print(f"\n  Overall Interpretation:")
        if valid_results:
            avg_gap = sum(abs(r['gap_percent']) for r in valid_results) / len(valid_results)
        print(f"  - Average absolute gap: {avg_gap:.2f}%")
        
        if avg_gap < 0.5:
            print(f"  ✓ Discretized formulation provides excellent approximation across all scales")
        elif avg_gap < 2.0:
            print(f"  ✓ Discretized formulation provides good approximation across most scales")
        elif avg_gap < 5.0:
            print(f"  ⚠ Discretized formulation shows moderate approximation quality")
        else:
            print(f"  ✗ Discretized formulation shows significant approximation error")
        
        # Check if gap improves with refinement
        if len(valid_results) >= 3:
            gaps = [abs(r['gap_percent']) for r in valid_results]
            if gaps[-1] < gaps[0]:
                print(f"  ✓ Gap improves with finer discretization ({gaps[0]:.2f}% → {gaps[-1]:.2f}%)")
            else:
                print(f"  ⚠ Gap does not consistently improve with refinement")
        
        # Note about negative gaps
        negative_gaps = [r for r in valid_results if r['gap_percent'] < 0]
        if negative_gaps:
            print(f"\n  Note: {len(negative_gaps)} cases show negative gaps (discrete > continuous)")
            print(f"        This may indicate:")
            print(f"        - Numerical precision differences between solvers")
            print(f"        - Different constraint handling (discrete more permissive)")
            print(f"        - Solver convergence tolerances")
        
        # Note about infeasible cases
        infeasible_results = [r for r in results if r['gap_percent'] is None]
        if infeasible_results:
            infeasible_configs = [r['n_units'] for r in infeasible_results]
            print(f"\n  Note: {len(infeasible_results)} config(s) had infeasible discretized formulations: {infeasible_configs}")
            print(f"        This indicates the patch formulation constraints are too restrictive for small grids.")
    
    print("\n" + "="*80)
    
    return results


def main():
    """Main entry point."""
    results = run_grid_refinement_analysis(total_land=100.0)


if __name__ == "__main__":
    main()

