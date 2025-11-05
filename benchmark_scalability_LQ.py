"""
Scalability Benchmark Script for Linear-Quadratic (LQ) Solver
Tests different combinations of farms and food groups to analyze solver performance
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from farm_sampler import generate_farms
from src.scenarios import load_food_data
from solver_runner_LQ import create_cqm, solve_with_pulp, solve_with_pyomo, solve_with_dwave
from benchmark_cache import BenchmarkCache, serialize_cqm
import pulp as pl

# Helper function to clean solution data for JSON serialization
def clean_solution_for_json(solution):
    """Convert solution dictionary to JSON-serializable format."""
    if not solution:
        return {}
    
    cleaned = {}
    for key, value in solution.items():
        # Convert tuple keys to strings
        if isinstance(key, tuple):
            key_str = '_'.join(str(k) for k in key)
        else:
            key_str = str(key)
        
        # Convert values to basic types
        if hasattr(value, 'item'):  # numpy types
            cleaned[key_str] = value.item()
        elif isinstance(value, (int, float, str, bool, type(None))):
            cleaned[key_str] = value
        else:
            cleaned[key_str] = str(value)
    
    return cleaned

# Benchmark configurations
# Format: number of farms to test with full_family scenario
# 6 points logarithmically scaled from 5 to 1535 farms
# Reduced from 30 points for faster testing with multiple runs
BENCHMARK_CONFIGS = [
    5, 19, 72, 279#, 1096, 1535
]

# Number of runs per configuration for statistical analysis
NUM_RUNS = 1

def load_full_family_with_n_farms(n_farms, seed=42):
    """
    Load full_family scenario with specified number of farms.
    Uses the same logic as the scaling analysis but with synergy matrix.
    """
    import pandas as pd
    
    # Generate farms
    L = generate_farms(n_farms=n_farms, seed=seed)
    farms = list(L.keys())
    
    # Load food data from Excel or use fallback
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, "Inputs", "Combined_Food_Data.xlsx")
    
    if not os.path.exists(excel_path):
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
    
    # --- Generate synergy matrix (NEW for LQ solver) ---
    synergy_matrix = {}
    default_boost = 0.1  # A default boost value for pairs in the same group

    for group_name, crops_in_group in food_groups.items():
        for i in range(len(crops_in_group)):
            for j in range(i + 1, len(crops_in_group)):
                crop1 = crops_in_group[i]
                crop2 = crops_in_group[j]

                if crop1 not in synergy_matrix:
                    synergy_matrix[crop1] = {}
                if crop2 not in synergy_matrix:
                    synergy_matrix[crop2] = {}

                # Add symmetric entries for the pair
                synergy_matrix[crop1][crop2] = default_boost
                synergy_matrix[crop2][crop1] = default_boost
    # --- End synergy matrix generation ---
    
    # Set minimum planting areas based on smallest farm and number of food groups
    smallest_farm = min(L.values())
    n_food_groups = len(food_groups)
    # Each farm must plant at least 1 crop from each food group
    # Reserve some margin for safety
    min_area_per_crop = (smallest_farm / n_food_groups) * 0.9  # 90% safety margin
    
    min_areas = {food: min_area_per_crop for food in foods.keys()}
    
    # Build config with synergy matrix
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': 1, 'max_foods': len(lst)}  # At least 1 food per group
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15,
            'synergy_bonus': 0.1  # NEW weight for synergy bonus
        },
        'synergy_matrix': synergy_matrix  # NEW parameter
    }
    
    config = {'parameters': parameters}
    
    return farms, foods, food_groups, config

def run_benchmark(n_farms, run_number=1, total_runs=1, cache=None, save_to_cache=True):
    """
    Run a single benchmark test with full_family scenario using LQ solver.
    Returns timing results and problem size metrics for all three solvers.
    
    Args:
        n_farms: Number of farms to test
        run_number: Current run number (for display)
        total_runs: Total number of runs (for display)
        cache: BenchmarkCache instance for saving results
        save_to_cache: Whether to save results to cache (default: True)
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: full_family scenario with {n_farms} Farms (Run {run_number}/{total_runs})")
    print(f"{'='*80}")
    
    try:
        # Load full_family scenario with specified number of farms
        farms, foods, food_groups, config = load_full_family_with_n_farms(n_farms, seed=42 + run_number)
        
        n_foods = len(foods)
        # For LQ solver, we don't have lambda variables (simpler than NLN)
        n_vars_base = 2 * n_farms * n_foods  # Binary + continuous (A and Y)
        
        # Count synergy pairs (for PuLP linearization, we add Z variables)
        synergy_matrix = config['parameters'].get('synergy_matrix', {})
        n_synergy_pairs = 0
        for crop1, pairs in synergy_matrix.items():
            n_synergy_pairs += len(pairs)
        n_synergy_pairs = n_synergy_pairs // 2  # Each pair counted twice
        
        n_z_vars_pulp = n_farms * n_synergy_pairs  # Z variables for PuLP linearization
        n_vars_pulp = n_vars_base + n_z_vars_pulp
        n_vars_quadratic = n_vars_base  # For CQM and Pyomo (native quadratic)
        
        n_constraints_base = n_farms + 2*n_farms*n_foods + 2*len(food_groups)*n_farms
        n_linearization_constraints = n_z_vars_pulp * 3  # 3 constraints per Z variable (McCormick)
        n_constraints_pulp = n_constraints_base + n_linearization_constraints
        n_constraints_quadratic = n_constraints_base
        
        problem_size = n_farms * n_foods  # n = farms × foods
        
        print(f"  Foods: {n_foods}")
        print(f"  Synergy Pairs: {n_synergy_pairs}")
        print(f"  Base Variables (A+Y): {n_vars_base}")
        print(f"  PuLP Variables (A+Y+Z): {n_vars_pulp}")
        print(f"  CQM/Pyomo Variables: {n_vars_quadratic}")
        print(f"  PuLP Constraints: {n_constraints_pulp}")
        print(f"  CQM/Pyomo Constraints: {n_constraints_quadratic}")
        print(f"  Problem Size (n): {problem_size}")
        
        # Create CQM (needed for DWave)
        print(f"\n  Creating CQM model...")
        cqm_start = time.time()
        cqm, A, Y, constraint_metadata = create_cqm(
            farms, foods, food_groups, config
        )
        cqm_time = time.time() - cqm_start
        print(f"    ✅ CQM created: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints ({cqm_time:.2f}s)")
        
        # Save CQM to cache if requested
        if save_to_cache and cache:
            cqm_data = serialize_cqm(cqm)
            cqm_result = {
                'cqm_time': cqm_time,
                'num_variables': len(cqm.variables),
                'num_constraints': len(cqm.constraints),
                'n_foods': n_foods,
                'problem_size': problem_size,
                'n_vars_quadratic': n_vars_quadratic,
                'n_constraints_quadratic': n_constraints_quadratic,
                'n_synergy_pairs': n_synergy_pairs
            }
            cache.save_result('LQ', 'CQM', n_farms, run_number, cqm_result, cqm_data=cqm_data)
        
        # Solve with PuLP (linearized quadratic)
        print(f"\n  Solving with PuLP (Linearized Quadratic)...")
        pulp_start = time.time()
        pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config)
        pulp_time = time.time() - pulp_start
        
        print(f"    Status: {pulp_results['status']}")
        print(f"    Objective: {pulp_results.get('objective_value', 'N/A')}")
        print(f"    Time: {pulp_time:.3f}s")
        
        # Save PuLP results to cache
        if save_to_cache and cache:
            pulp_cache_result = {
                'solve_time': pulp_time,
                'status': pulp_results['status'],
                'objective_value': pulp_results.get('objective_value'),
                'normalized_objective': pulp_results.get('normalized_objective'),
                'total_area': pulp_results.get('total_area'),
                'solution': clean_solution_for_json(pulp_results.get('solution', {})),
                'n_foods': n_foods,
                'problem_size': problem_size,
                'n_vars_pulp': n_vars_pulp,
                'n_constraints_pulp': n_constraints_pulp,
                'n_synergy_pairs': n_synergy_pairs
            }
            cache.save_result('LQ', 'PuLP', n_farms, run_number, pulp_cache_result)
        
        # Solve with Pyomo (native quadratic)
        print(f"\n  Solving with Pyomo (Native Quadratic)...")
        pyomo_start = time.time()
        pyomo_model, pyomo_results = solve_with_pyomo(farms, foods, food_groups, config)
        pyomo_time = time.time() - pyomo_start
        
        if pyomo_results.get('error'):
            print(f"    Status: {pyomo_results['status']}")
            print(f"    Error: {pyomo_results.get('error')}")
            pyomo_time = None
            pyomo_objective = None
        else:
            print(f"    Status: {pyomo_results['status']}")
            print(f"    Objective: {pyomo_results.get('objective_value', 'N/A')}")
            print(f"    Time: {pyomo_time:.3f}s")
            pyomo_objective = pyomo_results.get('objective_value')
        
        # Save Pyomo results to cache
        if save_to_cache and cache:
            pyomo_cache_result = {
                'solve_time': pyomo_time,
                'status': pyomo_results.get('status', 'Error'),
                'objective_value': pyomo_objective,
                'normalized_objective': pyomo_results.get('normalized_objective'),
                'total_area': pyomo_results.get('total_area'),
                'solution': clean_solution_for_json(pyomo_results.get('solution', {})),
                'error': pyomo_results.get('error'),
                'n_foods': n_foods,
                'problem_size': problem_size,
                'n_vars_quadratic': n_vars_quadratic,
                'n_constraints_quadratic': n_constraints_quadratic,
                'n_synergy_pairs': n_synergy_pairs
            }
            cache.save_result('LQ', 'Pyomo', n_farms, run_number, pyomo_cache_result)
        
        # Solve with DWave
        print(f"\n  Solving with DWave...")
        token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
        
        dwave_time = None
        qpu_time = None
        hybrid_time = None
        dwave_feasible = False
        dwave_objective = None
        dwave_total_area = None
        dwave_normalized_objective = None

        if not token:
            print(f"    SKIPPED: DWAVE_API_TOKEN not found.")
        else:
            try:
                sampleset, dwave_time = solve_with_dwave(cqm, token)

                feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
                dwave_feasible = len(feasible_sampleset) > 0

                if dwave_feasible:
                    best = feasible_sampleset.first
                    dwave_objective = -best.energy
                    
                    # Calculate total area from the solution
                    dwave_total_area = 0.0
                    for var_name, var_value in best.sample.items():
                        # Area variables are named like "Area_farm_crop"
                        if var_name.startswith('Area_') and var_value > 1e-6:
                            dwave_total_area += var_value
                    
                    # Calculate normalized objective
                    dwave_normalized_objective = dwave_objective / dwave_total_area if dwave_total_area > 1e-6 else 0.0
                    
                    # Extract timing info correctly - DWave returns times in seconds already
                    qpu_time = sampleset.info.get('qpu_access_time', 0)  / 1e6
                    hybrid_time = sampleset.info.get('charge_time', 0)   / 1e6
                    run_time = sampleset.info.get('run_time', 0) / 1e6

                    print(f"    Status: {len(feasible_sampleset)} feasible solutions")
                    print(f"    Objective: {dwave_objective:.6f}")
                    print(f"    Total Area: {dwave_total_area:.2f}")
                    print(f"    Normalized Objective: {dwave_normalized_objective:.6f}")
                    print(f"    Total Time: {dwave_time:.3f}s")
                    if hybrid_time and hybrid_time > 0:
                        print(f"    Charge Time: {hybrid_time:.3f}s")
                    if qpu_time and qpu_time > 0:
                        print(f"    QPU Access Time: {qpu_time:.4f}s")
                    if run_time and run_time > 0:
                        print(f"    Run Time: {run_time:.3f}s")
                else:
                    print("    Status: No feasible solutions found")

            except Exception as e:
                print(f"    ERROR: DWave solving failed: {str(e)}")
        
        # Save DWave results to cache
        if save_to_cache and cache:
            dwave_cache_result = {
                'dwave_time': dwave_time,
                'qpu_time': qpu_time,
                'hybrid_time': hybrid_time,
                'feasible': dwave_feasible,
                'objective_value': dwave_objective,
                'normalized_objective': dwave_normalized_objective,
                'total_area': dwave_total_area,
            }
            cache.save_result('LQ', 'DWave', n_farms, run_number, dwave_cache_result)
        
        # Calculate difference between PuLP and Pyomo (should be exact)
        pulp_error = None
        if pyomo_objective is not None and pulp_results.get('objective_value') is not None:
            pulp_error = abs(pulp_results['objective_value'] - pyomo_objective) / abs(pyomo_objective) * 100
            print(f"\n  Solution Comparison:")
            print(f"    PuLP vs Pyomo: {pulp_error:.4f}% (should be ~0% - exact)")
        
        result = {
            'n_farms': n_farms,
            'n_foods': n_foods,
            'n_synergy_pairs': n_synergy_pairs,
            'n_vars_base': n_vars_base,
            'n_z_vars_pulp': n_z_vars_pulp,
            'n_vars_pulp': n_vars_pulp,
            'n_vars_quadratic': n_vars_quadratic,
            'n_constraints_pulp': n_constraints_pulp,
            'n_constraints_quadratic': n_constraints_quadratic,
            'problem_size': problem_size,
            'cqm_time': cqm_time,
            'pulp_time': pulp_time,
            'pulp_status': pulp_results['status'],
            'pulp_objective': pulp_results.get('objective_value'),
            'pulp_normalized_objective': pulp_results.get('normalized_objective'),
            'pulp_total_area': pulp_results.get('total_area'),
            'pyomo_time': pyomo_time,
            'pyomo_status': pyomo_results.get('status', 'Error'),
            'pyomo_objective': pyomo_objective,
            'pyomo_normalized_objective': pyomo_results.get('normalized_objective'),
            'pyomo_total_area': pyomo_results.get('total_area'),
            'pulp_diff_percent': pulp_error,
            'dwave_time': dwave_time,
            'qpu_time': qpu_time,
            'hybrid_time': hybrid_time,
            'dwave_objective': dwave_objective,
            'dwave_normalized_objective': dwave_normalized_objective,
            'dwave_total_area': dwave_total_area,
            'dwave_feasible': dwave_feasible,
        }
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(results, output_file='scalability_benchmark_lq.png'):
    """
    Create beautiful plots for presentation with error bars.
    Results should be aggregated statistics with mean and std.
    """
    # Filter valid results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to plot!")
        return
    
    # Extract data
    problem_sizes = [r['problem_size'] for r in valid_results]
    
    # PuLP times
    pulp_times = [r['pulp_time_mean'] for r in valid_results]
    pulp_errors = [r['pulp_time_std'] for r in valid_results]
    
    # Pyomo times
    pyomo_times = [r['pyomo_time_mean'] for r in valid_results if r['pyomo_time_mean'] is not None]
    pyomo_errors = [r['pyomo_time_std'] for r in valid_results if r['pyomo_time_mean'] is not None]
    pyomo_problem_sizes = [r['problem_size'] for r in valid_results if r['pyomo_time_mean'] is not None]
    
    # Solution difference (should be near zero)
    solution_diffs = [r['pulp_diff_mean'] for r in valid_results if r['pulp_diff_mean'] is not None]
    solution_diffs_std = [r['pulp_diff_std'] for r in valid_results if r['pulp_diff_mean'] is not None]
    diff_problem_sizes = [r['problem_size'] for r in valid_results if r['pulp_diff_mean'] is not None]
    
    # Create figure with professional styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Solve times with error bars
    ax1.errorbar(problem_sizes, pulp_times, yerr=pulp_errors, marker='o', linestyle='-', 
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                label='PuLP (Linearized)', color='#2E86AB', alpha=0.9)
    
    if pyomo_times:
        ax1.errorbar(pyomo_problem_sizes, pyomo_times, yerr=pyomo_errors, marker='s', linestyle='-',
                    linewidth=2.5, markersize=8, capsize=5, capthick=2,
                    label='Pyomo (Native Quadratic)', color='#A23B72', alpha=0.9)
    
    ax1.set_xlabel('Problem Size (n = Farms × Foods)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax1.set_title('Linear-Quadratic Solver Performance (Linear + Synergy Bonus)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Add annotations for key points
    if len(problem_sizes) > 0:
        # Annotate largest problem
        max_idx = problem_sizes.index(max(problem_sizes))
        ax1.annotate(f'n={problem_sizes[max_idx]}\nPuLP: {pulp_times[max_idx]:.2f}s',
                    xy=(problem_sizes[max_idx], pulp_times[max_idx]),
                    xytext=(-60, -30), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Solution accuracy (should be near zero)
    if solution_diffs:
        ax2.errorbar(diff_problem_sizes, solution_diffs, yerr=solution_diffs_std,
                    marker='o', linestyle='-', linewidth=2.5, markersize=8,
                    capsize=5, capthick=2, color='#06A77D', alpha=0.9,
                    label='PuLP vs Pyomo')
        
        ax2.set_xlabel('Problem Size (n = Farms × Foods)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Solution Difference (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Linearization Accuracy (McCormick Relaxation)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xscale('log')
        
        # Add horizontal line at y=0 (perfect match)
        ax2.axhline(y=0.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Exact Match')
        ax2.axhline(y=0.01, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='0.01% Diff')
        ax2.legend(loc='best', fontsize=11, framealpha=0.95)
        
        # Add average difference annotation
        avg_diff = np.mean(solution_diffs)
        ax2.text(0.98, 0.98, f'Avg: {avg_diff:.4f}%\n± {np.mean(solution_diffs_std):.4f}%', 
                transform=ax2.transAxes, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    else:
        ax2.text(0.5, 0.5, 'No Pyomo Data\nfor Comparison', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    
    # Also create a summary table plot in the same folder as the main plot
    table_file = output_file.replace('scalability_benchmark_lq', 'scalability_table_lq')
    create_summary_table(valid_results, table_file)

def create_summary_table(results, output_file='scalability_table_lq.png'):
    """
    Create a beautiful summary table for LQ benchmark.
    """
    fig, ax = plt.subplots(figsize=(18, len(results) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Farms', 'Foods', 'n', 'Synergy\nPairs', 'PuLP\nVars', 'Quad\nVars', 
               'PuLP\nTime (s)', 'Pyomo\nTime (s)', 'Diff\n(%)', 'Runs', 'Winner']
    
    table_data = []
    for r in results:
        # Determine winner (faster solver)
        if r.get('pyomo_time_mean') is not None and r.get('pulp_time_mean') is not None:
            if r['pulp_time_mean'] < r['pyomo_time_mean']:
                winner = '🏆 PuLP'
            elif r['pyomo_time_mean'] < r['pulp_time_mean']:
                winner = '🏆 Pyomo'
            else:
                winner = 'Tie'
        else:
            winner = 'PuLP'
        
        row = [
            r['n_farms'],
            r['n_foods'],
            r['problem_size'],
            r.get('n_synergy_pairs', 'N/A'),
            r.get('n_vars_pulp', 'N/A'),
            r.get('n_vars_quadratic', 'N/A'),
            f"{r['pulp_time_mean']:.3f} ± {r['pulp_time_std']:.3f}",
            f"{r['pyomo_time_mean']:.3f} ± {r['pyomo_time_std']:.3f}" if r.get('pyomo_time_mean') else 'N/A',
            f"{r['pulp_diff_mean']:.4f}" if r.get('pulp_diff_mean') else 'N/A',
            r['num_runs'],
            winner
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Linear-Quadratic Scalability Benchmark Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Table saved to: {output_file}")

def main():
    """
    Run all benchmarks with multiple runs and calculate statistics.
    Uses intelligent caching to avoid redundant runs.
    """
    # Initialize cache
    cache = BenchmarkCache()
    
    print("="*80)
    print("LINEAR-QUADRATIC SCALABILITY BENCHMARK")
    print("="*80)
    print(f"Configurations: {len(BENCHMARK_CONFIGS)} points")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Total benchmarks: {len(BENCHMARK_CONFIGS) * NUM_RUNS}")
    print(f"Objective: Linear area + Quadratic synergy bonus")
    print("="*80)
    
    # Check cache status
    print("\n" + "="*80)
    print("CHECKING CACHE STATUS")
    print("="*80)
    cache.print_cache_status('LQ', BENCHMARK_CONFIGS, NUM_RUNS)
    
    all_results = []
    aggregated_results = []
    
    for n_farms in BENCHMARK_CONFIGS:
        print(f"\n" + "="*80)
        print(f"TESTING CONFIGURATION: {n_farms} Farms")
        print("="*80)
        
        # Check which runs are needed
        runs_needed = cache.get_runs_needed('LQ', n_farms, NUM_RUNS)
        
        # Load existing results from cache for PuLP (our primary solver)
        existing_pulp_results = cache.get_all_results('LQ', 'PuLP', n_farms)
        config_results = []
        
        # Convert cached results to the format expected by aggregation
        for cached in existing_pulp_results:
            result_data = cached['result']
            run_num = cached['metadata']['run_number']
            
            run_result = {
                'n_farms': n_farms,
                'n_foods': result_data.get('n_foods', 10),
                'pulp_time': result_data['solve_time'],
                'pulp_status': result_data['status'],
                'pulp_objective': result_data['objective_value'],
                'problem_size': result_data.get('problem_size', n_farms * 10),
                'n_synergy_pairs': result_data.get('n_synergy_pairs', 0),
                'n_vars_base': result_data.get('n_vars_base', 0),
                'n_z_vars_pulp': result_data.get('n_z_vars_pulp', 0),
                'n_vars_pulp': result_data.get('n_vars_pulp', 0),
                'n_vars_quadratic': result_data.get('n_vars_quadratic', 0),
                'n_constraints_pulp': result_data.get('n_constraints_pulp', 0),
                'n_constraints_quadratic': result_data.get('n_constraints_quadratic', 0)
            }
            
            # Load corresponding CQM result
            cqm_cached = cache.load_result('LQ', 'CQM', n_farms, run_num)
            if cqm_cached:
                run_result['cqm_time'] = cqm_cached['result']['cqm_time']
            else:
                run_result['cqm_time'] = None
            
            # Load corresponding Pyomo result
            pyomo_cached = cache.load_result('LQ', 'Pyomo', n_farms, run_num)
            if pyomo_cached and pyomo_cached['result'].get('solve_time'):
                run_result['pyomo_time'] = pyomo_cached['result']['solve_time']
                run_result['pyomo_status'] = pyomo_cached['result'].get('status', 'Error')
                run_result['pyomo_objective'] = pyomo_cached['result'].get('objective_value')
                
                # Calculate diff if both objectives exist
                if run_result['pyomo_objective'] is not None and run_result['pulp_objective'] is not None:
                    run_result['pulp_diff_percent'] = abs(run_result['pulp_objective'] - run_result['pyomo_objective']) / abs(run_result['pyomo_objective']) * 100
                else:
                    run_result['pulp_diff_percent'] = None
            else:
                run_result['pyomo_time'] = None
                run_result['pyomo_status'] = 'Error'
                run_result['pyomo_objective'] = None
                run_result['pulp_diff_percent'] = None
            
            # Load corresponding DWave result
            dwave_cached = cache.load_result('LQ', 'DWave', n_farms, run_num)
            if dwave_cached and dwave_cached['result'].get('dwave_time'):
                run_result['dwave_time'] = dwave_cached['result']['dwave_time']
                run_result['qpu_time'] = dwave_cached['result'].get('qpu_time')
                run_result['hybrid_time'] = dwave_cached['result'].get('hybrid_time')
                run_result['dwave_feasible'] = dwave_cached['result'].get('feasible', False)
                run_result['dwave_objective'] = dwave_cached['result'].get('objective_value')
            else:
                run_result['dwave_time'] = None
                run_result['qpu_time'] = None
                run_result['hybrid_time'] = None
                run_result['dwave_feasible'] = False
                run_result['dwave_objective'] = None
            
            config_results.append(run_result)
        
        print(f"\n  Loaded {len(config_results)} existing runs from cache")
        
        # Determine which runs still need to be executed
        # We need to find the union of all missing runs across all solvers
        all_missing_runs = set()
        for solver, missing in runs_needed.items():
            all_missing_runs.update(missing)
        
        all_missing_runs = sorted(all_missing_runs)
        
        if all_missing_runs:
            print(f"  Need to run: {all_missing_runs}")
            print(f"  Details by solver:")
            for solver, missing in runs_needed.items():
                if missing:
                    print(f"    {solver:12s}: missing runs {missing}")
            
            # Run the missing benchmarks
            for run_num in all_missing_runs:
                result = run_benchmark(n_farms, run_number=run_num, total_runs=NUM_RUNS, 
                                     cache=cache, save_to_cache=True)
                if result:
                    config_results.append(result)
                    all_results.append(result)
        else:
            print(f"  ✓ All {NUM_RUNS} runs already completed for all solvers!")
        
        # Calculate statistics for this configuration
        if config_results:
            pulp_times = [r['pulp_time'] for r in config_results if r['pulp_time'] is not None]
            pyomo_times = [r['pyomo_time'] for r in config_results if r['pyomo_time'] is not None]
            cqm_times = [r['cqm_time'] for r in config_results if r['cqm_time'] is not None]
            pulp_diffs = [r['pulp_diff_percent'] for r in config_results if r['pulp_diff_percent'] is not None]
            
            aggregated = {
                'n_farms': n_farms,
                'n_foods': config_results[0]['n_foods'],
                'problem_size': config_results[0]['problem_size'],
                'n_synergy_pairs': config_results[0]['n_synergy_pairs'],
                'n_vars_base': config_results[0]['n_vars_base'],
                'n_z_vars_pulp': config_results[0]['n_z_vars_pulp'],
                'n_vars_pulp': config_results[0]['n_vars_pulp'],
                'n_vars_quadratic': config_results[0]['n_vars_quadratic'],
                'n_constraints_pulp': config_results[0]['n_constraints_pulp'],
                'n_constraints_quadratic': config_results[0]['n_constraints_quadratic'],
                
                # CQM creation stats
                'cqm_time_mean': float(np.mean(cqm_times)) if cqm_times else None,
                'cqm_time_std': float(np.std(cqm_times)) if cqm_times else None,
                
                # PuLP stats
                'pulp_time_mean': float(np.mean(pulp_times)) if pulp_times else None,
                'pulp_time_std': float(np.std(pulp_times)) if pulp_times else None,
                'pulp_time_min': float(np.min(pulp_times)) if pulp_times else None,
                'pulp_time_max': float(np.max(pulp_times)) if pulp_times else None,
                
                # Pyomo stats
                'pyomo_time_mean': float(np.mean(pyomo_times)) if pyomo_times else None,
                'pyomo_time_std': float(np.std(pyomo_times)) if pyomo_times else None,
                'pyomo_time_min': float(np.min(pyomo_times)) if pyomo_times else None,
                'pyomo_time_max': float(np.max(pyomo_times)) if pyomo_times else None,
                
                # Solution difference stats (should be near zero)
                'pulp_diff_mean': float(np.mean(pulp_diffs)) if pulp_diffs else None,
                'pulp_diff_std': float(np.std(pulp_diffs)) if pulp_diffs else None,
                
                'num_runs': len(config_results),
            }
            
            aggregated_results.append(aggregated)
            
            # Print statistics
            print(f"\n  Statistics for {n_farms} farms ({len(config_results)} runs):")
            print(f"    CQM Creation: {aggregated['cqm_time_mean']:.3f}s ± {aggregated['cqm_time_std']:.3f}s")
            print(f"    PuLP:         {aggregated['pulp_time_mean']:.3f}s ± {aggregated['pulp_time_std']:.3f}s")
            if aggregated['pyomo_time_mean']:
                print(f"    Pyomo:        {aggregated['pyomo_time_mean']:.3f}s ± {aggregated['pyomo_time_std']:.3f}s")
            if aggregated['pulp_diff_mean']:
                print(f"    Solution Diff: {aggregated['pulp_diff_mean']:.4f}% ± {aggregated['pulp_diff_std']:.4f}%")
    
    # Save results to Benchmarks/LQ folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.join(script_dir, "Benchmarks", "LQ")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all individual runs
    all_results_file = os.path.join(benchmark_dir, f'benchmark_lq_all_runs_{timestamp}.json')
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save aggregated statistics
    aggregated_file = os.path.join(benchmark_dir, f'benchmark_lq_aggregated_{timestamp}.json')
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"All runs saved to: {all_results_file}")
    print(f"Aggregated stats saved to: {aggregated_file}")
    
    # Create plots in Plots folder
    plots_dir = os.path.join(script_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\nGenerating plots...")
    plot_results(aggregated_results, os.path.join(plots_dir, f'scalability_benchmark_lq_{timestamp}.png'))
    
    print(f"\n🎉 All done! Ready for your presentation!")

if __name__ == "__main__":
    main()
