"""
Scalability Benchmark Script for BQM_PATCH (Plot-Crop Assignment)
Tests different combinations of patches and crops to analyze QPU-enabled solver performance
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.patch_sampler import generate_farms as generate_patches
from src.scenarios import load_food_data
from solver_runner_PATCH import create_cqm, solve_with_pulp, solve_with_dwave, solve_with_simulated_annealing, solve_with_gurobi_qubo
from dimod import cqm_to_bqm
from Utils.benchmark_cache import BenchmarkCache, serialize_cqm
from Utils.constraint_validator import validate_bqm_patch_constraints, validate_pulp_patch_constraints, print_validation_report
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
# Format: number of patches to test with simple scenario
# Using patches (1/10 the size of farms)
BENCHMARK_CONFIGS = [
    5, 10, 15, 25
]

# Number of runs per configuration for statistical analysis
NUM_RUNS = 1

def load_full_family_with_n_patches(n_patches, seed=42):
    """
    Load full_family scenario with specified number of patches (using patch_sampler).
    Adjusts food group constraints to be feasible for the given number of patches.
    """
    import pandas as pd
    
    # Generate patches (1/10 the size of farms)
    L = generate_patches(n_farms=n_patches, seed=seed)
    patches = list(L.keys())
    
    # Load food data from Excel or use fallback
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    excel_path = os.path.join(project_root, "Inputs", "Combined_Food_Data.xlsx")
    
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
        }
        food_groups = {
            'Grains': ['Wheat', 'Corn', 'Rice'],
            'Legumes': ['Soybeans'],
            'Vegetables': ['Potatoes'],
            'Fruits': ['Apples'],
        }
    else:
        # Load from Excel - USE ALL FOODS (not just 2 per group)
        df = pd.read_excel(excel_path)
        
        # Use ALL foods from the dataset
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
    
    # Calculate total land
    total_land = sum(L.values())
    n_food_groups = len(food_groups)
    
    # Adjust food group constraints based on number of patches
    # Only require food groups if there's enough capacity
    min_capacity = min(L.values())
    
    if n_patches >= n_food_groups:
        # Enough patches to potentially have at least 1 crop per group
        food_group_config = {
            g: {'min_foods': 1, 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        }
    else:
        # Not enough patches - relax constraints (make all optional)
        food_group_config = {
            g: {'min_foods': 0, 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        }
    
    # Build config with BQM_PATCH parameters
    parameters = {
        'land_availability': L,
        'minimum_planting_area': {food: 0.0 for food in foods},
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'food_group_constraints': food_group_config,
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        },
        'idle_penalty_lambda': 0.1
    }
    
    config = {'parameters': parameters}
    
    return patches, foods, food_groups, config

def calculate_objective_from_bqm_sample(sample, invert, patches, foods, config):
    """
    Calculate the actual objective value from a BQM solution.
    
    Args:
        sample: BQM sample dictionary
        invert: Invert function from cqm_to_bqm
        patches: List of patch names
        foods: Dictionary of food data
        config: Configuration dictionary
    
    Returns:
        Objective value (maximization)
    """
    # Convert BQM sample back to CQM variables
    cqm_sample = invert(sample)
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    idle_penalty = params.get('idle_penalty_lambda', 0.1)
    
    # Calculate objective: sum_{p,c} (B_c + λ) * s_p * X_{p,c}
    objective = 0.0
    
    for plot in patches:
        s_p = land_availability[plot]
        
        for crop in foods:
            # Get X_{p,c} value from sample
            x_var_name = f"X_{plot}_{crop}"
            x_val = cqm_sample.get(x_var_name, 0)
            
            # Calculate B_c (weighted benefit for crop c)
            B_c = (
                weights['nutritional_value'] * foods[crop]['nutritional_value'] +
                weights['nutrient_density'] * foods[crop]['nutrient_density'] +
                weights['environmental_impact'] * (1 - foods[crop]['environmental_impact']) +
                weights['affordability'] * foods[crop]['affordability'] +
                weights['sustainability'] * foods[crop]['sustainability']
            )
            
            # Add contribution
            objective += (B_c + idle_penalty) * s_p * x_val
    
    return objective

def run_benchmark(n_patches, run_number=1, total_runs=1, dwave_token=None, cache=None, save_to_cache=True):
    """
    Run a single benchmark test with simple scenario.
    Returns timing results and problem size metrics for all solvers including DWave BQUBO.
    
    Args:
        n_patches: Number of patches to test
        run_number: Current run number (for display)
        total_runs: Total number of runs (for display)
        dwave_token: DWave API token (optional)
        cache: BenchmarkCache instance for saving results
        save_to_cache: Whether to save results to cache (default: True)
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: full_family scenario with {n_patches} Patches (Run {run_number}/{total_runs})")
    print(f"{'='*80}")
    
    try:
        # Load full_family scenario with specified number of patches
        patches, foods, food_groups, config = load_full_family_with_n_patches(n_patches, seed=42 + run_number)
        
        n_foods = len(foods)
        # BQM_PATCH formulation: X_{p,c} + Y_c variables
        n_x_vars = n_patches * n_foods  # X variables
        n_y_vars = n_foods  # Y variables
        n_vars = n_x_vars + n_y_vars
        # Constraints: at most one per plot + X-Y linking + Y activation + area bounds + food groups
        n_constraints = n_patches + (n_patches * n_foods) + n_foods + (n_foods * 2) + (2 * len(food_groups))
        problem_size = n_patches * n_foods  # n = patches × foods
        
        print(f"  Foods: {n_foods}")
        print(f"  Variables: {n_vars} (X: {n_x_vars}, Y: {n_y_vars})")
        print(f"  Constraints: ~{n_constraints}")
        print(f"  Problem Size (n): {problem_size}")
        print(f"  Formulation: BQM_PATCH (plot-crop assignment with implicit idle)")
        
        # Create CQM (needed for DWave)
        print(f"\n  Creating CQM model...")
        cqm_start = time.time()
        cqm, (X, Y), constraint_metadata = create_cqm(
            patches, foods, food_groups, config
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
                'n_vars': n_vars,
                'n_constraints': len(cqm.constraints)
            }
            cache.save_result('PATCH', 'CQM', n_patches, run_number, cqm_result, cqm_data=cqm_data)
        
        # Solve with PuLP
        print(f"\n  Solving with PuLP...")
        pulp_start = time.time()
        pulp_model, pulp_results = solve_with_pulp(patches, foods, food_groups, config)
        pulp_time = time.time() - pulp_start
        
        print(f"    Status: {pulp_results['status']}")
        print(f"    Objective: {pulp_results.get('objective_value', 'N/A')}")
        print(f"    Time: {pulp_time:.3f}s")
        
        # Validate PuLP constraints
        print(f"\n    Validating PuLP solution constraints...")
        pulp_validation = validate_pulp_patch_constraints(
            pulp_results.get('X_variables', {}),
            pulp_results.get('Y_variables', {}),
            patches, foods, food_groups, config
        )
        print_validation_report(pulp_validation, verbose=False)
        
        # Save PuLP results to cache
        if save_to_cache and cache:
            pulp_cache_result = {
                'solve_time': pulp_time,
                'status': pulp_results['status'],
                'objective_value': pulp_results.get('objective_value'),
                'X_variables': clean_solution_for_json(pulp_results.get('X_variables', {})),
                'Y_variables': clean_solution_for_json(pulp_results.get('Y_variables', {})),
                'n_foods': n_foods,
                'problem_size': problem_size,
                'n_vars': n_vars,
                'n_constraints': len(cqm.constraints),
                'validation': pulp_validation if 'pulp_validation' in locals() else None
            }
            cache.save_result('PATCH', 'PuLP', n_patches, run_number, pulp_cache_result)
            print(f"✓ Saved PuLP result: config_{n_patches}_run_{run_number}")
        
        # Only convert to BQM and run DWave/SA if DWave token is available
        # This prevents overwriting existing DWave results when running PuLP-only benchmarks
        if dwave_token:
            # Convert CQM to BQM once for both samplers
            print(f"\n  Converting CQM to BQM...")
            convert_start = time.time()
            bqm, invert = cqm_to_bqm(cqm)
            bqm_conversion_time = time.time() - convert_start
            print(f"    ✅ BQM created: {len(bqm.variables)} vars, {len(bqm.quadratic)} interactions ({bqm_conversion_time:.2f}s)")
        else:
            # Skip BQM conversion and DWave/SA when token is disabled
            bqm_conversion_time = None
            print(f"\n  ⚠️  Skipping BQM conversion and DWave/SA (token disabled)")
        
        # DWave HybridBQM solving
        hybrid_time = None
        qpu_time = None
        dwave_feasible = False
        dwave_objective = None
        
        if dwave_token:
            print(f"\n  Solving with DWave HybridBQM...")
            try:
                from dwave.system import LeapHybridBQMSampler
                sampler = LeapHybridBQMSampler(token=dwave_token)
                
                hybrid_start = time.time()
                sampleset = sampler.sample(bqm, label="BQM_PATCH Benchmark")
                hybrid_time = time.time() - hybrid_start
                
                # Extract timing from sampleset
                timing_info = sampleset.info.get('timing', {})
                qpu_time = (timing_info.get('qpu_access_time') or 
                           sampleset.info.get('qpu_access_time'))
                if qpu_time is not None:
                    qpu_time = qpu_time / 1e6  # Convert to seconds
                
                print(f"    Status: {'Optimal' if len(sampleset) > 0 else 'No solutions'}")
                print(f"    Samples: {len(sampleset)}")
                print(f"    Hybrid Time: {hybrid_time:.3f}s")
                if qpu_time is not None:
                    print(f"    QPU Access: {qpu_time:.4f}s")
                
                if len(sampleset) > 0:
                    best = sampleset.first
                    # Calculate actual objective using the invert function
                    dwave_objective = calculate_objective_from_bqm_sample(
                        best.sample, invert, patches, foods, config
                    )
                    dwave_feasible = True
                    print(f"    BQM Energy: {best.energy:.6f}")
                    print(f"    Actual Objective: {dwave_objective:.6f}")
                    
                    # Validate constraints
                    print(f"\n    Validating DWave solution constraints...")
                    dwave_validation = validate_bqm_patch_constraints(
                        best.sample, invert, patches, foods, food_groups, config
                    )
                    print_validation_report(dwave_validation, verbose=False)
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n  DWave: SKIPPED (no token)")
        
        # Save DWave results to cache ONLY if DWave actually ran
        # Don't overwrite existing DWave results with Simulated Annealing-only runs
        if save_to_cache and cache and dwave_token is not None:
            dwave_cache_result = {
                'hybrid_time': hybrid_time,
                'qpu_time': qpu_time,
                'bqm_conversion_time': bqm_conversion_time,
                'feasible': dwave_feasible,
                'objective_value': dwave_objective,
                'num_samples': len(sampleset) if 'sampleset' in locals() else 0,
                'dwave_validation': dwave_validation if 'dwave_validation' in locals() else None
            }
            cache.save_result('PATCH', 'DWave', n_patches, run_number, dwave_cache_result)
            print(f"✓ Saved DWave result: config_{n_patches}_run_{run_number}")
        elif save_to_cache and cache and dwave_token is None:
            print(f"⚠️  Skipping DWave cache save (token disabled) - preserving existing DWave results")
        
        # Gurobi QUBO solving
        gurobi_time = None
        gurobi_feasible = False
        gurobi_objective = None
        gurobi_validation = None
        
        if dwave_token:  # Only run if BQM was created
            print(f"\n  Solving with Gurobi QUBO...")
            try:
                gurobi_result = solve_with_gurobi_qubo(bqm)
                gurobi_time = gurobi_result['solve_time']
                gurobi_feasible = gurobi_result['status'] == 'Optimal'
                
                if gurobi_feasible:
                    # Calculate actual objective using the invert function
                    gurobi_objective = calculate_objective_from_bqm_sample(
                        gurobi_result['solution'], invert, patches, foods, config
                    )
                    print(f"    Status: {gurobi_result['status']}")
                    print(f"    BQM Energy: {gurobi_result['bqm_energy']:.6f}")
                    print(f"    Actual Objective: {gurobi_objective:.6f}")
                    print(f"    Solve Time: {gurobi_time:.3f}s")
                    
                    # Validate constraints
                    print(f"\n    Validating Gurobi QUBO solution constraints...")
                    gurobi_validation = validate_bqm_patch_constraints(
                        gurobi_result['solution'], invert, patches, foods, food_groups, config
                    )
                    print_validation_report(gurobi_validation, verbose=False)
                else:
                    print(f"    Status: {gurobi_result['status']}")
                    print(f"    Solve Time: {gurobi_time:.3f}s")
                    
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n  Gurobi QUBO: SKIPPED (no BQM available)")
        
        # Save Gurobi results to cache 
        if save_to_cache and cache and dwave_token is not None:
            gurobi_cache_result = {
                'gurobi_time': gurobi_time,
                'feasible': gurobi_feasible,
                'objective_value': gurobi_objective,
                'gurobi_validation': gurobi_validation
            }
            cache.save_result('PATCH', 'GurobiQUBO', n_patches, run_number, gurobi_cache_result)
            print(f"✓ Saved Gurobi QUBO result: config_{n_patches}_run_{run_number}")
        elif save_to_cache and cache and dwave_token is None:
            print(f"⚠️  Skipping Gurobi QUBO cache save (no BQM available)")
        
        result = {
            'n_patches': n_patches,
            'n_foods': n_foods,
            'n_vars': n_vars,
            'n_constraints': len(cqm.constraints),
            'problem_size': problem_size,
            'cqm_time': cqm_time,
            'pulp_time': pulp_time,
            'pulp_status': pulp_results['status'],
            'pulp_objective': pulp_results.get('objective_value'),
            'hybrid_time': hybrid_time,
            'qpu_time': qpu_time,
            'bqm_conversion_time': bqm_conversion_time,
            'dwave_feasible': dwave_feasible,
            'dwave_objective': dwave_objective,
            'gurobi_time': gurobi_time,
            'gurobi_feasible': gurobi_feasible,
            'gurobi_objective': gurobi_objective
        }
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(results, output_file='scalability_benchmark_patch.png'):
    """
    Create beautiful plots for presentation with error bars including DWave BQM_PATCH results.
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
    
    # DWave times
    hybrid_times = [r['hybrid_time_mean'] for r in valid_results if r.get('hybrid_time_mean') is not None]
    hybrid_errors = [r['hybrid_time_std'] for r in valid_results if r.get('hybrid_time_mean') is not None]
    hybrid_problem_sizes = [r['problem_size'] for r in valid_results if r.get('hybrid_time_mean') is not None]
    
    # QPU times
    qpu_times = [r['qpu_time_mean'] for r in valid_results if r['qpu_time_mean'] is not None]
    qpu_errors = [r['qpu_time_std'] for r in valid_results if r['qpu_time_mean'] is not None]
    qpu_problem_sizes = [r['problem_size'] for r in valid_results if r['qpu_time_mean'] is not None]
    
    # Gurobi QUBO times
    gurobi_times = [r['gurobi_time_mean'] for r in valid_results if r.get('gurobi_time_mean') is not None]
    gurobi_errors = [r['gurobi_time_std'] for r in valid_results if r.get('gurobi_time_mean') is not None]
    gurobi_problem_sizes = [r['problem_size'] for r in valid_results if r.get('gurobi_time_mean') is not None]
    
    # Solution quality (we can plot objective values if needed)
    
    # Create figure with professional styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Solve times with error bars
    ax1.errorbar(problem_sizes, pulp_times, yerr=pulp_errors, marker='o', linestyle='-', 
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                label='PuLP (Linear)', color='#2E86AB', alpha=0.9)
    
    if hybrid_times:
        ax1.errorbar(hybrid_problem_sizes, hybrid_times, yerr=hybrid_errors, marker='D', linestyle='-',
                    linewidth=2.5, markersize=8, capsize=5, capthick=2,
                    label='DWave BQM_PATCH (Hybrid)', color='#F18F01', alpha=0.9)
    
    if gurobi_times:
        ax1.errorbar(gurobi_problem_sizes, gurobi_times, yerr=gurobi_errors, marker='s', linestyle='-',
                    linewidth=2.5, markersize=8, capsize=5, capthick=2,
                    label='Gurobi QUBO (GPU)', color='#A4243B', alpha=0.9)
    
    ax1.set_xlabel('Problem Size (n = Patches × Crops)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax1.set_title('BQM_PATCH Solver Performance (Linear Objective)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Plot 2: QPU Access Time
    if qpu_times:
        ax2.errorbar(qpu_problem_sizes, qpu_times, yerr=qpu_errors,
                    marker='D', linestyle='-', linewidth=2.5, markersize=8,
                    capsize=5, capthick=2, color='#06A77D', alpha=0.9,
                    label='QPU Access Time')
        
        ax2.set_xlabel('Problem Size (n = Patches × Crops)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('QPU Access Time (seconds)', fontsize=14, fontweight='bold')
        ax2.set_title('DWave QPU Utilization (BQM_PATCH)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend(loc='best', fontsize=11, framealpha=0.95)
        
        # Add average QPU time annotation
        avg_qpu = np.mean(qpu_times)
        ax2.text(0.98, 0.98, f'Avg QPU: {avg_qpu:.4f}s', 
                transform=ax2.transAxes, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    else:
        ax2.text(0.5, 0.5, 'No DWave QPU Data', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    # Plot 3: Solution quality comparison (placeholder for now)
    ax3.text(0.5, 0.5, 'Linear Objective\nNo Approximation Error', 
            ha='center', va='center', fontsize=14, transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Solution Quality', fontsize=16, fontweight='bold', pad=20)
    
    # Plot 4: Speedup Analysis (DWave vs PuLP)
    if hybrid_times and len(hybrid_times) == len(pulp_times):
        speedups = [pulp_times[i] / hybrid_times[i] for i in range(len(hybrid_times))]
        ax4.plot(hybrid_problem_sizes, speedups, marker='D', linestyle='-',
                linewidth=2.5, markersize=8, color='#C73E1D', alpha=0.9,
                label='DWave Speedup Factor')
        
        ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='No Speedup')
        
        ax4.set_xlabel('Problem Size (n = Patches × Crops)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Speedup Factor (PuLP / DWave)', fontsize=14, fontweight='bold')
        ax4.set_title('DWave BQM_PATCH Speedup vs Classical Solver', 
                     fontsize=16, fontweight='bold', pad=20)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xscale('log')
        ax4.legend(loc='best', fontsize=11, framealpha=0.95)
        
        # Add annotations
        max_speedup = max(speedups)
        max_idx = speedups.index(max_speedup)
        ax4.annotate(f'Max: {max_speedup:.2f}x\n@ n={hybrid_problem_sizes[max_idx]}',
                    xy=(hybrid_problem_sizes[max_idx], max_speedup),
                    xytext=(20, 20), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    else:
        ax4.text(0.5, 0.5, 'Insufficient DWave Data\nfor Speedup Analysis', 
                ha='center', va='center', fontsize=14, transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    
    # Also create a summary table plot
    create_summary_table(valid_results, 'scalability_table_patch.png')

def create_summary_table(results, output_file='scalability_table_patch.png'):
    """
    Create a beautiful summary table for BQM_PATCH benchmark.
    """
    fig, ax = plt.subplots(figsize=(18, len(results) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Patches', 'Crops', 'n', 'Vars', 'Constraints', 
               'PuLP\nTime (s)', 'DWave\nTime (s)', 'QPU\nTime (s)', 'BQM Conv\nTime (s)', 'Runs', 'Winner']
    
    table_data = []
    for r in results:
        # Determine winner (faster solver)
        times = []
        if r.get('pulp_time_mean') is not None:
            times.append(('PuLP', r['pulp_time_mean']))
        if r.get('hybrid_time_mean') is not None:
            times.append(('DWave', r['hybrid_time_mean']))
        
        if times:
            winner_name, winner_time = min(times, key=lambda x: x[1])
            winner = f'🏆 {winner_name}'
        else:
            winner = 'N/A'
        
        row = [
            r['n_patches'],
            r['n_foods'],
            r['problem_size'],
            r.get('n_vars', 'N/A'),
            r['n_constraints'],
            f"{r['pulp_time_mean']:.3f} ± {r['pulp_time_std']:.3f}",
            f"{r['hybrid_time_mean']:.3f} ± {r['hybrid_time_std']:.3f}" if r.get('hybrid_time_mean') else 'N/A',
            f"{r['qpu_time_mean']:.4f}" if r.get('qpu_time_mean') else 'N/A',
            f"{r['bqm_conversion_time_mean']:.3f}" if r.get('bqm_conversion_time_mean') else 'N/A',
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
    
    plt.title('BQM_PATCH Scalability Benchmark Results', fontsize=16, fontweight='bold', pad=20)
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
    print("BQM_PATCH SCALABILITY BENCHMARK (Plot-Crop Assignment)")
    print("="*80)
    print(f"Configurations: {len(BENCHMARK_CONFIGS)} points")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Total benchmarks: {len(BENCHMARK_CONFIGS) * NUM_RUNS}")
    print(f"Objective: Linear (BQM_PATCH formulation)")
    print("="*80)
    
    # Check cache status
    print("\n" + "="*80)
    print("CHECKING CACHE STATUS")
    print("="*80)
    cache.print_cache_status('PATCH', BENCHMARK_CONFIGS, NUM_RUNS)
    
    # Get DWave token - COMMENTED OUT TO PRESERVE BUDGET
    # dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
    dwave_token = None  # Disable DWave to preserve budget
    print(f"\n⚠️  DWave disabled to preserve budget - skipping all DWave tests")
    print(f"   To enable: uncomment dwave_token line and set DWAVE_API_TOKEN")
    
    all_results = []
    aggregated_results = []
    
    for n_patches in BENCHMARK_CONFIGS:
        print(f"\n" + "="*80)
        print(f"TESTING CONFIGURATION: {n_patches} Patches")
        print("="*80)
        
        # Check which runs are needed
        runs_needed = cache.get_runs_needed('PATCH', n_patches, NUM_RUNS)
        
        # Load existing results from cache for PuLP (our primary solver)
        existing_pulp_results = cache.get_all_results('PATCH', 'PuLP', n_patches)
        config_results = []
        
        # Convert cached results to the format expected by aggregation
        for cached in existing_pulp_results:
            result_data = cached['result']
            run_num = cached['metadata']['run_number']
            
            run_result = {
                'n_patches': n_patches,
                'n_foods': result_data.get('n_foods', 6),
                'pulp_time': result_data['solve_time'],
                'pulp_status': result_data['status'],
                'pulp_objective': result_data['objective_value'],
                'problem_size': result_data.get('problem_size', n_patches * 6),
                'n_vars': result_data.get('n_vars', 0),
                'n_constraints': result_data.get('n_constraints', 0)
            }
            
            # Load corresponding CQM result
            cqm_cached = cache.load_result('PATCH', 'CQM', n_patches, run_num)
            if cqm_cached:
                run_result['cqm_time'] = cqm_cached['result']['cqm_time']
            
            # Load corresponding DWave result
            dwave_cached = cache.load_result('PATCH', 'DWave', n_patches, run_num)
            if dwave_cached and dwave_cached['result'].get('hybrid_time'):
                run_result['hybrid_time'] = dwave_cached['result']['hybrid_time']
                run_result['qpu_time'] = dwave_cached['result'].get('qpu_time')
                run_result['bqm_conversion_time'] = dwave_cached['result'].get('bqm_conversion_time')
                run_result['dwave_feasible'] = dwave_cached['result'].get('feasible', False)
                run_result['dwave_objective'] = dwave_cached['result'].get('objective_value')
                run_result['sa_time'] = dwave_cached['result'].get('sa_time')
                run_result['sa_objective'] = dwave_cached['result'].get('sa_objective')
            else:
                run_result['hybrid_time'] = None
                run_result['qpu_time'] = None
                run_result['bqm_conversion_time'] = None
                run_result['dwave_feasible'] = False
                run_result['dwave_objective'] = None
                run_result['sa_time'] = None
                run_result['sa_objective'] = None
            
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
                result = run_benchmark(n_patches, run_number=run_num, total_runs=NUM_RUNS, 
                                     dwave_token=dwave_token, cache=cache, save_to_cache=True)
                if result:
                    config_results.append(result)
                    all_results.append(result)
        else:
            print(f"  ✓ All {NUM_RUNS} runs already completed for all solvers!")
        
        # Calculate statistics for this configuration
        if config_results:
            pulp_times = [r['pulp_time'] for r in config_results if r['pulp_time'] is not None]
            cqm_times = [r['cqm_time'] for r in config_results if r['cqm_time'] is not None]
            hybrid_times = [r['hybrid_time'] for r in config_results if r['hybrid_time'] is not None]
            qpu_times = [r['qpu_time'] for r in config_results if r['qpu_time'] is not None]
            bqm_conv_times = [r['bqm_conversion_time'] for r in config_results if r['bqm_conversion_time'] is not None]
            
            aggregated = {
                'n_patches': n_patches,
                'n_foods': config_results[0]['n_foods'],
                'problem_size': config_results[0]['problem_size'],
                'n_vars': config_results[0]['n_vars'],
                'n_constraints': config_results[0]['n_constraints'],
                
                # CQM creation stats
                'cqm_time_mean': float(np.mean(cqm_times)) if cqm_times else None,
                'cqm_time_std': float(np.std(cqm_times)) if cqm_times else None,
                
                # PuLP stats
                'pulp_time_mean': float(np.mean(pulp_times)) if pulp_times else None,
                'pulp_time_std': float(np.std(pulp_times)) if pulp_times else None,
                'pulp_time_min': float(np.min(pulp_times)) if pulp_times else None,
                'pulp_time_max': float(np.max(pulp_times)) if pulp_times else None,
                
                # DWave hybrid solver stats
                'hybrid_time_mean': float(np.mean(hybrid_times)) if hybrid_times else None,
                'hybrid_time_std': float(np.std(hybrid_times)) if hybrid_times else None,
                'hybrid_time_min': float(np.min(hybrid_times)) if hybrid_times else None,
                'hybrid_time_max': float(np.max(hybrid_times)) if hybrid_times else None,
                
                # QPU stats
                'qpu_time_mean': float(np.mean(qpu_times)) if qpu_times else None,
                'qpu_time_std': float(np.std(qpu_times)) if qpu_times else None,
                
                # BQM conversion stats
                'bqm_conversion_time_mean': float(np.mean(bqm_conv_times)) if bqm_conv_times else None,
                'bqm_conversion_time_std': float(np.std(bqm_conv_times)) if bqm_conv_times else None,
                
                'num_runs': len(config_results)
            }
            
            aggregated_results.append(aggregated)
            
            # Print statistics
            print(f"\n  Statistics for {n_patches} patches ({len(config_results)} runs):")
            print(f"    CQM Creation:     {aggregated['cqm_time_mean']:.3f}s ± {aggregated['cqm_time_std']:.3f}s")
            print(f"    PuLP:             {aggregated['pulp_time_mean']:.3f}s ± {aggregated['pulp_time_std']:.3f}s")
            if aggregated['hybrid_time_mean']:
                print(f"    DWave (Total):    {aggregated['hybrid_time_mean']:.3f}s ± {aggregated['hybrid_time_std']:.3f}s")
                print(f"    BQM Conversion:   {aggregated['bqm_conversion_time_mean']:.3f}s ± {aggregated['bqm_conversion_time_std']:.3f}s")
                print(f"    QPU Access:       {aggregated['qpu_time_mean']:.4f}s ± {aggregated['qpu_time_std']:.4f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all individual runs
    all_results_file = f'benchmark_patch_all_runs_{timestamp}.json'
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save aggregated statistics
    aggregated_file = f'benchmark_patch_aggregated_{timestamp}.json'
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"All runs saved to: {all_results_file}")
    print(f"Aggregated stats saved to: {aggregated_file}")
    
    # Create plots
    print(f"\nGenerating plots...")
    plot_results(aggregated_results, f'scalability_benchmark_patch_{timestamp}.png')
    
    print(f"\n🎉 BQM_PATCH Benchmark Complete! QPU-enabled scaling analysis ready!")

if __name__ == "__main__":
    main()
