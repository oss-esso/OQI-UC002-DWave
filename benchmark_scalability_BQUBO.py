"""
Scalability Benchmark Script for BQUBO (CQM→BQM with Gurobi QUBO Solver)
Tests different combinations of farms and food groups to analyze QUBO solver performance
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
from solver_runner_BQUBO import create_cqm
from solver_runner_PATCH import solve_with_gurobi_qubo
from dimod import cqm_to_bqm
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
    5, 19, 72, 279, 1096, 1535
]

# Number of runs per configuration for statistical analysis
NUM_RUNS = 1

def load_full_family_with_n_farms(n_farms, seed=42):
    """
    Load full_family scenario with specified number of farms.
    Uses the same logic as the scaling analysis.
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
    
    # For binary formulation: each plantation is exactly 1 acre
    # No minimum planting area needed - it's either 1 acre or 0
    
    # Relax food group constraints to make problem feasible for small farms
    # Only require food groups if there's enough capacity
    min_capacity = min(L.values())
    n_food_groups = len(food_groups)
    
    # Only add food group constraints if farm can accommodate them
    if min_capacity >= n_food_groups:
        food_group_config = {
            g: {'min_foods': 1, 'max_foods': len(lst)}  # At least 1 food per group
            for g, lst in food_groups.items()
        }
    else:
        # Too many food groups for small farms - make it optional
        food_group_config = {
            g: {'min_foods': 0, 'max_foods': len(lst)}  # No minimum requirement
            for g, lst in food_groups.items()
        }
    
    # Build config
    parameters = {
        'land_availability': L,  # Max number of 1-acre plantations per farm
        'minimum_planting_area': {},  # Not used in binary formulation
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': food_group_config,
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }
    
    config = {'parameters': parameters}
    
    return farms, foods, food_groups, config

def run_benchmark(n_farms, run_number=1, total_runs=1, dwave_token=None, cache=None, save_to_cache=True):
    """
    Run a single benchmark test with full_family scenario.
    Returns timing results and problem size metrics for Gurobi QUBO and optionally DWave BQUBO.
    
    Args:
        n_farms: Number of farms to test
        run_number: Current run number (for display)
        total_runs: Total number of runs (for display)
        dwave_token: DWave API token (optional, set to None to skip DWave)
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
        # Binary formulation - only binary variables (each farm-crop is a 1-acre plantation)
        n_vars = n_farms * n_foods  # Only Y (binary plantation decisions)
        n_constraints = n_farms + 2*len(food_groups)*n_farms  # Plantation limits + food group constraints
        problem_size = n_farms * n_foods  # n = farms × foods
        
        print(f"  Foods: {n_foods}")
        print(f"  Variables: {n_vars} (all binary)")
        print(f"  Constraints: {n_constraints}")
        print(f"  Problem Size (n): {problem_size}")
        print(f"  Formulation: Binary - each farm-crop = 1 acre plantation if selected")
        
        # Create CQM (needed for conversion to BQM)
        print(f"\n  Creating CQM model...")
        cqm_start = time.time()
        cqm, Y, constraint_metadata = create_cqm(
            farms, foods, food_groups, config
        )
        cqm_time = time.time() - cqm_start
        print(f"    ✅ CQM created: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints ({cqm_time:.2f}s)")
        
        # Convert CQM to BQM for QUBO solving
        print(f"\n  Converting CQM to BQM...")
        bqm_convert_start = time.time()
        bqm, invert = cqm_to_bqm(cqm)
        bqm_conversion_time = time.time() - bqm_convert_start
        print(f"    ✅ BQM created: {len(bqm.variables)} vars, {len(bqm.quadratic)} quadratic terms ({bqm_conversion_time:.2f}s)")
        
        # Save CQM to cache if requested
        if save_to_cache and cache:
            cqm_data = serialize_cqm(cqm)
            cqm_result = {
                'cqm_time': cqm_time,
                'bqm_conversion_time': bqm_conversion_time,
                'num_variables': len(cqm.variables),
                'num_constraints': len(cqm.constraints),
                'num_bqm_variables': len(bqm.variables),
                'num_bqm_quadratic': len(bqm.quadratic),
                'n_foods': n_foods,
                'problem_size': problem_size,
                'n_vars': n_vars,
                'n_constraints': n_constraints
            }
            cache.save_result('BQUBO', 'CQM', n_farms, run_number, cqm_result, cqm_data=cqm_data)
        
        # Solve with Gurobi QUBO
        print(f"\n  Solving with Gurobi QUBO (BQM formulation)...")
        gurobi_start = time.time()
        
        # Get list of farms as list (not dict keys)
        farms_list = list(farms) if isinstance(farms, dict) else farms
        
        gurobi_result = solve_with_gurobi_qubo(
            bqm,
            farms=farms_list,
            foods=foods,
            food_groups=food_groups,
            land_availability=config['parameters']['land_availability'],
            weights=config['parameters']['weights'],
            idle_penalty=config['parameters'].get('idle_penalty_lambda', 0.1),
            config=config
        )
        gurobi_time = time.time() - gurobi_start
        
        print(f"    Status: {gurobi_result['status']}")
        print(f"    BQM Energy: {gurobi_result.get('bqm_energy', 'N/A')}")
        print(f"    Original Objective: {gurobi_result.get('objective_value', 'N/A')}")
        print(f"    Time: {gurobi_time:.3f}s")
        
        # Save Gurobi QUBO results to cache
        if save_to_cache and cache:
            gurobi_cache_result = {
                'solve_time': gurobi_time,
                'status': gurobi_result['status'],
                'bqm_energy': gurobi_result.get('bqm_energy'),
                'objective_value': gurobi_result.get('objective_value'),
                'solution': clean_solution_for_json(gurobi_result.get('solution', {})),
                'n_foods': n_foods,
                'problem_size': problem_size,
                'n_vars': n_vars,
                'n_constraints': n_constraints,
                'num_bqm_variables': len(bqm.variables),
                'num_bqm_quadratic': len(bqm.quadratic)
            }
            cache.save_result('BQUBO', 'GurobiQUBO', n_farms, run_number, gurobi_cache_result)
        
            cache.save_result('BQUBO', 'GurobiQUBO', n_farms, run_number, gurobi_cache_result)
        
        # DWave solving with BQUBO (CQM→BQM + HybridBQM)
        hybrid_time = None
        qpu_time = None
        bqm_conversion_time_dwave = None
        dwave_feasible = False
        dwave_objective = None
        
        if dwave_token:
            print(f"\n  Solving with DWave (BQUBO: CQM→BQM + HybridBQM)...")
            try:
                from solver_runner_BQUBO import solve_with_dwave
                sampleset, hybrid_time, qpu_time, bqm_conversion_time_dwave, invert = solve_with_dwave(cqm, dwave_token)
                
                # BQM samplesets don't have feasibility - all samples are valid
                print(f"    Status: {'Optimal' if len(sampleset) > 0 else 'No solutions'}")
                print(f"    Samples: {len(sampleset)}")
                if hybrid_time is not None:
                    print(f"    Hybrid Time: {hybrid_time:.3f}s")
                print(f"    BQM Conversion: {bqm_conversion_time_dwave:.3f}s")
                if qpu_time is not None:
                    print(f"    QPU Access: {qpu_time:.4f}s")
                
                if len(sampleset) > 0:
                    best = sampleset.first
                    dwave_objective = -best.energy
                    dwave_feasible = True
                    print(f"    Objective: {dwave_objective:.6f}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n  DWave: SKIPPED (no token)")
        
        # Save DWave results to cache
        if save_to_cache and cache:
            dwave_cache_result = {
                'hybrid_time': hybrid_time,
                'qpu_time': qpu_time,
                'bqm_conversion_time': bqm_conversion_time_dwave,
                'feasible': dwave_feasible,
                'objective_value': dwave_objective,
                'num_samples': len(sampleset) if 'sampleset' in locals() else 0
            }
            cache.save_result('BQUBO', 'DWave', n_farms, run_number, dwave_cache_result)
        
        result = {
            'n_farms': n_farms,
            'n_foods': n_foods,
            'n_vars': n_vars,
            'n_constraints': n_constraints,
            'problem_size': problem_size,
            'cqm_time': cqm_time,
            'bqm_conversion_time': bqm_conversion_time,
            'gurobi_time': gurobi_time,
            'gurobi_status': gurobi_result['status'],
            'gurobi_objective': gurobi_result.get('objective_value'),
            'hybrid_time': hybrid_time,
            'qpu_time': qpu_time,
            'bqm_conversion_time_dwave': bqm_conversion_time_dwave,
            'dwave_feasible': dwave_feasible,
            'dwave_objective': dwave_objective,
            'num_bqm_variables': len(bqm.variables),
            'num_bqm_quadratic': len(bqm.quadratic)
        }
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(results, output_file='scalability_benchmark_bqubo.png'):
    """
    Create beautiful plots for presentation with error bars including DWave BQUBO results.
    Results should be aggregated statistics with mean and std.
    """
    # Filter valid results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to plot!")
        return
    
    # Extract data
    problem_sizes = [r['problem_size'] for r in valid_results]
    
    # Gurobi QUBO times
    gurobi_times = [r['gurobi_time_mean'] for r in valid_results]
    gurobi_errors = [r['gurobi_time_std'] for r in valid_results]
    
    # DWave times
    hybrid_times = [r['hybrid_time_mean'] for r in valid_results if r.get('hybrid_time_mean') is not None]
    hybrid_errors = [r['hybrid_time_std'] for r in valid_results if r.get('hybrid_time_mean') is not None]
    hybrid_problem_sizes = [r['problem_size'] for r in valid_results if r.get('hybrid_time_mean') is not None]
    
    # QPU times
    qpu_times = [r['qpu_time_mean'] for r in valid_results if r.get('qpu_time_mean') is not None]
    qpu_errors = [r['qpu_time_std'] for r in valid_results if r.get('qpu_time_mean') is not None]
    qpu_problem_sizes = [r['problem_size'] for r in valid_results if r.get('qpu_time_mean') is not None]
    
    # Solution quality (we can plot objective values if needed)
    
    # Create figure with professional styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Solve times with error bars
    ax1.errorbar(problem_sizes, gurobi_times, yerr=gurobi_errors, marker='o', linestyle='-', 
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                label='Gurobi QUBO (BQM)', color='#2E86AB', alpha=0.9)
    
    if hybrid_times:
        ax1.errorbar(hybrid_problem_sizes, hybrid_times, yerr=hybrid_errors, marker='D', linestyle='-',
                    linewidth=2.5, markersize=8, capsize=5, capthick=2,
                    label='DWave BQUBO (Hybrid)', color='#F18F01', alpha=0.9)
    
    ax1.set_xlabel('Problem Size (n = Farms × Foods)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax1.set_title('BQUBO Solver Performance (Linear Objective)', 
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
        
        ax2.set_xlabel('Problem Size (n = Farms × Foods)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('QPU Access Time (seconds)', fontsize=14, fontweight='bold')
        ax2.set_title('DWave QPU Utilization (BQUBO Advantage)', 
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
    
    # Plot 4: Speedup Analysis (DWave vs Gurobi QUBO)
    if hybrid_times and len(hybrid_times) == len(gurobi_times):
        speedups = [gurobi_times[i] / hybrid_times[i] for i in range(len(hybrid_times))]
        ax4.plot(hybrid_problem_sizes, speedups, marker='D', linestyle='-',
                linewidth=2.5, markersize=8, color='#C73E1D', alpha=0.9,
                label='DWave Speedup Factor')
        
        ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='No Speedup')
        
        ax4.set_xlabel('Problem Size (n = Farms × Foods)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Speedup Factor (Gurobi / DWave)', fontsize=14, fontweight='bold')
        ax4.set_title('DWave BQUBO Speedup vs Gurobi QUBO', 
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
    create_summary_table(valid_results, 'scalability_table_bqubo.png')

def create_summary_table(results, output_file='scalability_table_bqubo.png'):
    """
    Create a beautiful summary table for BQUBO benchmark.
    """
    fig, ax = plt.subplots(figsize=(18, len(results) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Farms', 'Foods', 'n', 'BQM Vars', 'BQM Quad', 
               'Gurobi\nTime (s)', 'DWave\nTime (s)', 'QPU\nTime (s)', 'BQM Conv\nTime (s)', 'Runs', 'Winner']
    
    table_data = []
    for r in results:
        # Determine winner (faster solver)
        times = []
        if r.get('gurobi_time_mean') is not None:
            times.append(('Gurobi', r['gurobi_time_mean']))
        if r.get('hybrid_time_mean') is not None:
            times.append(('DWave', r['hybrid_time_mean']))
        
        if times:
            winner_name, winner_time = min(times, key=lambda x: x[1])
            winner = f'🏆 {winner_name}'
        else:
            winner = 'N/A'
        
        row = [
            r['n_farms'],
            r['n_foods'],
            r['problem_size'],
            r.get('num_bqm_variables', 'N/A'),
            r.get('num_bqm_quadratic', 'N/A'),
            f"{r['gurobi_time_mean']:.3f} ± {r['gurobi_time_std']:.3f}",
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
    
    plt.title('BQUBO Scalability Benchmark Results', fontsize=16, fontweight='bold', pad=20)
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
    print("BQUBO SCALABILITY BENCHMARK (CQM→BQM + Gurobi QUBO)")
    print("="*80)
    print(f"Configurations: {len(BENCHMARK_CONFIGS)} points")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Total benchmarks: {len(BENCHMARK_CONFIGS) * NUM_RUNS}")
    print(f"Objective: Linear")
    print("="*80)
    
    # Check cache status
    print("\n" + "="*80)
    print("CHECKING CACHE STATUS")
    print("="*80)
    cache.print_cache_status('BQUBO', BENCHMARK_CONFIGS, NUM_RUNS)
    
    # Get DWave token
    dwave_token = None #os.getenv('DWAVE_API_TOKEN', '45FS-7b81782896495d7c6a061bda257a9d9b03b082cd')
    if dwave_token:
        print(f"\n✅ DWave API token found - QPU-enabled benchmarking active")
    else:
        print(f"\n⚠️  No DWave API token - DWave benchmarks will be skipped")
        print(f"   Set DWAVE_API_TOKEN environment variable to enable DWave")
    
    all_results = []
    aggregated_results = []
    
    for n_farms in BENCHMARK_CONFIGS:
        print(f"\n" + "="*80)
        print(f"TESTING CONFIGURATION: {n_farms} Farms")
        print("="*80)
        
        # Check which runs are needed
        runs_needed = cache.get_runs_needed('BQUBO', n_farms, NUM_RUNS)
        
        # Load existing results from cache for GurobiQUBO (our primary solver)
        existing_gurobi_results = cache.get_all_results('BQUBO', 'GurobiQUBO', n_farms)
        config_results = []
        
        # Convert cached results to the format expected by aggregation
        for cached in existing_gurobi_results:
            result_data = cached['result']
            run_num = cached['metadata']['run_number']
            
            run_result = {
                'n_farms': n_farms,
                'n_foods': result_data.get('n_foods', 10),
                'gurobi_time': result_data['solve_time'],
                'gurobi_status': result_data['status'],
                'gurobi_objective': result_data['objective_value'],
                'problem_size': result_data.get('problem_size', n_farms * 10),
                'n_vars': result_data.get('n_vars', 0),
                'n_constraints': result_data.get('n_constraints', 0),
                'num_bqm_variables': result_data.get('num_bqm_variables', 0),
                'num_bqm_quadratic': result_data.get('num_bqm_quadratic', 0)
            }
            
            # Load corresponding CQM result
            cqm_cached = cache.load_result('BQUBO', 'CQM', n_farms, run_num)
            if cqm_cached:
                run_result['cqm_time'] = cqm_cached['result']['cqm_time']
                run_result['bqm_conversion_time'] = cqm_cached['result'].get('bqm_conversion_time', 0)
            
            # Load corresponding DWave result
            dwave_cached = cache.load_result('BQUBO', 'DWave', n_farms, run_num)
            if dwave_cached and dwave_cached['result'].get('hybrid_time'):
                run_result['hybrid_time'] = dwave_cached['result']['hybrid_time']
                run_result['qpu_time'] = dwave_cached['result'].get('qpu_time')
                run_result['bqm_conversion_time'] = dwave_cached['result'].get('bqm_conversion_time')
                run_result['dwave_feasible'] = dwave_cached['result'].get('feasible', False)
                run_result['dwave_objective'] = dwave_cached['result'].get('objective_value')
            else:
                run_result['hybrid_time'] = None
                run_result['qpu_time'] = None
                run_result['bqm_conversion_time'] = None
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
                                     dwave_token=dwave_token, cache=cache, save_to_cache=True)
                if result:
                    config_results.append(result)
                    all_results.append(result)
        else:
            print(f"  ✓ All {NUM_RUNS} runs already completed for all solvers!")
        
        # Calculate statistics for this configuration
        if config_results:
            gurobi_times = [r['gurobi_time'] for r in config_results if r.get('gurobi_time') is not None]
            cqm_times = [r.get('cqm_time') for r in config_results if r.get('cqm_time') is not None]
            hybrid_times = [r.get('hybrid_time') for r in config_results if r.get('hybrid_time') is not None]
            qpu_times = [r.get('qpu_time') for r in config_results if r.get('qpu_time') is not None]
            bqm_conv_times = [r.get('bqm_conversion_time') for r in config_results if r.get('bqm_conversion_time') is not None]
            
            aggregated = {
                'n_farms': n_farms,
                'n_foods': config_results[0]['n_foods'],
                'problem_size': config_results[0]['problem_size'],
                'n_vars': config_results[0].get('n_vars', 0),
                'n_constraints': config_results[0].get('n_constraints', 0),
                'num_bqm_variables': config_results[0].get('num_bqm_variables', 0),
                'num_bqm_quadratic': config_results[0].get('num_bqm_quadratic', 0),
                
                # CQM creation stats
                'cqm_time_mean': float(np.mean(cqm_times)) if cqm_times else None,
                'cqm_time_std': float(np.std(cqm_times)) if cqm_times else None,
                
                # Gurobi QUBO stats
                'gurobi_time_mean': float(np.mean(gurobi_times)) if gurobi_times else None,
                'gurobi_time_std': float(np.std(gurobi_times)) if gurobi_times else None,
                'gurobi_time_min': float(np.min(gurobi_times)) if gurobi_times else None,
                'gurobi_time_max': float(np.max(gurobi_times)) if gurobi_times else None,
                
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
            print(f"\n  Statistics for {n_farms} farms ({len(config_results)} runs):")
            print(f"    CQM Creation:     {aggregated['cqm_time_mean']:.3f}s ± {aggregated['cqm_time_std']:.3f}s")
            print(f"    Gurobi QUBO:      {aggregated['gurobi_time_mean']:.3f}s ± {aggregated['gurobi_time_std']:.3f}s")
            if aggregated['hybrid_time_mean']:
                print(f"    DWave (Total):    {aggregated['hybrid_time_mean']:.3f}s ± {aggregated['hybrid_time_std']:.3f}s")
                print(f"    BQM Conversion:   {aggregated['bqm_conversion_time_mean']:.3f}s ± {aggregated['bqm_conversion_time_std']:.3f}s")
                print(f"    QPU Access:       {aggregated['qpu_time_mean']:.4f}s ± {aggregated['qpu_time_std']:.4f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all individual runs
    all_results_file = f'benchmark_bqubo_all_runs_{timestamp}.json'
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save aggregated statistics
    aggregated_file = f'benchmark_bqubo_aggregated_{timestamp}.json'
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"All runs saved to: {all_results_file}")
    print(f"Aggregated stats saved to: {aggregated_file}")
    
    # Create plots
    print(f"\nGenerating plots...")
    plot_results(aggregated_results, f'scalability_benchmark_bqubo_{timestamp}.png')
    
    print(f"\n🎉 BQUBO Benchmark Complete! Gurobi QUBO scaling analysis ready!")

if __name__ == "__main__":
    main()
