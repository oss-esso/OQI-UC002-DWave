"""
Scalability Benchmark Script
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
from solver_runner_NLN import create_cqm, solve_with_pulp, solve_with_pyomo, solve_with_dwave
import pulp as pl

# Benchmark configurations
# Format: number of farms to test with full_family scenario
# 6 points logarithmically scaled from 5 to 1535 farms
# Reduced from 30 points for faster testing with multiple runs
BENCHMARK_CONFIGS = [
    5, 19, 72, 279, 1096, 1535
]

# Number of runs per configuration for statistical analysis
NUM_RUNS = 5

# Power for non-linear objective
POWER = 0.548
NUM_BREAKPOINTS = 10  # For piecewise approximation

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
    
    # Set minimum planting areas based on smallest farm and number of food groups
    smallest_farm = min(L.values())
    n_food_groups = len(food_groups)
    # Each farm must plant at least 1 crop from each food group
    # Reserve some margin for safety
    min_area_per_crop = (smallest_farm / n_food_groups) * 0.9  # 90% safety margin
    
    min_areas = {food: min_area_per_crop for food in foods.keys()}
    
    # Build config
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
            'sustainability': 0.15
        }
    }
    
    config = {'parameters': parameters}
    
    return farms, foods, food_groups, config

def run_benchmark(n_farms, run_number=1, total_runs=1):
    """
    Run a single benchmark test with full_family scenario.
    Returns timing results and problem size metrics for all three solvers.
    
    Args:
        n_farms: Number of farms to test
        run_number: Current run number (for display)
        total_runs: Total number of runs (for display)
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: full_family scenario with {n_farms} Farms (Run {run_number}/{total_runs})")
    print(f"{'='*80}")
    
    try:
        # Load full_family scenario with specified number of farms
        farms, foods, food_groups, config = load_full_family_with_n_farms(n_farms, seed=42 + run_number)
        
        n_foods = len(foods)
        # For NLN solver, we have additional lambda variables
        n_vars_base = 2 * n_farms * n_foods  # Binary + continuous (A and Y)
        n_lambda_vars = n_farms * n_foods * (NUM_BREAKPOINTS + 2)  # Lambda variables for piecewise
        n_vars = n_vars_base + n_lambda_vars
        n_constraints = n_farms + 2*n_farms*n_foods + 2*len(food_groups)*n_farms + 2*n_farms*n_foods  # Added piecewise constraints
        problem_size = n_farms * n_foods  # n = farms × foods
        
        print(f"  Foods: {n_foods}")
        print(f"  Base Variables: {n_vars_base}")
        print(f"  Lambda Variables: {n_lambda_vars}")
        print(f"  Total Variables: {n_vars}")
        print(f"  Constraints: {n_constraints}")
        print(f"  Problem Size (n): {problem_size}")
        
        # Create CQM (needed for DWave, but we'll skip DWave solving)
        print(f"\n  Creating CQM model...")
        cqm_start = time.time()
        cqm, A, Y, Lambda, constraint_metadata, approximation_metadata = create_cqm(
            farms, foods, food_groups, config, power=POWER, num_breakpoints=NUM_BREAKPOINTS
        )
        cqm_time = time.time() - cqm_start
        print(f"    ✅ CQM created: {len(cqm.variables)} vars, {len(cqm.constraints)} constraints ({cqm_time:.2f}s)")
        
        # Solve with PuLP (piecewise approximation)
        print(f"\n  Solving with PuLP (Piecewise Approximation)...")
        pulp_start = time.time()
        pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config, power=POWER, num_breakpoints=NUM_BREAKPOINTS)
        pulp_time = time.time() - pulp_start
        
        print(f"    Status: {pulp_results['status']}")
        print(f"    Objective: {pulp_results.get('objective_value', 'N/A')}")
        print(f"    Time: {pulp_time:.3f}s")
        
        # Solve with Pyomo (true non-linear)
        print(f"\n  Solving with Pyomo (True Non-Linear)...")
        pyomo_start = time.time()
        pyomo_model, pyomo_results = solve_with_pyomo(farms, foods, food_groups, config, power=POWER)
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
        
        # DWave solving is SKIPPED (no token)
        dwave_time = None
        qpu_time = None
        hybrid_time = None
        dwave_feasible = False
        dwave_objective = None
        
        print(f"\n  DWave: SKIPPED (no token)")
        
        # Calculate approximation error if we have Pyomo solution
        pulp_error = None
        if pyomo_objective is not None and pulp_results.get('objective_value') is not None:
            pulp_error = abs(pulp_results['objective_value'] - pyomo_objective) / abs(pyomo_objective) * 100
            print(f"\n  Approximation Error:")
            print(f"    PuLP vs Pyomo: {pulp_error:.2f}%")
        
        result = {
            'n_farms': n_farms,
            'n_foods': n_foods,
            'n_vars_base': n_vars_base,
            'n_lambda_vars': n_lambda_vars,
            'n_vars_total': n_vars,
            'n_constraints': n_constraints,
            'problem_size': problem_size,
            'cqm_time': cqm_time,
            'pulp_time': pulp_time,
            'pulp_status': pulp_results['status'],
            'pulp_objective': pulp_results.get('objective_value'),
            'pyomo_time': pyomo_time,
            'pyomo_status': pyomo_results.get('status', 'Error'),
            'pyomo_objective': pyomo_objective,
            'pulp_error_percent': pulp_error,
            'dwave_time': dwave_time,
            'qpu_time': qpu_time,
            'hybrid_time': hybrid_time,
            'dwave_feasible': dwave_feasible,
            'dwave_objective': dwave_objective,
            'power': POWER,
            'num_breakpoints': NUM_BREAKPOINTS
        }
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(results, output_file='scalability_benchmark_nln.png'):
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
    
    # Approximation error
    approx_errors = [r['pulp_error_mean'] for r in valid_results if r['pulp_error_mean'] is not None]
    approx_errors_std = [r['pulp_error_std'] for r in valid_results if r['pulp_error_mean'] is not None]
    approx_problem_sizes = [r['problem_size'] for r in valid_results if r['pulp_error_mean'] is not None]
    
    # Create figure with professional styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Solve times with error bars
    ax1.errorbar(problem_sizes, pulp_times, yerr=pulp_errors, marker='o', linestyle='-', 
                linewidth=2.5, markersize=8, capsize=5, capthick=2,
                label=f'PuLP (Piecewise Approx, {NUM_BREAKPOINTS} pts)', color='#2E86AB', alpha=0.9)
    
    if pyomo_times:
        ax1.errorbar(pyomo_problem_sizes, pyomo_times, yerr=pyomo_errors, marker='s', linestyle='-',
                    linewidth=2.5, markersize=8, capsize=5, capthick=2,
                    label='Pyomo (True Non-Linear)', color='#A23B72', alpha=0.9)
    
    ax1.set_xlabel('Problem Size (n = Farms × Foods)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Non-Linear Solver Performance (f(A) = A^{POWER})', 
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
    
    # Plot 2: Approximation error with error bars
    if approx_errors:
        ax2.errorbar(approx_problem_sizes, approx_errors, yerr=approx_errors_std,
                    marker='o', linestyle='-', linewidth=2.5, markersize=8,
                    capsize=5, capthick=2, color='#F18F01', alpha=0.9,
                    label='PuLP vs Pyomo')
        
        ax2.set_xlabel('Problem Size (n = Farms × Foods)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Approximation Error (%)', fontsize=14, fontweight='bold')
        ax2.set_title(f'Piecewise Approximation Quality ({NUM_BREAKPOINTS} breakpoints)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xscale('log')
        
        # Add horizontal lines for acceptable error thresholds
        ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='1% Error')
        ax2.axhline(y=5.0, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='5% Error')
        ax2.legend(loc='best', fontsize=11, framealpha=0.95)
        
        # Add average error annotation
        avg_error = np.mean(approx_errors)
        ax2.text(0.98, 0.98, f'Avg: {avg_error:.2f}%\n± {np.mean(approx_errors_std):.2f}%', 
                transform=ax2.transAxes, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    else:
        ax2.text(0.5, 0.5, 'No Pyomo Data\nfor Comparison', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    
    # Also create a summary table plot
    create_summary_table(valid_results, 'scalability_table.png')

def create_summary_table(results, output_file='scalability_table_nln.png'):
    """
    Create a beautiful summary table for NLN benchmark.
    """
    fig, ax = plt.subplots(figsize=(16, len(results) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Farms', 'Foods', 'n', 'Vars\n(Total)', 'Constraints', 
               'PuLP\nTime (s)', 'Pyomo\nTime (s)', 'Error\n(%)', 'Runs', 'Winner']
    
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
            r.get('n_vars_total', 'N/A'),
            r['n_constraints'],
            f"{r['pulp_time_mean']:.3f} ± {r['pulp_time_std']:.3f}",
            f"{r['pyomo_time_mean']:.3f} ± {r['pyomo_time_std']:.3f}" if r.get('pyomo_time_mean') else 'N/A',
            f"{r['pulp_error_mean']:.2f}" if r.get('pulp_error_mean') else 'N/A',
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
    
    plt.title('Scalability Benchmark Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Table saved to: {output_file}")

def main():
    """
    Run all benchmarks with multiple runs and calculate statistics.
    """
    print("="*80)
    print("NON-LINEAR SCALABILITY BENCHMARK")
    print("="*80)
    print(f"Configurations: {len(BENCHMARK_CONFIGS)} points")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Total benchmarks: {len(BENCHMARK_CONFIGS) * NUM_RUNS}")
    print(f"Power: {POWER}")
    print(f"Breakpoints: {NUM_BREAKPOINTS}")
    print("="*80)
    
    all_results = []
    aggregated_results = []
    
    for n_farms in BENCHMARK_CONFIGS:
        print(f"\n" + "="*80)
        print(f"TESTING CONFIGURATION: {n_farms} Farms")
        print("="*80)
        
        config_results = []
        
        # Run multiple times for this configuration
        for run_num in range(1, NUM_RUNS + 1):
            result = run_benchmark(n_farms, run_number=run_num, total_runs=NUM_RUNS)
            if result:
                config_results.append(result)
                all_results.append(result)
        
        # Calculate statistics for this configuration
        if config_results:
            pulp_times = [r['pulp_time'] for r in config_results if r['pulp_time'] is not None]
            pyomo_times = [r['pyomo_time'] for r in config_results if r['pyomo_time'] is not None]
            cqm_times = [r['cqm_time'] for r in config_results if r['cqm_time'] is not None]
            pulp_errors = [r['pulp_error_percent'] for r in config_results if r['pulp_error_percent'] is not None]
            
            aggregated = {
                'n_farms': n_farms,
                'n_foods': config_results[0]['n_foods'],
                'problem_size': config_results[0]['problem_size'],
                'n_vars_base': config_results[0]['n_vars_base'],
                'n_lambda_vars': config_results[0]['n_lambda_vars'],
                'n_vars_total': config_results[0]['n_vars_total'],
                'n_constraints': config_results[0]['n_constraints'],
                
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
                
                # Error stats
                'pulp_error_mean': float(np.mean(pulp_errors)) if pulp_errors else None,
                'pulp_error_std': float(np.std(pulp_errors)) if pulp_errors else None,
                
                'num_runs': len(config_results),
                'power': POWER,
                'num_breakpoints': NUM_BREAKPOINTS
            }
            
            aggregated_results.append(aggregated)
            
            # Print statistics
            print(f"\n  Statistics for {n_farms} farms ({len(config_results)} runs):")
            print(f"    CQM Creation: {aggregated['cqm_time_mean']:.3f}s ± {aggregated['cqm_time_std']:.3f}s")
            print(f"    PuLP:         {aggregated['pulp_time_mean']:.3f}s ± {aggregated['pulp_time_std']:.3f}s")
            if aggregated['pyomo_time_mean']:
                print(f"    Pyomo:        {aggregated['pyomo_time_mean']:.3f}s ± {aggregated['pyomo_time_std']:.3f}s")
            if aggregated['pulp_error_mean']:
                print(f"    Approx Error: {aggregated['pulp_error_mean']:.2f}% ± {aggregated['pulp_error_std']:.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all individual runs
    all_results_file = f'benchmark_nln_all_runs_{timestamp}.json'
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save aggregated statistics
    aggregated_file = f'benchmark_nln_aggregated_{timestamp}.json'
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"All runs saved to: {all_results_file}")
    print(f"Aggregated stats saved to: {aggregated_file}")
    
    # Create plots
    print(f"\nGenerating plots...")
    plot_results(aggregated_results, f'scalability_benchmark_nln_{timestamp}.png')
    
    print(f"\n🎉 All done! Ready for your presentation!")

if __name__ == "__main__":
    main()
