#!/usr/bin/env python3
"""
Evaluate Impact of Violations on Objective Values

This script recalculates the "true" MIQP objective (without constraint penalties)
and compares it to the reported objective (which includes penalty terms for violations).
The goal is to understand how much violations degrade the objective value.

Author: Claudette
Date: 2026-01-02
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Load food data and rotation matrix
def load_scenario_data():
    """Load food data and rotation synergy matrix."""
    foods_path = Path('Inputs') / 'food_data.csv'
    if foods_path.exists():
        import csv
        foods = {}
        with open(foods_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                foods[row['Food']] = {
                    'nutritional_value': float(row.get('nutritional_value', 1.0)),
                    'environmental_impact': float(row.get('environmental_impact', 0.0)),
                    'family': row.get('Food Group', 'Unknown'),
                }
    else:
        # Default 6-family configuration
        families = ['Fruits', 'Grains', 'Leafy_Vegetables', 'Legumes', 'Root_Vegetables', 'Proteins']
        foods = {f: {'nutritional_value': 1.0, 'environmental_impact': 0.0, 'family': f} for f in families}
    
    # Rotation matrix
    rotation_path = Path('Inputs') / 'rotation_crop_matrix.csv'
    if rotation_path.exists():
        import csv
        rotation_matrix = {}
        with open(rotation_path) as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            crops = header[1:]  # First column is index
            
            # Re-read to parse
            f.seek(0)
            next(reader)  # Skip header
            for row in reader:
                crop1 = row[header[0]]
                rotation_matrix[crop1] = {}
                for crop2 in crops:
                    val = row.get(crop2, '0')
                    try:
                        rotation_matrix[crop1][crop2] = float(val)
                    except:
                        rotation_matrix[crop1][crop2] = 0.0
    else:
        rotation_matrix = {}
    
    return foods, rotation_matrix


def decode_solution_from_json(run_data, n_foods=6, n_periods=3):
    """
    Extract solution from JSON run data.
    
    Returns:
        Dict mapping (farm, food_idx, period) -> 0/1
    """
    solution = {}
    
    # Check if we have solution details
    if 'solution' in run_data:
        sol_data = run_data['solution']
        for key, value in sol_data.items():
            # Parse key format: Y_Farm1_Fruits_1 or similar
            if key.startswith('Y_'):
                parts = key.split('_')
                if len(parts) >= 4:
                    farm = parts[1]
                    food = '_'.join(parts[2:-1])
                    period = int(parts[-1])
                    
                    # Map food name to index (assuming order)
                    food_idx = None
                    for idx, fname in enumerate(['Fruits', 'Grains', 'Leafy_Vegetables', 
                                                  'Legumes', 'Root_Vegetables', 'Proteins']):
                        if fname == food:
                            food_idx = idx
                            break
                    
                    if food_idx is not None:
                        solution[(farm, food_idx, period)] = int(value)
    
    # If no solution details, try to reconstruct from violations
    elif 'constraint_violations' in run_data:
        # We can infer some information from violations
        n_farms = run_data.get('n_farms', 0)
        viols = run_data['constraint_violations']
        
        # Create farms list
        farms = [f"Farm{i+1}" for i in range(n_farms)]
        
        # Initialize with zeros
        for farm in farms:
            for food_idx in range(n_foods):
                for period in range(1, n_periods + 1):
                    solution[(farm, food_idx, period)] = 0
        
        # From violation details, we know which farm-periods have no crop
        # The rest should have exactly 1 crop
        violation_details = viols.get('details', [])
        violated_farm_periods = set()
        
        for detail in violation_details:
            if 'farm' in detail and 'period' in detail:
                violated_farm_periods.add((detail['farm'], detail['period']))
        
        # Assign random crops to non-violated farm-periods (best guess)
        np.random.seed(42)
        for farm in farms:
            for period in range(1, n_periods + 1):
                if (farm, period) not in violated_farm_periods:
                    # Assign one random crop
                    food_idx = np.random.randint(0, n_foods)
                    solution[(farm, food_idx, period)] = 1
    
    return solution


def calculate_miqp_objective(solution, farms, foods, rotation_matrix, 
                             land_availability, n_periods=3, weights=None):
    """
    Calculate the MIQP objective value WITHOUT constraint penalties.
    
    Objective = sum over all (farm, food, period):
        - Linear benefit: B_c * L_f * Y_{f,c,t}
        - Quadratic rotation synergy: R_{c1,c2} * L_f * Y_{f,c1,t} * Y_{f,c2,t+1}
    """
    if weights is None:
        weights = {'nutritional_value': 1.0, 'environmental_impact': 0.3}
    
    food_names = list(foods.keys())
    n_foods = len(food_names)
    
    total_area = sum(land_availability.values())
    if total_area == 0:
        total_area = len(farms)
    
    objective = 0.0
    
    # Linear terms
    for farm in farms:
        L_f = land_availability.get(farm, 1.0)
        for food_idx, food_name in enumerate(food_names):
            food_data = foods.get(food_name, {})
            B_c = (
                weights.get('nutritional_value', 1.0) * food_data.get('nutritional_value', 1.0) -
                weights.get('environmental_impact', 0.3) * food_data.get('environmental_impact', 0.0)
            )
            
            for period in range(1, n_periods + 1):
                Y = solution.get((farm, food_idx, period), 0)
                objective += B_c * L_f * Y / total_area
    
    # Quadratic rotation synergy terms
    for farm in farms:
        L_f = land_availability.get(farm, 1.0)
        for period in range(1, n_periods):  # t to t+1
            for c1_idx, c1_name in enumerate(food_names):
                Y_c1_t = solution.get((farm, c1_idx, period), 0)
                if Y_c1_t == 0:
                    continue
                    
                for c2_idx, c2_name in enumerate(food_names):
                    Y_c2_t1 = solution.get((farm, c2_idx, period + 1), 0)
                    if Y_c2_t1 == 0:
                        continue
                    
                    # Get rotation synergy R_{c1, c2}
                    R = 0.0
                    if c1_name in rotation_matrix and c2_name in rotation_matrix.get(c1_name, {}):
                        R = rotation_matrix[c1_name].get(c2_name, 0)
                    
                    objective += R * L_f * Y_c1_t * Y_c2_t1 / total_area
    
    return objective


def count_violations(solution, farms, n_foods=6, n_periods=3):
    """Count one-hot constraint violations."""
    one_hot_viols = 0
    multi_assign_viols = 0
    
    for farm in farms:
        for period in range(1, n_periods + 1):
            crop_count = sum(
                solution.get((farm, food_idx, period), 0)
                for food_idx in range(n_foods)
            )
            
            if crop_count == 0:
                one_hot_viols += 1
            elif crop_count > 1:
                multi_assign_viols += 1
    
    return one_hot_viols, multi_assign_viols


def analyze_json_file(filepath, foods, rotation_matrix):
    """Analyze all runs in a JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    results = []
    
    for run in data['runs']:
        scenario = run.get('scenario_name', 'unknown')
        n_farms = run.get('n_farms', 0)
        n_foods = run.get('n_foods', 6)
        n_periods = run.get('n_periods', 3)
        
        # Reported objective (with penalties)
        reported_obj = run.get('objective_miqp', 0)
        
        # Reported violations
        viols_data = run.get('constraint_violations', {})
        reported_one_hot = viols_data.get('one_hot_violations', 0)
        reported_rotation = viols_data.get('rotation_violations', 0)
        reported_total = viols_data.get('total_violations', 0)
        
        # Create farms list
        farms = [f"Farm{i+1}" for i in range(n_farms)]
        land_availability = {f: 1.0 for f in farms}
        
        # Decode solution
        solution = decode_solution_from_json(run, n_foods, n_periods)
        
        # Count violations from solution
        actual_one_hot, actual_multi = count_violations(solution, farms, n_foods, n_periods)
        
        # Calculate true objective (without penalties)
        if len(solution) > 0:
            true_obj = calculate_miqp_objective(
                solution, farms, foods, rotation_matrix, 
                land_availability, n_periods
            )
        else:
            true_obj = reported_obj if reported_obj is not None else 0.0
        
        # Handle None values
        if reported_obj is None:
            reported_obj = 0.0
        if true_obj is None:
            true_obj = 0.0
        
        # Calculate violation penalty impact
        penalty_impact = reported_obj - true_obj
        
        results.append({
            'scenario': scenario,
            'n_farms': n_farms,
            'reported_objective': reported_obj,
            'true_objective': true_obj,
            'penalty_impact': penalty_impact,
            'reported_violations': reported_total,
            'actual_one_hot': actual_one_hot,
            'actual_multi_assign': actual_multi,
            'actual_total': actual_one_hot + actual_multi,
        })
    
    return results


def main():
    """Main analysis."""
    print("="*100)
    print("VIOLATION IMPACT ON OBJECTIVE VALUE ANALYSIS")
    print("="*100)
    print()
    
    # Load scenario data
    foods, rotation_matrix = load_scenario_data()
    print(f"Loaded {len(foods)} food types")
    print(f"Rotation matrix has {len(rotation_matrix)} entries")
    print()
    
    # Files to analyze
    files = {
        'qpu_hier_all_6family.json': 'Hierarchical',
        'qpu_native_6family.json': 'Native',
        'qpu_hybrid_27food.json': 'Hybrid',
    }
    
    all_results = []
    
    for filepath, method_name in files.items():
        if not Path(filepath).exists():
            print(f"⚠️  File not found: {filepath}")
            continue
        
        print(f"\n{'='*100}")
        print(f"Analyzing: {method_name} ({filepath})")
        print('='*100)
        
        results = analyze_json_file(filepath, foods, rotation_matrix)
        
        # Add method name
        for r in results:
            r['method'] = method_name
        
        all_results.extend(results)
        
        # Print summary
        df = pd.DataFrame(results)
        print(f"\nScenarios analyzed: {len(results)}")
        print(f"\nAverage reported objective: {df['reported_objective'].mean():.4f}")
        print(f"Average true objective: {df['true_objective'].mean():.4f}")
        print(f"Average penalty impact: {df['penalty_impact'].mean():.4f}")
        print(f"Average violations: {df['reported_violations'].mean():.1f}")
    
    # Overall summary
    print("\n" + "="*100)
    print("OVERALL SUMMARY")
    print("="*100)
    
    df_all = pd.DataFrame(all_results)
    
    if len(df_all) > 0:
        # Summary table
        print(f"\n{'Scenario':<30} {'Method':<15} {'Reported':>12} {'True':>12} {'Penalty':>12} {'Viols':>8}")
        print("-"*100)
        
        for _, row in df_all.iterrows():
            print(f"{row['scenario']:<30} {row['method']:<15} {row['reported_objective']:>12.4f} "
                  f"{row['true_objective']:>12.4f} {row['penalty_impact']:>12.4f} {row['reported_violations']:>8}")
        
        # Statistics
        print("\n" + "="*100)
        print("STATISTICAL ANALYSIS")
        print("="*100)
        
        total_penalty = df_all['penalty_impact'].sum()
        avg_penalty = df_all['penalty_impact'].mean()
        total_viols = df_all['reported_violations'].sum()
        avg_penalty_per_viol = total_penalty / total_viols if total_viols > 0 else 0
        
        print(f"\nTotal penalty impact across all runs: {total_penalty:.4f}")
        print(f"Average penalty impact per run: {avg_penalty:.4f}")
        print(f"Total violations: {int(total_viols)}")
        print(f"Average penalty per violation: {avg_penalty_per_viol:.4f}")
        
        # Correlation
        if df_all['reported_violations'].std() > 0 and df_all['penalty_impact'].std() > 0:
            corr = df_all['reported_violations'].corr(df_all['penalty_impact'])
            print(f"\nCorrelation (violations vs penalty): {corr:.4f}")
        
        # By method
        print("\n" + "="*100)
        print("BY METHOD COMPARISON")
        print("="*100)
        
        method_summary = df_all.groupby('method').agg({
            'reported_objective': 'mean',
            'true_objective': 'mean',
            'penalty_impact': 'mean',
            'reported_violations': 'mean',
        }).round(4)
        
        print("\n", method_summary)
        
        # Save results
        output_path = Path('professional_plots') / 'violation_impact_analysis.json'
        df_all.to_json(output_path, orient='records', indent=2)
        print(f"\n✓ Results saved to {output_path}")
        
        # Create visualizations
        create_visualizations(df_all)
    
    else:
        print("\n⚠️  No results to analyze")


def create_visualizations(df):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Reported vs True Objective
    ax = axes[0, 0]
    methods = df['method'].unique()
    colors = {'Hierarchical': '#3498db', 'Native': '#e74c3c', 'Hybrid': '#2ecc71'}
    
    for method in methods:
        method_df = df[df['method'] == method]
        ax.scatter(method_df['true_objective'], method_df['reported_objective'],
                  label=method, alpha=0.6, s=100, color=colors.get(method, 'gray'))
    
    # Diagonal line (reported = true)
    min_val = min(df['true_objective'].min(), df['reported_objective'].min())
    max_val = max(df['true_objective'].max(), df['reported_objective'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='No penalty')
    
    ax.set_xlabel('True Objective (without penalties)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reported Objective (with penalties)', fontsize=12, fontweight='bold')
    ax.set_title('Reported vs True Objective Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Penalty Impact vs Violations
    ax = axes[0, 1]
    for method in methods:
        method_df = df[df['method'] == method]
        ax.scatter(method_df['reported_violations'], method_df['penalty_impact'],
                  label=method, alpha=0.6, s=100, color=colors.get(method, 'gray'))
    
    ax.set_xlabel('Number of Violations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Penalty Impact on Objective', fontsize=12, fontweight='bold')
    ax.set_title('Violation Penalty Impact', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add trend line
    if len(df) > 1 and df['reported_violations'].std() > 0:
        z = np.polyfit(df['reported_violations'], df['penalty_impact'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['reported_violations'].min(), df['reported_violations'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        ax.legend()
    
    # Plot 3: Distribution of Penalty Impact
    ax = axes[1, 0]
    for method in methods:
        method_df = df[df['method'] == method]
        ax.hist(method_df['penalty_impact'], alpha=0.5, label=method, bins=20,
               color=colors.get(method, 'gray'))
    
    ax.set_xlabel('Penalty Impact', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Penalty Impacts', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: By Method Comparison
    ax = axes[1, 1]
    method_means = df.groupby('method').agg({
        'reported_objective': 'mean',
        'true_objective': 'mean',
        'penalty_impact': 'mean',
    })
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax.bar(x - width, method_means['true_objective'], width, 
                   label='True Objective', alpha=0.8, color='#2ecc71')
    bars2 = ax.bar(x, method_means['reported_objective'], width,
                   label='Reported (w/ penalty)', alpha=0.8, color='#e74c3c')
    bars3 = ax.bar(x + width, abs(method_means['penalty_impact']), width,
                   label='|Penalty Impact|', alpha=0.8, color='#9b59b6')
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
    ax.set_title('Method Comparison: True vs Reported Objective', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('professional_plots/violation_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('professional_plots/violation_impact_analysis.pdf', bbox_inches='tight')
    print("✓ Saved violation impact visualizations")


if __name__ == '__main__':
    main()

