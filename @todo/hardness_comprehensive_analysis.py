#!/usr/bin/env python3
"""
Comprehensive Hardness Analysis with Constant Area Sampling

Goal: Systematically test the hardness region with CONSISTENT total area
      to isolate the farms/food ratio as the primary hardness factor.

Key Innovation:
- Sample N farms from large pool to achieve constant total area (~100 ha)
- This normalizes objective coefficients while preserving farms/food ratio
- Test 19 farm counts from 3 to 100 farms
- All with 6 food families, 3 periods

Expected Results:
- TRIVIAL: 3 farms (0.5 farms/food) - too small to be hard
- HARDNESS ZONE: 5-25 farms (0.83-4.17 farms/food) - Gurobi struggles
- EASY: 50+ farms (8.33+ farms/food) - Gurobi solves quickly

Author: OQI-UC002-DWave
Date: December 14, 2025
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Try to import seaborn, but don't fail if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib only")

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.scenarios import load_food_data
import gurobipy as gp
from gurobipy import GRB

# ============================================================================
# CONFIGURATION
# ============================================================================

N_PERIODS = 3
N_FAMILIES = 6  # Fixed: using 6-family scenarios
TARGET_AREA_PER_FARM = 1.0  # Target area per farm in hectares
AREA_TOLERANCE = 0.05  # ±5% tolerance

# Test farm counts (19 datapoints)
FARM_COUNTS = [3, 5, 7, 10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]

# Gurobi parameters
GUROBI_TIMEOUT = 300  # 5 minutes
GUROBI_GAP = 0.01  # 1% MIP gap

# Output
OUTPUT_DIR = Path(__file__).parent / 'hardness_analysis_results'
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42  # For reproducibility

print("="*80)
print("COMPREHENSIVE HARDNESS ANALYSIS: Constant Area Per Farm Sampling")
print("="*80)
print(f"\nConfiguration:")
print(f"  - Target area per farm: {TARGET_AREA_PER_FARM} ha (±{AREA_TOLERANCE*100}%)")
print(f"  - Food families: {N_FAMILIES}")
print(f"  - Time periods: {N_PERIODS}")
print(f"  - Farm counts: {len(FARM_COUNTS)} test points")
print(f"  - Gurobi timeout: {GUROBI_TIMEOUT}s")
print(f"  - Gurobi gap: {GUROBI_GAP*100}%")
print()

# ============================================================================
# CONSTANT-AREA FARM SAMPLING
# ============================================================================

def sample_farms_constant_area(
    all_farms: List[str],
    land_availability: Dict[str, float],
    n_farms: int,
    target_area_per_farm: float = TARGET_AREA_PER_FARM,
    tolerance: float = AREA_TOLERANCE,
    seed: int = SEED,
    max_attempts: int = 1000
) -> Tuple[List[str], Dict[str, float], float]:
    """
    Sample N farms from pool to achieve target area per farm.
    
    Strategy:
    1. Randomly sample N farms
    2. Calculate total area
    3. If within tolerance, accept
    4. Otherwise, try again (up to max_attempts)
    5. If no solution found, scale areas to match target
    
    Returns:
        selected_farms: List of farm names
        selected_land: Dict of farm -> area
        actual_area: Actual total area achieved
    """
    rng = np.random.RandomState(seed)
    
    target_total_area = n_farms * target_area_per_farm
    min_area = target_total_area * (1 - tolerance)
    max_area = target_total_area * (1 + tolerance)
    
    # Try random sampling first
    for attempt in range(max_attempts):
        # Sample N farms
        sampled_farms = rng.choice(all_farms, size=min(n_farms, len(all_farms)), replace=False)
        sampled_land = {f: land_availability[f] for f in sampled_farms}
        total = sum(sampled_land.values())
        
        # Check if within tolerance
        if min_area <= total <= max_area:
            return list(sampled_farms), sampled_land, total
    
    # If no solution found, scale to match target exactly
    sampled_farms = rng.choice(all_farms, size=min(n_farms, len(all_farms)), replace=False)
    sampled_land = {f: land_availability[f] for f in sampled_farms}
    total = sum(sampled_land.values())
    
    # Scale to match target
    scale_factor = target_total_area / total
    scaled_land = {f: area * scale_factor for f, area in sampled_land.items()}
    actual_total = sum(scaled_land.values())
    
    print(f"    Note: Scaled areas by {scale_factor:.3f} to achieve target area")
    
    return list(sampled_farms), scaled_land, actual_total

# ============================================================================
# PROBLEM LOADING
# ============================================================================

def load_rotation_problem(n_farms: int) -> Dict:
    """
    Load rotation problem with N farms and constant total area.
    """
    print(f"\n{'='*80}")
    print(f"Loading problem: {n_farms} farms")
    print(f"{'='*80}")
    
    # Use rotation_large_200 which has 200 farms available
    scenario = 'rotation_large_200'
    farms, foods, food_groups, config = load_food_data(scenario)
    
    params = config.get('parameters', {})
    weights = params.get('weights', {})
    land_full = params.get('land_availability', {})
    
    # Sample farms to achieve constant area per farm
    all_farms = list(land_full.keys())
    selected_farms, selected_land, actual_area = sample_farms_constant_area(
        all_farms, land_full, n_farms, TARGET_AREA_PER_FARM, AREA_TOLERANCE, SEED + n_farms
    )
    
    # Calculate food benefits
    food_names = list(foods.keys())
    food_benefits = {}
    for food in food_names:
        benefit = (
            weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
            weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
            weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
            weights.get('affordability', 0) * foods[food].get('affordability', 0) +
            weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
        )
        food_benefits[food] = benefit
    
    # Calculate metrics
    areas = list(selected_land.values())
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    cv_area = std_area / mean_area if mean_area > 0 else 0
    
    n_vars = n_farms * N_FAMILIES * N_PERIODS
    farms_per_food = n_farms / N_FAMILIES
    area_per_var = actual_area / n_vars
    
    print(f"  Selected farms: {n_farms}")
    print(f"  Total area: {actual_area:.2f} ha (target: {n_farms * TARGET_AREA_PER_FARM:.2f} ha)")
    print(f"  Mean area/farm: {mean_area:.2f} ha (target: {TARGET_AREA_PER_FARM} ha/farm)")
    print(f"  CV (variability): {cv_area:.3f}")
    print(f"  Variables: {n_vars}")
    print(f"  Farms/Food ratio: {farms_per_food:.2f}")
    print(f"  Area/Variable: {area_per_var:.3f}")
    
    return {
        'n_farms': n_farms,
        'farm_names': selected_farms,
        'food_names': food_names,
        'food_benefits': food_benefits,
        'land_availability': selected_land,
        'total_area': actual_area,
        'mean_area': mean_area,
        'cv_area': cv_area,
        'n_vars': n_vars,
        'farms_per_food': farms_per_food,
        'area_per_var': area_per_var,
        'config': config,
        'scenario': scenario,
    }

# ============================================================================
# GUROBI SOLVER
# ============================================================================

def build_gurobi_model(data: Dict) -> gp.Model:
    """
    Build Gurobi model for crop rotation optimization.
    
    BQM formulation with quadratic rotation and spatial terms.
    """
    farm_names = data['farm_names']
    food_names = data['food_names']
    food_benefits = data['food_benefits']
    land_availability = data['land_availability']
    total_area = data['total_area']
    config = data['config']
    params = config.get('parameters', {})
    
    # Build rotation matrix (frustration-based)
    rotation_gamma = params.get('rotation_gamma', 0.35)
    frustration_ratio = params.get('frustration_ratio', 0.88)
    negative_strength = params.get('negative_synergy_strength', -1.5)
    k_neighbors = params.get('spatial_k_neighbors', 4)
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    n_periods = N_PERIODS
    
    # Construct rotation matrix R
    np.random.seed(42)
    R = np.zeros((n_foods, n_foods))
    for i in range(n_foods):
        for j in range(n_foods):
            if i == j:
                R[i, j] = negative_strength * 1.5
            elif np.random.random() < frustration_ratio:
                R[i, j] = np.random.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                R[i, j] = np.random.uniform(0.02, 0.20)
    
    # Construct spatial neighbor graph
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {farm: (i // side, i % side) for i, farm in enumerate(farm_names)}
    
    neighbor_edges = []
    for f1 in farm_names:
        distances = [(np.sqrt((positions[f1][0] - positions[f2][0])**2 + 
                             (positions[f1][1] - positions[f2][1])**2), f2)
                    for f2 in farm_names if f1 != f2]
        distances.sort()
        for _, f2 in distances[:k_neighbors]:
            if (f2, f1) not in neighbor_edges:
                neighbor_edges.append((f1, f2))
    
    # Create model
    model = gp.Model("CropRotation")
    model.setParam('OutputFlag', 0)  # Suppress output
    model.setParam('TimeLimit', GUROBI_TIMEOUT)
    model.setParam('MIPGap', GUROBI_GAP)
    model.setParam('Threads', 8)
    
    # Decision variables: x[farm, food, period] ∈ {0, 1} BINARY
    x = {}
    for f in farm_names:
        for c in food_names:
            for t in range(n_periods):
                x[f, c, t] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, 
                                          name=f'x_{f}_{c}_{t}')
    
    model.update()
    
    # Objective: Maximize benefit
    obj = gp.QuadExpr()
    
    # Linear terms: food benefits weighted by area (normalized)
    for f in farm_names:
        area = land_availability[f]
        for c_idx, c in enumerate(food_names):
            benefit = food_benefits[c]
            for t in range(n_periods):
                obj += (benefit * area * x[f, c, t]) / total_area
    
    # Rotation benefits (temporal quadratics)
    for f in farm_names:
        area = land_availability[f]
        for t in range(n_periods - 1):
            for c1_idx, c1 in enumerate(food_names):
                for c2_idx, c2 in enumerate(food_names):
                    synergy = R[c1_idx, c2_idx]
                    if abs(synergy) > 1e-6:
                        obj += (rotation_gamma * synergy * area * 
                               x[f, c1, t] * x[f, c2, t+1]) / total_area
    
    # Spatial synergy (spatial quadratics)
    spatial_gamma = rotation_gamma * 0.5
    for (f1, f2) in neighbor_edges:
        for t in range(n_periods):
            for c1_idx, c1 in enumerate(food_names):
                for c2_idx, c2 in enumerate(food_names):
                    spatial_synergy = R[c1_idx, c2_idx] * 0.3
                    if abs(spatial_synergy) > 1e-6:
                        obj += (spatial_gamma * spatial_synergy * 
                               x[f1, c1, t] * x[f2, c2, t]) / total_area
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    
    # 1. One-hot constraint: Each farm-period plants at most one crop family
    for f in farm_names:
        for t in range(n_periods):
            model.addConstr(
                gp.quicksum(x[f, c, t] for c in food_names) <= 1,
                name=f'onehot_{f}_{t}'
            )
    
    # 2. Land capacity: Total area planted per farm across all periods ≤ available land
    for f in farm_names:
        area = land_availability[f]
        model.addConstr(
            gp.quicksum(x[f, c, t] for c in food_names for t in range(n_periods)) <= area / (area / n_periods),
            name=f'capacity_{f}'
        )
    
    # 3. Minimum planting constraint (optional, if in config)
    min_planting = params.get('minimum_planting_area', {})
    if min_planting:
        for c in food_names:
            min_area = min_planting.get(c, 0)
            if min_area > 0:
                model.addConstr(
                    gp.quicksum(land_availability[f] * x[f, c, t] 
                               for f in farm_names for t in range(n_periods)) >= min_area,
                    name=f'min_planting_{c}'
                )
    
    # 4. Diversity constraint: Each food family planted at least once
    for c in food_names:
        model.addConstr(
            gp.quicksum(x[f, c, t] for f in farm_names for t in range(n_periods)) >= 0.1,
            name=f'diversity_{c}'
        )
    
    model.update()
    
    return model

def solve_gurobi(data: Dict) -> Dict:
    """
    Solve problem with Gurobi and collect metrics.
    """
    print(f"\n  Building Gurobi model...")
    build_start = time.time()
    model = build_gurobi_model(data)
    build_time = time.time() - build_start
    
    n_vars = model.NumVars
    n_constraints = model.NumConstrs
    n_quadratic = model.NumQNZs  # Quadratic non-zeros
    
    print(f"  Model built: {n_vars} vars, {n_constraints} constrs, {n_quadratic} quadratic non-zeros")
    print(f"  Build time: {build_time:.2f}s")
    print(f"\n  Solving with Gurobi (timeout: {GUROBI_TIMEOUT}s)...")
    
    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    
    # Collect results
    status = model.Status
    status_name = {
        GRB.OPTIMAL: 'OPTIMAL',
        GRB.TIME_LIMIT: 'TIMEOUT',
        GRB.INFEASIBLE: 'INFEASIBLE',
        GRB.INF_OR_UNBD: 'INF_OR_UNBD',
        GRB.UNBOUNDED: 'UNBOUNDED',
    }.get(status, f'UNKNOWN_{status}')
    
    obj_value = model.ObjVal if status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None
    obj_bound = model.ObjBound if status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None
    
    # Try to get MIP gap (may not be available for continuous problems)
    try:
        gap = model.MIPGap if status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None
    except AttributeError:
        # Calculate gap manually if objective and bound are available
        if obj_value is not None and obj_bound is not None and abs(obj_value) > 1e-6:
            gap = abs(obj_value - obj_bound) / abs(obj_value)
        else:
            gap = None
    
    # Classify by solve time
    if status == GRB.TIME_LIMIT:
        time_category = 'TIMEOUT'
    elif solve_time > 100:
        time_category = 'SLOW'
    elif solve_time > 10:
        time_category = 'MEDIUM'
    else:
        time_category = 'FAST'
    
    print(f"  Status: {status_name}")
    print(f"  Solve time: {solve_time:.2f}s ({time_category})")
    if obj_value is not None:
        print(f"  Objective: {obj_value:.2f}")
    if gap is not None:
        print(f"  MIP gap: {gap*100:.2f}%")
    
    return {
        'status': status_name,
        'status_code': status,
        'time_category': time_category,
        'solve_time': solve_time,
        'build_time': build_time,
        'obj_value': obj_value,
        'obj_bound': obj_bound,
        'gap': gap,
        'n_vars': n_vars,
        'n_constraints': n_constraints,
        'n_quadratic': n_quadratic,
    }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run comprehensive hardness analysis."""
    results = []
    
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {len(FARM_COUNTS)} test points")
    print(f"{'='*80}")
    
    for i, n_farms in enumerate(FARM_COUNTS, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(FARM_COUNTS)}: {n_farms} farms")
        print(f"{'='*80}")
        
        try:
            # Load problem
            data = load_rotation_problem(n_farms)
            
            # Solve with Gurobi
            solve_results = solve_gurobi(data)
            
            # Combine results
            result = {
                **data,
                **solve_results,
            }
            results.append(result)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'n_farms': n_farms,
                'status': 'ERROR',
                'error': str(e)
            })
    
    return results

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================

def analyze_results(results: List[Dict]):
    """Analyze and visualize results."""
    print(f"\n{'='*80}")
    print("ANALYSIS & VISUALIZATION")
    print(f"{'='*80}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Filter out errors
    df_valid = df[df['status'] != 'ERROR'].copy()
    
    # Save raw results
    results_file = OUTPUT_DIR / 'hardness_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results: {results_file}")
    
    csv_file = OUTPUT_DIR / 'hardness_analysis_results.csv'
    df_valid.to_csv(csv_file, index=False)
    print(f"Saved CSV results: {csv_file}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"\n{'Farms':<7} {'Vars':<7} {'F/Food':<8} {'Area':<8} {'Time(s)':<10} {'Status':<10} {'Category'}")
    print("-"*80)
    for _, row in df_valid.iterrows():
        print(f"{int(row['n_farms']):<7} {int(row['n_vars']):<7} {row['farms_per_food']:<8.2f} "
              f"{row['total_area']:<8.1f} {row['solve_time']:<10.2f} {row['status']:<10} {row['time_category']}")
    
    # Statistical analysis
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    
    # Group by time category
    for category in ['TIMEOUT', 'SLOW', 'MEDIUM', 'FAST']:
        subset = df_valid[df_valid['time_category'] == category]
        if len(subset) > 0:
            print(f"\n{category} ({len(subset)} instances):")
            print(f"  Farms/Food ratio: {subset['farms_per_food'].min():.2f} - {subset['farms_per_food'].max():.2f} "
                  f"(mean={subset['farms_per_food'].mean():.2f})")
            print(f"  Number of farms: {subset['n_farms'].min():.0f} - {subset['n_farms'].max():.0f} "
                  f"(mean={subset['n_farms'].mean():.0f})")
            print(f"  Solve time: {subset['solve_time'].min():.2f}s - {subset['solve_time'].max():.2f}s "
                  f"(mean={subset['solve_time'].mean():.2f}s)")
    
    # Identify hardness zone
    timeout_df = df_valid[df_valid['time_category'] == 'TIMEOUT']
    slow_df = df_valid[df_valid['time_category'] == 'SLOW']
    hard_df = pd.concat([timeout_df, slow_df])
    
    if len(hard_df) > 0:
        print(f"\n{'='*80}")
        print("HARDNESS ZONE IDENTIFIED")
        print(f"{'='*80}")
        print(f"  Farms/Food ratio: {hard_df['farms_per_food'].min():.2f} - {hard_df['farms_per_food'].max():.2f}")
        print(f"  Number of farms: {hard_df['n_farms'].min():.0f} - {hard_df['n_farms'].max():.0f}")
        print(f"  Variables: {hard_df['n_vars'].min():.0f} - {hard_df['n_vars'].max():.0f}")
    
    # Create visualizations
    create_visualizations(df_valid)

def create_visualizations(df: pd.DataFrame):
    """Create comprehensive visualizations."""
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Set style
    if HAS_SEABORN:
        sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Color mapping
    color_map = {
        'TIMEOUT': 'red',
        'SLOW': 'orange',
        'MEDIUM': 'yellow',
        'FAST': 'green'
    }
    
    # ========================================================================
    # Plot 1: Solve Time vs Farms/Food Ratio
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for category in ['TIMEOUT', 'SLOW', 'MEDIUM', 'FAST']:
        subset = df[df['time_category'] == category]
        if len(subset) > 0:
            # For timeout, use timeout value
            y_vals = subset['solve_time'].copy()
            if category == 'TIMEOUT':
                y_vals = [GUROBI_TIMEOUT] * len(y_vals)
            
            ax.scatter(subset['farms_per_food'], y_vals, 
                      c=color_map[category], s=100, alpha=0.7, label=category, edgecolors='black')
    
    # Mark hardness zone
    ax.axvspan(0.8, 4.5, alpha=0.2, color='red', label='Hardness Zone (0.8-4.5)')
    ax.axvline(4.2, color='red', linestyle='--', alpha=0.5, label='Hard/Easy Threshold (4.2)')
    
    ax.set_xlabel('Farms per Food Family', fontsize=14)
    ax.set_ylabel('Solve Time (seconds)', fontsize=14)
    ax.set_yscale('log')
    ax.set_title('Gurobi Solve Time vs Farms/Food Ratio\n(Constant Area Per Farm: 1 ha/farm)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plot_file = OUTPUT_DIR / 'plot_solve_time_vs_ratio.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    plt.close()
    
    # ========================================================================
    # Plot 2: Solve Time vs Number of Farms
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for category in ['TIMEOUT', 'SLOW', 'MEDIUM', 'FAST']:
        subset = df[df['time_category'] == category]
        if len(subset) > 0:
            y_vals = subset['solve_time'].copy()
            if category == 'TIMEOUT':
                y_vals = [GUROBI_TIMEOUT] * len(y_vals)
            
            ax.scatter(subset['n_farms'], y_vals, 
                      c=color_map[category], s=100, alpha=0.7, label=category, edgecolors='black')
    
    ax.set_xlabel('Number of Farms', fontsize=14)
    ax.set_ylabel('Solve Time (seconds)', fontsize=14)
    ax.set_yscale('log')
    ax.set_title('Gurobi Solve Time vs Number of Farms\n(6 Families, 3 Periods, 1 ha/farm)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plot_file = OUTPUT_DIR / 'plot_solve_time_vs_farms.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    plt.close()
    
    # ========================================================================
    # Plot 3: MIP Gap vs Farms/Food Ratio (for non-optimal solutions)
    # ========================================================================
    df_gap = df[df['gap'].notna()].copy()
    if len(df_gap) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for category in ['TIMEOUT', 'SLOW', 'MEDIUM', 'FAST']:
            subset = df_gap[df_gap['time_category'] == category]
            if len(subset) > 0:
                ax.scatter(subset['farms_per_food'], subset['gap'] * 100, 
                          c=color_map[category], s=100, alpha=0.7, label=category, edgecolors='black')
        
        ax.axvspan(0.8, 4.5, alpha=0.2, color='red', label='Hardness Zone (0.8-4.5)')
        ax.set_xlabel('Farms per Food Family', fontsize=14)
        ax.set_ylabel('MIP Gap (%)', fontsize=14)
        ax.set_title('MIP Gap vs Farms/Food Ratio\n(Higher gap = harder problem)', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plot_file = OUTPUT_DIR / 'plot_gap_vs_ratio.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {plot_file}")
        plt.close()
    
    # ========================================================================
    # Plot 4: Heatmap - Farms vs Time Category
    # ========================================================================
    # Create pivot table
    pivot = df.pivot_table(values='solve_time', index='time_category', 
                          columns='n_farms', aggfunc='count', fill_value=0)
    
    # Reorder rows
    row_order = ['FAST', 'MEDIUM', 'SLOW', 'TIMEOUT']
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    if HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt='g', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Count'})
    else:
        # Matplotlib heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, int(pivot.values[i, j]),
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax, label='Count')
    
    ax.set_xlabel('Number of Farms', fontsize=14)
    ax.set_ylabel('Time Category', fontsize=14)
    ax.set_title('Hardness Distribution Heatmap\n(Count of instances by farm size and difficulty)', fontsize=16)
    
    plot_file = OUTPUT_DIR / 'plot_heatmap_hardness.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    plt.close()
    
    # ========================================================================
    # Plot 5: Combined analysis - Ratio, Farms, Variables
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 5a: Solve time vs ratio
    ax = axes[0, 0]
    for category in ['TIMEOUT', 'SLOW', 'MEDIUM', 'FAST']:
        subset = df[df['time_category'] == category]
        if len(subset) > 0:
            y_vals = subset['solve_time'].copy()
            if category == 'TIMEOUT':
                y_vals = [GUROBI_TIMEOUT] * len(y_vals)
            ax.scatter(subset['farms_per_food'], y_vals, c=color_map[category], 
                      s=80, alpha=0.7, label=category)
    ax.axvspan(0.8, 4.5, alpha=0.2, color='red')
    ax.set_xlabel('Farms per Food')
    ax.set_ylabel('Solve Time (s)')
    ax.set_yscale('log')
    ax.set_title('Solve Time vs Farms/Food Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5b: Solve time vs farms
    ax = axes[0, 1]
    for category in ['TIMEOUT', 'SLOW', 'MEDIUM', 'FAST']:
        subset = df[df['time_category'] == category]
        if len(subset) > 0:
            y_vals = subset['solve_time'].copy()
            if category == 'TIMEOUT':
                y_vals = [GUROBI_TIMEOUT] * len(y_vals)
            ax.scatter(subset['n_farms'], y_vals, c=color_map[category], 
                      s=80, alpha=0.7, label=category)
    ax.set_xlabel('Number of Farms')
    ax.set_ylabel('Solve Time (s)')
    ax.set_yscale('log')
    ax.set_title('Solve Time vs Number of Farms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5c: Solve time vs variables
    ax = axes[1, 0]
    for category in ['TIMEOUT', 'SLOW', 'MEDIUM', 'FAST']:
        subset = df[df['time_category'] == category]
        if len(subset) > 0:
            y_vals = subset['solve_time'].copy()
            if category == 'TIMEOUT':
                y_vals = [GUROBI_TIMEOUT] * len(y_vals)
            ax.scatter(subset['n_vars'], y_vals, c=color_map[category], 
                      s=80, alpha=0.7, label=category)
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('Solve Time (s)')
    ax.set_yscale('log')
    ax.set_title('Solve Time vs Problem Size (Variables)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5d: Farms vs Ratio colored by time
    ax = axes[1, 1]
    scatter_data = []
    for category in ['TIMEOUT', 'SLOW', 'MEDIUM', 'FAST']:
        subset = df[df['time_category'] == category]
        if len(subset) > 0:
            for _, row in subset.iterrows():
                scatter_data.append({
                    'n_farms': row['n_farms'],
                    'farms_per_food': row['farms_per_food'],
                    'category': category,
                    'solve_time': row['solve_time'] if category != 'TIMEOUT' else GUROBI_TIMEOUT
                })
    
    if scatter_data:
        scatter_df = pd.DataFrame(scatter_data)
        for category in ['TIMEOUT', 'SLOW', 'MEDIUM', 'FAST']:
            subset = scatter_df[scatter_df['category'] == category]
            if len(subset) > 0:
                ax.scatter(subset['n_farms'], subset['farms_per_food'], 
                          c=color_map[category], s=100, alpha=0.7, label=category)
    
    ax.axhspan(0.8, 4.5, alpha=0.2, color='red', label='Hardness Zone')
    ax.set_xlabel('Number of Farms')
    ax.set_ylabel('Farms per Food Ratio')
    ax.set_title('Farms vs Farms/Food Ratio\n(Color = Hardness)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = OUTPUT_DIR / 'plot_combined_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run experiment
    results = run_experiment()
    
    # Analyze and visualize
    analyze_results(results)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE HARDNESS ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nKey findings:")
    print(f"  - Tested {len(FARM_COUNTS)} farm counts with constant area per farm ({TARGET_AREA_PER_FARM} ha/farm)")
    print(f"  - Identified hardness region based on farms/food ratio")
    print(f"  - Generated comprehensive visualizations")
    print(f"  - Ready for QPU testing in identified hardness zone")
    print()
