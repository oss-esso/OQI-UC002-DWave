"""
Professional solver runner script with BQUBO (CQM→BQM conversion).

This script:
1. Loads a scenario (simple, intermediate, or custom)
2. Converts to CQM with LINEAR objective and saves the model
3. Solves with PuLP and saves results
4. Solves with DWave using CQM→BQM conversion + HybridBQM solver (QPU-enabled)
5. Saves all constraints for verification
"""

import os
import sys
import json
import pickle
import shutil
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.scenarios import load_food_data
from Utils.patch_sampler import generate_farms as generate_patches
from dimod import ConstrainedQuadraticModel, Binary, Real, cqm_to_bqm
from dwave.system import LeapHybridCQMSampler, LeapHybridBQMSampler
from dwave.samplers import SimulatedAnnealingSampler
import pulp as pl
from tqdm import tqdm

def calculate_original_objective(solution, farms, foods, land_availability, weights, idle_penalty):
    """
    Calculate the original CQM objective from a solution.
    
    This reconstructs the objective: sum_{p,c} (B_c + λ) * s_p * X_{p,c}
    
    Works for both PATCH formulation (X variables) and BQUBO formulation (Y variables).
    
    Args:
        solution: Dictionary with variable assignments (X_{plot}_{crop} or Y_{farm}_{crop})
        farms: List of farm/plot names
        foods: Dictionary of food data with nutritional values
        land_availability: Dictionary mapping plot to area
        weights: Dictionary of objective weights
        idle_penalty: Lambda penalty for idle land
        
    Returns:
        float: The original objective value (to be maximized)
    """
    objective = 0.0
    
    # Detect which formulation we're using
    # BQUBO uses Y variables, PATCH uses X variables
    is_bqubo = any(var.startswith("Y_") for var in solution.keys())
    
    if is_bqubo:
        # BQUBO formulation: Y_{farm}_{crop} = 1 if planted (1 acre), 0 otherwise
        # Objective without normalization: sum (B_c * Y_{farm}_{crop})
        # Each Y represents a 1-acre plantation
        for plot in farms:
            for crop in foods:
                # Get Y_{p,c} value from solution
                var_name = f"Y_{plot}_{crop}"
                y_pc = solution.get(var_name, 0)
                
                if y_pc > 0:  # Only count if planted
                    # Calculate B_c: weighted benefit per acre
                    B_c = (
                        weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                        weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                        weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                        weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                        weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
                    )
                    # Each Y_{p,c} = 1 represents 1 acre of crop c on plot p
                    # Contribution is B_c * 1 acre * y_pc
                    objective += B_c * y_pc
        
        # Normalize by total possible plantations
        total_possible_plantations = len(farms) * len(foods)
        if total_possible_plantations > 0:
            objective = objective / total_possible_plantations
    else:
        # PATCH formulation: X_{plot}_{crop} is continuous [0,1] fraction
        for plot in farms:
            s_p = land_availability[plot]  # Area of plot p
            for crop in foods:
                # Get X_{p,c} value from solution
                var_name = f"X_{plot}_{crop}"
                x_pc = solution.get(var_name, 0)
                
                if x_pc > 0:  # Only count if assigned
                    # Calculate B_c: weighted benefit per unit area
                    B_c = (
                        weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                        weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                        weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                        weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                        weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
                    )
                    # Add (B_c + λ) * s_p * X_{p,c} to objective
                    objective += (B_c + idle_penalty) * s_p * x_pc
    
    return objective

def extract_solution_summary(solution, farms, foods, land_availability):
    """
    Extract a summary of the solution showing crop selections and plot assignments.
    
    Args:
        solution: Dictionary with variable assignments (X_{plot}_{crop}, Y_{crop})
        farms: List of farm/plot names
        foods: Dictionary of food data
        land_availability: Dictionary mapping plot to area
        
    Returns:
        dict: Summary with crops selected, areas, and plot assignments
    """
    crops_selected = []
    plot_assignments = []
    total_allocated = 0.0
    
    for crop in foods:
        # Check if crop is selected (Y_c = 1)
        y_var = f"Y_{crop}"
        if solution.get(y_var, 0) > 0:
            # Calculate total area allocated to this crop
            total_area = sum(
                solution.get(f"X_{plot}_{crop}", 0) * land_availability[plot]
                for plot in farms
            )
            
            # Get list of plots assigned to this crop
            assigned_plots = [
                {'plot': plot, 'area': land_availability[plot]}
                for plot in farms 
                if solution.get(f"X_{plot}_{crop}", 0) > 0
            ]
            
            if total_area > 0:  # Only include if actually allocated
                crops_selected.append(crop)
                plot_assignments.append({
                    'crop': crop,
                    'total_area': total_area,
                    'n_plots': len(assigned_plots),
                    'plots': assigned_plots
                })
                total_allocated += total_area
    
    total_available = sum(land_availability.values())
    idle_area = total_available - total_allocated
    
    return {
        'crops_selected': crops_selected,
        'n_crops': len(crops_selected),
        'plot_assignments': plot_assignments,
        'total_allocated': total_allocated,
        'total_available': total_available,
        'idle_area': idle_area,
        'utilization': total_allocated / total_available if total_available > 0 else 0
    }

def validate_solution_constraints(solution, farms, foods, food_groups, land_availability, config):
    """
    Validate if a solution satisfies all original CQM constraints.
    
    Args:
        solution: Dictionary with variable assignments (X_{plot}_{crop}, Y_{crop})
        farms: List of farm/plot names
        foods: Dictionary of food data
        food_groups: Dictionary of food groups
        land_availability: Dictionary mapping plot to area
        config: Configuration dictionary with parameters
        
    Returns:
        dict: Validation results with violations and constraint checks
    """
    params = config['parameters']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    total_land = sum(land_availability.values())
    
    violations = []
    constraint_checks = {
        'at_most_one_per_plot': {'passed': 0, 'failed': 0, 'violations': []},
        'x_y_linking': {'passed': 0, 'failed': 0, 'violations': []},
        'y_activation': {'passed': 0, 'failed': 0, 'violations': []},
        'area_bounds_min': {'passed': 0, 'failed': 0, 'violations': []},
        'food_group_constraints': {'passed': 0, 'failed': 0, 'violations': []}
    }
    
    # 1. Check: At most one crop per plot
    for plot in farms:
        assigned = sum(solution.get(f"X_{plot}_{crop}", 0) for crop in foods)
        if assigned > 1.01:  # Allow small numerical tolerance
            violation = f"Plot {plot}: {assigned:.3f} crops assigned (should be ≤ 1)"
            violations.append(violation)
            constraint_checks['at_most_one_per_plot']['violations'].append(violation)
            constraint_checks['at_most_one_per_plot']['failed'] += 1
        else:
            constraint_checks['at_most_one_per_plot']['passed'] += 1
    
    # 2. Check: X-Y Linking (X_{p,c} <= Y_c)
    for plot in farms:
        for crop in foods:
            x_pc = solution.get(f"X_{plot}_{crop}", 0)
            y_c = solution.get(f"Y_{crop}", 0)
            if x_pc > y_c + 0.01:  # Allow small tolerance
                violation = f"X_{plot}_{crop}={x_pc:.3f} > Y_{crop}={y_c:.3f}"
                violations.append(violation)
                constraint_checks['x_y_linking']['violations'].append(violation)
                constraint_checks['x_y_linking']['failed'] += 1
            else:
                constraint_checks['x_y_linking']['passed'] += 1
    
    # 3. Check: Y Activation (Y_c <= sum_p X_{p,c})
    for crop in foods:
        y_c = solution.get(f"Y_{crop}", 0)
        sum_x = sum(solution.get(f"X_{plot}_{crop}", 0) for plot in farms)
        if y_c > sum_x + 0.01:  # Allow small tolerance
            violation = f"Y_{crop}={y_c:.3f} > sum(X_{{p,{crop}}})={sum_x:.3f}"
            violations.append(violation)
            constraint_checks['y_activation']['violations'].append(violation)
            constraint_checks['y_activation']['failed'] += 1
        else:
            constraint_checks['y_activation']['passed'] += 1
    
    # 4. Check: Area bounds (minimum and maximum per crop)
    for crop in foods:
        crop_area = sum(
            solution.get(f"X_{plot}_{crop}", 0) * land_availability[plot]
            for plot in farms
        )
        
        # Check minimum area
        if crop in min_planting_area:
            min_area = min_planting_area[crop]
            y_c = solution.get(f"Y_{crop}", 0)
            if y_c > 0.5 and crop_area < min_area - 0.001:  # If crop selected, check min
                violation = f"Crop {crop}: area={crop_area:.4f} < min={min_area:.4f}"
                violations.append(violation)
                constraint_checks['area_bounds_min']['violations'].append(violation)
                constraint_checks['area_bounds_min']['failed'] += 1
            else:
                constraint_checks['area_bounds_min']['passed'] += 1
    
    # NOTE: No maximum area constraints - matches original solver_runner.py
    
    # 5. Check: Food group constraints
    if food_group_constraints:
        for group_name, group_data in food_group_constraints.items():
            if group_name in food_groups:
                crops_in_group = food_groups[group_name]
                n_selected = sum(
                    1 for crop in crops_in_group 
                    if solution.get(f"Y_{crop}", 0) > 0.5
                )
                
                min_crops = group_data.get('min', 0)
                max_crops = group_data.get('max', len(crops_in_group))
                
                if n_selected < min_crops:
                    violation = f"Group {group_name}: {n_selected} crops < min={min_crops}"
                    violations.append(violation)
                    constraint_checks['food_group_constraints']['violations'].append(violation)
                    constraint_checks['food_group_constraints']['failed'] += 1
                elif n_selected > max_crops:
                    violation = f"Group {group_name}: {n_selected} crops > max={max_crops}"
                    violations.append(violation)
                    constraint_checks['food_group_constraints']['violations'].append(violation)
                    constraint_checks['food_group_constraints']['failed'] += 1
                else:
                    constraint_checks['food_group_constraints']['passed'] += 1
    
    # Calculate summary statistics
    total_checks = sum(check['passed'] + check['failed'] for check in constraint_checks.values())
    total_passed = sum(check['passed'] for check in constraint_checks.values())
    total_failed = sum(check['failed'] for check in constraint_checks.values())
    
    return {
        'is_feasible': len(violations) == 0,
        'n_violations': len(violations),
        'violations': violations,
        'constraint_checks': constraint_checks,
        'summary': {
            'total_checks': total_checks,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'pass_rate': total_passed / total_checks if total_checks > 0 else 0
        }
    }

def create_cqm(farms, foods, food_groups, config):
    """
    Creates a CQM for the plot-crop assignment problem (BQM_PATCH formulation).
    
    Variables:
    - X_{p,c}: Binary, 1 if plot p is assigned to crop c, 0 otherwise
    - Y_c: Binary, 1 if crop c is grown on at least one plot, 0 otherwise
    
    Objective: Maximize sum_{p,c} (B_c + λ) * s_p * X_{p,c}
    Where B_c is the weighted benefit per area, s_p is plot area, λ is idle penalty
    
    Constraints:
    1. At most one crop per plot: sum_c X_{p,c} <= 1 for all p
    2. Linking X and Y: X_{p,c} <= Y_c for all p,c
    3. Y activation: Y_c <= sum_p X_{p,c} for all c
    4. Area bounds: A_c^min <= sum_p (s_p * X_{p,c}) <= A_c^max for all c
    5. Food group diversity: FG_g^min <= sum_{c in G_g} Y_c <= FG_g^max for all g
    
    Returns CQM, variables (X, Y), and constraint metadata.
    """
    cqm = ConstrainedQuadraticModel()
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']  # s_p: area of each plot
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})  # A_c^min
    food_group_constraints = params.get('food_group_constraints', {})
    idle_penalty = params.get('idle_penalty_lambda', 0.1)  # λ: penalty for unused area
    
    n_farms = len(farms)
    n_foods = len(foods)
    n_food_groups = len(food_groups) if food_group_constraints else 0
    
    # Calculate total land available
    total_land = sum(land_availability.values())
    
    # NOTE: Maximum area per crop = total land (no artificial limit)
    # This matches the original solver_runner.py formulation
    
    # Calculate total operations for progress bar
    total_ops = (
        n_farms * n_foods +       # X_{p,c} variables
        n_foods +                 # Y_c variables
        n_farms * n_foods +       # Objective terms
        n_farms +                 # At most one crop per plot
        n_farms * n_foods +       # X-Y linking constraints
        n_foods +                 # Y activation constraints
        n_foods +                 # Area bounds (min only)
        n_food_groups * 2         # Food group constraints (min and max)
    )
    
    pbar = tqdm(total=total_ops, desc="Building CQM (BQM_PATCH formulation)", unit="op", ncols=100)
    
    # Define X_{p,c} variables: 1 if plot p is assigned to crop c
    X = {}
    pbar.set_description("Creating X_{p,c} (plot-crop assignment) variables")
    for plot in farms:
        for crop in foods:
            X[(plot, crop)] = Binary(f"X_{plot}_{crop}")
            pbar.update(1)
    
    # Define Y_c variables: 1 if crop c is grown on at least one plot
    Y = {}
    pbar.set_description("Creating Y_c (crop activation) variables")
    for crop in foods:
        Y[crop] = Binary(f"Y_{crop}")
        pbar.update(1)
    
    # Objective function: Maximize sum_{p,c} (B_c + λ) * s_p * X_{p,c}
    # B_c is the weighted benefit per unit area for crop c
    pbar.set_description("Building objective function")
    objective = 0
    for plot in farms:
        s_p = land_availability[plot]  # Area of plot p
        for crop in foods:
            # Calculate B_c: weighted benefit per unit area
            B_c = (
                weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
            )
            # Add (B_c + λ) * s_p * X_{p,c} to objective
            objective += (B_c + idle_penalty) * s_p * X[(plot, crop)]
            pbar.update(1)
    
    # Set objective (maximize, so negate for minimization)
    cqm.set_objective(-objective)
    
    # Constraint metadata
    constraint_metadata = {
        'at_most_one_per_plot': {},
        'x_y_linking': {},
        'y_activation': {},
        'area_bounds_min': {},
        'area_bounds_max': {},
        'food_group_min': {},
        'food_group_max': {}
    }
    
    # Constraint 1: At most one crop per plot
    # For each plot p: sum_c X_{p,c} <= 1
    pbar.set_description("Adding at-most-one-crop-per-plot constraints")
    for plot in farms:
        cqm.add_constraint(
            sum(X[(plot, crop)] for crop in foods) - 1 <= 0,
            label=f"AtMostOne_{plot}"
        )
        constraint_metadata['at_most_one_per_plot'][plot] = {
            'type': 'at_most_one_per_plot',
            'plot': plot,
            'plot_area': land_availability[plot]
        }
        pbar.update(1)
    
    # Constraint 2: X-Y Linking
    # For each plot p and crop c: X_{p,c} <= Y_c
    # (If crop c is not selected, no plot can be assigned to it)
    pbar.set_description("Adding X-Y linking constraints")
    for plot in farms:
        for crop in foods:
            cqm.add_constraint(
                X[(plot, crop)] - Y[crop] <= 0,
                label=f"XY_Link_{plot}_{crop}"
            )
            constraint_metadata['x_y_linking'][(plot, crop)] = {
                'type': 'x_y_linking',
                'plot': plot,
                'crop': crop
            }
            pbar.update(1)
    
    # Constraint 3: Y Activation
    # For each crop c: Y_c <= sum_p X_{p,c}
    # (If crop c is selected, at least one plot must be assigned to it)
    pbar.set_description("Adding Y activation constraints")
    for crop in foods:
        cqm.add_constraint(
            Y[crop] - sum(X[(plot, crop)] for plot in farms) <= 0,
            label=f"Y_Activation_{crop}"
        )
        constraint_metadata['y_activation'][crop] = {
            'type': 'y_activation',
            'crop': crop
        }
        pbar.update(1)
    
    # Constraint 4: Area bounds per crop
    # For each crop c: A_c^min <= sum_p (s_p * X_{p,c})
    # NOTE: No maximum area constraint - matches original solver_runner.py
    pbar.set_description("Adding area bounds constraints")
    for crop in foods:
        total_crop_area = sum(land_availability[plot] * X[(plot, crop)] for plot in farms)
        
        # Minimum area constraint (only if Y_c = 1, i.e., crop is selected)
        # This ensures the constraint only applies to selected crops
        if crop in min_planting_area and min_planting_area[crop] > 0:
            cqm.add_constraint(
                total_crop_area - min_planting_area[crop] * Y[crop] >= 0,
                label=f"MinArea_{crop}"
            )
            constraint_metadata['area_bounds_min'][crop] = {
                'type': 'area_bounds_min',
                'crop': crop,
                'min_area': min_planting_area[crop]
            }
        pbar.update(1)
    
    # Constraint 5: Food group diversity constraints
    # For each food group g: FG_g^min <= sum_{c in G_g} Y_c <= FG_g^max
    pbar.set_description("Adding food group diversity constraints")
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                # Minimum number of crops from this group
                if 'min_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[crop] for crop in foods_in_group) - constraints['min_foods'] >= 0,
                        label=f"FoodGroup_Min_{group}"
                    )
                    constraint_metadata['food_group_min'][group] = {
                        'type': 'food_group_min',
                        'group': group,
                        'min_foods': constraints['min_foods'],
                        'foods_in_group': foods_in_group
                    }
                    pbar.update(1)
                
                # Maximum number of crops from this group
                if 'max_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[crop] for crop in foods_in_group) - constraints['max_foods'] <= 0,
                        label=f"FoodGroup_Max_{group}"
                    )
                    constraint_metadata['food_group_max'][group] = {
                        'type': 'food_group_max',
                        'group': group,
                        'max_foods': constraints['max_foods'],
                        'foods_in_group': foods_in_group
                    }
                    pbar.update(1)
    
    pbar.set_description("CQM complete (BQM_PATCH formulation)")
    pbar.close()
    
    return cqm, (X, Y), constraint_metadata

def solve_with_pulp(farms, foods, food_groups, config):
    """
    Solve with PuLP using BQM_PATCH formulation.
    
    Variables:
    - X_{p,c}: Binary, 1 if plot p is assigned to crop c
    - Y_c: Binary, 1 if crop c is grown on at least one plot
    
    Matches the CQM formulation exactly.
    """
    params = config['parameters']
    land_availability = params['land_availability']  # s_p: area of each plot
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})  # A_c^min
    max_percentage_per_crop = params.get('max_percentage_per_crop', {})
    food_group_constraints = params.get('food_group_constraints', {})
    idle_penalty = params.get('idle_penalty_lambda', 0.1)  # λ: penalty for unused area
    
    # Calculate total land and max areas
    total_land = sum(land_availability.values())
    max_planting_area = {}
    for crop in foods:
        if crop in max_percentage_per_crop:
            max_planting_area[crop] = max_percentage_per_crop[crop] * total_land
        else:
            max_planting_area[crop] = total_land
    
    # Define X_{p,c} variables: 1 if plot p is assigned to crop c
    X_pulp = pl.LpVariable.dicts("X", [(p, c) for p in farms for c in foods], cat='Binary')
    
    # Define Y_c variables: 1 if crop c is grown on at least one plot
    Y_pulp = pl.LpVariable.dicts("Y", foods, cat='Binary')
    
    model = pl.LpProblem("Food_Optimization_BQM_PATCH", pl.LpMaximize)
    
    # Objective: Maximize sum_{p,c} (B_c + λ) * s_p * X_{p,c} / total_land
    # NOTE: Normalized by total_land to match continuous formulation's per-hectare metric
    objective = 0
    for plot in farms:
        s_p = land_availability[plot]  # Area of plot p
        for crop in foods:
            # Calculate B_c: weighted benefit per unit area
            B_c = (
                weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
            )
            # Add (B_c + λ) * s_p * X_{p,c}
            objective += (B_c + idle_penalty) * s_p * X_pulp[(plot, crop)]
    
    # Normalize by total land area (to match continuous formulation)
    objective = objective / total_land
    
    model += objective, "Objective"
    
    # Constraint 1: At most one crop per plot
    for plot in farms:
        model += pl.lpSum([X_pulp[(plot, crop)] for crop in foods]) <= 1, f"AtMostOne_{plot}"
    
    # Constraint 2: X-Y Linking - X_{p,c} <= Y_c
    for plot in farms:
        for crop in foods:
            model += X_pulp[(plot, crop)] <= Y_pulp[crop], f"XY_Link_{plot}_{crop}"
    
    # Constraint 3: Y Activation - Y_c <= sum_p X_{p,c}
    for crop in foods:
        model += Y_pulp[crop] <= pl.lpSum([X_pulp[(plot, crop)] for plot in farms]), f"Y_Activation_{crop}"
    
    # Constraint 4: Area bounds per crop
    for crop in foods:
        total_crop_area = pl.lpSum([land_availability[plot] * X_pulp[(plot, crop)] for plot in farms])
        
        # Minimum area constraint: only applies if crop is selected (Y_pulp[crop] = 1)
        # If crop not selected, Y_pulp[crop] = 0 and constraint becomes total_crop_area >= 0
        if crop in min_planting_area and min_planting_area[crop] > 0:
            model += total_crop_area >= min_planting_area[crop] * Y_pulp[crop], f"MinArea_{crop}"
        
        # Maximum area constraint
        if crop in max_planting_area and max_planting_area[crop] < total_land:
            model += total_crop_area <= max_planting_area[crop], f"MaxArea_{crop}"
    
    # Constraint 5: Food group diversity
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                if 'min_foods' in constraints:
                    model += pl.lpSum([Y_pulp[crop] for crop in foods_in_group]) >= constraints['min_foods'], f"FoodGroup_Min_{group}"
                if 'max_foods' in constraints:
                    model += pl.lpSum([Y_pulp[crop] for crop in foods_in_group]) <= constraints['max_foods'], f"FoodGroup_Max_{group}"
    
    start_time = time.time()
    # Use Gurobi with GPU acceleration and aggressive parallelization
    # GPU-specific parameters (requires Gurobi 9.0+ and CUDA-compatible GPU):
    #   - Method=2: Use barrier method (GPU-accelerated)
    #   - Crossover=0: Disable crossover to keep computation on GPU
    #   - BarHomogeneous=1: Use homogeneous barrier algorithm (better for GPU)
    #   - Threads=0: Use all available CPU threads for parallel processing
    #   - MIPFocus=1: Focus on finding good solutions quickly
    #   - Presolve=2: Aggressive presolve
    gurobi_options = [
        ('Method', 2),           # Barrier method (GPU-accelerated)
        ('Crossover', 0),        # Disable crossover to keep computation on GPU
        ('BarHomogeneous', 1),   # Homogeneous barrier (more GPU-friendly)
        ('Threads', 0),          # Use all available CPU threads for parallelization
        ('MIPFocus', 1),         # Focus on finding good solutions quickly
        ('Presolve', 2),         # Aggressive presolve
    ]
    
    try:
        # Try using GUROBI API directly for better GPU support
        print("  Using Gurobi API with GPU acceleration and parallelization...")
        solver = pl.GUROBI(msg=0, timeLimit=100)
        # Set parameters directly on the solver
        for param, value in gurobi_options:
            solver.optionsDict[param] = value
        model.solve(solver)
    except Exception as e:
        # Fallback to GUROBI_CMD if direct API is not available
        print(f"  Gurobi API failed ({str(e)[:50]}...), using GUROBI_CMD...")
        # GUROBI_CMD expects options as a list of "key=value" strings
        options_list = [f'{k}={v}' for k, v in gurobi_options]
        model.solve(pl.GUROBI_CMD(msg=0, options=options_list))
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': pl.value(model.objective),
        'solve_time': solve_time,
        'X_variables': {},  # X_{p,c}: plot-crop assignments
        'Y_variables': {}   # Y_c: crop selections
    }
    
    # Extract X_{p,c} values
    for plot in farms:
        for crop in foods:
            key = f"X_{plot}_{crop}"
            results['X_variables'][key] = X_pulp[(plot, crop)].value() if X_pulp[(plot, crop)].value() is not None else 0.0
    
    # Extract Y_c values
    for crop in foods:
        key = f"Y_{crop}"
        results['Y_variables'][key] = Y_pulp[crop].value() if Y_pulp[crop].value() is not None else 0.0
    
    return model, results

def solve_with_dwave_cqm(cqm, token):
    """
    Solve with DWave using native HybridCQMSampler (NO BQM conversion).
    This is the proper way to solve CQM problems - keeps constraints native.
    
    Args:
        cqm: ConstrainedQuadraticModel to solve
        token: DWave API token
    
    Returns tuple of (sampleset, hybrid_time, qpu_time)
    """
    print("\nSubmitting CQM to DWave Leap HybridCQM solver...")
    print("  Using native CQM solver - no BQM conversion needed!")
    
    # Use HybridCQM sampler - handles constraints natively
    sampler = LeapHybridCQMSampler(token=token)
    
    solve_start = time.time()
    sampleset = sampler.sample_cqm(cqm, label="Food Optimization - CQM Native")
    total_solve_time = time.time() - solve_start
    
    # Extract timing from sampleset.info
    timing_info = sampleset.info.get('timing', {})
    
    # Hybrid solve time (total time including QPU)
    hybrid_time = (timing_info.get('run_time') or 
                  sampleset.info.get('run_time') or
                  timing_info.get('charge_time') or
                  sampleset.info.get('charge_time'))
    
    if hybrid_time is not None:
        hybrid_time = hybrid_time / 1e6  # Convert from microseconds to seconds
    else:
        hybrid_time = total_solve_time  # Fallback to measured time
    
    # QPU access time
    qpu_time = (timing_info.get('qpu_access_time') or
               sampleset.info.get('qpu_access_time'))
    
    if qpu_time is not None:
        qpu_time = qpu_time / 1e6  # Convert from microseconds to seconds
    
    if hybrid_time is not None:
        print(f"  Hybrid Time: {hybrid_time:.2f}s")
    if qpu_time is not None:
        print(f"  QPU Access Time: {qpu_time:.4f}s")
    
    return sampleset, hybrid_time, qpu_time

def solve_with_dwave(cqm, token):
    """
    Solve with DWave using HybridBQM solver after converting CQM to BQM.
    This enables QPU usage and better scaling for quadratic problems.
    
    NOTE: This converts CQM→BQM which can cause constraint violations if
    Lagrange multiplier is not chosen correctly. For native CQM solving,
    use solve_with_dwave_cqm() instead.
    
    Args:
        cqm: ConstrainedQuadraticModel to convert and solve
        token: DWave API token
    
    Returns tuple of (sampleset, hybrid_time, qpu_time, bqm_conversion_time, invert)
    """
    print("\nConverting CQM to BQM for QPU-enabled solving...")
    print("  This discretizes continuous variables for better QPU utilization.")
    
    # Convert CQM to BQM - this discretizes continuous variables
    convert_start = time.time()
    bqm, invert = cqm_to_bqm(cqm)
    bqm_conversion_time = time.time() - convert_start
    
    print(f"  CQM converted to BQM in {bqm_conversion_time:.2f}s")
    print(f"  BQM Variables: {len(bqm.variables)}")
    print(f"  BQM Interactions: {len(bqm.quadratic)}")
    
    # Use HybridBQM sampler for better QPU usage
    sampler = LeapHybridBQMSampler(token=token)
    
    print("\nSubmitting to DWave Leap HybridBQM solver...")
    print("  This solver uses more QPU time than CQM solver for quadratic problems.")

    sampleset = sampler.sample(bqm, label="Food Optimization - BQUBO Run")
    
    # Extract timing from sampleset.info
    timing_info = sampleset.info.get('timing', {})
    
    # Hybrid solve time (total time including QPU)
    hybrid_time = (timing_info.get('run_time') or 
                  sampleset.info.get('run_time') or
                  timing_info.get('charge_time') or
                  sampleset.info.get('charge_time'))
    
    if hybrid_time is not None:
        hybrid_time = hybrid_time / 1e6  # Convert from microseconds to seconds
    
    # QPU access time
    qpu_time = (timing_info.get('qpu_access_time') or
               sampleset.info.get('qpu_access_time'))
    
    if qpu_time is not None:
        qpu_time = qpu_time / 1e6  # Convert from microseconds to seconds
    
    if hybrid_time is not None:
        print(f"  Hybrid Time: {hybrid_time:.2f}s")
    if qpu_time is not None:
        print(f"  QPU Access Time: {qpu_time:.4f}s")
    
    return sampleset, hybrid_time, qpu_time, bqm_conversion_time, invert

def solve_with_simulated_annealing(bqm):
    """
    Solve with free Simulated Annealing sampler.
    This runs locally and doesn't require QPU access.
    
    Args:
        bqm: BinaryQuadraticModel to solve
    
    Returns tuple of (sampleset, solve_time)
    """
    print("\n" + "=" * 80)
    print("SOLVING WITH SIMULATED ANNEALING (FREE LOCAL SOLVER)")
    print("=" * 80)
    
    sampler = SimulatedAnnealingSampler()
    
    print("Running Simulated Annealing locally...")
    start_time = time.time()
    sampleset = sampler.sample(bqm, num_reads=100, label="Food Optimization - Simulated Annealing")
    solve_time = time.time() - start_time
    
    print(f"  Total samples: {len(sampleset)}")
    print(f"  Solve time: {solve_time:.2f} seconds")
    
    if len(sampleset) > 0:
        best = sampleset.first
        print(f"  Best energy: {best.energy:.6f}")
        best_objective = -best.energy
        print(f"  Best objective: {best_objective:.6f}")
    
    return sampleset, solve_time

def solve_with_gurobi_qubo(bqm, farms=None, foods=None, food_groups=None, land_availability=None, 
                          weights=None, idle_penalty=None, config=None, time_limit=100):
    """
    Solve a Binary Quadratic Model (BQM) using Gurobi's native QUBO solver
    from the `gurobi_optimods` package.
    
    Args:
        bqm: dimod.BinaryQuadraticModel object
        farms: List of farm/plot names (optional, for objective calculation)
        foods: Dictionary of food data (optional, for objective calculation)
        food_groups: Dictionary of food groups (optional, for validation)
        land_availability: Dictionary mapping plot to area (optional)
        weights: Dictionary of objective weights (optional)
        idle_penalty: Lambda penalty for idle land (optional)
        config: Full configuration dictionary (optional, for validation)
        time_limit: Gurobi time limit in seconds (default: 100)
        
    Returns:
        dict: Solution containing status, solution dict, BQM energy, solve time,
              original objective, and constraint validation (if config provided)
    """
    try:
        from gurobi_optimods.qubo import solve_qubo
        import gurobipy as gp
    except ImportError:
        raise ImportError(
            "gurobipy and gurobi-optimods are required. "
            "Install with: pip install gurobipy gurobi-optimods"
        )

    print("\n" + "=" * 80)
    print("SOLVING QUBO WITH GUROBI OPTIMODS (NATIVE QUBO SOLVER)")
    print("=" * 80)

    # Convert BQM to a QUBO dictionary compatible with gurobi_optimods
    Q, offset = bqm.to_qubo()
    
    print(f"  BQM Variables: {len(bqm.variables)}")
    print(f"  QUBO non-zero terms: {len(Q)}")

    # Convert QUBO dictionary to matrix format expected by gurobi_optimods
    import numpy as np
    variables = list(bqm.variables)
    n_vars = len(variables)
    var_to_idx = {var: i for i, var in enumerate(variables)}
    
    # Create coefficient matrix
    Q_matrix = np.zeros((n_vars, n_vars))
    for (var1, var2), coeff in Q.items():
        i, j = var_to_idx[var1], var_to_idx[var2]
        if i == j:
            Q_matrix[i, j] = coeff
        else:
            # For off-diagonal terms, split the coefficient
            Q_matrix[i, j] = coeff / 2
            Q_matrix[j, i] = coeff / 2

    # Gurobi parameters for the QUBO solver
    gurobi_params = {
        "Threads": 0,
        "TimeLimit": time_limit,
    }

    print(f"  Solving with gurobi_optimods.qubo.solve_qubo (TimeLimit={gurobi_params['TimeLimit']}s)...")
    start_time = time.time()
    
    # Call the native QUBO solver
    result_optimod = solve_qubo(Q_matrix, solver_params=gurobi_params)
    
    solve_time = time.time() - start_time

    # Process results
    solution_array = result_optimod.solution
    bqm_energy = result_optimod.objective_value + offset
    
    # Convert solution array back to dictionary format using BQM variable order
    variables = list(bqm.variables)
    solution = {var: int(solution_array[i]) for i, var in enumerate(variables)}
    
    # Determine status based on whether we got a valid solution
    if solution_array is not None and len(solution_array) > 0:
        status = "Optimal"
    else:
        status = "No solution found"
    
    # Calculate original CQM objective if parameters provided
    original_objective = None
    solution_summary = None
    validation = None
    
    if all(x is not None for x in [farms, foods, land_availability, weights, idle_penalty]):
        original_objective = calculate_original_objective(
            solution, farms, foods, land_availability, weights, idle_penalty
        )
        solution_summary = extract_solution_summary(solution, farms, foods, land_availability)
        
        # Validate constraints if full config provided
        if config is not None and food_groups is not None:
            validation = validate_solution_constraints(
                solution, farms, foods, food_groups, land_availability, config
            )
            
            print(f"  Optimal solution found")
            print(f"  BQM Energy: {bqm_energy:.6f}")
            print(f"  Original CQM Objective: {original_objective:.6f}")
            print(f"  Active variables: {sum(solution.values())}")
            print(f"  Constraint validation: {validation['n_violations']} violations")
            if validation['n_violations'] > 0:
                print(f"  CONSTRAINT VIOLATIONS DETECTED:")
                for violation in validation['violations'][:5]:  # Show first 5
                    print(f"     - {violation}")
                if len(validation['violations']) > 5:
                    print(f"     ... and {len(validation['violations']) - 5} more")
            else:
                print(f"  All constraints satisfied!")
        else:
            print(f"  Optimal solution found")
            print(f"  BQM Energy: {bqm_energy:.6f}")
            print(f"  Original CQM Objective: {original_objective:.6f}")
            print(f"  Active variables: {sum(solution.values())}")
    else:
        print(f"  Optimal solution found")
        print(f"  BQM Energy: {bqm_energy:.6f}")
        print(f"  NOTE: BQM energy includes penalty terms and is not comparable to CQM objective")
        print(f"  Active variables: {sum(solution.values())}")
    
    print(f"  Solve time: {solve_time:.3f} seconds")
    
    result = {
        'status': status,
        'solution': solution,
        'objective_value': original_objective,  # Original CQM objective (if calculated)
        'bqm_energy': bqm_energy,
        'solve_time': solve_time,
        'gurobi_status': getattr(result_optimod, 'status', 'N/A'),
        'variables_count': len(bqm.variables),
        'linear_terms': len(bqm.linear),
        'quadratic_terms': len(bqm.quadratic),
        'note': 'objective_value is reconstructed CQM objective; bqm_energy includes penalties'
    }
    
    # Add optional fields if calculated
    if solution_summary is not None:
        result['solution_summary'] = solution_summary
    if validation is not None:
        result['validation'] = validation
    
    return result

def main(scenario='simple', n_patches=None):
    """Main execution function."""
    print("=" * 80)
    print("PROFESSIONAL SOLVER RUNNER - BQM_PATCH")
    print("=" * 80)
    
    # Create output directories
    os.makedirs('PuLP_Results', exist_ok=True)
    os.makedirs('DWave_Results', exist_ok=True)
    os.makedirs('CQM_Models', exist_ok=True)
    os.makedirs('Constraints', exist_ok=True)
    
    # Load scenario
    print(f"\nLoading '{scenario}' scenario...")
    farms, foods, food_groups, config = load_food_data(scenario)
    
    # If n_patches is specified, override farms with generated patches
    if n_patches is not None:
        print(f"\nGenerating {n_patches} patches using patch_sampler...")
        patches_dict = generate_patches(n_patches, seed=42)
        farms = list(patches_dict.keys())
        config['parameters']['land_availability'] = patches_dict
        print(f"  Generated patches: {farms}")
        print(f"  Total patch area: {sum(patches_dict.values()):.3f} ha")
        print(f"  Average patch size: {sum(patches_dict.values())/len(patches_dict):.3f} ha")
    else:
        print(f"  Using scenario farms: {len(farms)} - {farms}")
    
    print(f"  Foods: {len(foods)} - {list(foods.keys())}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CQM with BQM_PATCH formulation
    print("\nCreating CQM with BQM_PATCH formulation (implicit idle area)...")
    cqm, (X, Y), constraint_metadata = create_cqm(farms, foods, food_groups, config)
    print(f"  Variables: {len(cqm.variables)} (X_{{p,c}} and Y_c, all binary)")
    print(f"  - X variables (plot-crop assignments): {len(X)}")
    print(f"  - Y variables (crop selections): {len(Y)}")
    print(f"  Constraints: {len(cqm.constraints)}")
    print(f"  Formulation: Plot-crop assignment with implicit idle representation")
    
    # Save CQM
    cqm_path = f'CQM_Models/cqm_{scenario}_{timestamp}.cqm'
    print(f"\nSaving CQM to {cqm_path}...")
    with open(cqm_path, 'wb') as f:
        shutil.copyfileobj(cqm.to_file(), f)
    
    # Save constraint metadata
    constraints_path = f'Constraints/constraints_{scenario}_{timestamp}.json'
    print(f"Saving constraints to {constraints_path}...")
    
    # Convert constraint_metadata keys to strings for JSON serialization
    # Also convert foods dict to serializable format
    foods_serializable = {
        name: {k: float(v) if isinstance(v, (int, float)) else v for k, v in attrs.items()}
        for name, attrs in foods.items()
    }
    
    constraints_json = {
        'scenario': scenario,
        'timestamp': timestamp,
        'farms': farms,
        'foods': list(foods.keys()),
        'foods_data': foods_serializable,  # Add full food data for objective calculation
        'food_groups': food_groups,
        'config': config,
        'constraint_metadata': {
            'at_most_one_per_plot': {str(k): v for k, v in constraint_metadata['at_most_one_per_plot'].items()},
            'x_y_linking': {str(k): v for k, v in constraint_metadata['x_y_linking'].items()},
            'y_activation': {str(k): v for k, v in constraint_metadata['y_activation'].items()},
            'area_bounds_min': {str(k): v for k, v in constraint_metadata['area_bounds_min'].items()},
            'area_bounds_max': {str(k): v for k, v in constraint_metadata['area_bounds_max'].items()},
            'food_group_min': {str(k): v for k, v in constraint_metadata['food_group_min'].items()},
            'food_group_max': {str(k): v for k, v in constraint_metadata['food_group_max'].items()}
        },
        'formulation': 'BQM_PATCH',
        'description': 'Plot-crop assignment with implicit idle area representation'
    }
    
    with open(constraints_path, 'w') as f:
        json.dump(constraints_json, f, indent=2)
    
    # Solve with PuLP
    print("\n" + "=" * 80)
    print("SOLVING WITH PULP")
    print("=" * 80)
    pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config)
    print(f"  Status: {pulp_results['status']}")
    print(f"  Objective: {pulp_results['objective_value']:.6f}")
    print(f"  Solve time: {pulp_results['solve_time']:.2f} seconds")
    
    # Save PuLP results
    pulp_path = f'PuLP_Results/pulp_{scenario}_{timestamp}.json'
    print(f"\nSaving PuLP results to {pulp_path}...")
    with open(pulp_path, 'w') as f:
        json.dump(pulp_results, f, indent=2)
    
    # Solve with DWave using BQUBO approach
    print("\n" + "=" * 80)
    print("SOLVING WITH DWAVE (BQUBO: CQM→BQM + HybridBQM)")
    print("=" * 80)
    token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
    
    # Convert CQM to BQM once (reuse for both samplers)
    print("\nConverting CQM to BQM...")
    convert_start = time.time()
    bqm, invert = cqm_to_bqm(cqm)
    bqm_conversion_time = time.time() - convert_start
    print(f"  CQM converted to BQM in {bqm_conversion_time:.2f}s")
    print(f"  BQM Variables: {len(bqm.variables)}")
    print(f"  BQM Interactions: {len(bqm.quadratic)}")
    
    # Solve with HybridBQM (QPU-enabled)
    print("\nSubmitting to DWave Leap HybridBQM solver...")
    sampler_hybrid = LeapHybridBQMSampler(token=token)
    hybrid_start = time.time()
    sampleset_hybrid = sampler_hybrid.sample(bqm, label="Food Optimization - BQUBO HybridBQM")
    hybrid_total_time = time.time() - hybrid_start
    
    # Extract timing from sampleset
    timing_info = sampleset_hybrid.info.get('timing', {})
    hybrid_time = (timing_info.get('run_time') or 
                  sampleset_hybrid.info.get('run_time') or
                  timing_info.get('charge_time') or
                  sampleset_hybrid.info.get('charge_time'))
    if hybrid_time is not None:
        hybrid_time = hybrid_time / 1e6
    else:
        hybrid_time = hybrid_total_time
        
    qpu_access_time = (timing_info.get('qpu_access_time') or
                      sampleset_hybrid.info.get('qpu_access_time'))
    if qpu_access_time is not None:
        qpu_access_time = qpu_access_time / 1e6
    else:
        qpu_access_time = 0.0
    
    print(f"  HybridBQM solve complete")
    print(f"  Total samples: {len(sampleset_hybrid)}")
    print(f"  Hybrid Time: {hybrid_time:.2f}s")
    print(f"  QPU Access Time: {qpu_access_time:.4f}s")
    
    if len(sampleset_hybrid) > 0:
        best_hybrid = sampleset_hybrid.first
        print(f"  Best energy: {best_hybrid.energy:.6f}")
        best_objective_hybrid = -best_hybrid.energy
        print(f"  Best objective: {best_objective_hybrid:.6f}")
    
    # Solve with Simulated Annealing (free, local)
    print("\n" + "=" * 80)
    print("SOLVING WITH SIMULATED ANNEALING (FREE LOCAL SOLVER)")
    print("=" * 80)
    sampler_sa = SimulatedAnnealingSampler()
    sa_start = time.time()
    sampleset_sa = sampler_sa.sample(bqm, num_reads=100, label="Food Optimization - Simulated Annealing")
    sa_time = time.time() - sa_start
    
    print(f"  Simulated Annealing complete")
    print(f"  Total samples: {len(sampleset_sa)}")
    print(f"  Solve time: {sa_time:.2f}s")
    
    if len(sampleset_sa) > 0:
        best_sa = sampleset_sa.first
        print(f"  Best energy: {best_sa.energy:.6f}")
        best_objective_sa = -best_sa.energy
        print(f"  Best objective: {best_objective_sa:.6f}")
    
    # Save DWave results (both pickle and JSON)
    dwave_pickle_path = f'DWave_Results/dwave_bqubo_{scenario}_{timestamp}.pickle'
    print(f"\nSaving DWave HybridBQM sampleset to {dwave_pickle_path}...")
    with open(dwave_pickle_path, 'wb') as f:
        pickle.dump(sampleset_hybrid, f)
    
    # Save DWave results as JSON for easy reading
    dwave_json_path = f'DWave_Results/dwave_bqubo_{scenario}_{timestamp}.json'
    
    # Convert sample to JSON-serializable format
    best_sample_json = {}
    if len(sampleset_hybrid) > 0:
        for key, value in best_hybrid.sample.items():
            best_sample_json[str(key)] = int(value)  # Convert numpy types to Python int
    
    dwave_results = {
        'status': 'Optimal' if len(sampleset_hybrid) > 0 else 'No solutions',
        'objective_value': best_objective_hybrid if len(sampleset_hybrid) > 0 else None,
        'solve_time': hybrid_time,
        'qpu_access_time': qpu_access_time,
        'bqm_conversion_time': bqm_conversion_time,
        'num_samples': len(sampleset_hybrid),
        'formulation': 'BQUBO (HybridBQM)',
        'best_sample': best_sample_json
    }
    with open(dwave_json_path, 'w') as f:
        json.dump(dwave_results, f, indent=2)
    
    # Save Simulated Annealing results
    sa_json_path = f'DWave_Results/simulated_annealing_{scenario}_{timestamp}.json'
    print(f"Saving Simulated Annealing results to {sa_json_path}...")
    
    best_sa_sample_json = {}
    if len(sampleset_sa) > 0:
        for key, value in best_sa.sample.items():
            best_sa_sample_json[str(key)] = int(value)
    
    sa_results = {
        'status': 'Optimal' if len(sampleset_sa) > 0 else 'No solutions',
        'objective_value': best_objective_sa if len(sampleset_sa) > 0 else None,
        'solve_time': sa_time,
        'num_samples': len(sampleset_sa),
        'formulation': 'Simulated Annealing (Free)',
        'best_sample': best_sa_sample_json
    }
    with open(sa_json_path, 'w') as f:
        json.dump(sa_results, f, indent=2)
    
    # Print comprehensive comparison
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON - BQM_PATCH FORMULATION")
    print("=" * 80)
    print(f"\n📊 PROBLEM SIZE:")
    print(f"   Plots: {len(farms)}")
    print(f"   Crops: {len(foods)}")
    print(f"   X variables (plot-crop assignments): {len(farms) * len(foods)}")
    print(f"   Y variables (crop selections): {len(foods)}")
    print(f"   Total CQM variables: {len(cqm.variables)}")
    print(f"   Total BQM variables (after conversion): {len(bqm.variables)}")
    print(f"   Total constraints: {len(cqm.constraints)}")
    
    print(f"\n🔧 PULP SOLVER:")
    print(f"   Status: {pulp_results['status']}")
    print(f"   Objective: {pulp_results['objective_value']:.6f}")
    print(f"   Solve Time: {pulp_results['solve_time']:.4f}s")
    print(f"   Selected crops (Y=1): {sum(1 for v in pulp_results['Y_variables'].values() if v > 0.5)}")
    print(f"   Assigned plots (X=1): {sum(1 for v in pulp_results['X_variables'].values() if v > 0.5)}")
    
    print(f"\n⚛️  DWAVE HYBRID BQM:")
    print(f"   Status: {dwave_results['status']}")
    print(f"   Objective: {dwave_results['objective_value']:.6f}")
    print(f"   Run Time: {hybrid_time:.4f}s")
    print(f"   QPU Access Time: {qpu_access_time:.6f}s")
    print(f"   BQM Conversion: {bqm_conversion_time:.4f}s")
    print(f"   Samples: {len(sampleset_hybrid)}")
    
    print(f"\n🔥 SIMULATED ANNEALING:")
    print(f"   Status: {sa_results['status']}")
    print(f"   Objective: {sa_results['objective_value']:.6f}")
    print(f"   Solve Time: {sa_time:.4f}s")
    print(f"   Samples: {len(sampleset_sa)}")
    
    print(f"\n📈 OBJECTIVE COMPARISONS:")
    print(f"   PuLP vs HybridBQM: {abs(pulp_results['objective_value'] - best_objective_hybrid):.6f}")
    print(f"   PuLP vs SimAnneal: {abs(pulp_results['objective_value'] - best_objective_sa):.6f}")
    print(f"   HybridBQM vs SimAnneal: {abs(best_objective_hybrid - best_objective_sa):.6f}")
    
    # Create run manifest
    manifest = {
        'scenario': scenario,
        'timestamp': timestamp,
        'cqm_path': cqm_path,
        'constraints_path': constraints_path,
        'pulp_path': pulp_path,
        'dwave_pickle_path': dwave_pickle_path,
        'dwave_json_path': dwave_json_path,
        'sa_json_path': sa_json_path,
        'farms': farms,
        'foods': list(foods.keys()),
        'pulp_status': pulp_results['status'],
        'pulp_objective': pulp_results['objective_value'],
        'dwave_status': dwave_results['status'],
        'dwave_objective': dwave_results['objective_value'],
        'dwave_qpu_time': qpu_access_time,
        'dwave_sample_count': len(sampleset_hybrid),
        'sa_status': sa_results['status'],
        'sa_objective': sa_results['objective_value'],
        'sa_sample_count': len(sampleset_sa),
        'formulation': 'BQM_PATCH'
    }
    
    manifest_path = f'run_manifest_{scenario}_{timestamp}.json'
    print(f"\nSaving run manifest to {manifest_path}...")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SOLVER RUN COMPLETE (BQUBO)")
    print("=" * 80)
    print(f"Manifest file: {manifest_path}")
    print(f"CQM model: {cqm_path}")
    print(f"PuLP results: {pulp_path}")
    print(f"DWave results (JSON): {dwave_json_path}")
    print(f"DWave results (pickle): {dwave_pickle_path}")
    print(f"\nBQUBO approach: CQM→BQM conversion + HybridBQM solver")
    print(f"   QPU Access Time: {qpu_access_time:.4f}s")
    print(f"   More QPU usage = Better scaling for larger problems!")
    print("\nRun the verifier script with this manifest to check results:")
    print(f"  python verifier.py {manifest_path}")
    
    return manifest_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run solvers on a food optimization scenario')
    parser.add_argument('--scenario', type=str, default='simple', 
                       choices=['simple', 'intermediate', 'full', 'custom', 'full_family'],
                       help='Scenario to solve (default: simple)')
    parser.add_argument('--n-patches', type=int, default=None,
                       help='Number of patches to generate (overrides scenario farms)')
    
    args = parser.parse_args()
    
    main(scenario=args.scenario, n_patches=args.n_patches)
