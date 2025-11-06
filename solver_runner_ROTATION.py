"""
Professional solver runner script for 3-period crop rotation optimization.

This script implements the binary (plot-level) formulation from crop_rotation.tex:
1. Time-indexed variables: Y_{p,c,t} for plot p, crop c, period t ∈ {1,2,3}
2. Objective includes linear crop values + quadratic rotation synergy between consecutive periods
3. Per-period constraints: each plot assigned to at most one crop per period
4. Food group constraints applied per period
5. Rotation synergy matrix R loaded from rotation_crop_matrix.csv
"""

import os
import sys
import json
import pickle
import shutil
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.scenarios import load_food_data
import patch_sampler
import farm_sampler
from dimod import ConstrainedQuadraticModel, Binary, Real, cqm_to_bqm
from dwave.system import LeapHybridCQMSampler, LeapHybridBQMSampler
import pulp as pl
from tqdm import tqdm
import math

# Global rotation matrix - loaded once at module level
ROTATION_MATRIX = None
ROTATION_GAMMA = 0.1  # Default synergy weight coefficient

def load_rotation_matrix(matrix_path="rotation_data/rotation_crop_matrix.csv"):
    """
    Load the rotation synergy matrix from CSV.
    
    Args:
        matrix_path: Path to the rotation_crop_matrix.csv file
        
    Returns:
        pandas DataFrame with crop-to-crop rotation synergy values
    """
    global ROTATION_MATRIX
    if ROTATION_MATRIX is None:
        full_path = os.path.join(project_root, matrix_path)
        if not os.path.exists(full_path):
            print(f"Warning: Rotation matrix not found at {full_path}")
            print("Run rotation_matrix.py first to generate it.")
            return None
        ROTATION_MATRIX = pd.read_csv(full_path, index_col=0)
        print(f"Loaded rotation matrix with shape {ROTATION_MATRIX.shape}")
    return ROTATION_MATRIX

def calculate_original_objective(solution, farms, foods, land_availability, weights, idle_penalty):
    """
    Calculate the original CQM objective from a solution (binary formulation).
    
    This reconstructs the objective for the binary (plots) formulation:
    sum_{p,c} B_c * s_p * Y_{p,c}
    
    Args:
        solution: Dictionary with variable assignments (Y_{plot}_{crop})
        farms: List of farm/plot names
        foods: Dictionary of food data with nutritional values
        land_availability: Dictionary mapping plot to area
        weights: Dictionary of objective weights
        idle_penalty: Lambda penalty for idle land (not used in binary formulation)
        
    Returns:
        float: The original objective value (to be maximized)
    """
    objective = 0.0
    total_land = sum(land_availability.values())
    
    for plot in farms:
        s_p = land_availability[plot]  # Area of plot p
        for crop in foods:
            # Get Y_{p,c} value from solution
            var_name = f"Y_{plot}_{crop}"
            y_pc = solution.get(var_name, 0)
            
            if y_pc > 0:  # Only count if assigned
                # Calculate B_c: weighted benefit per unit area
                B_c = (
                    weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                    weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                    weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
                )
                # Add B_c * s_p * Y_{p,c} to objective (normalized)
                objective += B_c * s_p * y_pc / total_land
    
    return objective

def extract_solution_summary(solution, farms, foods, land_availability):
    """
    Extract a summary of the solution showing crop selections and plot assignments (binary formulation).
    
    Args:
        solution: Dictionary with variable assignments (Y_{plot}_{crop})
        farms: List of farm/plot names
        foods: Dictionary of food data
        land_availability: Dictionary mapping plot to area
        
    Returns:
        dict: Summary with crops selected, areas, and plot assignments
    """
    crops_selected = set()
    plot_assignments = []
    total_allocated = 0.0
    
    for crop in foods:
        # Calculate total area allocated to this crop
        total_area = 0.0
        assigned_plots = []
        
        for plot in farms:
            y_var = f"Y_{plot}_{crop}"
            if solution.get(y_var, 0) > 0:
                area = land_availability[plot]
                total_area += area
                assigned_plots.append({'plot': plot, 'area': area})
        
        if total_area > 0:  # Only include if actually allocated
            crops_selected.add(crop)
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
        'crops_selected': list(crops_selected),
        'n_crops': len(crops_selected),
        'plot_assignments': plot_assignments,
        'total_allocated': total_allocated,
        'total_available': total_available,
        'idle_area': idle_area,
        'utilization': total_allocated / total_available if total_available > 0 else 0
    }

def calculate_model_complexity(formulation_type, n_farms, n_foods, n_food_groups=0, 
                               n_crops_with_min_area=0, n_crops_with_max_area=0,
                               has_food_group_constraints=False):
    """
    Calculate the number of variables, constraints, and coefficients for a given formulation.
    
    This function provides metrics for comparison with benchmark papers in optimization literature.
    
    Args:
        formulation_type: 'continuous' or 'binary'
        n_farms: Number of farms/plots
        n_foods: Number of crops/foods
        n_food_groups: Number of food groups
        n_crops_with_min_area: Number of crops with minimum area constraints
        n_crops_with_max_area: Number of crops with maximum area constraints
        has_food_group_constraints: Whether food group constraints are active
    
    Returns:
        Dictionary with complexity metrics:
        - n_variables: Total number of decision variables
        - n_binary_vars: Number of binary variables
        - n_continuous_vars: Number of continuous variables
        - n_constraints: Total number of constraints
        - n_linear_coefficients: Number of non-zero linear coefficients
        - n_quadratic_coefficients: Number of non-zero quadratic coefficients (bilinear terms)
        - problem_class: Classification (LP, MILP, MINLP, BIP, etc.)
    """
    complexity = {}
    
    if formulation_type == 'continuous':
        # Continuous formulation with area (A) and selection (Y) variables
        n_area_vars = n_farms * n_foods  # A_{f,c}
        n_selection_vars = n_farms * n_foods  # Y_{f,c}
        
        complexity['n_variables'] = n_area_vars + n_selection_vars
        complexity['n_continuous_vars'] = n_area_vars
        complexity['n_binary_vars'] = n_selection_vars
        
        # Constraints:
        # 1. Land availability: n_farms constraints
        n_land_constraints = n_farms
        
        # 2. Minimum area linking: n_farms * n_foods constraints (A >= A_min * Y)
        n_min_linking = n_farms * n_foods
        
        # 3. Maximum area linking: n_farms * n_foods constraints (A <= L_f * Y)
        n_max_linking = n_farms * n_foods
        
        # 4. Food group constraints: 2 * n_farms * n_food_groups (min and max)
        n_food_group = 2 * n_farms * n_food_groups if has_food_group_constraints else 0
        
        complexity['n_constraints'] = (n_land_constraints + n_min_linking + 
                                      n_max_linking + n_food_group)
        
        # Linear coefficients in objective: n_farms * n_foods (for A variables)
        # Linear coefficients in constraints:
        # - Land: n_farms * n_foods (sum of A per farm)
        # - Min linking: n_farms * n_foods (A terms) + n_farms * n_foods (Y terms)
        # - Max linking: n_farms * n_foods (A terms) + n_farms * n_foods (Y terms)
        # - Food group: varies, approximately 2 * n_farms * (avg foods per group)
        avg_foods_per_group = n_foods / max(n_food_groups, 1) if n_food_groups > 0 else 0
        complexity['n_linear_coefficients'] = (
            n_farms * n_foods +  # Objective
            n_farms * n_foods +  # Land constraints
            2 * n_farms * n_foods +  # Min linking (A and Y)
            2 * n_farms * n_foods +  # Max linking (A and Y)
            2 * n_farms * int(avg_foods_per_group) * n_food_groups  # Food group
        )
        
        # Quadratic coefficients: bilinear terms A * Y in linking constraints
        # Each linking constraint has one bilinear term
        complexity['n_quadratic_coefficients'] = 2 * n_farms * n_foods
        
        complexity['problem_class'] = 'MINLP (Mixed-Integer Nonlinear Program with bilinear terms)'
        
    elif formulation_type == 'binary':
        # Binary formulation with only selection variables Y
        n_selection_vars = n_farms * n_foods  # Y_{p,c}
        
        complexity['n_variables'] = n_selection_vars
        complexity['n_continuous_vars'] = 0
        complexity['n_binary_vars'] = n_selection_vars
        
        # Constraints:
        # 1. Plot assignment: n_farms constraints (each plot assigned to at most one crop)
        n_plot_constraints = n_farms
        
        # 2. Minimum plots per crop: n_crops_with_min_area constraints
        n_min_plots = n_crops_with_min_area
        
        # 3. Maximum plots per crop: n_crops_with_max_area constraints
        n_max_plots = n_crops_with_max_area
        
        # 4. Food group constraints: 2 * n_farms * n_food_groups (min and max)
        n_food_group = 2 * n_farms * n_food_groups if has_food_group_constraints else 0
        
        complexity['n_constraints'] = (n_plot_constraints + n_min_plots + 
                                      n_max_plots + n_food_group)
        
        # Linear coefficients in objective: n_farms * n_foods (all Y variables)
        # Linear coefficients in constraints:
        # - Plot assignment: n_farms * n_foods (sum of Y per plot)
        # - Min plots: n_farms per constraint * n_crops_with_min
        # - Max plots: n_farms per constraint * n_crops_with_max
        # - Food group: 2 * n_farms * (avg foods per group) * n_food_groups
        avg_foods_per_group = n_foods / max(n_food_groups, 1) if n_food_groups > 0 else 0
        complexity['n_linear_coefficients'] = (
            n_farms * n_foods +  # Objective
            n_farms * n_foods +  # Plot assignment
            n_farms * n_crops_with_min_area +  # Min plots per crop
            n_farms * n_crops_with_max_area +  # Max plots per crop
            2 * n_farms * int(avg_foods_per_group) * n_food_groups  # Food group
        )
        
        # No quadratic terms in binary formulation
        complexity['n_quadratic_coefficients'] = 0
        
        complexity['problem_class'] = 'BIP (Binary Integer Program - pure 0-1 optimization)'
    
    else:
        raise ValueError(f"Unknown formulation_type: {formulation_type}")
    
    return complexity

def print_model_complexity_comparison(continuous_complexity, binary_complexity):
    """
    Print a formatted comparison table of model complexities.
    
    Args:
        continuous_complexity: Dictionary from calculate_model_complexity for continuous
        binary_complexity: Dictionary from calculate_model_complexity for binary
    """
    print("\n" + "="*80)
    print("MODEL COMPLEXITY COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<40} {'Continuous':<20} {'Binary':<20}")
    print("-"*80)
    
    print(f"{'Problem Class':<40} {continuous_complexity['problem_class']:<20} {binary_complexity['problem_class']:<20}")
    print(f"{'Total Variables':<40} {continuous_complexity['n_variables']:<20} {binary_complexity['n_variables']:<20}")
    print(f"{'  - Continuous Variables':<40} {continuous_complexity['n_continuous_vars']:<20} {binary_complexity['n_continuous_vars']:<20}")
    print(f"{'  - Binary Variables':<40} {continuous_complexity['n_binary_vars']:<20} {binary_complexity['n_binary_vars']:<20}")
    print(f"{'Total Constraints':<40} {continuous_complexity['n_constraints']:<20} {binary_complexity['n_constraints']:<20}")
    print(f"{'Linear Coefficients':<40} {continuous_complexity['n_linear_coefficients']:<20} {binary_complexity['n_linear_coefficients']:<20}")
    print(f"{'Quadratic Coefficients (bilinear)':<40} {continuous_complexity['n_quadratic_coefficients']:<20} {binary_complexity['n_quadratic_coefficients']:<20}")
    
    print("\n" + "="*80)
    print("COMPLEXITY REDUCTION ANALYSIS")
    print("="*80)
    
    var_reduction = (1 - binary_complexity['n_variables'] / continuous_complexity['n_variables']) * 100
    const_reduction = (1 - binary_complexity['n_constraints'] / continuous_complexity['n_constraints']) * 100
    coeff_reduction = (1 - binary_complexity['n_linear_coefficients'] / continuous_complexity['n_linear_coefficients']) * 100
    
    print(f"{'Variable Reduction':<40} {var_reduction:>6.2f}%")
    print(f"{'Constraint Reduction':<40} {const_reduction:>6.2f}%")
    print(f"{'Linear Coefficient Reduction':<40} {coeff_reduction:>6.2f}%")
    print(f"{'Quadratic Terms Eliminated':<40} {'YES (100%)' if binary_complexity['n_quadratic_coefficients'] == 0 else 'NO'}")
    
    print("\n" + "="*80)

def create_cqm_farm(farms, foods, food_groups, config):
    """
    Creates a CQM for the food optimization problem.
    Returns CQM, variables, and constraint metadata.
    """
    cqm = ConstrainedQuadraticModel()

    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})

    # Define variables
    A = {}
    Y = {}

    for farm in tqdm(farms, desc="Creating variables"):
        for food in foods:
            A[(farm, food)] = Real(
                f"A_{farm}_{food}", lower_bound=0, upper_bound=land_availability[farm])
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")

    # Objective function
    total_area = sum(land_availability[farm] for farm in farms)

    objective = sum(
        weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * A[(farm, food)] +
        weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * A[(farm, food)] -
        weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * A[(farm, food)] +
        weights.get('affordability', 0) * foods[food].get('affordability', 0) * A[(farm, food)] +
        weights.get('sustainability', 0) *
        foods[food].get('sustainability', 0) * A[(farm, food)]
        for farm in tqdm(farms) for food in foods
    )

    cqm.set_objective(-objective / total_area)  # Maximize normalized objective

    # Constraint metadata
    constraint_metadata = {
        'land_availability': {},
        'min_area_if_selected': {},
        'max_area_if_selected': {},
        'food_group_min': {},
        'food_group_max': {}
    }

    # Land availability constraints
    for farm in tqdm(farms, desc="Adding land availability constraints"):
        cqm.add_constraint(
            sum(A[(farm, food)]
                for food in foods) - land_availability[farm] <= 0,
            label=f"Land_Availability_{farm}"
        )
        constraint_metadata['land_availability'][farm] = {
            'type': 'land_availability',
            'farm': farm,
            'max_land': land_availability[farm]
        }

    # Linking constraints
    for farm in tqdm(farms, desc="Adding linking constraints"):
        for food in foods:
            A_min = min_planting_area.get(food, 0)

            cqm.add_constraint(
                A[(farm, food)] - A_min * Y[(farm, food)] >= 0,
                label=f"Min_Area_If_Selected_{farm}_{food}"
            )
            constraint_metadata['min_area_if_selected'][(farm, food)] = {
                'type': 'min_area_if_selected',
                'farm': farm,
                'food': food,
                'min_area': A_min
            }

            cqm.add_constraint(
                A[(farm, food)] - land_availability[farm] * Y[(farm, food)] <= 0,
                label=f"Max_Area_If_Selected_{farm}_{food}"
            )
            constraint_metadata['max_area_if_selected'][(farm, food)] = {
                'type': 'max_area_if_selected',
                'farm': farm,
                'food': food,
                'max_land': land_availability[farm]
            }

    # Food group constraints - GLOBAL across all farms
    if food_group_constraints:
        for group, constraints in tqdm(food_group_constraints.items(), desc="Adding food group constraints"):
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                # Global minimum: across ALL farms, at least min_foods from this group
                if 'min_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) -
                        constraints['min_foods'] >= 0,
                        label=f"Food_Group_Min_{group}_Global"
                    )
                    constraint_metadata['food_group_min'][group] = {
                        'type': 'food_group_min_global',
                        'group': group,
                        'min_foods': constraints['min_foods'],
                        'foods_in_group': foods_in_group,
                        'scope': 'global'
                    }

                # Global maximum: across ALL farms, at most max_foods from this group
                if 'max_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) -
                        constraints['max_foods'] <= 0,
                        label=f"Food_Group_Max_{group}_Global"
                    )
                    constraint_metadata['food_group_max'][group] = {
                        'type': 'food_group_max_global',
                        'group': group,
                        'max_foods': constraints['max_foods'],
                        'foods_in_group': foods_in_group,
                        'scope': 'global'
                    }

    return cqm, A, Y, constraint_metadata


def create_cqm_farm_rotation_3period(farms, foods, food_groups, config, gamma=None):
    """
    Creates a CQM for the 3-period crop rotation optimization problem (FARM/CONTINUOUS formulation).
    
    Implements the continuous formulation with rotation across 3 time periods:
    - Time periods: t ∈ {1, 2, 3}
    - Variables: A_{f,c,t} ∈ [0, land_f] (continuous area), Y_{f,c,t} ∈ {0,1} (binary indicator)
    - Objective: Maximize area-normalized sum of:
      * Linear crop values across all periods
      * Quadratic rotation synergy between consecutive periods (t-1 to t)
    - Constraints (per period):
      * Total area per farm ≤ land_availability[f]
      * Linking: A_{f,c,t} >= min_area * Y_{f,c,t} and A_{f,c,t} <= land_f * Y_{f,c,t}
      * Food group min/max constraints per period
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food group mappings
        config: Configuration including land_availability and parameters
        gamma: Rotation synergy weight (default: ROTATION_GAMMA global)
    
    Returns:
        Tuple of (CQM, (A_vars, Y_vars), constraint_metadata)
    """
    # Load rotation matrix
    rotation_matrix = load_rotation_matrix()
    if rotation_matrix is None:
        raise FileNotFoundError("Rotation matrix not found. Run rotation_matrix.py first.")
    
    if gamma is None:
        gamma = ROTATION_GAMMA
    
    cqm = ConstrainedQuadraticModel()
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    n_farms = len(farms)
    n_crops = len(foods)
    n_periods = 3
    n_food_groups = len(food_groups) if food_group_constraints else 0
    
    total_land_area = sum(land_availability.values())
    crop_list = list(foods.keys())
    
    # Count operations for progress bar
    total_ops = (
        n_farms * n_crops * n_periods * 2 +     # Variables (A and Y)
        n_farms * n_crops * n_periods +         # Linear objective terms
        n_farms * n_crops * n_crops * 2 +       # Quadratic rotation terms (periods 2 and 3)
        n_farms * n_periods +                   # Land availability constraints
        n_farms * n_crops * n_periods * 2 +     # Linking constraints
        n_food_groups * 2 * n_periods           # Food group constraints
    )
    
    pbar = tqdm(total=total_ops, desc="Building 3-period rotation CQM (farm formulation)", unit="op", ncols=100)
    
    # Define time-indexed variables A_{f,c,t} and Y_{f,c,t}
    A = {}
    Y = {}
    
    pbar.set_description("Creating time-indexed variables")
    for farm in farms:
        for crop in foods:
            for t in range(1, n_periods + 1):
                A[(farm, crop, t)] = Real(
                    f"A_{farm}_{crop}_t{t}", 
                    lower_bound=0, 
                    upper_bound=land_availability[farm]
                )
                Y[(farm, crop, t)] = Binary(f"Y_{farm}_{crop}_t{t}")
                pbar.update(2)
    
    # Build objective function
    pbar.set_description("Building objective - linear terms")
    objective = 0
    
    # Part 1: Linear crop value terms across all periods
    for t in range(1, n_periods + 1):
        for farm in farms:
            for crop in foods:
                # Calculate B_c: weighted benefit per unit area
                B_c = (
                    weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                    weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                    weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
                )
                # Add normalized term: (B_c * A_{f,c,t}) / A_tot
                objective += B_c * A[(farm, crop, t)]
                pbar.update(1)
    
    # Part 2: Quadratic rotation synergy terms between consecutive periods
    # NOTE: For CQM, we use binary Y variables for rotation synergy (not continuous A variables)
    # This represents synergy based on crop selection rather than exact area allocation
    pbar.set_description("Building objective - rotation synergy terms")
    for t in range(2, n_periods + 1):  # t = 2, 3 (comparing to t-1)
        for farm in farms:
            for crop_prev in crop_list:
                for crop_curr in crop_list:
                    # Get rotation synergy R_{c,c'} from matrix
                    try:
                        R_cc = rotation_matrix.loc[crop_prev, crop_curr]
                    except KeyError:
                        R_cc = 0.0
                    
                    # Add quadratic term: gamma * R_{c,c'} * Y_{f,c,t-1} * Y_{f,c',t}
                    # Scaled by farm land availability to approximate area impact
                    if R_cc != 0:
                        farm_area = land_availability[farm]
                        objective += (gamma * R_cc * farm_area * Y[(farm, crop_prev, t-1)] * Y[(farm, crop_curr, t)]) / total_land_area
                pbar.update(1)
    
    # Set objective (CQM minimizes, so negate for maximization)
    cqm.set_objective(-objective/total_land_area)
    
    # Constraint metadata
    constraint_metadata = {
        'land_availability_per_period': {},
        'min_area_if_selected_per_period': {},
        'max_area_if_selected_per_period': {},
        'food_group_min_per_period': {},
        'food_group_max_per_period': {}
    }
    
    # Constraint 1: Land availability per farm per period
    pbar.set_description("Adding land availability constraints (per period)")
    for t in range(1, n_periods + 1):
        for farm in farms:
            cqm.add_constraint(
                sum(A[(farm, crop, t)] for crop in foods) - land_availability[farm] <= 0,
                label=f"Land_Availability_{farm}_t{t}"
            )
            constraint_metadata['land_availability_per_period'][f"{farm}_t{t}"] = {
                'type': 'land_availability',
                'farm': farm,
                'period': t,
                'max_land': land_availability[farm]
            }
            pbar.update(1)
    
    # Constraint 2: Linking constraints (min and max area if selected)
    pbar.set_description("Adding linking constraints (per period)")
    for t in range(1, n_periods + 1):
        for farm in farms:
            for crop in foods:
                A_min = min_planting_area.get(crop, 0)
                
                # Min area if selected: A_{f,c,t} >= A_min * Y_{f,c,t}
                cqm.add_constraint(
                    A[(farm, crop, t)] - A_min * Y[(farm, crop, t)] >= 0,
                    label=f"Min_Area_If_Selected_{farm}_{crop}_t{t}"
                )
                constraint_metadata['min_area_if_selected_per_period'][f"{farm}_{crop}_t{t}"] = {
                    'type': 'min_area_if_selected',
                    'farm': farm,
                    'crop': crop,
                    'period': t,
                    'min_area': A_min
                }
                
                # Max area if selected: A_{f,c,t} <= land_f * Y_{f,c,t}
                cqm.add_constraint(
                    A[(farm, crop, t)] - land_availability[farm] * Y[(farm, crop, t)] <= 0,
                    label=f"Max_Area_If_Selected_{farm}_{crop}_t{t}"
                )
                constraint_metadata['max_area_if_selected_per_period'][f"{farm}_{crop}_t{t}"] = {
                    'type': 'max_area_if_selected',
                    'farm': farm,
                    'crop': crop,
                    'period': t,
                    'max_land': land_availability[farm]
                }
                pbar.update(2)
    
    # Constraint 3: Food group constraints per period
    pbar.set_description("Adding food group constraints (per period)")
    if food_group_constraints:
        for t in range(1, n_periods + 1):
            for group, constraints in food_group_constraints.items():
                foods_in_group = food_groups.get(group, [])
                if foods_in_group:
                    # Minimum foods from group in this period (global across all farms)
                    if 'min_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, crop, t)] for farm in farms for crop in foods_in_group) - constraints['min_foods'] >= 0,
                            label=f"Food_Group_Min_{group}_t{t}"
                        )
                        constraint_metadata['food_group_min_per_period'][f"{group}_t{t}"] = {
                            'type': 'food_group_min_global',
                            'group': group,
                            'period': t,
                            'min_foods': constraints['min_foods'],
                            'foods_in_group': foods_in_group,
                            'scope': 'global'
                        }
                        pbar.update(1)
                    
                    # Maximum foods from group in this period (global across all farms)
                    if 'max_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, crop, t)] for farm in farms for crop in foods_in_group) - constraints['max_foods'] <= 0,
                            label=f"Food_Group_Max_{group}_t{t}"
                        )
                        constraint_metadata['food_group_max_per_period'][f"{group}_t{t}"] = {
                            'type': 'food_group_max_global',
                            'group': group,
                            'period': t,
                            'max_foods': constraints['max_foods'],
                            'foods_in_group': foods_in_group,
                            'scope': 'global'
                        }
                        pbar.update(1)
    
    pbar.set_description("3-period rotation CQM (farm) complete")
    pbar.close()
    
    return cqm, (A, Y), constraint_metadata


def create_cqm_plots(farms, foods, food_groups, config):
    """
    Creates a CQM for the BINARY food optimization problem.
    
    This function supports land representation method:
    1. Even Grid: land_availability[farm] represents area per patch (all equal)
    
    For crops with minimum/maximum planting areas, the constraints are converted:
    - min_plots_for_crop_c = ceil(min_planting_area[c] / plot_area)
    - max_plots_for_crop_c = floor(max_planting_area[c] / plot_area)
    
    Args:
        farms: List of patch names
        foods: Dictionary of food data
        food_groups: Dictionary of food group mappings  
        config: Configuration including land_availability and generation method
    
    Returns:
        Tuple of (CQM, variables_dict, constraint_metadata)
    """
    cqm = ConstrainedQuadraticModel()
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    max_planting_area = params.get('maximum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    n_farms = len(farms)
    n_foods = len(foods)
    n_food_groups = len(food_groups) if food_group_constraints else 0
    
    # Calculate plot area (assuming even grid, all plots have same area)
    plot_area = list(land_availability.values())[0] if farms else 1.0
    
    # Count crops with min/max constraints for progress bar
    n_crops_with_min = len([c for c in foods if c in min_planting_area and min_planting_area[c] > 0])
    n_crops_with_max = len([c for c in foods if c in max_planting_area])
    
    # Calculate total operations for progress bar
    total_ops = (
        n_farms * n_foods +       # Binary variables only (Y)
        n_farms * n_foods +       # Objective terms
        n_farms +                 # Land availability constraints
        n_crops_with_min +        # Minimum plot constraints per crop
        n_crops_with_max +        # Maximum plot constraints per crop
        n_farms * n_food_groups * 2  # Food group constraints (min and max)
    )
    
    pbar = tqdm(total=total_ops, desc="Building CQM (Binary formulation)", unit="op", ncols=100)
    
    # Define variables - ONLY BINARY (each plantation is 1 acre if selected, 0 otherwise)
    Y = {}
    
    pbar.set_description("Creating binary variables")
    for farm in farms:
        for food in foods:
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
            pbar.update(1)
    
    # Objective function - each selected crop contributes its area-weighted value
    pbar.set_description("Building objective")
    total_land_area = sum(land_availability.values())  # Normalization factor
    
    objective = 0
    for farm in farms:
        farm_area = land_availability[farm]  # Area of this farm/plot
        for food in foods:
            # Y is binary: 1 if crop assigned to this unit, 0 if not
            # Each assignment contributes its area-weighted value
            area_weighted_value = farm_area * (
                weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[food].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
            )
            objective += area_weighted_value * Y[(farm, food)]
            pbar.update(1)
    
    # Normalize by total land area to make objectives comparable
    objective = objective / total_land_area
    cqm.set_objective(-objective)
    
    # Constraint metadata
    constraint_metadata = {
        'plantation_limit': {},
        'min_plots_per_crop': {},
        'max_plots_per_crop': {},
        'food_group_min': {},
        'food_group_max': {}
    }
    
    # Land unit constraints - each unit can be assigned to at most one crop
    # The area of each unit affects the objective contribution
    pbar.set_description("Adding land unit constraints")
    
    for farm in farms:
        # Each unit can have exactly one crop assigned (or remain idle)
        cqm.add_constraint(
            sum(Y[(farm, food)] for food in foods) - 1 <= 0,
            label=f"Max_Assignment_{farm}"
        )
        constraint_metadata['plantation_limit'][farm] = {
            'type': 'land_unit_assignment',
            'farm': farm,
            'area_ha': land_availability[farm]
        }
        pbar.update(1)
    
    # Minimum plots per crop constraints
    # If a crop requires minimum area, convert to minimum number of plots
    pbar.set_description("Adding minimum plot constraints")
    import math
    for food in foods:
        if food in min_planting_area and min_planting_area[food] > 0:
            min_plots = math.ceil(min_planting_area[food] / plot_area)
            # If crop is planted anywhere, it must be planted on at least min_plots
            cqm.add_constraint(
                sum(Y[(farm, food)] for farm in farms) >= min_plots,
                label=f"Min_Plots_{food}"
            )
            constraint_metadata['min_plots_per_crop'][food] = {
                'type': 'min_plots_per_crop',
                'food': food,
                'min_area_ha': min_planting_area[food],
                'plot_area_ha': plot_area,
                'min_plots': min_plots
            }
            pbar.update(1)
    
    # Maximum plots per crop constraints
    # If a crop has maximum area, convert to maximum number of plots
    pbar.set_description("Adding maximum plot constraints")
    for food in foods:
        if food in max_planting_area:
            max_plots = math.floor(max_planting_area[food] / plot_area)
            cqm.add_constraint(
                sum(Y[(farm, food)] for farm in farms) <= max_plots,
                label=f"Max_Plots_{food}"
            )
            constraint_metadata['max_plots_per_crop'][food] = {
                'type': 'max_plots_per_crop',
                'food': food,
                'max_area_ha': max_planting_area[food],
                'plot_area_ha': plot_area,
                'max_plots': max_plots
            }
            pbar.update(1)
    
    # Food group constraints - GLOBAL across all plots
    # Fixed: Was incorrectly applying per-plot, which created infeasible problems
    # Now applies globally: "across all plots, we need at least min_foods from each group"
    pbar.set_description("Adding food group constraints")
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                # Global minimum: across ALL plots, at least min_foods from this group
                if 'min_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) - constraints['min_foods'] >= 0,
                        label=f"Food_Group_Min_{group}_Global"
                    )
                    constraint_metadata['food_group_min'][group] = {
                        'type': 'food_group_min_global',
                        'group': group,
                        'min_foods': constraints['min_foods'],
                        'foods_in_group': foods_in_group,
                        'scope': 'global'
                    }
                    pbar.update(1)
                
                # Global maximum: across ALL plots, at most max_foods from this group
                if 'max_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) - constraints['max_foods'] <= 0,
                        label=f"Food_Group_Max_{group}_Global"
                    )
                    constraint_metadata['food_group_max'][group] = {
                        'type': 'food_group_max_global',
                        'group': group,
                        'max_foods': constraints['max_foods'],
                        'foods_in_group': foods_in_group,
                        'scope': 'global'
                    }
                    pbar.update(1)
    
    pbar.set_description("CQM complete")
    pbar.close()
    
    return cqm, Y, constraint_metadata


def create_cqm_plots_rotation_3period(farms, foods, food_groups, config, gamma=None):
    """
    Creates a CQM for the 3-period crop rotation optimization problem (PLOTS/BINARY formulation).
    
    Implements the binary formulation from crop_rotation.tex:
    - Time periods: t ∈ {1, 2, 3}
    - Variables: Y_{p,c,t} ∈ {0,1} for plot p, crop c, period t
    - Objective: Maximize area-normalized sum of:
      * Linear crop values across all periods
      * Quadratic rotation synergy between consecutive periods (t-1 to t)
    - Constraints (per period):
      * Each plot assigned to at most one crop per period
      * Food group min/max constraints per period
      * Minimum/maximum plot constraints per crop per period
    
    Args:
        farms: List of plot names
        foods: Dictionary of food data
        food_groups: Dictionary of food group mappings
        config: Configuration including land_availability and parameters
        gamma: Rotation synergy weight (default: ROTATION_GAMMA global)
    
    Returns:
        Tuple of (CQM, variables_dict, constraint_metadata)
    """
    # Load rotation matrix
    rotation_matrix = load_rotation_matrix()
    if rotation_matrix is None:
        raise FileNotFoundError("Rotation matrix not found. Run rotation_matrix.py first.")
    
    if gamma is None:
        gamma = ROTATION_GAMMA
    
    cqm = ConstrainedQuadraticModel()
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    max_planting_area = params.get('maximum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    n_plots = len(farms)
    n_crops = len(foods)
    n_periods = 3
    n_food_groups = len(food_groups) if food_group_constraints else 0
    
    # Calculate plot area (assuming even grid)
    plot_area = list(land_availability.values())[0] if farms else 1.0
    total_land_area = sum(land_availability.values())
    
    # Map crop names to indices for rotation matrix lookup
    crop_list = list(foods.keys())
    
    # Count operations for progress bar
    n_crops_with_min = len([c for c in foods if c in min_planting_area and min_planting_area[c] > 0])
    n_crops_with_max = len([c for c in foods if c in max_planting_area])
    
    total_ops = (
        n_plots * n_crops * n_periods +     # Binary variables
        n_plots * n_crops * n_periods +     # Linear objective terms
        n_plots * n_crops * n_crops * 2 +   # Quadratic rotation terms (periods 2 and 3)
        n_plots * n_periods +               # Plot assignment constraints (per period)
        n_crops_with_min * n_periods +      # Min plot constraints (per crop per period)
        n_crops_with_max * n_periods +      # Max plot constraints (per crop per period)
        n_food_groups * 2 * n_periods       # Food group constraints (min/max per period)
    )
    
    pbar = tqdm(total=total_ops, desc="Building 3-period rotation CQM", unit="op", ncols=100)
    
    # Define time-indexed binary variables Y_{p,c,t}
    Y = {}
    pbar.set_description("Creating time-indexed binary variables")
    for plot in farms:
        for crop in foods:
            for t in range(1, n_periods + 1):
                Y[(plot, crop, t)] = Binary(f"Y_{plot}_{crop}_t{t}")
                pbar.update(1)
    
    # Build objective function
    pbar.set_description("Building objective - linear terms")
    objective = 0
    
    # Part 1: Linear crop value terms across all periods
    for t in range(1, n_periods + 1):
        for plot in farms:
            plot_area_val = land_availability[plot]
            for crop in foods:
                # Calculate B_c: weighted benefit per unit area
                B_c = (
                    weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                    weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                    weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
                )
                # Add normalized term: (a_p * B_c * Y_{p,c,t}) / A_tot
                objective += plot_area_val * B_c * Y[(plot, crop, t)]
                pbar.update(1)
    
    # Part 2: Quadratic rotation synergy terms between consecutive periods
    pbar.set_description("Building objective - rotation synergy terms")
    for t in range(2, n_periods + 1):  # t = 2, 3 (comparing to t-1)
        for plot in farms:
            plot_area_val = land_availability[plot]
            for crop_prev in crop_list:
                for crop_curr in crop_list:
                    # Get rotation synergy R_{c,c'} from matrix
                    try:
                        R_cc = rotation_matrix.loc[crop_prev, crop_curr]
                    except KeyError:
                        # If crop not in rotation matrix, use 0
                        R_cc = 0.0
                    
                    # Add quadratic term: gamma * a_p * R_{c,c'} * Y_{p,c,t-1} * Y_{p,c',t} / A_tot
                    if R_cc != 0:
                        objective += (gamma * plot_area_val * R_cc * 
                                    Y[(plot, crop_prev, t-1)] * Y[(plot, crop_curr, t)]) / total_land_area
                pbar.update(1)
    
    # Set objective (CQM minimizes, so negate for maximization)
    objective = objective / total_land_area  # Extra normalization as in original
    cqm.set_objective(-objective)
    
    # Constraint metadata
    constraint_metadata = {
        'plot_assignment_per_period': {},
        'min_plots_per_crop_per_period': {},
        'max_plots_per_crop_per_period': {},
        'food_group_min_per_period': {},
        'food_group_max_per_period': {}
    }
    
    # Constraint 1: Plot single assignment per period
    # Each plot can be assigned to at most one crop per period
    pbar.set_description("Adding plot assignment constraints (per period)")
    for t in range(1, n_periods + 1):
        for plot in farms:
            cqm.add_constraint(
                sum(Y[(plot, crop, t)] for crop in foods) - 1 <= 0,
                label=f"Plot_Assignment_{plot}_t{t}"
            )
            constraint_metadata['plot_assignment_per_period'][f"{plot}_t{t}"] = {
                'type': 'plot_assignment',
                'plot': plot,
                'period': t,
                'area_ha': land_availability[plot]
            }
            pbar.update(1)
    
    # Constraint 2: Minimum plots per crop per period
    pbar.set_description("Adding minimum plot constraints (per crop per period)")
    for t in range(1, n_periods + 1):
        for crop in foods:
            if crop in min_planting_area and min_planting_area[crop] > 0:
                min_plots = math.ceil(min_planting_area[crop] / plot_area)
                cqm.add_constraint(
                    sum(Y[(plot, crop, t)] for plot in farms) - min_plots >= 0,
                    label=f"Min_Plots_{crop}_t{t}"
                )
                constraint_metadata['min_plots_per_crop_per_period'][f"{crop}_t{t}"] = {
                    'type': 'min_plots_per_crop',
                    'crop': crop,
                    'period': t,
                    'min_area_ha': min_planting_area[crop],
                    'plot_area_ha': plot_area,
                    'min_plots': min_plots
                }
                pbar.update(1)
    
    # Constraint 3: Maximum plots per crop per period
    pbar.set_description("Adding maximum plot constraints (per crop per period)")
    for t in range(1, n_periods + 1):
        for crop in foods:
            if crop in max_planting_area:
                max_plots = math.floor(max_planting_area[crop] / plot_area)
                cqm.add_constraint(
                    sum(Y[(plot, crop, t)] for plot in farms) - max_plots <= 0,
                    label=f"Max_Plots_{crop}_t{t}"
                )
                constraint_metadata['max_plots_per_crop_per_period'][f"{crop}_t{t}"] = {
                    'type': 'max_plots_per_crop',
                    'crop': crop,
                    'period': t,
                    'max_area_ha': max_planting_area[crop],
                    'plot_area_ha': plot_area,
                    'max_plots': max_plots
                }
                pbar.update(1)
    
    # Constraint 4: Food group constraints per period
    pbar.set_description("Adding food group constraints (per period)")
    if food_group_constraints:
        for t in range(1, n_periods + 1):
            for group, constraints in food_group_constraints.items():
                foods_in_group = food_groups.get(group, [])
                if foods_in_group:
                    # Minimum foods from group in this period
                    if 'min_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(plot, crop, t)] for plot in farms for crop in foods_in_group) - constraints['min_foods'] >= 0,
                            label=f"Food_Group_Min_{group}_t{t}"
                        )
                        constraint_metadata['food_group_min_per_period'][f"{group}_t{t}"] = {
                            'type': 'food_group_min',
                            'group': group,
                            'period': t,
                            'min_foods': constraints['min_foods'],
                            'foods_in_group': foods_in_group
                        }
                        pbar.update(1)
                    
                    # Maximum foods from group in this period
                    if 'max_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(plot, crop, t)] for plot in farms for crop in foods_in_group) - constraints['max_foods'] <= 0,
                            label=f"Food_Group_Max_{group}_t{t}"
                        )
                        constraint_metadata['food_group_max_per_period'][f"{group}_t{t}"] = {
                            'type': 'food_group_max',
                            'group': group,
                            'period': t,
                            'max_foods': constraints['max_foods'],
                            'foods_in_group': foods_in_group
                        }
                        pbar.update(1)
    
    pbar.set_description("3-period rotation CQM complete")
    pbar.close()
    
    return cqm, Y, constraint_metadata


def calculate_rotation_objective(solution, farms, foods, land_availability, weights, gamma=None):
    """
    Calculate the 3-period rotation objective from a solution.
    
    Computes both linear crop value terms and quadratic rotation synergy.
    
    Args:
        solution: Dictionary with variable assignments (Y_{plot}_{crop}_t{t})
        farms: List of plot names
        foods: Dictionary of food data
        land_availability: Dictionary mapping plot to area
        weights: Dictionary of objective weights
        gamma: Rotation synergy weight (default: ROTATION_GAMMA)
        
    Returns:
        dict with 'total', 'linear_value', and 'rotation_synergy' components
    """
    if gamma is None:
        gamma = ROTATION_GAMMA
    
    rotation_matrix = load_rotation_matrix()
    total_land = sum(land_availability.values())
    n_periods = 3
    
    linear_value = 0.0
    rotation_synergy = 0.0
    
    # Part 1: Linear crop values across all periods
    for t in range(1, n_periods + 1):
        for plot in farms:
            s_p = land_availability[plot]
            for crop in foods:
                var_name = f"Y_{plot}_{crop}_t{t}"
                y_pct = solution.get(var_name, 0)
                
                if y_pct > 0:
                    B_c = (
                        weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                        weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                        weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                        weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                        weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
                    )
                    linear_value += B_c * s_p * y_pct / total_land
    
    # Part 2: Rotation synergy between consecutive periods
    if rotation_matrix is not None:
        crop_list = list(foods.keys())
        for t in range(2, n_periods + 1):
            for plot in farms:
                s_p = land_availability[plot]
                for crop_prev in crop_list:
                    var_prev = f"Y_{plot}_{crop_prev}_t{t-1}"
                    y_prev = solution.get(var_prev, 0)
                    
                    if y_prev > 0:
                        for crop_curr in crop_list:
                            var_curr = f"Y_{plot}_{crop_curr}_t{t}"
                            y_curr = solution.get(var_curr, 0)
                            
                            if y_curr > 0:
                                try:
                                    R_cc = rotation_matrix.loc[crop_prev, crop_curr]
                                    rotation_synergy += gamma * s_p * R_cc * y_prev * y_curr / total_land
                                except KeyError:
                                    pass
    
    total_objective = linear_value + rotation_synergy
    
    return {
        'total': total_objective,
        'linear_value': linear_value,
        'rotation_synergy': rotation_synergy
    }


def extract_rotation_solution_summary(solution, farms, foods, land_availability):
    """
    Extract a summary of the 3-period rotation solution.
    
    Args:
        solution: Dictionary with variable assignments (Y_{plot}_{crop}_t{t})
        farms: List of plot names
        foods: Dictionary of food data
        land_availability: Dictionary mapping plot to area
        
    Returns:
        dict: Summary with per-period crop selections and plot assignments
    """
    n_periods = 3
    periods_summary = []
    
    for t in range(1, n_periods + 1):
        crops_selected = set()
        plot_assignments = []
        total_allocated = 0.0
        
        for crop in foods:
            total_area = 0.0
            assigned_plots = []
            
            for plot in farms:
                var_name = f"Y_{plot}_{crop}_t{t}"
                if solution.get(var_name, 0) > 0:
                    area = land_availability[plot]
                    total_area += area
                    assigned_plots.append({'plot': plot, 'area': area})
            
            if total_area > 0:
                crops_selected.add(crop)
                plot_assignments.append({
                    'crop': crop,
                    'total_area': total_area,
                    'n_plots': len(assigned_plots),
                    'plots': assigned_plots
                })
                total_allocated += total_area
        
        total_available = sum(land_availability.values())
        idle_area = total_available - total_allocated
        
        periods_summary.append({
            'period': t,
            'crops_selected': list(crops_selected),
            'n_crops': len(crops_selected),
            'plot_assignments': plot_assignments,
            'total_allocated': total_allocated,
            'total_available': total_available,
            'idle_area': idle_area,
            'utilization': total_allocated / total_available if total_available > 0 else 0
        })
    
    return {
        'n_periods': n_periods,
        'periods': periods_summary,
        'total_land': sum(land_availability.values())
    }


def solve_with_pulp_farm(farms, foods, food_groups, config):
    """Solve with PuLP and return model and results."""
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})

    A_pulp = pl.LpVariable.dicts(
        "Area", [(f, c) for f in farms for c in foods], lowBound=0)
    Y_pulp = pl.LpVariable.dicts(
        "Choose", [(f, c) for f in farms for c in foods], cat='Binary')

    total_area = sum(land_availability[f] for f in farms)

    goal = (
        weights.get('nutritional_value', 0) * pl.lpSum([(foods[c].get('nutritional_value', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('nutrient_density', 0) * pl.lpSum([(foods[c].get('nutrient_density', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area -
        weights.get('environmental_impact', 0) * pl.lpSum([(foods[c].get('environmental_impact', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('affordability', 0) * pl.lpSum([(foods[c].get('affordability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('sustainability', 0) * pl.lpSum([(foods[c].get(
            'sustainability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area
    )

    model = pl.LpProblem("Food_Optimization", pl.LpMaximize)

    for f in farms:
        model += pl.lpSum([A_pulp[(f, c)] for c in foods]
                          ) <= land_availability[f], f"Max_Area_{f}"

    for f in farms:
        for c in foods:
            A_min = min_planting_area.get(c, 0)
            model += A_pulp[(f, c)] >= A_min * \
                Y_pulp[(f, c)], f"MinArea_{f}_{c}"
            model += A_pulp[(f, c)] <= land_availability[f] * \
                Y_pulp[(f, c)], f"MaxArea_{f}_{c}"

    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                # Global constraints: across ALL farms
                if 'min_foods' in constraints:
                    model += pl.lpSum([Y_pulp[(f, c)] for f in farms for c in foods_in_group]
                                      ) >= constraints['min_foods'], f"MinFoodGroup_Global_{g}"
                if 'max_foods' in constraints:
                    model += pl.lpSum([Y_pulp[(f, c)] for f in farms for c in foods_in_group]
                                      ) <= constraints['max_foods'], f"MaxFoodGroup_Global_{g}"

    model += goal, "Objective"

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
        'areas': {},
        'selections': {}
    }

    for f in farms:
        for c in foods:
            key = f"{f}_{c}"
            results['areas'][key] = A_pulp[(f, c)].value(
            ) if A_pulp[(f, c)].value() is not None else 0.0
            results['selections'][key] = Y_pulp[(f, c)].value(
            ) if Y_pulp[(f, c)].value() is not None else 0.0

    return model, results


def solve_with_pulp_farm_rotation(farms, foods, food_groups, config, gamma=None):
    """
    Solve 3-period rotation problem with PuLP using FARM formulation (continuous areas).
    
    Uses McCormick relaxation to linearize the quadratic rotation synergy terms.
    For each Y_{f,c,t-1} * Y_{f,c',t} product, creates auxiliary binary variable Z_{f,c,c',t}.
    
    Implements time-indexed variables: A_{f,c,t} and Y_{f,c,t} for farm f, crop c, period t ∈ {1,2,3}
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food group mappings
        config: Configuration including land_availability and parameters
        gamma: Rotation synergy weight (default: ROTATION_GAMMA)
    
    Returns:
        Tuple of (model, results)
    """
    print("  NOTE: PuLP using McCormick relaxation to linearize rotation synergy")
    print("        Quadratic terms Y_{t-1} * Y_t converted to auxiliary variables Z")
    
    if gamma is None:
        gamma = ROTATION_GAMMA
    
    # Load rotation matrix
    rotation_matrix = load_rotation_matrix()
    if rotation_matrix is None:
        print("  WARNING: Rotation matrix not found. Proceeding with linear objective only.")
        rotation_matrix = None
    
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    n_periods = 3
    total_land_area = sum(land_availability.values())
    crop_list = list(foods.keys())
    
    # Create time-indexed continuous and binary variables
    A_pulp = {}
    Y_pulp = {}
    
    for farm in farms:
        for crop in foods:
            for t in range(1, n_periods + 1):
                var_name_a = f"A_{farm}_{crop}_t{t}"
                var_name_y = f"Y_{farm}_{crop}_t{t}"
                A_pulp[var_name_a] = pl.LpVariable(var_name_a, lowBound=0, upBound=land_availability[farm])
                Y_pulp[var_name_y] = pl.LpVariable(var_name_y, cat='Binary')
    
    # Create auxiliary binary variables Z for linearizing rotation synergy
    # Z_{f,c_prev,c_curr,t} represents Y_{f,c_prev,t-1} * Y_{f,c_curr,t}
    Z_pulp = {}
    rotation_pairs = []
    
    if rotation_matrix is not None:
        for t in range(2, n_periods + 1):  # t = 2, 3
            for farm in farms:
                for crop_prev in crop_list:
                    for crop_curr in crop_list:
                        try:
                            R_cc = rotation_matrix.loc[crop_prev, crop_curr]
                            if abs(R_cc) > 1e-6:  # Only create Z if rotation benefit exists
                                var_name_z = f"Z_{farm}_{crop_prev}_{crop_curr}_t{t}"
                                Z_pulp[var_name_z] = pl.LpVariable(var_name_z, cat='Binary')
                                rotation_pairs.append((farm, crop_prev, crop_curr, t, R_cc))
                        except KeyError:
                            continue  # Crop not in rotation matrix
    
    print(f"  Created {len(rotation_pairs)} rotation synergy pairs to linearize")
    
    # Build objective function
    goal = 0
    
    # Part 1: Linear crop value terms
    for t in range(1, n_periods + 1):
        for farm in farms:
            for crop in foods:
                var_name_a = f"A_{farm}_{crop}_t{t}"
                B_c = (
                    weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                    weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                    weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
                )
                goal += B_c * A_pulp[var_name_a]
    
    # Part 2: Linearized quadratic rotation synergy
    # Use Z variables instead of Y_{t-1} * Y_t products
    if rotation_pairs:
        synergy_terms = []
        for farm, crop_prev, crop_curr, t, R_cc in rotation_pairs:
            var_name_z = f"Z_{farm}_{crop_prev}_{crop_curr}_t{t}"
            # Synergy contribution: gamma * R_{c,c'} * Z (Z approximates Y_{t-1} * Y_t)
            synergy_terms.append(gamma * R_cc * Z_pulp[var_name_z])
        goal += pl.lpSum(synergy_terms)
    
    # Normalize by total land area
    goal = goal / total_land_area
    
    model = pl.LpProblem("Rotation_3Period_Farm_Linearized", pl.LpMaximize)
    model += goal, "Objective"
    
    # McCormick relaxation constraints for Z_{f,c_prev,c_curr,t} = Y_{f,c_prev,t-1} * Y_{f,c_curr,t}
    # Constraints: Z <= Y_{t-1}, Z <= Y_t, Z >= Y_{t-1} + Y_t - 1
    if rotation_pairs:
        for farm, crop_prev, crop_curr, t, R_cc in rotation_pairs:
            var_name_z = f"Z_{farm}_{crop_prev}_{crop_curr}_t{t}"
            var_name_y_prev = f"Y_{farm}_{crop_prev}_t{t-1}"
            var_name_y_curr = f"Y_{farm}_{crop_curr}_t{t}"
            
            model += Z_pulp[var_name_z] <= Y_pulp[var_name_y_prev], \
                    f"Z_upper1_{farm}_{crop_prev}_{crop_curr}_t{t}"
            model += Z_pulp[var_name_z] <= Y_pulp[var_name_y_curr], \
                    f"Z_upper2_{farm}_{crop_prev}_{crop_curr}_t{t}"
            model += Z_pulp[var_name_z] >= Y_pulp[var_name_y_prev] + Y_pulp[var_name_y_curr] - 1, \
                    f"Z_lower_{farm}_{crop_prev}_{crop_curr}_t{t}"
    
    # Constraint 1: Land availability per farm per period
    for t in range(1, n_periods + 1):
        for farm in farms:
            area_vars = [A_pulp[f"A_{farm}_{crop}_t{t}"] for crop in foods]
            model += pl.lpSum(area_vars) <= land_availability[farm], f"Land_{farm}_Period_{t}"
    
    # Constraint 2: Linking constraints (min and max area if selected)
    for t in range(1, n_periods + 1):
        for farm in farms:
            for crop in foods:
                var_name_a = f"A_{farm}_{crop}_t{t}"
                var_name_y = f"Y_{farm}_{crop}_t{t}"
                A_min = min_planting_area.get(crop, 0)
                
                # Min area if selected
                model += A_pulp[var_name_a] >= A_min * Y_pulp[var_name_y], \
                        f"MinArea_{farm}_{crop}_Period_{t}"
                
                # Max area if selected
                model += A_pulp[var_name_a] <= land_availability[farm] * Y_pulp[var_name_y], \
                        f"MaxArea_{farm}_{crop}_Period_{t}"
    
    # Constraint 3: Food group constraints per period (if specified)
    if food_group_constraints:
        for t in range(1, n_periods + 1):
            for group, constraints in food_group_constraints.items():
                foods_in_group = food_groups.get(group, [])
                if foods_in_group:
                    group_vars = [Y_pulp[f"Y_{farm}_{crop}_t{t}"] 
                                  for farm in farms for crop in foods_in_group]
                    if 'min_foods' in constraints:
                        model += pl.lpSum(group_vars) >= constraints['min_foods'], \
                                f"FoodGroup_{group}_Period_{t}_Min"
                    if 'max_foods' in constraints:
                        model += pl.lpSum(group_vars) <= constraints['max_foods'], \
                                f"FoodGroup_{group}_Period_{t}_Max"
    
    # Solve with Gurobi
    print("  Solving 3-period farm rotation with PuLP/Gurobi (linearized)...")
    start_time = time.time()
    gurobi_options = [
        ('Method', 2),           # Barrier method
        ('Crossover', 0),        # Disable crossover
        ('BarHomogeneous', 1),   # Homogeneous barrier
        ('Threads', 0),          # Use all threads
        ('MIPFocus', 1),         # Focus on solutions
        ('Presolve', 2),         # Aggressive presolve
    ]
    
    try:
        solver = pl.GUROBI(msg=1, timeLimit=300)
        for param, value in gurobi_options:
            solver.optionsDict[param] = value
        model.solve(solver)
    except Exception as e:
        print(f"  Gurobi failed, trying default solver: {e}")
        model.solve()
    
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': pl.value(model.objective),
        'solve_time': solve_time,
        'areas': {},
        'selections': {},
        'rotation_pairs': {}  # Store Z values for analysis
    }
    
    # Extract solution values
    for var_name, var in A_pulp.items():
        results['areas'][var_name] = var.value() if var.value() is not None else 0.0
    
    for var_name, var in Y_pulp.items():
        results['selections'][var_name] = var.value() if var.value() is not None else 0.0
    
    for var_name, var in Z_pulp.items():
        results['rotation_pairs'][var_name] = var.value() if var.value() is not None else 0.0
    
    print(f"  PuLP/Gurobi: {results['status']} in {solve_time:.3f}s")
    if results['objective_value'] is not None:
        print(f"  Objective (with linearized rotation): {results['objective_value']:.6f}")
    
    return model, results


def solve_with_pulp_plots(farms, foods, food_groups, config):
    """
    Solve with PuLP using BINARY formulation.
    
    Supports land generation methods:
    - even_grid: land_availability represents area per plot
    """
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    food_group_constraints = params.get('food_group_constraints', {})
    land_method = params.get('land_generation_method', 'even_grid')
    
    # Binary variables - each represents assignment of a crop to a land unit
    Y_pulp = pl.LpVariable.dicts("Assignment", [(f, c) for f in farms for c in foods], cat='Binary')
    
    total_land_area = sum(land_availability.values())
    
    # Objective: maximize area-weighted value of crop assignments
    goal = 0
    for f in farms:
        farm_area = land_availability[f]
        for c in foods:
            area_weighted_value = farm_area * (
                weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[c].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[c].get('sustainability', 0)
            )
            goal += area_weighted_value * Y_pulp[(f, c)]
    
    # Normalize by total land area
    goal = goal / total_land_area
    
    model = pl.LpProblem("Food_Optimization_Binary", pl.LpMaximize)
    
    # Land unit assignment: each unit can be assigned to at most one crop
    # The area of each unit affects the objective contribution
    for f in farms:
        # Each unit can have exactly one crop assigned (or remain idle)
        model += pl.lpSum([Y_pulp[(f, c)] for c in foods]) <= 1, f"Max_Assignment_{f}"
    
    # Food group constraints - GLOBAL across all plots
    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                # Global constraints: across ALL plots
                if 'min_foods' in constraints:
                    model += pl.lpSum([Y_pulp[(f, c)] for f in farms for c in foods_in_group]) >= constraints['min_foods'], f"MinFoodGroup_Global_{g}"
                if 'max_foods' in constraints:
                    model += pl.lpSum([Y_pulp[(f, c)] for f in farms for c in foods_in_group]) <= constraints['max_foods'], f"MaxFoodGroup_Global_{g}"
    
    model += goal, "Objective"
    
    start_time = time.time()
    gurobi_options = [
        ('Method', 2),           # Barrier method (GPU-accelerated)
        ('Crossover', 0),        # Disable crossover to keep computation on GPU
        ('BarHomogeneous', 1),   # Homogeneous barrier (more GPU-friendly)
        ('Threads', 0),          # Use all available CPU threads for parallelization
        ('MIPFocus', 1),         # Focus on finding good solutions quickly
        ('Presolve', 2),         # Aggressive presolve
    ]
    solver = pl.GUROBI(msg=0, timeLimit=300)
        # Set parameters directly on the solver
    for param, value in gurobi_options:
        solver.optionsDict[param] = value
    model.solve(solver)
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': pl.value(model.objective),
        'solve_time': solve_time,
        'plantations': {}  # Binary: 1 if planted, 0 if not
    }
    
    for f in farms:
        for c in foods:
            key = f"{f}_{c}"
            results['plantations'][key] = Y_pulp[(f, c)].value() if Y_pulp[(f, c)].value() is not None else 0.0
    
    return model, results


def solve_with_pulp_plots_rotation(farms, foods, food_groups, config, gamma=None):
    """
    Solve 3-period rotation problem with PuLP using PLOTS formulation (binary).
    
    Uses McCormick relaxation to linearize the quadratic rotation synergy terms.
    For each Y_{p,c,t-1} * Y_{p,c',t} product, creates auxiliary binary variable Z_{p,c,c',t}.
    
    Implements time-indexed binary variables: Y_{p,c,t} for plot p, crop c, period t ∈ {1,2,3}
    
    Args:
        farms: List of plot names
        foods: Dictionary of food data
        food_groups: Dictionary of food group mappings
        config: Configuration including land_availability and parameters
        gamma: Rotation synergy weight (default: ROTATION_GAMMA)
    
    Returns:
        Tuple of (model, results)
    """
    print("  NOTE: PuLP using McCormick relaxation to linearize rotation synergy")
    print("        Quadratic terms Y_{t-1} * Y_t converted to auxiliary variables Z")
    
    if gamma is None:
        gamma = ROTATION_GAMMA
    
    # Load rotation matrix
    rotation_matrix = load_rotation_matrix()
    if rotation_matrix is None:
        print("  WARNING: Rotation matrix not found. Proceeding with linear objective only.")
        rotation_matrix = None
    
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    food_group_constraints = params.get('food_group_constraints', {})
    
    n_periods = 3
    total_land_area = sum(land_availability.values())
    crop_list = list(foods.keys())
    
    # Create time-indexed binary variables Y_{p,c,t}
    Y_pulp = {}
    for plot in farms:
        for crop in foods:
            for t in range(1, n_periods + 1):
                var_name = f"Y_{plot}_{crop}_t{t}"
                Y_pulp[var_name] = pl.LpVariable(var_name, cat='Binary')
    
    # Create auxiliary binary variables Z for linearizing rotation synergy
    # Z_{p,c_prev,c_curr,t} represents Y_{p,c_prev,t-1} * Y_{p,c_curr,t}
    Z_pulp = {}
    rotation_pairs = []
    
    if rotation_matrix is not None:
        for t in range(2, n_periods + 1):  # t = 2, 3
            for plot in farms:
                plot_area = land_availability[plot]
                for crop_prev in crop_list:
                    for crop_curr in crop_list:
                        try:
                            R_cc = rotation_matrix.loc[crop_prev, crop_curr]
                            if abs(R_cc) > 1e-6:  # Only create Z if rotation benefit exists
                                var_name_z = f"Z_{plot}_{crop_prev}_{crop_curr}_t{t}"
                                Z_pulp[var_name_z] = pl.LpVariable(var_name_z, cat='Binary')
                                rotation_pairs.append((plot, crop_prev, crop_curr, t, R_cc, plot_area))
                        except KeyError:
                            continue  # Crop not in rotation matrix
    
    print(f"  Created {len(rotation_pairs)} rotation synergy pairs to linearize")
    
    # Build objective function
    goal = 0
    
    # Part 1: Linear crop value terms
    for t in range(1, n_periods + 1):
        for plot in farms:
            plot_area = land_availability[plot]
            for crop in foods:
                var_name = f"Y_{plot}_{crop}_t{t}"
                area_weighted_value = plot_area * (
                    weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                    weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                    weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
                )
                goal += area_weighted_value * Y_pulp[var_name]
    
    # Part 2: Linearized quadratic rotation synergy
    # Use Z variables instead of Y_{t-1} * Y_t products
    if rotation_pairs:
        synergy_terms = []
        for plot, crop_prev, crop_curr, t, R_cc, plot_area in rotation_pairs:
            var_name_z = f"Z_{plot}_{crop_prev}_{crop_curr}_t{t}"
            # Synergy contribution: gamma * plot_area * R_{c,c'} * Z
            synergy_terms.append(gamma * plot_area * R_cc * Z_pulp[var_name_z])
        goal += pl.lpSum(synergy_terms)
    
    # Normalize by total land area
    goal = goal / total_land_area
    
    model = pl.LpProblem("Rotation_3Period_Plots_Linearized", pl.LpMaximize)
    model += goal, "Objective"
    
    # McCormick relaxation constraints for Z_{p,c_prev,c_curr,t} = Y_{p,c_prev,t-1} * Y_{p,c_curr,t}
    # Constraints: Z <= Y_{t-1}, Z <= Y_t, Z >= Y_{t-1} + Y_t - 1
    if rotation_pairs:
        for plot, crop_prev, crop_curr, t, R_cc, plot_area in rotation_pairs:
            var_name_z = f"Z_{plot}_{crop_prev}_{crop_curr}_t{t}"
            var_name_y_prev = f"Y_{plot}_{crop_prev}_t{t-1}"
            var_name_y_curr = f"Y_{plot}_{crop_curr}_t{t}"
            
            model += Z_pulp[var_name_z] <= Y_pulp[var_name_y_prev], \
                    f"Z_upper1_{plot}_{crop_prev}_{crop_curr}_t{t}"
            model += Z_pulp[var_name_z] <= Y_pulp[var_name_y_curr], \
                    f"Z_upper2_{plot}_{crop_prev}_{crop_curr}_t{t}"
            model += Z_pulp[var_name_z] >= Y_pulp[var_name_y_prev] + Y_pulp[var_name_y_curr] - 1, \
                    f"Z_lower_{plot}_{crop_prev}_{crop_curr}_t{t}"
    
    # Constraint 1: Plot single assignment per period
    for t in range(1, n_periods + 1):
        for plot in farms:
            constraint_vars = [Y_pulp[f"Y_{plot}_{crop}_t{t}"] for crop in foods]
            model += pl.lpSum(constraint_vars) <= 1, f"Plot_{plot}_Period_{t}_SingleAssignment"
    
    # Constraint 2: Food group constraints per period (if specified)
    if food_group_constraints:
        for t in range(1, n_periods + 1):
            for group, constraints in food_group_constraints.items():
                foods_in_group = food_groups.get(group, [])
                if foods_in_group:
                    group_vars = [Y_pulp[f"Y_{plot}_{crop}_t{t}"] 
                                  for plot in farms for crop in foods_in_group]
                    if 'min_foods' in constraints:
                        model += pl.lpSum(group_vars) >= constraints['min_foods'], \
                                f"FoodGroup_{group}_Period_{t}_Min"
                    if 'max_foods' in constraints:
                        model += pl.lpSum(group_vars) <= constraints['max_foods'], \
                                f"FoodGroup_{group}_Period_{t}_Max"
    
    # Solve with Gurobi
    print("  Solving 3-period plots rotation with PuLP/Gurobi (linearized)...")
    start_time = time.time()
    gurobi_options = [
        ('Method', 2),           # Barrier method
        ('Crossover', 0),        # Disable crossover
        ('BarHomogeneous', 1),   # Homogeneous barrier
        ('Threads', 0),          # Use all threads
        ('MIPFocus', 1),         # Focus on solutions
        ('Presolve', 2),         # Aggressive presolve
    ]
    
    try:
        solver = pl.GUROBI(msg=1, timeLimit=300)
        for param, value in gurobi_options:
            solver.optionsDict[param] = value
        model.solve(solver)
    except Exception as e:
        print(f"  Gurobi failed, trying default solver: {e}")
        model.solve()
    
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': pl.value(model.objective),
        'solve_time': solve_time,
        'solution': {},
        'rotation_pairs': {}  # Store Z values for analysis
    }
    
    # Extract solution values
    for var_name, var in Y_pulp.items():
        results['solution'][var_name] = var.value() if var.value() is not None else 0.0
    
    for var_name, var in Z_pulp.items():
        results['rotation_pairs'][var_name] = var.value() if var.value() is not None else 0.0
    
    print(f"  PuLP/Gurobi: {results['status']} in {solve_time:.3f}s")
    if results['objective_value'] is not None:
        print(f"  Objective (with linearized rotation): {results['objective_value']:.6f}")
    
    return model, results
    




def solve_with_pyomo_farm_rotation(farms, foods, food_groups, config, gamma=None):
    """Solve 3-period rotation with Pyomo (farm formulation, native quadratic)."""
    print('  NOTE: Pyomo handles quadratic rotation terms natively (no linearization needed)')
    
    if gamma is None:
        gamma = ROTATION_GAMMA
    
    rotation_matrix = load_rotation_matrix()
    if rotation_matrix is None:
        print('  WARNING: Rotation matrix not found.')
        rotation_matrix = None
    
    try:
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory
    except ImportError:
        raise ImportError('Pyomo is required. Install with: pip install pyomo')
    
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    n_periods = 3
    total_land_area = sum(land_availability.values())
    crop_list = list(foods.keys())
    
    model = pyo.ConcreteModel()
    model.farms = pyo.Set(initialize=farms)
    model.crops = pyo.Set(initialize=crop_list)
    model.periods = pyo.Set(initialize=range(1, n_periods + 1))
    
    model.A = pyo.Var(model.farms, model.crops, model.periods, 
                      domain=pyo.NonNegativeReals, 
                      bounds=lambda m, f, c, t: (0, land_availability[f]))
    model.Y = pyo.Var(model.farms, model.crops, model.periods, domain=pyo.Binary)
    
    def objective_rule(m):
        obj = 0.0
        for t in m.periods:
            for f in m.farms:
                for c in m.crops:
                    B_c = (weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
                           weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
                           weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
                           weights.get('affordability', 0) * foods[c].get('affordability', 0) +
                           weights.get('sustainability', 0) * foods[c].get('sustainability', 0))
                    obj += B_c * m.A[f, c, t]
        if rotation_matrix is not None:
            for t in range(2, n_periods + 1):
                for f in m.farms:
                    for c_prev in m.crops:
                        for c_curr in m.crops:
                            try:
                                R_cc = rotation_matrix.loc[c_prev, c_curr]
                                if abs(R_cc) > 1e-6:
                                    obj += (gamma * R_cc * m.A[f, c_prev, t-1] * m.A[f, c_curr, t]) / total_land_area
                            except KeyError:
                                continue
        return obj / total_land_area
    
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    def land_constraint_rule(m, f, t):
        return sum(m.A[f, c, t] for c in m.crops) <= land_availability[f]
    model.land_constraint = pyo.Constraint(model.farms, model.periods, rule=land_constraint_rule)
    
    def min_area_rule(m, f, c, t):
        if c in min_planting_area:
            return m.A[f, c, t] >= min_planting_area[c] * m.Y[f, c, t]
        return pyo.Constraint.Skip
    model.min_area_constraint = pyo.Constraint(model.farms, model.crops, model.periods, rule=min_area_rule)
    
    def max_area_rule(m, f, c, t):
        return m.A[f, c, t] <= land_availability[f] * m.Y[f, c, t]
    model.max_area_constraint = pyo.Constraint(model.farms, model.crops, model.periods, rule=max_area_rule)
    
    if food_group_constraints:
        def food_group_min_rule(m, g, t):
            if g in food_groups:
                crops_in_group = food_groups[g]
                min_foods = food_group_constraints[g].get('min_foods', 0)
                if min_foods > 0:
                    return sum(m.Y[f, c, t] for f in m.farms for c in crops_in_group) >= min_foods
            return pyo.Constraint.Skip
        model.food_group_min = pyo.Constraint(food_groups.keys(), model.periods, rule=food_group_min_rule)
        
        def food_group_max_rule(m, g, t):
            if g in food_groups:
                crops_in_group = food_groups[g]
                max_foods = food_group_constraints[g].get('max_foods', len(crops_in_group))
                if max_foods < len(crops_in_group):
                    return sum(m.Y[f, c, t] for f in m.farms for c in crops_in_group) <= max_foods
            return pyo.Constraint.Skip
        model.food_group_max = pyo.Constraint(food_groups.keys(), model.periods, rule=food_group_max_rule)
    
    print('  Solving 3-period farm rotation with Pyomo/Gurobi (native quadratic)...')
    start_time = time.time()
    
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 300
    solver.options['NonConvex'] = 2
    
    results_obj = solver.solve(model, tee=False)
    solve_time = time.time() - start_time
    
    status_map = {
        pyo.TerminationCondition.optimal: 'Optimal',
        pyo.TerminationCondition.feasible: 'Feasible',
        pyo.TerminationCondition.infeasible: 'Infeasible',
        pyo.TerminationCondition.unbounded: 'Unbounded',
        pyo.TerminationCondition.maxTimeLimit: 'Time Limit',
        pyo.TerminationCondition.maxIterations: 'Max Iterations'
    }
    status = status_map.get(results_obj.solver.termination_condition, 'Unknown')
    
    results = {
        'status': status,
        'objective_value': pyo.value(model.objective) if status in ['Optimal', 'Feasible'] else None,
        'solve_time': solve_time,
        'solution': {},
        'areas': {},
        'selections': {}
    }
    
    if status in ['Optimal', 'Feasible']:
        for f in model.farms:
            for c in model.crops:
                for t in model.periods:
                    var_name_a = f'A_{f}_{c}_t{t}'
                    var_name_y = f'Y_{f}_{c}_t{t}'
                    results['areas'][var_name_a] = pyo.value(model.A[f, c, t])
                    results['selections'][var_name_y] = pyo.value(model.Y[f, c, t])
                    results['solution'][var_name_y] = pyo.value(model.Y[f, c, t])
    
    print(f'  Pyomo/Gurobi: {results["status"]} in {solve_time:.3f}s')
    if results['objective_value'] is not None:
        print(f'  Objective (native quadratic): {results["objective_value"]:.6f}')
    
    return model, results


def solve_with_pyomo_plots_rotation(farms, foods, food_groups, config, gamma=None):
    """Solve 3-period rotation with Pyomo (plots formulation, native quadratic)."""
    print('  NOTE: Pyomo handles quadratic rotation terms natively (no linearization needed)')
    
    if gamma is None:
        gamma = ROTATION_GAMMA
    
    rotation_matrix = load_rotation_matrix()
    if rotation_matrix is None:
        print('  WARNING: Rotation matrix not found.')
        rotation_matrix = None
    
    try:
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory
    except ImportError:
        raise ImportError('Pyomo is required. Install with: pip install pyomo')
    
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    food_group_constraints = params.get('food_group_constraints', {})
    
    n_periods = 3
    total_land_area = sum(land_availability.values())
    crop_list = list(foods.keys())
    
    model = pyo.ConcreteModel()
    model.plots = pyo.Set(initialize=farms)
    model.crops = pyo.Set(initialize=crop_list)
    model.periods = pyo.Set(initialize=range(1, n_periods + 1))
    model.Y = pyo.Var(model.plots, model.crops, model.periods, domain=pyo.Binary)
    
    def objective_rule(m):
        obj = 0.0
        for t in m.periods:
            for p in m.plots:
                for c in m.crops:
                    B_c = (weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
                           weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
                           weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
                           weights.get('affordability', 0) * foods[c].get('affordability', 0) +
                           weights.get('sustainability', 0) * foods[c].get('sustainability', 0))
                    obj += land_availability[p] * B_c * m.Y[p, c, t]
        if rotation_matrix is not None:
            for t in range(2, n_periods + 1):
                for p in m.plots:
                    for c_prev in m.crops:
                        for c_curr in m.crops:
                            try:
                                R_cc = rotation_matrix.loc[c_prev, c_curr]
                                if abs(R_cc) > 1e-6:
                                    obj += (gamma * (land_availability[p]**2) * R_cc * m.Y[p, c_prev, t-1] * m.Y[p, c_curr, t]) / total_land_area
                            except KeyError:
                                continue
        return obj / total_land_area
    
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    def plot_assignment_rule(m, p, t):
        return sum(m.Y[p, c, t] for c in m.crops) <= 1
    model.plot_assignment = pyo.Constraint(model.plots, model.periods, rule=plot_assignment_rule)
    
    if food_group_constraints:
        def food_group_min_rule(m, g, t):
            if g in food_groups:
                crops_in_group = food_groups[g]
                min_foods = food_group_constraints[g].get('min_foods', 0)
                if min_foods > 0:
                    return sum(m.Y[p, c, t] for p in m.plots for c in crops_in_group) >= min_foods
            return pyo.Constraint.Skip
        model.food_group_min = pyo.Constraint(food_groups.keys(), model.periods, rule=food_group_min_rule)
        
        def food_group_max_rule(m, g, t):
            if g in food_groups:
                crops_in_group = food_groups[g]
                max_foods = food_group_constraints[g].get('max_foods', len(crops_in_group))
                if max_foods < len(crops_in_group):
                    return sum(m.Y[p, c, t] for p in m.plots for c in crops_in_group) <= max_foods
            return pyo.Constraint.Skip
        model.food_group_max = pyo.Constraint(food_groups.keys(), model.periods, rule=food_group_max_rule)
    
    print('  Solving 3-period plots rotation with Pyomo/Gurobi (native quadratic)...')
    start_time = time.time()
    
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 300
    solver.options['NonConvex'] = 2
    
    results_obj = solver.solve(model, tee=False)
    solve_time = time.time() - start_time
    
    status_map = {
        pyo.TerminationCondition.optimal: 'Optimal',
        pyo.TerminationCondition.feasible: 'Feasible',
        pyo.TerminationCondition.infeasible: 'Infeasible',
        pyo.TerminationCondition.unbounded: 'Unbounded',
        pyo.TerminationCondition.maxTimeLimit: 'Time Limit',
        pyo.TerminationCondition.maxIterations: 'Max Iterations'
    }
    status = status_map.get(results_obj.solver.termination_condition, 'Unknown')
    
    results = {
        'status': status,
        'objective_value': pyo.value(model.objective) if status in ['Optimal', 'Feasible'] else None,
        'solve_time': solve_time,
        'solution': {}
    }
    
    if status in ['Optimal', 'Feasible']:
        for p in model.plots:
            for c in model.crops:
                for t in model.periods:
                    var_name = f'Y_{p}_{c}_t{t}'
                    results['solution'][var_name] = pyo.value(model.Y[p, c, t])
    
    print(f'  Pyomo/Gurobi: {results["status"]} in {solve_time:.3f}s')
    if results['objective_value'] is not None:
        print(f'  Objective (native quadratic): {results["objective_value"]:.6f}')
    
    return model, results
    

def validate_solution_constraints(solution, farms, foods, food_groups, land_availability, config):
    """
    Validate if a solution satisfies all original CQM constraints.
    Handles both variable formats:
    - Patch format: Y_{farm}_{food} (binary variables only)
    - Farm format: X_{plot}_{crop} and Y_{crop} (continuous + binary)
    
    Args:
        solution: Dictionary with variable assignments
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
    
    # Detect variable format
    has_x_vars = any(k.startswith('X_') for k in solution.keys())
    has_separate_y = any(k.startswith('Y_') and k.count('_') == 1 for k in solution.keys())
    
    # Patch format: only Y_{farm}_{food} variables
    # Farm format: X_{plot}_{crop} and Y_{crop} variables
    is_patch_format = not has_x_vars
    
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
        if is_patch_format:
            # Patch format: Y_{farm}_{food}
            assigned = sum(solution.get(f"Y_{plot}_{crop}", 0) for crop in foods)
        else:
            # Farm format: X_{plot}_{crop}
            assigned = sum(solution.get(f"X_{plot}_{crop}", 0) for crop in foods)
            
        if assigned > 1.01:  # Allow small numerical tolerance
            violation = f"Plot {plot}: {assigned:.3f} crops assigned (should be ≤ 1)"
            violations.append(violation)
            constraint_checks['at_most_one_per_plot']['violations'].append(violation)
            constraint_checks['at_most_one_per_plot']['failed'] += 1
        else:
            constraint_checks['at_most_one_per_plot']['passed'] += 1
    
    # 2. Check: X-Y Linking (only for farm format)
    if not is_patch_format:
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
    
        # 3. Check: Y Activation (Y_c <= sum_p X_{p,c}) - only for farm format
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
        if is_patch_format:
            # Patch format: Y_{farm}_{food}
            crop_area = sum(
                solution.get(f"Y_{plot}_{crop}", 0) * land_availability[plot]
                for plot in farms
            )
        else:
            # Farm format: X_{plot}_{crop}
            crop_area = sum(
                solution.get(f"X_{plot}_{crop}", 0) * land_availability[plot]
                for plot in farms
            )
        
        # Check minimum area
        if crop in min_planting_area:
            min_area = min_planting_area[crop]
            
            if is_patch_format:
                # Check if any plot has this crop
                y_c = sum(solution.get(f"Y_{plot}_{crop}", 0) for plot in farms)
            else:
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
                
                if is_patch_format:
                    # Count how many plots have each crop
                    n_selected = sum(
                        1 for crop in crops_in_group 
                        if sum(solution.get(f"Y_{plot}_{crop}", 0) for plot in farms) > 0.5
                    )
                else:
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


def solve_with_dwave_cqm(cqm, token):
    """Solve with DWave and return sampleset."""
    sampler = LeapHybridCQMSampler(token=token)

    print("Submitting to DWave Leap hybrid solver...")
    
    sampleset = sampler.sample_cqm(
        cqm, label="Food Optimization - Professional Run")
    

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

    return sampleset, hybrid_time, qpu_time


def solve_with_dwave_bqm(cqm, token):
    """
    Solve with DWave using HybridBQM solver after converting CQM to BQM.
    This enables QPU usage and better scaling for quadratic problems.
    
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
    
    print(f"  ✅ CQM converted to BQM in {bqm_conversion_time:.2f}s")
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


def main(scenario='simple', land_method='even_grid', n_units=None, total_land=100.0):
    """
    Main execution function.
    
    Args:
        scenario: Scenario type ('simple', 'intermediate', etc.)
        land_method: Land generation method ('even_grid' or 'uneven_distribution')
        n_units: Number of units to generate (if None, use scenario default)
        total_land: Total land area in hectares (default: 100.0)
    """
    print("=" * 80)
    print("PROFESSIONAL SOLVER RUNNER - BINARY FORMULATION")
    print("=" * 80)
    print(f"Land Generation Method: {land_method}")
    print(f"Total Land Area: {total_land} ha")
    
    # Create output directories
    os.makedirs('PuLP_Results', exist_ok=True)
    os.makedirs('DWave_Results', exist_ok=True)
    os.makedirs('CQM_Models', exist_ok=True)
    os.makedirs('Constraints', exist_ok=True)
    
    # Load scenario data first to get foods and food groups
    print(f"\nLoading food data for '{scenario}' scenario...")
    _, foods, food_groups, base_config = load_food_data(scenario)
    print(f"  Foods: {len(foods)} - {list(foods.keys())}")
    print(f"  Food Groups: {len(food_groups)} - {list(food_groups.keys())}")
    
    # Generate land data based on method
    print(f"\nGenerating land data using '{land_method}' method...")
    if land_method == 'even_grid':
        # Even grid: all patches have equal size
        n_patches = n_units or 25  # Default to 25 patches if not specified
        land_availability = patch_sampler.generate_grid(
            n_farms=n_patches, 
            area=total_land, 
            seed=42
        )
        farms = list(land_availability.keys())
        print(f"  ✅ Even Grid: {len(farms)} patches of {total_land/len(farms):.3f} ha each")
        print(f"  Formulation: Binary variables X_{{p,c}} ∈ {{0,1}} (discrete optimization)")
        
    elif land_method == 'uneven_distribution':
        # Uneven distribution: farms have realistic size distribution
        n_farms = n_units or 15  # Default to 15 farms if not specified  
        land_availability = farm_sampler.generate_farms(n_farms=n_farms, seed=42)
        
        # Scale to match total_land
        current_total = sum(land_availability.values())
        scale_factor = total_land / current_total
        land_availability = {k: v * scale_factor for k, v in land_availability.items()}
        
        farms = list(land_availability.keys())
        areas = list(land_availability.values())
        print(f"  ✅ Uneven Distribution: {len(farms)} farms")
        print(f"  Farm sizes: min={min(areas):.2f} ha, max={max(areas):.2f} ha, avg={sum(areas)/len(areas):.2f} ha")
        print(f"  Formulation: Continuous area variables A_{{f,c}} ∈ [0, farm_area] (same as solver_runner.py)")
        
    else:
        raise ValueError(f"Unknown land_method: {land_method}. Use 'even_grid' or 'uneven_distribution'")
    
    # Update configuration with generated land data
    config = {
        'parameters': {
            'land_availability': land_availability,
            'minimum_planting_area': base_config['parameters'].get('minimum_planting_area', {}),
            'food_group_constraints': base_config['parameters'].get('food_group_constraints', {}),
            'weights': base_config['parameters'].get('weights', {}),
            'land_generation_method': land_method,
            'total_land_ha': total_land,
            'n_units': len(farms)
        }
    }
    
    print(f"  Total Land: {sum(land_availability.values()):.2f} ha")
    print(f"  Units (farms/plots): {len(farms)}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Choose formulation based on land method
    if land_method == 'even_grid':
        # Binary formulation for even grid
        print(f"\nCreating CQM with BINARY formulation (even grid)...")
        print(f"  Land method: {land_method}")
        print(f"  Total plots: {len(farms)}")
        cqm, Y, constraint_metadata = create_cqm_plots(farms, foods, food_groups, config)
        print(f"  Variables: {len(cqm.variables)} (all binary)")
        print(f"  Constraints: {len(cqm.constraints)}")
        
        # Calculate complexity metrics
        n_crops_with_min = len([c for c in foods if c in config['parameters'].get('minimum_planting_area', {}) 
                                and config['parameters']['minimum_planting_area'][c] > 0])
        n_crops_with_max = len([c for c in foods if c in config['parameters'].get('maximum_planting_area', {})])
        n_food_groups = len(food_groups) if config['parameters'].get('food_group_constraints') else 0
        has_fg_constraints = config['parameters'].get('food_group_constraints') is not None
        
        binary_complexity = calculate_model_complexity(
            'binary', len(farms), len(foods), n_food_groups,
            n_crops_with_min, n_crops_with_max, has_fg_constraints
        )
        
        # Save CQM for binary formulation
        cqm_path = f'CQM_Models/cqm_binary_{scenario}_{timestamp}.cqm'
        print(f"\nSaving Binary CQM to {cqm_path}...")
        with open(cqm_path, 'wb') as f:
            shutil.copyfileobj(cqm.to_file(), f)
            
    elif land_method == 'uneven_distribution':
        # Continuous formulation for uneven farms (same as solver_runner.py)
        print(f"\nUsing CONTINUOUS formulation (uneven farms)...")
        print(f"  Land method: {land_method}")
        print(f"  Total farms: {len(farms)}")
        print(f"  Formulation: Continuous area variables (same as solver_runner.py)")
        
        # Import continuous solver functions
        from solver_runner import create_cqm as create_cqm_continuous
        
        cqm, A, Y, constraint_metadata = create_cqm_continuous(farms, foods, food_groups, config)
        print(f"  Variables: {len(cqm.variables)} (continuous + binary)")
        print(f"  Constraints: {len(cqm.constraints)}")
        
        # Calculate complexity metrics
        n_food_groups = len(food_groups) if config['parameters'].get('food_group_constraints') else 0
        has_fg_constraints = config['parameters'].get('food_group_constraints') is not None
        
        continuous_complexity = calculate_model_complexity(
            'continuous', len(farms), len(foods), n_food_groups,
            0, 0, has_fg_constraints
        )
        
        # Save CQM for continuous formulation
        cqm_path = f'CQM_Models/cqm_continuous_{scenario}_{timestamp}.cqm'
        print(f"\nSaving Continuous CQM to {cqm_path}...")
        with open(cqm_path, 'wb') as f:
            shutil.copyfileobj(cqm.to_file(), f)
    
    # Print complexity comparison if both are available (for documentation purposes)
    if land_method == 'even_grid':
        # Calculate continuous complexity for comparison
        n_food_groups = len(food_groups) if config['parameters'].get('food_group_constraints') else 0
        has_fg_constraints = config['parameters'].get('food_group_constraints') is not None
        continuous_complexity = calculate_model_complexity(
            'continuous', len(farms), len(foods), n_food_groups,
            0, 0, has_fg_constraints
        )
        print_model_complexity_comparison(continuous_complexity, binary_complexity)
    
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
        'land_generation_method': land_method,
        'total_land_ha': total_land,
        'n_units': len(farms),
        'farms': farms,
        'foods': list(foods.keys()),
        'foods_data': foods_serializable,  # Add full food data for objective calculation
        'food_groups': food_groups,
        'config': config,
        'constraint_metadata': {
            'plantation_limit': {str(k): v for k, v in constraint_metadata.get('plantation_limit', {}).items()},
            'min_plots_per_crop': {str(k): v for k, v in constraint_metadata.get('min_plots_per_crop', {}).items()},
            'max_plots_per_crop': {str(k): v for k, v in constraint_metadata.get('max_plots_per_crop', {}).items()},
            'land_availability': {str(k): v for k, v in constraint_metadata.get('land_availability', {}).items()},
            'min_area_if_selected': {str(k): v for k, v in constraint_metadata.get('min_area_if_selected', {}).items()},
            'max_area_if_selected': {str(k): v for k, v in constraint_metadata.get('max_area_if_selected', {}).items()},
            'food_group_min': {str(k): v for k, v in constraint_metadata.get('food_group_min', {}).items()},
            'food_group_max': {str(k): v for k, v in constraint_metadata.get('food_group_max', {}).items()}
        },
        'formulation': 'binary' if land_method == 'even_grid' else 'continuous',
        'variable_type': 'X_{p,c} ∈ {0,1}' if land_method == 'even_grid' else 'A_{f,c} ∈ [0, farm_area]',
        'model_complexity': binary_complexity if land_method == 'even_grid' else continuous_complexity
    }
    
    with open(constraints_path, 'w') as f:
        json.dump(constraints_json, f, indent=2)
    
    # Solve with PuLP
    print("\n" + "=" * 80)
    if land_method == 'even_grid':
        print("SOLVING WITH PULP (BINARY FORMULATION)")
        print("=" * 80)
        pulp_model, pulp_results = solve_with_pulp_plots(farms, foods, food_groups, config)
        formulation_type = "binary"
    else:
        print("SOLVING WITH PULP (CONTINUOUS FORMULATION)")  
        print("=" * 80)
        # Import and use continuous solver
        from solver_runner import solve_with_pulp as solve_with_pulp_continuous
        pulp_model, pulp_results = solve_with_pulp_continuous(farms, foods, food_groups, config)
        formulation_type = "continuous"
        
    print(f"  Formulation: {formulation_type}")
    print(f"  Status: {pulp_results['status']}")
    print(f"  Objective: {pulp_results['objective_value']:.6f}")
    print(f"  Solve time: {pulp_results['solve_time']:.2f} seconds")
    
    # Save PuLP results
    pulp_path = f'PuLP_Results/pulp_{formulation_type}_{scenario}_{timestamp}.json'
    print(f"\nSaving PuLP results to {pulp_path}...")
    with open(pulp_path, 'w') as f:
        json.dump(pulp_results, f, indent=2)
    
    # Solve with DWave (only for binary formulation)
    if land_method == 'even_grid':
        print("\n" + "=" * 80)
        print("SOLVING WITH DWAVE (BQUBO: CQM→BQM + HybridBQM)")
        print("=" * 80)
        token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
        sampleset, dwave_solve_time, qpu_access_time, bqm_conversion_time, invert = solve_with_dwave_bqm(cqm, token)
        
        # BQM samplesets don't have feasibility - all samples are valid (constraints are penalties)
        print(f"  Total samples: {len(sampleset)}")
        print(f"  Total solve time: {dwave_solve_time:.2f} seconds")
        print(f"  BQM conversion time: {bqm_conversion_time:.2f} seconds")
        print(f"  QPU access time: {qpu_access_time:.4f} seconds")
        
        if len(sampleset) > 0:
            best = sampleset.first
            print(f"  Best energy: {best.energy:.6f}")
            best_objective = -best.energy
            print(f"  Best objective: {best_objective:.6f}")
    else:
        print("\n" + "=" * 80)
        print("SKIPPING DWAVE (Continuous formulation not quantum-compatible)")
        print("=" * 80)
        sampleset = None
        dwave_solve_time = 0
        qpu_access_time = 0
        bqm_conversion_time = 0
        best_objective = None
    
    # Save DWave results (only for binary formulation)
    if land_method == 'even_grid' and sampleset is not None:
        dwave_pickle_path = f'DWave_Results/dwave_bqubo_{scenario}_{timestamp}.pickle'
        print(f"\nSaving DWave sampleset to {dwave_pickle_path}...")
        with open(dwave_pickle_path, 'wb') as f:
            pickle.dump(sampleset, f)
        
        # Save DWave results as JSON for easy reading
        dwave_json_path = f'DWave_Results/dwave_bqubo_{scenario}_{timestamp}.json'
        dwave_results = {
            'status': 'Optimal' if len(sampleset) > 0 else 'No solutions',
            'objective_value': best_objective if len(sampleset) > 0 else None,
            'solve_time': dwave_solve_time,
            'qpu_access_time': qpu_access_time,
            'bqm_conversion_time': bqm_conversion_time,
            'num_samples': len(sampleset) if sampleset else 0,
            'formulation': 'BQUBO (binary only)'
        }
        with open(dwave_json_path, 'w') as f:
            json.dump(dwave_results, f, indent=2)
    else:
        dwave_pickle_path = None
        dwave_json_path = None
    
    # Create run manifest
    manifest = {
        'scenario': scenario,
        'timestamp': timestamp,
        'cqm_path': cqm_path,
        'constraints_path': constraints_path,
        'pulp_path': pulp_path,
        'dwave_pickle_path': dwave_pickle_path,
        'dwave_json_path': dwave_json_path,
        'farms': farms,
        'foods': list(foods.keys()),
        'pulp_status': pulp_results['status'],
        'pulp_objective': pulp_results['objective_value'],
        'dwave_status': dwave_results['status'],
        'dwave_objective': dwave_results['objective_value'],
        'dwave_qpu_time': qpu_access_time,
        'dwave_sample_count': len(sampleset),
        'formulation': 'binary_plantation'
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
    print(f"\n✅ BQUBO approach: CQM→BQM conversion + HybridBQM solver")
    print(f"   QPU Access Time: {qpu_access_time:.4f}s")
    print(f"   More QPU usage = Better scaling for larger problems!")
    print("\nRun the verifier script with this manifest to check results:")
    print(f"  python verifier.py {manifest_path}")
    
    return manifest_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run solvers on a food optimization scenario with configurable land generation')
    parser.add_argument('--scenario', type=str, default='simple', 
                       choices=['simple', 'intermediate', 'full', 'custom', 'full_family'],
                       help='Scenario to solve (default: simple)')
    parser.add_argument('--land-method', type=str, default='even_grid',
                       choices=['even_grid', 'uneven_distribution'], 
                       help='Land generation method: even_grid (equal patches) or uneven_distribution (realistic farms) (default: even_grid)')
    parser.add_argument('--n-units', type=int, default=None,
                       help='Number of units to generate (patches for even_grid, farms for uneven_distribution). If not specified, uses defaults.')
    parser.add_argument('--total-land', type=float, default=100.0,
                       help='Total land area in hectares (default: 100.0)')
    
    args = parser.parse_args()
    
    main(scenario=args.scenario, land_method=args.land_method, n_units=args.n_units, total_land=args.total_land)
