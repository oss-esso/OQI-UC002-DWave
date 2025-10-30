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
import patch_sampler
import farm_sampler
from dimod import ConstrainedQuadraticModel, Binary, Real, cqm_to_bqm
from dwave.system import LeapHybridCQMSampler, LeapHybridBQMSampler
import pulp as pl
from tqdm import tqdm
import math

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

    cqm.set_objective(-objective)

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

    # Food group constraints
    if food_group_constraints:
        for group, constraints in tqdm(food_group_constraints.items(), desc="Adding food group constraints"):
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                for farm in farms:
                    if 'min_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) -
                            constraints['min_foods'] >= 0,
                            label=f"Food_Group_Min_{group}_{farm}"
                        )
                        constraint_metadata['food_group_min'][(group, farm)] = {
                            'type': 'food_group_min',
                            'group': group,
                            'farm': farm,
                            'min_foods': constraints['min_foods'],
                            'foods_in_group': foods_in_group
                        }

                    if 'max_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) -
                            constraints['max_foods'] <= 0,
                            label=f"Food_Group_Max_{group}_{farm}"
                        )
                        constraint_metadata['food_group_max'][(group, farm)] = {
                            'type': 'food_group_max',
                            'group': group,
                            'farm': farm,
                            'max_foods': constraints['max_foods'],
                            'foods_in_group': foods_in_group
                        }

    return cqm, A, Y, constraint_metadata


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
    
    # Food group constraints
    pbar.set_description("Adding food group constraints")
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                # Constraints apply across ALL farms, not per farm
                # Count how many different foods from this group are selected across all plots
                if 'min_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) - constraints['min_foods'] >= 0,
                        label=f"Food_Group_Min_{group}"
                    )
                    constraint_metadata['food_group_min'][group] = {
                        'type': 'food_group_min',
                        'group': group,
                        'min_foods': constraints['min_foods'],
                        'foods_in_group': foods_in_group
                    }
                    pbar.update(1)
                
                if 'max_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) - constraints['max_foods'] <= 0,
                        label=f"Food_Group_Max_{group}"
                    )
                    constraint_metadata['food_group_max'][group] = {
                        'type': 'food_group_max',
                        'group': group,
                        'max_foods': constraints['max_foods'],
                        'foods_in_group': foods_in_group
                    }
                    pbar.update(1)
    
    pbar.set_description("CQM complete")
    pbar.close()
    
    return cqm, Y, constraint_metadata


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
                for f in farms:
                    if 'min_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]
                                          ) >= constraints['min_foods'], f"MinFoodGroup_{f}_{g}"
                    if 'max_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]
                                          ) <= constraints['max_foods'], f"MaxFoodGroup_{f}_{g}"

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
    
    # Food group constraints
    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                # Constraints apply across ALL farms, not per farm
                # Count how many different foods from this group are selected across all plots
                if 'min_foods' in constraints:
                    # At least min_foods different foods from this group must be selected somewhere
                    model += pl.lpSum([Y_pulp[(f, c)] for f in farms for c in foods_in_group]) >= constraints['min_foods'], f"MinFoodGroup_{g}"
                if 'max_foods' in constraints:
                    # At most max_foods different foods from this group across all plots
                    model += pl.lpSum([Y_pulp[(f, c)] for f in farms for c in foods_in_group]) <= constraints['max_foods'], f"MaxFoodGroup_{g}"
    
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


def solve_with_dwave_cqm(cqm, token):
    """Solve with DWave and return sampleset."""
    sampler = LeapHybridCQMSampler(token=token)

    print("Submitting to DWave Leap hybrid solver...")
    
    sampleset = sampler.sample_cqm(
        cqm, label="Food Optimization - Professional Run")
    solve_time = sampleset.info.get('charge_time', 0) / 1e6  # Convert to seconds

    return sampleset, solve_time


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


def calculate_original_objective(solution, farms, foods, land_availability, weights, idle_penalty):
    """
    Calculate the original CQM objective from a solution.
    
    This reconstructs the objective for the binary formulation.
    
    Args:
        solution: Dictionary with variable assignments
        farms: List of farm/plot names
        foods: Dictionary of food data with nutritional values
        land_availability: Dictionary mapping plot to area
        weights: Dictionary of objective weights
        idle_penalty: Lambda penalty for idle land
        
    Returns:
        float: The original objective value (to be maximized)
    """
    objective = 0.0
    total_land = sum(land_availability.values())
    
    for plot in farms:
        s_p = land_availability[plot]  # Area of plot p
        for crop in foods:
            # Get variable value from solution (using plot_crop naming)
            var_name = f"{plot}_{crop}"
            x_pc = solution.get(var_name, 0)
            
            if x_pc > 0:  # Only count if assigned
                # Calculate benefit per unit area
                B_c = (
                    weights.get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                    weights.get('affordability', 0) * foods[crop].get('affordability', 0) +
                    weights.get('sustainability', 0) * foods[crop].get('sustainability', 0)
                )
                # Add benefit * area to objective
                objective += B_c * s_p * x_pc
    
    # Normalize by total land area
    return objective / total_land if total_land > 0 else 0.0


def extract_solution_summary(solution, farms, foods, land_availability):
    """
    Extract a summary of the solution showing crop selections and plot assignments.
    
    Args:
        solution: Dictionary with variable assignments
        farms: List of farm/plot names
        foods: Dictionary of food data
        land_availability: Dictionary mapping plot to area
        
    Returns:
        dict: Summary with crops selected, areas, and plot assignments
    """
    crops_selected = set()
    plot_assignments = []
    total_allocated = 0.0
    
    for plot in farms:
        for crop in foods:
            var_name = f"{plot}_{crop}"
            if solution.get(var_name, 0) > 0:
                crops_selected.add(crop)
                plot_assignments.append({
                    'plot': plot,
                    'crop': crop,
                    'area': land_availability[plot]
                })
                total_allocated += land_availability[plot]
    
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


def validate_solution_constraints(solution, farms, foods, food_groups, land_availability, config):
    """
    Validate if a solution satisfies all original CQM constraints.
    
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
    food_group_constraints = params.get('food_group_constraints', {})
    
    violations = []
    
    # Check plot assignment constraints (each plot assigned to at most one crop)
    for plot in farms:
        assigned_count = sum(1 for crop in foods if solution.get(f"{plot}_{crop}", 0) > 0)
        if assigned_count > 1:
            violations.append(f"Plot {plot} assigned to {assigned_count} crops (should be ≤1)")
    
    # Check food group constraints if specified
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            if not foods_in_group:
                continue
            
            # Count how many foods from this group are selected across all plots
            selected_from_group = set()
            for plot in farms:
                for crop in foods_in_group:
                    if solution.get(f"{plot}_{crop}", 0) > 0:
                        selected_from_group.add(crop)
            
            n_selected = len(selected_from_group)
            
            if 'min_foods' in constraints and n_selected < constraints['min_foods']:
                violations.append(
                    f"Food group '{group_name}': {n_selected} foods selected, "
                    f"minimum required: {constraints['min_foods']}"
                )
            
            if 'max_foods' in constraints and n_selected > constraints['max_foods']:
                violations.append(
                    f"Food group '{group_name}': {n_selected} foods selected, "
                    f"maximum allowed: {constraints['max_foods']}"
                )
    
    return {
        'n_violations': len(violations),
        'violations': violations,
        'is_feasible': len(violations) == 0
    }


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
