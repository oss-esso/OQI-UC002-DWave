"""
Professional solver runner script with DECOMPOSED QPU (Strategic Problem Decomposition).

This script implements Alternative 2: Strategic Problem Decomposition
- Farm scenario: Classical-only (Gurobi) for continuous optimization
- Patch scenario: Quantum-only (low-level QPU) for binary optimization

Key difference from other solvers:
- Uses DWaveSampler + EmbeddingComposite for direct QPU access
- Manual embedding, no hybrid solver overhead
- Maximizes QPU utilization for pure binary problems
- Explicit control over annealing parameters

IEEE Standard: Professional code with comprehensive documentation.
"""

import os
import sys
import json
import pickle
import shutil
import time
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.scenarios import load_food_data
from Utils import patch_sampler
from Utils import farm_sampler
from dimod import ConstrainedQuadraticModel, Binary, Real, cqm_to_bqm
from dwave.system import LeapHybridCQMSampler, LeapHybridBQMSampler, DWaveSampler, EmbeddingComposite
import pulp as pl
from tqdm import tqdm
import math

# Check for low-level QPU sampler availability
try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    LOWLEVEL_QPU_AVAILABLE = True
except ImportError:
    LOWLEVEL_QPU_AVAILABLE = False
    print("Warning: DWaveSampler not available. Install with: pip install dwave-system")

# Check for Neal (SimulatedAnnealingSampler) for testing without QPU
try:
    import neal
    NEAL_AVAILABLE = True
except ImportError:
    NEAL_AVAILABLE = False
    print("Warning: neal not available. Install with: pip install dwave-neal")

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
    max_planting_area = params.get('maximum_planting_area', {})
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

    # Linking constraints - minimum and maximum area if selected
    for farm in tqdm(farms, desc="Adding linking constraints"):
        for food in foods:
            # Minimum area if selected
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

            # Maximum area if selected (either explicit max or farm capacity)
            if food in max_planting_area:
                A_max = max_planting_area[food]
            else:
                A_max = land_availability[farm]
                
            cqm.add_constraint(
                A[(farm, food)] - A_max * Y[(farm, food)] <= 0,
                label=f"Max_Area_If_Selected_{farm}_{food}"
            )
            constraint_metadata['max_area_if_selected'][(farm, food)] = {
                'type': 'max_area_if_selected',
                'farm': farm,
                'food': food,
                'max_area': A_max
            }

    # Food group constraints - GLOBAL across all farms (use COUNT of Y, not area A)
    if food_group_constraints:
        for group, constraints in tqdm(food_group_constraints.items(), desc="Adding food group constraints"):
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                # Normalize group name for labels
                group_label = group.replace(' ', '_').replace(',', '').replace('-', '_')
                
                # Global minimum COUNT: across ALL farms, at least min_foods selections
                if 'min_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) - constraints['min_foods'] >= 0,
                        label=f"Food_Group_Min_{group_label}_Global"
                    )
                    constraint_metadata['food_group_min'][group] = {
                        'type': 'food_group_min_count_global',
                        'group': group,
                        'min_foods': constraints['min_foods'],
                        'foods_in_group': foods_in_group,
                        'scope': 'global'
                    }

                # Global maximum COUNT: across ALL farms, at most max_foods selections
                if 'max_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) - constraints['max_foods'] <= 0,
                        label=f"Food_Group_Max_{group_label}_Global"
                    )
                    constraint_metadata['food_group_max'][group] = {
                        'type': 'food_group_max_count_global',
                        'group': group,
                        'max_foods': constraints['max_foods'],
                        'foods_in_group': foods_in_group,
                        'scope': 'global'
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
        n_crops_with_min * n_farms +  # Conditional minimum plot constraints (per farm per crop)
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
            sum(Y[(farm, food)] for food in foods) <= 1,
            label=f"Max_Assignment_{farm}"
        )
        constraint_metadata['plantation_limit'][farm] = {
            'type': 'land_unit_assignment',
            'farm': farm,
            'area_ha': land_availability[farm]
        }
        pbar.update(1)
    
    # CONDITIONAL Minimum/Maximum plot constraints
    # These constraints only apply when a crop is actually planted
    # If crop is not planted anywhere: sum(Y) = 0 (automatically satisfied)
    # If crop is planted: min_plots <= sum(Y) <= max_plots
    
    pbar.set_description("Adding conditional min/max plot constraints")
    for food in foods:
        # Conditional minimum: IF planted, must use at least min_plots
        if food in min_planting_area and min_planting_area[food] > 0:
            min_plots = math.ceil(min_planting_area[food] / plot_area)
            total_assignments = sum(Y[(farm, food)] for farm in farms)
            
            # If Y_{farm,food} = 1, then sum(Y_{*,food}) >= min_plots
            for farm in farms:
                cqm.add_constraint(
                    total_assignments - min_plots * Y[(farm, food)] >= 0,
                    label=f"Min_Plots_If_Selected_{farm}_{food}"
                )
            
            constraint_metadata['min_plots_per_crop'][food] = {
                'type': 'conditional_min_plots_per_crop',
                'food': food,
                'min_area_ha': min_planting_area[food],
                'plot_area_ha': plot_area,
                'min_plots': min_plots
            }
            pbar.update(1)
        
        # Conditional maximum: always enforced (if not planted, sum=0 satisfies max)
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
                # Normalize group name for constraint labels (replace spaces and special chars with underscores)
                group_label = group.replace(' ', '_').replace(',', '').replace('-', '_')
                
                # Global minimum: across ALL plots, at least min_foods from this group
                if 'min_foods' in constraints:
                    cqm.add_constraint(
                        sum(Y[(farm, food)] for farm in farms for food in foods_in_group) - constraints['min_foods'] >= 0,
                        label=f"MinFoodGroup_Global_{group_label}"
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
                        label=f"MaxFoodGroup_Global_{group_label}"
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


def solve_with_pulp_plots(farms, foods, food_groups, config):
    """
    Solve with PuLP using streamlined binary formulation.
    
    Supports land generation methods:
    - even_grid: land_availability represents area per plot
    """
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    max_planting_area = params.get('max_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    land_method = params.get('land_generation_method', 'even_grid')
    
    # Calculate plot area (assuming even grid, all plots have same area)
    plot_area = list(land_availability.values())[0] if farms else 1.0
    
    # Binary variables - each represents assignment of a crop to a land unit
    X_pulp = pl.LpVariable.dicts("X", [(f, c) for f in farms for c in foods], cat='Binary')
    
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
            goal += area_weighted_value * X_pulp[(f, c)]
    
    # Normalize by total land area
    goal = goal / total_land_area
    
    model = pl.LpProblem("Food_Optimization_Binary_Streamlined", pl.LpMaximize)
    
    # Constraint 1: At most one crop per plot
    for f in farms:
        model += pl.lpSum([X_pulp[(f, c)] for c in foods]) <= 1, f"Max_Assignment_{f}"
    
    # Constraint 2: CONDITIONAL Minimum plots per crop
    # If a crop is planted anywhere (any X > 0), it must use at least min_plots
    import math
    for crop in foods:
        if crop in min_planting_area and min_planting_area[crop] > 0:
            min_plots = math.ceil(min_planting_area[crop] / plot_area)
            total_crop_assignments = pl.lpSum([X_pulp[(f, crop)] for f in farms])
            
            # For each farm: if X_{f,crop}=1, then total >= min_plots
            for f in farms:
                model += total_crop_assignments >= min_plots * X_pulp[(f, crop)], f"Min_Plots_If_{f}_{crop}"
    
    # Constraint 3: Maximum plots per crop
    for crop in foods:
        if crop in max_planting_area:
            max_plots = math.floor(max_planting_area[crop] / plot_area)
            model += pl.lpSum([X_pulp[(f, crop)] for f in farms]) <= max_plots, f"Max_Plots_{crop}"
    
    ## Constraint 4: Food group constraints - GLOBAL across all plots
    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                # Normalize group name for constraint labels
                group_label = g.replace(' ', '_').replace(',', '').replace('-', '_')
                
                # Global constraints: across ALL plots
                if 'min_foods' in constraints:
                    model += pl.lpSum([X_pulp[(f, c)] for f in farms for c in foods_in_group]) >= constraints['min_foods'], f"MinFoodGroup_Global_{group_label}"
                if 'max_foods' in constraints:
                    model += pl.lpSum([X_pulp[(f, c)] for f in farms for c in foods_in_group]) <= constraints['max_foods'], f"MaxFoodGroup_Global_{group_label}"
    
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
            results['plantations'][key] = X_pulp[(f, c)].value() if X_pulp[(f, c)].value() is not None else 0.0
    
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
    has_a_vars = any(k.startswith('A_') for k in solution.keys())
    has_y_vars = any(k.startswith('Y_') for k in solution.keys())
    
    # Farm format: A_{farm}_{food} (continuous) and Y_{farm}_{food} (binary)
    # Patch format: only Y_{farm}_{food} (binary)
    is_farm_format = has_a_vars and has_y_vars
    
    violations = []
    constraint_checks = {
        'land_availability': {'passed': 0, 'failed': 0, 'violations': []},
        'linking_constraints': {'passed': 0, 'failed': 0, 'violations': []},
        'food_group_constraints': {'passed': 0, 'failed': 0, 'violations': []}
    }
    
    # FARM FORMAT VALIDATION
    if is_farm_format:
        # 1. Check: Land availability per farm
        for farm in farms:
            farm_total = sum(solution.get(f"A_{farm}_{crop}", 0) for crop in foods)
            farm_capacity = land_availability[farm]
            
            if farm_total > farm_capacity + 0.01:
                violation = f"{farm}: {farm_total:.4f} ha > {farm_capacity:.4f} ha capacity"
                violations.append(violation)
                constraint_checks['land_availability']['violations'].append(violation)
                constraint_checks['land_availability']['failed'] += 1
            else:
                constraint_checks['land_availability']['passed'] += 1
        
        # 2. Check: Linking constraints A and Y
        # Constraint: A >= min_area * Y  AND  A <= farm_capacity * Y
        for farm in farms:
            farm_capacity = land_availability[farm]
            for crop in foods:
                a_val = solution.get(f"A_{farm}_{crop}", 0)
                y_val = solution.get(f"Y_{farm}_{crop}", 0)
                min_area = min_planting_area.get(crop, 0)
                
                # If Y=1 (selected), check A >= min_area
                if y_val > 0.5:
                    if a_val < min_area * 0.999:  # Relative tolerance
                        violation = f"A_{farm}_{crop}={a_val:.4f} < min_area={min_area:.4f} (Y=1)"
                        violations.append(violation)
                        constraint_checks['linking_constraints']['violations'].append(violation)
                        constraint_checks['linking_constraints']['failed'] += 1
                    elif a_val > farm_capacity + 0.001:
                        violation = f"A_{farm}_{crop}={a_val:.4f} > farm_capacity={farm_capacity:.4f} (Y=1)"
                        violations.append(violation)
                        constraint_checks['linking_constraints']['violations'].append(violation)
                        constraint_checks['linking_constraints']['failed'] += 1
                    else:
                        constraint_checks['linking_constraints']['passed'] += 1
                else:  # Y=0 (not selected), A must be 0
                    if a_val > 0.001:
                        violation = f"A_{farm}_{crop}={a_val:.4f} but Y_{farm}_{crop}={y_val:.4f} (should be 0)"
                        violations.append(violation)
                        constraint_checks['linking_constraints']['violations'].append(violation)
                        constraint_checks['linking_constraints']['failed'] += 1
                    else:
                        constraint_checks['linking_constraints']['passed'] += 1
    
    # PATCH FORMAT VALIDATION (binary-only formulation - not used for Farm)
    else:
        # For patch format, we don't have these linking constraints
        pass
    
    # 3. Check: Food group constraints (works for both formats)
    if food_group_constraints:
        for group_name, group_data in food_group_constraints.items():
            if group_name in food_groups:
                crops_in_group = food_groups[group_name]
                
                if is_farm_format:
                    # Farm format: count crops with Y_{farm}_{crop} > 0.5 anywhere
                    n_selected = sum(
                        1 for crop in crops_in_group 
                        if any(solution.get(f"Y_{farm}_{crop}", 0) > 0.5 for farm in farms)
                    )
                else:
                    # Patch format: count crops with Y_{farm}_{crop} > 0.5 anywhere
                    n_selected = sum(
                        1 for crop in crops_in_group 
                        if any(solution.get(f"Y_{farm}_{crop}", 0) > 0.5 for farm in farms)
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


def solve_with_decomposed_qpu(bqm, token, **kwargs):
    """
    Solve a BQM using low-level D-Wave QPU sampler with explicit embedding.
    
    This function implements strategic problem decomposition by using direct QPU access
    instead of hybrid solvers. It provides:
    - Explicit control over embedding via EmbeddingComposite
    - Direct annealing parameter control (num_reads, annealing_time)
    - Maximum QPU utilization without classical preprocessing
    - Detailed timing metrics (QPU access time, programming time, etc.)
    
    This approach is optimal for:
    - Pure binary optimization problems (BQM, QUBO)
    - Problems that fit directly on QPU after embedding
    - When you want maximum QPU time vs hybrid overhead
    
    Args:
        bqm: BinaryQuadraticModel to solve
        token: D-Wave API token (string, default: None uses environment variable)
        **kwargs: Additional parameters:
            - num_reads (int): Number of QPU reads (default: 1000)
            - annealing_time (int): Annealing time in microseconds (default: 20)
            - chain_strength (float): Chain strength for embedding (default: auto-calculated)
            - auto_scale (bool): Auto-scale BQM before submission (default: True)
    
    Returns:
        dict: Solution containing:
            - status (str): 'Optimal' (always, as QPU returns best found)
            - objective_value (float): Best BQM energy found
            - solve_time (float): Total solve time in seconds
            - qpu_access_time (float): QPU access time in seconds
            - qpu_programming_time (float): QPU programming time in seconds
            - qpu_sampling_time (float): Actual QPU sampling time in seconds
            - num_reads (int): Number of reads performed
            - num_variables (int): Number of BQM variables
            - num_interactions (int): Number of quadratic interactions
            - solver_name (str): 'dwave_decomposed_qpu'
            - solution (dict): Best variable assignments
    
    Raises:
        ImportError: If DWaveSampler is not available
        Exception: If QPU sampling fails
    """
    # Check if we should use simulated annealing (no token provided)
    use_simulated_annealing = (
        token is None or 
        token == 'YOUR_DWAVE_TOKEN_HERE' or
        (isinstance(token, str) and token.strip() == '')
    )
    
    if use_simulated_annealing:
        if not NEAL_AVAILABLE:
            raise ImportError(
                "neal is required for simulated annealing testing. "
                "Install with: pip install dwave-neal"
            )
        print("\n" + "="*80)
        print("SOLVING WITH SIMULATED ANNEALING (Testing Mode - No QPU)")
        print("Note: Using neal.SimulatedAnnealingSampler for testing without D-Wave token")
        print("="*80)
    else:
        if not LOWLEVEL_QPU_AVAILABLE:
            raise ImportError(
                "DWaveSampler is required for decomposed QPU solving. "
                "Install with: pip install dwave-system"
            )
        print("\n" + "="*80)
        print("SOLVING WITH DECOMPOSED QPU (Low-Level Sampler)")
        print("="*80)
    
    # Extract parameters
    num_reads = kwargs.get('num_reads', 1000)
    annealing_time = kwargs.get('annealing_time', 20)
    chain_strength = kwargs.get('chain_strength', None)
    auto_scale = kwargs.get('auto_scale', True)
    
    print(f"  Sampler Configuration:")
    print(f"    Num Reads: {num_reads}")
    if not use_simulated_annealing:
        print(f"    Annealing Time: {annealing_time} μs")
        print(f"    Chain Strength: {'Auto' if chain_strength is None else chain_strength}")
        print(f"    Auto Scale: {auto_scale}")
    print(f"    BQM Variables: {len(bqm.variables)}")
    print(f"    BQM Interactions: {len(bqm.quadratic)}")
    
    # Create sampler (QPU or Simulated Annealing)
    if use_simulated_annealing:
        print("\n  Initializing Simulated Annealing sampler...")
        try:
            sampler = neal.SimulatedAnnealingSampler()
            print(f"    ✓ SimulatedAnnealingSampler ready (neal)")
            print(f"    ✓ Testing mode: No QPU required")
        except Exception as e:
            print(f"    ❌ Simulated Annealing initialization failed: {e}")
            raise
    else:
        print("\n  Initializing QPU sampler...")
        try:
            sampler_qpu = DWaveSampler(token=token)
            print(f"    ✓ Connected to QPU: {sampler_qpu.properties.get('chip_id', 'Unknown')}")
            print(f"    QPU Topology: {sampler_qpu.properties.get('topology', {}).get('type', 'Unknown')}")
            
            # Wrap with EmbeddingComposite for automatic minor-embedding
            sampler = EmbeddingComposite(sampler_qpu)
            print(f"    ✓ EmbeddingComposite ready (automatic minor-embedding)")
            
        except Exception as e:
            print(f"    ❌ QPU initialization failed: {e}")
            raise
    
    # Sample on QPU or with Simulated Annealing
    if use_simulated_annealing:
        print("\n  Running Simulated Annealing...")
    else:
        print("\n  Submitting to QPU...")
    
    sample_start = time.time()
    
    try:
        if use_simulated_annealing:
            # Use simulated annealing (no QPU-specific parameters)
            sample_kwargs = {
                'num_reads': num_reads
            }
            sampleset = sampler.sample(bqm, **sample_kwargs)
            solve_time = time.time() - sample_start
            print(f"    ✓ Simulated Annealing complete in {solve_time:.2f}s")
        else:
            # Build sampling parameters for QPU
            sample_kwargs = {
                'num_reads': num_reads,
                'annealing_time': annealing_time,
                'auto_scale': auto_scale
            }
            
            if chain_strength is not None:
                sample_kwargs['chain_strength'] = chain_strength
            
            # Submit to QPU
            sampleset = sampler.sample(bqm, **sample_kwargs)
            solve_time = time.time() - sample_start
            print(f"    ✓ QPU sampling complete in {solve_time:.2f}s")
        
    except Exception as e:
        if use_simulated_annealing:
            print(f"    ❌ Simulated Annealing failed: {e}")
        else:
            print(f"    ❌ QPU sampling failed: {e}")
        raise
    
    # Extract timing information
    timing_info = sampleset.info.get('timing', {})
    
    if use_simulated_annealing:
        # No QPU timing for simulated annealing
        qpu_access_time = 0.0
        qpu_programming_time = 0.0
        qpu_sampling_time = 0.0
        qpu_anneal_time_per_sample = 0.0
        
        print(f"\n  Timing Summary:")
        print(f"    Total Solve Time: {solve_time:.3f}s")
        print(f"    Sampler: Simulated Annealing (neal)")
    else:
        # QPU access time (total time on QPU)
        qpu_access_time = timing_info.get('qpu_access_time', 0) / 1e6  # Convert μs to s
        
        # QPU programming time (time to program the QPU)
        qpu_programming_time = timing_info.get('qpu_programming_time', 0) / 1e6
        
        # QPU sampling time (actual annealing time)
        qpu_sampling_time = timing_info.get('qpu_sampling_time', 0) / 1e6
        
        # QPU anneal time per sample
        qpu_anneal_time_per_sample = timing_info.get('qpu_anneal_time_per_sample', 0) / 1e6
        
        print(f"\n  QPU Timing Breakdown:")
        print(f"    Total Solve Time: {solve_time:.3f}s")
        print(f"    QPU Access Time: {qpu_access_time:.4f}s")
        print(f"    QPU Programming Time: {qpu_programming_time:.4f}s")
        print(f"    QPU Sampling Time: {qpu_sampling_time:.4f}s")
        print(f"    QPU Anneal Time/Sample: {qpu_anneal_time_per_sample:.6f}s")
    
    # Extract best solution
    if len(sampleset) > 0:
        best = sampleset.first
        best_energy = best.energy
        best_sample = dict(best.sample)
        num_occurrences = best.num_occurrences
        
        print(f"\n  Solution Quality:")
        print(f"    Best Energy: {best_energy:.6f}")
        print(f"    Best Sample Occurrences: {num_occurrences}/{num_reads}")
        print(f"    Total Samples Returned: {len(sampleset)}")
        
        # Calculate objective value
        # NOTE: BQM energy is the NEGATIVE of the actual objective (because we minimize -objective)
        # So the true objective value is -energy
        objective_value = -best_energy
        
        result = {
            'status': 'Optimal',  # Sampler always returns best found
            'objective_value': objective_value,  # Actual objective (negative of BQM energy)
            'bqm_energy': best_energy,  # Raw BQM energy for debugging,
            'solve_time': solve_time,
            'qpu_access_time': qpu_access_time,
            'qpu_programming_time': qpu_programming_time,
            'qpu_sampling_time': qpu_sampling_time,
            'qpu_anneal_time_per_sample': qpu_anneal_time_per_sample,
            'num_reads': num_reads,
            'num_occurrences': num_occurrences,
            'num_variables': len(bqm.variables),
            'num_interactions': len(bqm.quadratic),
            'solver_name': 'simulated_annealing' if use_simulated_annealing else 'dwave_decomposed_qpu',
            'solution': best_sample,
            'sampler_type': 'simulated_annealing' if use_simulated_annealing else 'qpu',
            'timing_info': timing_info  # Full timing dict for analysis
        }
        
        if use_simulated_annealing:
            result['sampler_config'] = {
                'num_reads': num_reads,
                'sampler': 'neal.SimulatedAnnealingSampler'
            }
        else:
            result['qpu_config'] = {
                'num_reads': num_reads,
                'annealing_time': annealing_time,
                'chain_strength': chain_strength,
                'auto_scale': auto_scale,
                'chip_id': sampler_qpu.properties.get('chip_id', 'Unknown'),
                'topology': sampler_qpu.properties.get('topology', {}).get('type', 'Unknown')
            }
        
        if use_simulated_annealing:
            print(f"\n  ✓ Simulated Annealing solution complete")
        else:
            print(f"\n  ✓ Decomposed QPU solution complete")
        return result
        
    else:
        print(f"\n  ❌ No samples returned from QPU")
        raise ValueError("QPU returned empty sampleset")


def solve_farm_with_hybrid_decomposition(farms, foods, food_groups, config, token, **kwargs):
    """
    Hybrid decomposition approach for FARM scenario (continuous + binary variables).
    
    Strategy:
    1. Solve continuous relaxation (A continuous, Y relaxed to [0,1]) with Gurobi
    2. Extract optimal A* values from relaxation
    3. Fix A* and create binary subproblem for Y variables only
    4. Convert Y subproblem to BQM
    5. Solve Y subproblem on QPU
    6. Combine results: final solution uses A* (from Gurobi) and Y** (from QPU)
    
    This leverages:
    - Gurobi's strength: Continuous optimization for area allocation (A)
    - QPU's strength: Binary decision making for crop selection (Y)
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food group mappings
        config: Configuration dictionary with parameters
        token: D-Wave API token
        **kwargs: Additional QPU parameters (num_reads, annealing_time, etc.)
    
    Returns:
        dict: Solution with combined A* and Y** values
    """
    print("\n" + "="*80)
    print("FARM SCENARIO: HYBRID DECOMPOSITION (Gurobi + QPU)")
    print("="*80)
    print("Strategy:")
    print("  1. Solve continuous relaxation with Gurobi (A + relaxed Y)")
    print("  2. Extract A* (continuous area allocations)")
    print("  3. Create binary subproblem for Y only")
    print("  4. Solve Y subproblem on QPU")
    print("  5. Combine A* (Gurobi) + Y** (QPU)")
    print("="*80)
    
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    max_planting_area = params.get('maximum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    total_area = sum(land_availability.values())
    
    # STEP 1: Solve continuous relaxation with Gurobi
    print("\n[STEP 1: Continuous Relaxation]")
    print("  Building PuLP model with relaxed Y variables...")
    
    # Create relaxed problem (Y in [0, 1] instead of {0, 1})
    prob_relaxed = pl.LpProblem("Farm_Relaxation", pl.LpMaximize)
    
    # Variables: A (continuous), Y (relaxed to [0, 1])
    A_pulp = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in foods], lowBound=0)
    Y_pulp = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in foods], lowBound=0, upBound=1)
    
    # Objective (same as original, normalized by total_area)
    objective_relaxed = pl.lpSum(
        (weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
         weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
         weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
         weights.get('affordability', 0) * foods[c].get('affordability', 0) +
         weights.get('sustainability', 0) * foods[c].get('sustainability', 0)) * A_pulp[(f, c)]
        for f in farms for c in foods
    ) / total_area  # Normalize to match baseline formulation
    prob_relaxed += objective_relaxed
    
    # Constraints (same as original farm problem)
    # Land availability
    for f in farms:
        prob_relaxed += pl.lpSum(A_pulp[(f, c)] for c in foods) <= land_availability[f], f"Land_{f}"
    
    # Linking constraints with min and max area
    for f in farms:
        for c in foods:
            A_min = min_planting_area.get(c, 0)
            prob_relaxed += A_pulp[(f, c)] >= A_min * Y_pulp[(f, c)], f"Min_{f}_{c}"
            
            if c in max_planting_area:
                A_max = max_planting_area[c]
            else:
                A_max = land_availability[f]
            prob_relaxed += A_pulp[(f, c)] <= A_max * Y_pulp[(f, c)], f"Max_{f}_{c}"
    
    # Food group constraints (count-based, using Y variables)
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                group_label = group.replace(' ', '_').replace(',', '').replace('-', '_')
                if 'min_foods' in constraints:
                    prob_relaxed += pl.lpSum(Y_pulp[(f, c)] for f in farms for c in foods_in_group) >= constraints['min_foods'], f"FoodGroupMin_{group_label}"
                if 'max_foods' in constraints:
                    prob_relaxed += pl.lpSum(Y_pulp[(f, c)] for f in farms for c in foods_in_group) <= constraints['max_foods'], f"FoodGroupMax_{group_label}"
    
    # Solve relaxation
    print("  Solving with Gurobi...")
    start_gurobi = time.time()
    prob_relaxed.solve(pl.GUROBI(msg=0))
    gurobi_time = time.time() - start_gurobi
    
    if prob_relaxed.status != pl.LpStatusOptimal:
        print(f"  ❌ Relaxation failed: {pl.LpStatus[prob_relaxed.status]}")
        raise ValueError("Continuous relaxation failed")
    
    print(f"  ✓ Relaxation solved in {gurobi_time:.2f}s")
    print(f"  Relaxation objective: {pl.value(prob_relaxed.objective):.4f}")
    
    # STEP 2: Extract A* values
    print("\n[STEP 2: Extract Continuous Values]")
    A_star = {}
    Y_relaxed = {}
    for f in farms:
        for c in foods:
            A_star[(f, c)] = A_pulp[(f, c)].varValue or 0.0
            Y_relaxed[(f, c)] = Y_pulp[(f, c)].varValue or 0.0
    
    n_selected = sum(1 for y in Y_relaxed.values() if y > 0.5)
    print(f"  ✓ Extracted {len(A_star)} A* values")
    print(f"  Crops likely selected (Y > 0.5): {n_selected}")
    
    # STEP 3: Create binary subproblem for Y
    print("\n[STEP 3: Create Binary Subproblem for Y]")
    print("  Building CQM with only binary Y variables...")
    
    # Create new CQM with ONLY Y variables (A is fixed)
    cqm_binary = ConstrainedQuadraticModel()
    Y_binary = {}
    for f in farms:
        for c in foods:
            Y_binary[(f, c)] = Binary(f"Y_{f}_{c}")
    
    # Objective: Use fixed A* values, normalized by total_area
    # Objective = sum (B_c * A*_{f,c}) for Y_{f,c} = 1
    # This simplifies to: maximize sum (B_c * A*_{f,c} * Y_{f,c}) / total_area
    objective_binary = sum(
        (weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
         weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
         weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
         weights.get('affordability', 0) * foods[c].get('affordability', 0) +
         weights.get('sustainability', 0) * foods[c].get('sustainability', 0)) * A_star[(f, c)] * Y_binary[(f, c)]
        for f in farms for c in foods
    ) / total_area  # Normalize to match baseline formulation
    cqm_binary.set_objective(-objective_binary)  # Negative because CQM minimizes
    
    # Binary constraints (only Y-based constraints, A is fixed)
    # Food group constraints use COUNT (not area)
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                group_label = group.replace(' ', '_').replace(',', '').replace('-', '_')
                if 'min_foods' in constraints:
                    cqm_binary.add_constraint(
                        sum(Y_binary[(f, c)] for f in farms for c in foods_in_group) - constraints['min_foods'] >= 0,
                        label=f"FoodGroupMin_{group_label}_Binary"
                    )
                if 'max_foods' in constraints:
                    cqm_binary.add_constraint(
                        sum(Y_binary[(f, c)] for f in farms for c in foods_in_group) - constraints['max_foods'] <= 0,
                        label=f"FoodGroupMax_{group_label}_Binary"
                    )
    
    print(f"  ✓ Binary CQM created: {len(Y_binary)} Y variables, {len(cqm_binary.constraints)} constraints")
    
    # STEP 4: Convert to BQM
    print("\n[STEP 4: Convert to BQM]")
    print("  Converting CQM to BQM...")
    try:
        bqm, invert = cqm_to_bqm(cqm_binary)
        print(f"  ✓ BQM conversion complete")
        print(f"  BQM variables: {len(bqm.variables)}")
        print(f"  BQM quadratic terms: {len(bqm.quadratic)}")
    except Exception as e:
        print(f"  ❌ BQM conversion failed: {e}")
        raise
    
    # STEP 5: Solve on QPU
    print("\n[STEP 5: Solve Binary Subproblem on QPU]")
    qpu_result = solve_with_decomposed_qpu(bqm, token, **kwargs)
    
    # Invert BQM solution to get Y values
    y_solution_raw = qpu_result['solution']
    y_solution_cqm = invert(y_solution_raw)
    
    # STEP 6: Combine results
    print("\n[STEP 6: Combine Results]")
    print("  Combining A* (from Gurobi) and Y** (from QPU)...")
    
    # Create final solution with both A and Y
    final_solution = {}
    for f in farms:
        for c in foods:
            # Add A* values (from Gurobi)
            final_solution[f"A_{f}_{c}"] = A_star[(f, c)]
            # Add Y** values (from QPU)
            y_var_name = f"Y_{f}_{c}"
            final_solution[y_var_name] = y_solution_cqm.get(y_var_name, 0)
    
    # Calculate final objective using both A* and Y**
    # IMPORTANT: Don't use BQM energy! Evaluate the ORIGINAL objective function.
    # BQM energy includes penalties from constraints and is not comparable.
    final_objective = sum(
        (weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
         weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
         weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
         weights.get('affordability', 0) * foods[c].get('affordability', 0) +
         weights.get('sustainability', 0) * foods[c].get('sustainability', 0)) * A_star[(f, c)] * final_solution[f"Y_{f}_{c}"]
        for f in farms for c in foods
    ) / total_area  # Normalize to match baseline formulation
    
    # Count how many Y variables are selected
    n_selected = sum(1 for f in farms for c in foods if final_solution[f"Y_{f}_{c}"] > 0.5)
    print(f"  ✓ Final objective: {final_objective:.6f} ({n_selected} crops selected)")
    print(f"  Decomposition benefit: Gurobi handled {len(A_star)} continuous vars, QPU handled {len(Y_binary)} binary vars")
    
    result = {
        'status': 'Optimal',
        'objective_value': final_objective,
        'solve_time': gurobi_time + qpu_result['solve_time'],
        'gurobi_time': gurobi_time,
        'qpu_time': qpu_result['solve_time'],
        'qpu_access_time': qpu_result.get('qpu_access_time', 0),
        'relaxation_objective': pl.value(prob_relaxed.objective),
        'final_objective': final_objective,
        'solver_name': 'hybrid_decomposition_gurobi_qpu',
        'solution': final_solution,
        'A_star': {f"A_{f}_{c}": A_star[(f, c)] for f, c in A_star},  # Convert tuple keys to strings
        'Y_star': {f"Y_{f}_{c}": final_solution[f"Y_{f}_{c}"] for f in farms for c in foods}
    }
    
    print("\n" + "="*80)
    print("HYBRID DECOMPOSITION COMPLETE")
    print("="*80)
    
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
        #token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
        token = None
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
