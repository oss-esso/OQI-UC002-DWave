"""
Professional solver runner script with Non-Linear objective (A^0.548).

This script:
1. Loads a scenario (simple, intermediate, or custom)
2. Converts to CQM with piecewise linear approximation for non-linear objective
3. Saves the model
4. Solves with PuLP using piecewise approximation and saves results
5. (DWave solving disabled - token removed for testing)
6. Saves all constraints for verification
"""

import os
import sys
import json
import pickle
import shutil
import time
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.scenarios import load_food_data
from dimod import ConstrainedQuadraticModel, Binary, Real
from dwave.system import LeapHybridCQMSampler
import pulp as pl
from tqdm import tqdm
from Utils.piecewise_approximation import PiecewiseApproximation

# Try to import Pyomo for true non-linear solving
try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    print("Warning: Pyomo not available. Install with: pip install pyomo")

def create_cqm(farms, foods, food_groups, config, power=0.548, num_breakpoints=10):
    """
    Creates a CQM for the food optimization problem with piecewise linear approximation.
    Uses f(A) = A^power instead of linear objective.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food groups
        config: Configuration dictionary
        power: Power for non-linear objective (default: 0.548)
        num_breakpoints: Number of interior breakpoints for piecewise approximation
    
    Returns CQM, variables, constraint metadata, and approximation metadata.
    """
    cqm = ConstrainedQuadraticModel()
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    n_farms = len(farms)
    n_foods = len(foods)
    n_food_groups = len(food_groups) if food_group_constraints else 0
    
    # Determine max land value for piecewise approximation
    max_land = max(land_availability.values())
    
    # Create piecewise approximation for each farm-food pair
    print(f"\nCreating piecewise linear approximation: f(A) = A^{power}")
    print(f"  Using {num_breakpoints} interior points ({num_breakpoints + 2} total breakpoints)")
    print(f"  Domain: [0, {max_land}]")
    
    # Store approximations for each unique max_land value
    approximations = {}
    unique_max_lands = set(land_availability.values())
    
    for max_val in unique_max_lands:
        approx = PiecewiseApproximation(power=power, num_points=num_breakpoints, max_value=max_val)
        approximations[max_val] = approx
        max_abs, max_rel, avg_abs = approx.get_max_error()
        print(f"  Max land={max_val:.1f}: Max error={max_abs:.6f}, Avg error={avg_abs:.6f}")
    
    # Calculate total operations for progress bar
    # Additional variables needed: lambda variables for SOS2 (num_breakpoints+2 per farm-food pair)
    total_ops = (
        n_farms * n_foods * (2 + num_breakpoints + 2) +  # Variables (A, Y, and lambda_i for each breakpoint)
        n_farms * n_foods * 3 +       # Objective terms + SOS2 constraint + convexity constraint
        n_farms +                 # Land availability constraints
        n_farms * n_foods * 2 +   # Linking constraints (2 per farm-food pair)
        n_farms * n_food_groups * 2  # Food group constraints (min and max)
    )
    
    pbar = tqdm(total=total_ops, desc="Building CQM with piecewise approximation", unit="op", ncols=100)
    
    # Define variables
    A = {}
    Y = {}
    Lambda = {}  # Lambda variables for piecewise linear approximation (SOS2)
    f_approx = {}  # Approximated function values
    
    pbar.set_description("Creating area and binary variables")
    for farm in farms:
        for food in foods:
            A[(farm, food)] = Real(f"A_{farm}_{food}", lower_bound=0, upper_bound=land_availability[farm])
            pbar.update(1)
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
            pbar.update(1)
    
    # Create lambda variables and f_approx for piecewise approximation
    pbar.set_description("Creating piecewise approximation variables")
    for farm in farms:
        max_val = land_availability[farm]
        approx = approximations[max_val]
        n_breakpoints_total = len(approx.breakpoints)
        
        for food in foods:
            # Create lambda variables for each breakpoint (SOS2 constraint)
            Lambda[(farm, food)] = {}
            for i in range(n_breakpoints_total):
                Lambda[(farm, food)][i] = Real(
                    f"Lambda_{farm}_{food}_{i}", 
                    lower_bound=0, 
                    upper_bound=1
                )
                pbar.update(1)
            
            # f_approx will be computed as weighted sum of function values at breakpoints
            f_approx[(farm, food)] = sum(
                Lambda[(farm, food)][i] * approx.function_values[i] 
                for i in range(n_breakpoints_total)
            )
            pbar.update(1)
    
    # Piecewise approximation constraints
    pbar.set_description("Adding piecewise approximation constraints")
    for farm in farms:
        max_val = land_availability[farm]
        approx = approximations[max_val]
        n_breakpoints_total = len(approx.breakpoints)
        
        for food in foods:
            # Constraint 1: A[(farm, food)] = sum of lambda_i * breakpoint_i
            cqm.add_constraint(
                A[(farm, food)] - sum(
                    Lambda[(farm, food)][i] * approx.breakpoints[i] 
                    for i in range(n_breakpoints_total)
                ) == 0,
                label=f"Piecewise_A_Definition_{farm}_{food}"
            )
            pbar.update(1)
            
            # Constraint 2: Convexity constraint: sum of lambda_i = 1
            cqm.add_constraint(
                sum(Lambda[(farm, food)][i] for i in range(n_breakpoints_total)) == 1,
                label=f"Piecewise_Convexity_{farm}_{food}"
            )
            pbar.update(1)
            
            # Note: SOS2 constraint (at most 2 adjacent lambdas non-zero) 
            # is implicit in CQM solver - we rely on the convex combination
    
    # Objective function using piecewise approximation
    pbar.set_description("Building non-linear objective")
    total_area = sum(land_availability[farm] for farm in farms)
    
    objective = 0
    for farm in farms:
        for food in foods:
            # Use f_approx instead of A directly for non-linear objective
            objective += (
                weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * f_approx[(farm, food)] +
                weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * f_approx[(farm, food)] -
                weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * f_approx[(farm, food)] +
                weights.get('affordability', 0) * foods[food].get('affordability', 0) * f_approx[(farm, food)] +
                weights.get('sustainability', 0) * foods[food].get('sustainability', 0) * f_approx[(farm, food)]
            )
            pbar.update(1)
    
    # Normalize by total area to match PuLP and Pyomo formulations
    objective = objective / total_area
    cqm.set_objective(-objective)
    
    # Constraint metadata
    constraint_metadata = {
        'land_availability': {},
        'min_area_if_selected': {},
        'max_area_if_selected': {},
        'food_group_min': {},
        'food_group_max': {},
        'piecewise_a_definition': {},
        'piecewise_convexity': {}
    }
    
    # Store piecewise approximation metadata
    approximation_metadata = {
        'power': power,
        'num_interior_points': num_breakpoints,
        'approximations': {}
    }
    
    for max_val, approx in approximations.items():
        max_abs, max_rel, avg_abs = approx.get_max_error()
        approximation_metadata['approximations'][float(max_val)] = {
            'max_value': max_val,
            'total_breakpoints': len(approx.breakpoints),
            'breakpoints': approx.breakpoints.tolist(),
            'function_values': approx.function_values.tolist(),
            'max_absolute_error': max_abs,
            'max_relative_error_percent': max_rel * 100,
            'average_absolute_error': avg_abs
        }
    
    # Land availability constraints
    pbar.set_description("Adding land constraints")
    for farm in farms:
        cqm.add_constraint(
            sum(A[(farm, food)] for food in foods) - land_availability[farm] <= 0,
            label=f"Land_Availability_{farm}"
        )
        constraint_metadata['land_availability'][farm] = {
            'type': 'land_availability',
            'farm': farm,
            'max_land': land_availability[farm]
        }
        pbar.update(1)
    
    # Linking constraints
    pbar.set_description("Adding linking constraints")
    for farm in farms:
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
            pbar.update(1)
            
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
            pbar.update(1)
    
    # Food group constraints
    pbar.set_description("Adding food group constraints")
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                for farm in farms:
                    if 'min_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) - constraints['min_foods'] >= 0,
                            label=f"Food_Group_Min_{group}_{farm}"
                        )
                        constraint_metadata['food_group_min'][(group, farm)] = {
                            'type': 'food_group_min',
                            'group': group,
                            'farm': farm,
                            'min_foods': constraints['min_foods'],
                            'foods_in_group': foods_in_group
                        }
                        pbar.update(1)
                    
                    if 'max_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) - constraints['max_foods'] <= 0,
                            label=f"Food_Group_Max_{group}_{farm}"
                        )
                        constraint_metadata['food_group_max'][(group, farm)] = {
                            'type': 'food_group_max',
                            'group': group,
                            'farm': farm,
                            'max_foods': constraints['max_foods'],
                            'foods_in_group': foods_in_group
                        }
                        pbar.update(1)
    
    pbar.set_description("CQM complete")
    pbar.close()
    
    return cqm, A, Y, Lambda, constraint_metadata, approximation_metadata

def solve_with_pulp(farms, foods, food_groups, config, power=0.548, num_breakpoints=10):
    """
    Solve with PuLP using piecewise linear approximation for non-linear objective.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food groups
        config: Configuration dictionary
        power: Power for non-linear objective (default: 0.548)
        num_breakpoints: Number of interior breakpoints for piecewise approximation
    
    Returns model and results.
    """
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    print(f"\nCreating PuLP model with piecewise linear approximation...")
    print(f"  Power: {power}, Breakpoints: {num_breakpoints}")
    
    # Create piecewise approximations
    approximations = {}
    unique_max_lands = set(land_availability.values())
    
    for max_val in unique_max_lands:
        approx = PiecewiseApproximation(power=power, num_points=num_breakpoints, max_value=max_val)
        approximations[max_val] = approx
    
    # Decision variables
    A_pulp = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in foods], lowBound=0)
    Y_pulp = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in foods], cat='Binary')
    
    # Lambda variables for piecewise approximation
    Lambda_pulp = {}
    for f in farms:
        max_val = land_availability[f]
        approx = approximations[max_val]
        n_breakpoints_total = len(approx.breakpoints)
        
        for c in foods:
            Lambda_pulp[(f, c)] = {}
            for i in range(n_breakpoints_total):
                Lambda_pulp[(f, c)][i] = pl.LpVariable(
                    f"Lambda_{f}_{c}_{i}",
                    lowBound=0,
                    upBound=1,
                    cat='Continuous'
                )
    
    # Create model
    model = pl.LpProblem("Food_Optimization_NLN_PuLP", pl.LpMaximize)
    
    # Piecewise approximation constraints
    for f in farms:
        max_val = land_availability[f]
        approx = approximations[max_val]
        n_breakpoints_total = len(approx.breakpoints)
        
        for c in foods:
            # A = sum of lambda_i * breakpoint_i
            model += (
                A_pulp[(f, c)] == pl.lpSum([
                    Lambda_pulp[(f, c)][i] * approx.breakpoints[i] 
                    for i in range(n_breakpoints_total)
                ]),
                f"Piecewise_A_{f}_{c}"
            )
            
            # Convexity: sum of lambda_i = 1
            model += (
                pl.lpSum([Lambda_pulp[(f, c)][i] for i in range(n_breakpoints_total)]) == 1,
                f"Convexity_{f}_{c}"
            )
    
    # Objective function using piecewise approximation
    # f_approx = sum of lambda_i * f(breakpoint_i)
    total_area = sum(land_availability[f] for f in farms)
    
    objective_terms = []
    for f in farms:
        max_val = land_availability[f]
        approx = approximations[max_val]
        n_breakpoints_total = len(approx.breakpoints)
        
        for c in foods:
            # Approximated f(A) value
            f_approx = pl.lpSum([
                Lambda_pulp[(f, c)][i] * approx.function_values[i]
                for i in range(n_breakpoints_total)
            ])
            
            # Add to objective with weights
            coeff = (
                weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[c].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[c].get('sustainability', 0)
            )
            
            objective_terms.append(coeff * f_approx)
    
    goal = pl.lpSum(objective_terms) / total_area
    model += goal, "Objective"
    
    # Land availability constraints
    for f in farms:
        model += pl.lpSum([A_pulp[(f, c)] for c in foods]) <= land_availability[f], f"Max_Area_{f}"
    
    # Linking constraints (binary selection)
    for f in farms:
        for c in foods:
            A_min = min_planting_area.get(c, 0)
            model += A_pulp[(f, c)] >= A_min * Y_pulp[(f, c)], f"MinArea_{f}_{c}"
            model += A_pulp[(f, c)] <= land_availability[f] * Y_pulp[(f, c)], f"MaxArea_{f}_{c}"
    
    # Food group constraints
    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                for f in farms:
                    if 'min_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) >= constraints['min_foods'], f"MinFoodGroup_{f}_{g}"
                    if 'max_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) <= constraints['max_foods'], f"MaxFoodGroup_{f}_{g}"
    
    # Solve
    print("  Solving with CBC...")
    # For large problems, add time limit and allow optimality gap
    # This prevents CBC from getting stuck for hours
    n_vars = len(model.variables())
    n_constraints = len(model.constraints)
    
    if n_vars > 10000:
        # Large problem: use relaxed settings
        print(f"    Large problem detected ({n_vars} vars, {n_constraints} constraints)")
        print(f"    Using time limit and optimality gap for faster solving")
        solver = pl.PULP_CBC_CMD(
            msg=1,  # Show progress
            timeLimit=30,  # 10 minute limit
            gapRel=0.05,  # 5% optimality gap acceptable
            threads=4  # Use multiple threads
        )
    else:
        # Small problem: use default settings
        solver = pl.PULP_CBC_CMD(msg=0)
    
    start_time = time.time()
    model.solve(solver)
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': pl.value(model.objective),
        'solve_time': solve_time,
        'areas': {},
        'selections': {},
        'lambda_values': {}
    }
    
    for f in farms:
        for c in foods:
            key = f"{f}_{c}"
            results['areas'][key] = A_pulp[(f, c)].value() if A_pulp[(f, c)].value() is not None else 0.0
            results['selections'][key] = Y_pulp[(f, c)].value() if Y_pulp[(f, c)].value() is not None else 0.0
            
            # Store lambda values for verification
            max_val = land_availability[f]
            approx = approximations[max_val]
            n_breakpoints_total = len(approx.breakpoints)
            results['lambda_values'][key] = [
                Lambda_pulp[(f, c)][i].value() if Lambda_pulp[(f, c)][i].value() is not None else 0.0
                for i in range(n_breakpoints_total)
            ]
    
    return model, results

def solve_with_dwave(cqm, token):
    """Solve with DWave and return sampleset with hybrid solver timing."""
    sampler = LeapHybridCQMSampler(token=token)
    
    print("Submitting to DWave Leap hybrid solver...")
    
    sampleset = sampler.sample_cqm(cqm, label="Food Optimization - Professional Run")
    
    # Extract timing from sampleset.info
    # For hybrid solvers, the total solve time is in 'run_time' (microseconds)
    timing_info = sampleset.info.get('timing', {})
    hybrid_time = (timing_info.get('run_time') or 
                  sampleset.info.get('run_time') or 
                  sampleset.info.get('charge_time'))
    
    if hybrid_time is not None:
        hybrid_time = hybrid_time / 1e6  # convert microseconds to seconds
    
    # QPU access time (if available)
    qpu_time = (timing_info.get('qpu_access_time') or
               sampleset.info.get('qpu_access_time'))
    if qpu_time is not None:
        qpu_time = qpu_time / 1e6
    
    return sampleset, hybrid_time, qpu_time

def solve_with_pyomo(farms, foods, food_groups, config, power=0.548):
    """
    Solve with Pyomo using TRUE non-linear objective (no approximation).
    This uses the actual f(A) = A^power function directly.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food groups
        config: Configuration dictionary
        power: Power for non-linear objective (default: 0.548)
    
    Returns model and results.
    """
    if not PYOMO_AVAILABLE:
        print("ERROR: Pyomo is not installed. Skipping Pyomo solver.")
        return None, {
            'status': 'Not Available',
            'objective_value': None,
            'solve_time': 0.0,
            'areas': {},
            'selections': {},
            'error': 'Pyomo not installed'
        }
    
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    print(f"\nCreating Pyomo model with TRUE non-linear objective f(A) = A^{power}...")
    
    # Create model
    model = pyo.ConcreteModel(name="Food_Optimization_NLN_Pyomo")
    
    # Sets
    model.farms = pyo.Set(initialize=farms)
    model.foods = pyo.Set(initialize=list(foods.keys()))
    
    # Variables
    # Add small epsilon to avoid 0^0.548 evaluation issues in IPOPT
    epsilon = 1e-6
    model.A = pyo.Var(model.farms, model.foods, domain=pyo.NonNegativeReals,
                      bounds=lambda m, f, c: (epsilon, land_availability[f]))
    model.Y = pyo.Var(model.farms, model.foods, domain=pyo.Binary)
    
    # Objective function with TRUE power function
    total_area = sum(land_availability[f] for f in farms)
    
    def objective_rule(m):
        obj = 0
        for f in m.farms:
            for c in m.foods:
                # TRUE non-linear objective: A^power
                coeff = (
                    weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
                    weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
                    weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
                    weights.get('affordability', 0) * foods[c].get('affordability', 0) +
                    weights.get('sustainability', 0) * foods[c].get('sustainability', 0)
                )
                # Use the power function directly - this is the key difference!
                obj += coeff * (m.A[f, c] ** power)
        return obj / total_area
    
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    # Land availability constraints
    def land_constraint_rule(m, f):
        return sum(m.A[f, c] for c in m.foods) <= land_availability[f]
    model.land_constraint = pyo.Constraint(model.farms, rule=land_constraint_rule)
    
    # Linking constraints (binary selection)
    def min_area_rule(m, f, c):
        A_min = min_planting_area.get(c, 0)
        return m.A[f, c] >= A_min * m.Y[f, c]
    model.min_area = pyo.Constraint(model.farms, model.foods, rule=min_area_rule)
    
    def max_area_rule(m, f, c):
        return m.A[f, c] <= land_availability[f] * m.Y[f, c]
    model.max_area = pyo.Constraint(model.farms, model.foods, rule=max_area_rule)
    
    # Food group constraints
    if food_group_constraints:
        def min_food_group_rule(m, f, g):
            foods_in_group = food_groups.get(g, [])
            min_foods = food_group_constraints[g].get('min_foods', None)
            if min_foods is not None and foods_in_group:
                return sum(m.Y[f, c] for c in foods_in_group if c in m.foods) >= min_foods
            else:
                return pyo.Constraint.Skip
        
        def max_food_group_rule(m, f, g):
            foods_in_group = food_groups.get(g, [])
            max_foods = food_group_constraints[g].get('max_foods', None)
            if max_foods is not None and foods_in_group:
                return sum(m.Y[f, c] for c in foods_in_group if c in m.foods) <= max_foods
            else:
                return pyo.Constraint.Skip
        
        model.min_food_group = pyo.Constraint(
            model.farms, 
            list(food_group_constraints.keys()), 
            rule=min_food_group_rule
        )
        model.max_food_group = pyo.Constraint(
            model.farms, 
            list(food_group_constraints.keys()), 
            rule=max_food_group_rule
        )
    
    # Try to find an available MINLP solver
    # Order of preference: bonmin, couenne, ipopt
    solver_name = None
    solver = None
    
    print("  Searching for available MINLP solvers...")
    
    # First, try to find ipopt in conda environment
    conda_env_path = os.path.join(sys.prefix, 'Library', 'bin', 'ipopt.exe')
    if os.path.exists(conda_env_path):
        try:
            solver = pyo.SolverFactory('ipopt', executable=conda_env_path)
            solver_name = 'ipopt'
            print(f"  Found solver: {solver_name} at {conda_env_path}")
        except Exception as e:
            print(f"  Could not load ipopt from conda: {e}")
    
    # If not found, try standard solver detection
    if solver is None:
        solver_options = ['bonmin', 'couenne', 'ipopt']
        for solver_opt in solver_options:
            try:
                test_solver = pyo.SolverFactory(solver_opt)
                if test_solver.available():
                    solver_name = solver_opt
                    solver = test_solver
                    print(f"  Found solver: {solver_name}")
                    break
            except Exception as e:
                continue
    
    if solver is None:
        print("  ERROR: No suitable solver found.")
        print("  Install one of: bonmin, couenne, or ipopt")
        print("  For conda: conda install -c conda-forge ipopt")
        print("  For pip: pip install cyipopt")
        return model, {
            'status': 'No Solver',
            'objective_value': None,
            'solve_time': 0.0,
            'areas': {},
            'selections': {},
            'error': 'No MINLP solver available'
        }
    
    # Solve
    print(f"  Solving with {solver_name}...")
    start_time = time.time()
    
    try:
        results = solver.solve(model, tee=False)
        solve_time = time.time() - start_time
        
        # Extract results
        status = str(results.solver.status)
        termination = str(results.solver.termination_condition)
        
        output = {
            'status': f"{status} ({termination})",
            'solver': solver_name,
            'objective_value': pyo.value(model.obj) if pyo.value(model.obj) is not None else None,
            'solve_time': solve_time,
            'areas': {},
            'selections': {}
        }
        
        for f in farms:
            for c in foods:
                key = f"{f}_{c}"
                output['areas'][key] = pyo.value(model.A[f, c]) if model.A[f, c].value is not None else 0.0
                output['selections'][key] = pyo.value(model.Y[f, c]) if model.Y[f, c].value is not None else 0.0
        
        return model, output
        
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"  ERROR during solving: {str(e)}")
        return model, {
            'status': 'Error',
            'solver': solver_name,
            'objective_value': None,
            'solve_time': solve_time,
            'areas': {},
            'selections': {},
            'error': str(e)
        }

def main(scenario='simple', power=0.548, num_breakpoints=10):
    """Main execution function."""
    print("=" * 80)
    print("NON-LINEAR SOLVER RUNNER (A^0.548)")
    print("=" * 80)
    
    # Create output directories
    os.makedirs('PuLP_Results_NLN', exist_ok=True)
    os.makedirs('DWave_Results_NLN', exist_ok=True)
    os.makedirs('CQM_Models_NLN', exist_ok=True)
    os.makedirs('Constraints_NLN', exist_ok=True)
    
    # Load scenario
    print(f"\nLoading '{scenario}' scenario...")
    farms, foods, food_groups, config = load_food_data(scenario)
    print(f"  Farms: {len(farms)} - {farms}")
    print(f"  Foods: {len(foods)} - {list(foods.keys())}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CQM with piecewise approximation
    print("\nCreating CQM with non-linear objective...")
    cqm, A, Y, Lambda, constraint_metadata, approximation_metadata = create_cqm(
        farms, foods, food_groups, config, power=power, num_breakpoints=num_breakpoints
    )
    print(f"  Variables: {len(cqm.variables)}")
    print(f"  Constraints: {len(cqm.constraints)}")
    print(f"  Power function: f(A) = A^{power}")
    
    # Save CQM
    cqm_path = f'CQM_Models_NLN/cqm_nln_{scenario}_{timestamp}.cqm'
    print(f"\nSaving CQM to {cqm_path}...")
    with open(cqm_path, 'wb') as f:
        shutil.copyfileobj(cqm.to_file(), f)
    
    # Save constraint metadata
    constraints_path = f'Constraints_NLN/constraints_nln_{scenario}_{timestamp}.json'
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
        'power': power,
        'num_breakpoints': num_breakpoints,
        'approximation_metadata': approximation_metadata,
        'constraint_metadata': {
            'land_availability': {str(k): v for k, v in constraint_metadata['land_availability'].items()},
            'min_area_if_selected': {str(k): v for k, v in constraint_metadata['min_area_if_selected'].items()},
            'max_area_if_selected': {str(k): v for k, v in constraint_metadata['max_area_if_selected'].items()},
            'food_group_min': {str(k): v for k, v in constraint_metadata['food_group_min'].items()},
            'food_group_max': {str(k): v for k, v in constraint_metadata['food_group_max'].items()}
        }
    }
    
    with open(constraints_path, 'w') as f:
        json.dump(constraints_json, f, indent=2)
    
    # Solve with PuLP (piecewise approximation)
    print("\n" + "=" * 80)
    print("SOLVING WITH PULP (Piecewise Linear Approximation)")
    print("=" * 80)
    pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config, power=power, num_breakpoints=num_breakpoints)
    print(f"  Status: {pulp_results['status']}")
    print(f"  Objective: {pulp_results['objective_value']:.6f}")
    print(f"  Solve time: {pulp_results['solve_time']:.2f} seconds")
    
    # Save PuLP results
    pulp_path = f'PuLP_Results_NLN/pulp_nln_{scenario}_{timestamp}.json'
    print(f"\nSaving PuLP results to {pulp_path}...")
    with open(pulp_path, 'w') as f:
        json.dump(pulp_results, f, indent=2)
    
    # Solve with Pyomo (TRUE non-linear objective)
    print("\n" + "=" * 80)
    print("SOLVING WITH PYOMO (TRUE Non-Linear Objective)")
    print("=" * 80)
    pyomo_model, pyomo_results = solve_with_pyomo(farms, foods, food_groups, config, power=power)
    
    if pyomo_results.get('error'):
        print(f"  Status: {pyomo_results['status']}")
        print(f"  Error: {pyomo_results.get('error')}")
    else:
        print(f"  Solver: {pyomo_results.get('solver', 'Unknown')}")
        print(f"  Status: {pyomo_results['status']}")
        if pyomo_results['objective_value'] is not None:
            print(f"  Objective: {pyomo_results['objective_value']:.6f}")
        print(f"  Solve time: {pyomo_results['solve_time']:.2f} seconds")
    
    # Save Pyomo results
    pyomo_path = f'PuLP_Results_NLN/pyomo_nln_{scenario}_{timestamp}.json'
    print(f"\nSaving Pyomo results to {pyomo_path}...")
    
    # Convert to JSON-serializable format
    pyomo_results_serializable = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in pyomo_results.items()
        if k != 'lambda_values'  # Skip lambda values for Pyomo
    }
    
    with open(pyomo_path, 'w') as f:
        json.dump(pyomo_results_serializable, f, indent=2)
    
    # Solve with DWave
    print("\n" + "=" * 80)
    print("SOLVING WITH DWAVE (Piecewise Approximation)")
    print("=" * 80)
    
    #token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
    token = None
    if token:
        try:
            sampleset, dwave_solve_time = solve_with_dwave(cqm, token)
            
            feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
            print(f"  Feasible solutions: {len(feasible_sampleset)} of {len(sampleset)}")
            print(f"  Total solve time: {dwave_solve_time:.2f} seconds")
            
            if feasible_sampleset:
                best = feasible_sampleset.first
                print(f"  Best energy: {best.energy:.6f}")
                
                # Extract timing info
                timing_info = sampleset.info.get('timing', {})
                qpu_time = timing_info.get('qpu_access_time', 0) / 1e6  # Convert to seconds
                print(f"  QPU access time: {qpu_time:.4f} seconds")
            else:
                print("  WARNING: No feasible solutions found")
            
            # Save DWave results
            dwave_path = f'DWave_Results_NLN/dwave_nln_{scenario}_{timestamp}.pickle'
            print(f"\nSaving DWave results to {dwave_path}...")
            os.makedirs('DWave_Results_NLN', exist_ok=True)
            with open(dwave_path, 'wb') as f:
                pickle.dump(sampleset, f)
            
        except Exception as e:
            print(f"  ERROR: DWave solving failed: {str(e)}")
            dwave_path = None
    else:
        print("  DWave API token not found in environment")
        print("  Set DWAVE_API_TOKEN environment variable to enable DWave solving")
        dwave_path = None
    
    # Compare results if multiple solvers succeeded
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    if pulp_results['status'] == 'Optimal':
        print(f"  PuLP (approx):    {pulp_results['objective_value']:.6f}  |  {pulp_results['solve_time']:.2f}s")
    
    if pyomo_results.get('objective_value') is not None:
        print(f"  Pyomo (true NLN): {pyomo_results['objective_value']:.6f}  |  {pyomo_results['solve_time']:.2f}s")
    
    if dwave_path and feasible_sampleset:
        dwave_obj = -best.energy  # Convert energy back to objective
        print(f"  DWave (approx):   {dwave_obj:.6f}  |  {dwave_solve_time:.2f}s")
    
    # Detailed comparison if we have ground truth
    if pulp_results['status'] == 'Optimal' and pyomo_results.get('objective_value') is not None:
        print("\n  Approximation Quality:")
        pulp_obj = pulp_results['objective_value']
        pyomo_obj = pyomo_results['objective_value']
        diff = abs(pulp_obj - pyomo_obj)
        rel_diff = (diff / abs(pyomo_obj) * 100) if pyomo_obj != 0 else 0
        print(f"    PuLP vs Pyomo: {rel_diff:.2f}% difference")
        
        if dwave_path and feasible_sampleset:
            dwave_diff = abs(dwave_obj - pyomo_obj)
            dwave_rel_diff = (dwave_diff / abs(pyomo_obj) * 100) if pyomo_obj != 0 else 0
            print(f"    DWave vs Pyomo: {dwave_rel_diff:.2f}% difference")
    
    print("\n" + "=" * 80)
    print("SOLVER RUN COMPLETE")
    print("=" * 80)
    print(f"CQM saved to: {cqm_path}")
    print(f"Constraints saved to: {constraints_path}")
    print(f"PuLP results saved to: {pulp_path}")
    print(f"Pyomo results saved to: {pyomo_path}")
    if dwave_path:
        print(f"DWave results saved to: {dwave_path}")
    print(f"\nNon-linear objective: f(A) = A^{power}")
    print(f"PuLP & DWave: Piecewise approximation with {num_breakpoints} interior points")
    print(f"Pyomo: True non-linear objective (no approximation)")
    
    return cqm_path, constraints_path, pulp_path, pyomo_path, dwave_path if dwave_path else None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run solvers with non-linear objective (A^power) on a food optimization scenario'
    )
    parser.add_argument('--scenario', type=str, default='simple', 
                       choices=['simple', 'intermediate', 'full', 'custom', 'full_family'],
                       help='Scenario to solve (default: simple)')
    parser.add_argument('--power', type=float, default=0.548,
                       help='Power for non-linear objective f(A) = A^power (default: 0.548)')
    parser.add_argument('--breakpoints', type=int, default=10,
                       help='Number of interior breakpoints for piecewise approximation (default: 10)')
    
    args = parser.parse_args()
    
    main(scenario=args.scenario, power=args.power, num_breakpoints=args.breakpoints)
