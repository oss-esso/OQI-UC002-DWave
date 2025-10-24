"""
Professional solver runner script.

This script:
1. Loads a scenario (simple, intermediate, or custom)
2. Converts to CQM and saves the model
3. Solves with PuLP and saves results
4. Solves with DWave and saves results
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
from dimod import ConstrainedQuadraticModel, Binary, Real
from dwave.system import LeapHybridCQMSampler
import pulp as pl
from tqdm import tqdm

def create_cqm(farms, foods, food_groups, config):
    """
    Creates a CQM for the food optimization problem with progress bar.
    Returns CQM, variables, and constraint metadata.
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
    
    # Calculate total operations for progress bar
    total_ops = (
        n_farms * n_foods * 2 +  # Variables (A and Y)
        n_farms * n_foods +       # Objective terms
        n_farms +                 # Land availability constraints
        n_farms * n_foods * 2 +   # Linking constraints (2 per farm-food pair)
        n_farms * n_food_groups * 2  # Food group constraints (min and max)
    )
    
    pbar = tqdm(total=total_ops, desc="Building CQM", unit="op", ncols=100)
    
    # Define variables
    A = {}
    Y = {}
    
    pbar.set_description("Creating variables")
    for farm in farms:
        for food in foods:
            A[(farm, food)] = Real(f"A_{farm}_{food}", lower_bound=0, upper_bound=land_availability[farm])
            pbar.update(1)
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
            pbar.update(1)
    
    # Objective function - NONLINEAR (fractional programming)
    # Objective: max (weighted_sum / sum(A_{f,c}))
    # This creates a MINLP instead of MILP
    pbar.set_description("Building objective (MINLP)")
    
    # Build numerator: weighted sum of objectives * areas
    numerator = 0
    for farm in farms:
        for food in foods:
            numerator += (
                weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * A[(farm, food)] +
                weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * A[(farm, food)] -
                weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * A[(farm, food)] +
                weights.get('affordability', 0) * foods[food].get('affordability', 0) * A[(farm, food)] +
                weights.get('sustainability', 0) * foods[food].get('sustainability', 0) * A[(farm, food)]
            )
            pbar.update(1)
    
    # Build denominator: sum of all allocated areas (decision variables)
    denominator = sum(A[(farm, food)] for farm in farms for food in foods)
    
    # Fractional objective: numerator / denominator
    # NOTE: CQM doesn't support division directly, so this needs special handling
    # For now, we'll use the numerator only and handle the division in specialized solvers
    cqm.set_objective(-numerator)  # Placeholder - will be handled by Pyomo/Dinkelbach
    
    # Constraint metadata
    constraint_metadata = {
        'land_availability': {},
        'min_area_if_selected': {},
        'max_area_if_selected': {},
        'food_group_min': {},
        'food_group_max': {}
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
    
    return cqm, A, Y, constraint_metadata

def solve_with_pulp(farms, foods, food_groups, config):
    """
    Solve fractional program with PuLP using Dinkelbach's algorithm.
    
    Dinkelbach's iterative linearization:
    1. Initialize lambda_k = 0
    2. Solve LP: max f(x) - lambda_k * g(x)
    3. Update lambda_k = f(x*) / g(x*)
    4. Repeat until |f(x*) - lambda_k * g(x*)| < epsilon
    
    Returns model and results with convergence history.
    """
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    # Dinkelbach's algorithm parameters
    lambda_k = 0.0
    epsilon = 1e-6
    max_iterations = 100
    convergence_history = []
    
    # Calculate minimum total allocation based on food group constraints
    # This ensures denominator is never zero
    min_total_allocation = 0
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            min_foods_per_farm = constraints.get('min_foods', 0)
            if min_foods_per_farm > 0 and foods_in_group:
                # Each farm must plant at least min_foods crops from this group
                # Use the minimum planting area for this estimate
                min_areas_for_group = [min_planting_area.get(food, 0) for food in foods_in_group]
                if min_areas_for_group:
                    min_area = min([a for a in min_areas_for_group if a > 0] or [0])
                    min_total_allocation += len(farms) * min_foods_per_farm * min_area
    
    # Ensure at least some minimum allocation
    min_total_allocation = max(min_total_allocation, 0.1)
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Create fresh model for this iteration
        A_pulp = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in foods], lowBound=0)
        Y_pulp = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in foods], cat='Binary')
        
        # Build numerator: f(x) = weighted sum of objectives * areas
        numerator = (
            weights.get('nutritional_value', 0) * pl.lpSum([(foods[c].get('nutritional_value', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) +
            weights.get('nutrient_density', 0) * pl.lpSum([(foods[c].get('nutrient_density', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) -
            weights.get('environmental_impact', 0) * pl.lpSum([(foods[c].get('environmental_impact', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) +
            weights.get('affordability', 0) * pl.lpSum([(foods[c].get('affordability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) +
            weights.get('sustainability', 0) * pl.lpSum([(foods[c].get('sustainability', 0) * A_pulp[(f, c)]) for f in farms for c in foods])
        )
        
        # Build denominator: g(x) = sum of all allocated areas
        denominator = pl.lpSum([A_pulp[(f, c)] for f in farms for c in foods])
        
        # Dinkelbach's transformed objective: max (f(x) - lambda_k * g(x))
        goal = numerator - lambda_k * denominator
    
        model = pl.LpProblem(f"Food_Optimization_Dinkelbach_Iter{iteration}", pl.LpMaximize)
        
        # Add all constraints
        for f in farms:
            model += pl.lpSum([A_pulp[(f, c)] for c in foods]) <= land_availability[f], f"Max_Area_{f}"
        
        for f in farms:
            for c in foods:
                A_min = min_planting_area.get(c, 0)
                model += A_pulp[(f, c)] >= A_min * Y_pulp[(f, c)], f"MinArea_{f}_{c}"
                model += A_pulp[(f, c)] <= land_availability[f] * Y_pulp[(f, c)], f"MaxArea_{f}_{c}"
        
        if food_group_constraints:
            for g, constraints in food_group_constraints.items():
                foods_in_group = food_groups.get(g, [])
                if foods_in_group:
                    for f in farms:
                        if 'min_foods' in constraints:
                            model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) >= constraints['min_foods'], f"MinFoodGroup_{f}_{g}"
                        if 'max_foods' in constraints:
                            model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) <= constraints['max_foods'], f"MaxFoodGroup_{f}_{g}"
        
        # Add constraint to ensure minimum total allocation (prevents zero denominator)
        model += denominator >= min_total_allocation, "MinTotalAllocation"
        
        model += goal, "Objective"
        
        # Solve current iteration
        model.solve(pl.PULP_CBC_CMD(msg=0))
        
        if model.status != 1:  # Not optimal
            print(f"  Dinkelbach iteration {iteration}: Non-optimal status {pl.LpStatus[model.status]}")
            break
        
        # Compute f(x*) and g(x*)
        f_x_star = sum([
            (weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
             weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
             weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
             weights.get('affordability', 0) * foods[c].get('affordability', 0) +
             weights.get('sustainability', 0) * foods[c].get('sustainability', 0)) * A_pulp[(f, c)].value()
            for f in farms for c in foods
        ])
        
        g_x_star = sum([A_pulp[(f, c)].value() for f in farms for c in foods])
        
        # Avoid division by zero
        if g_x_star < 1e-10:
            print(f"  Dinkelbach iteration {iteration}: Denominator too small ({g_x_star}), stopping")
            break
        
        # Update lambda
        lambda_new = f_x_star / g_x_star
        
        # Check convergence
        residual = abs(f_x_star - lambda_k * g_x_star)
        convergence_history.append({
            'iteration': iteration,
            'lambda': lambda_new,
            'f_x': f_x_star,
            'g_x': g_x_star,
            'residual': residual,
            'objective_ratio': lambda_new
        })
        
        print(f"  Dinkelbach iteration {iteration}: lambda={lambda_new:.6f}, residual={residual:.2e}")
        
        if residual < epsilon:
            print(f"  Dinkelbach converged in {iteration+1} iterations!")
            break
        
        lambda_k = lambda_new
    
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': lambda_k,  # Final fractional objective value
        'solve_time': solve_time,
        'iterations': len(convergence_history),
        'convergence_history': convergence_history,
        'areas': {},
        'selections': {}
    }
    
    for f in farms:
        for c in foods:
            key = f"{f}_{c}"
            results['areas'][key] = A_pulp[(f, c)].value() if A_pulp[(f, c)].value() is not None else 0.0
            results['selections'][key] = Y_pulp[(f, c)].value() if Y_pulp[(f, c)].value() is not None else 0.0
    
    return model, results

def solve_with_pyomo(farms, foods, food_groups, config):
    """
    Solve fractional MINLP directly with Pyomo using MINLP solvers.
    
    Supports solvers: Ipopt, BARON, Couenne, SCIP
    Uses fractional objective: max f(x) / g(x)
    
    Returns results in same format as PuLP solver.
    """
    try:
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory
    except ImportError:
        return None, {
            'status': 'Failed',
            'error': 'Pyomo not installed. Install with: pip install pyomo',
            'objective_value': None,
            'solve_time': 0
        }
    
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    start_time = time.time()
    
    # Create Pyomo model
    model = pyo.ConcreteModel()
    
    # Define sets
    model.farms = pyo.Set(initialize=farms)
    model.foods = pyo.Set(initialize=list(foods.keys()))
    
    # Define variables
    model.A = pyo.Var(model.farms, model.foods, domain=pyo.NonNegativeReals, bounds=(0, None))
    model.Y = pyo.Var(model.farms, model.foods, domain=pyo.Binary)
    
    # Build numerator expression
    def numerator_rule(m):
        return sum(
            (weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
             weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
             weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
             weights.get('affordability', 0) * foods[c].get('affordability', 0) +
             weights.get('sustainability', 0) * foods[c].get('sustainability', 0)) * m.A[f, c]
            for f in m.farms for c in m.foods
        )
    model.numerator = pyo.Expression(rule=numerator_rule)
    
    # Build denominator expression
    def denominator_rule(m):
        return sum(m.A[f, c] for f in m.farms for c in m.foods)
    model.denominator = pyo.Expression(rule=denominator_rule)
    
    # Fractional objective: max numerator / denominator
    # Add small epsilon to denominator to avoid division by zero
    def objective_rule(m):
        return m.numerator / (m.denominator + 1e-8)
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    # Land availability constraints
    def land_constraint_rule(m, f):
        return sum(m.A[f, c] for c in m.foods) <= land_availability[f]
    model.land_constraints = pyo.Constraint(model.farms, rule=land_constraint_rule)
    
    # Linking constraints: minimum area if selected
    def min_area_rule(m, f, c):
        A_min = min_planting_area.get(c, 0)
        return m.A[f, c] >= A_min * m.Y[f, c]
    model.min_area_constraints = pyo.Constraint(model.farms, model.foods, rule=min_area_rule)
    
    # Linking constraints: maximum area if selected
    def max_area_rule(m, f, c):
        return m.A[f, c] <= land_availability[f] * m.Y[f, c]
    model.max_area_constraints = pyo.Constraint(model.farms, model.foods, rule=max_area_rule)
    
    # Food group constraints
    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                for f in farms:
                    if 'min_foods' in constraints:
                        min_val = constraints['min_foods']
                        model.add_component(
                            f'min_food_group_{g}_{f}',
                            pyo.Constraint(expr=sum(model.Y[f, c] for c in foods_in_group) >= min_val)
                        )
                    if 'max_foods' in constraints:
                        max_val = constraints['max_foods']
                        model.add_component(
                            f'max_food_group_{g}_{f}',
                            pyo.Constraint(expr=sum(model.Y[f, c] for c in foods_in_group) <= max_val)
                        )
    
    # Try to solve with available MINLP solvers
    solvers_to_try = ['ipopt', 'couenne', 'baron', 'scip']
    solved = False
    solver_used = None
    
    for solver_name in solvers_to_try:
        try:
            solver = SolverFactory(solver_name)
            if solver.available():
                print(f"  Solving with Pyomo/{solver_name}...")
                results = solver.solve(model, tee=False)
                
                if (results.solver.status == pyo.SolverStatus.ok and 
                    results.solver.termination_condition == pyo.TerminationCondition.optimal):
                    solved = True
                    solver_used = solver_name
                    print(f"  ✓ {solver_name} found optimal solution")
                    break
                else:
                    print(f"  ✗ {solver_name} failed: {results.solver.termination_condition}")
        except Exception as e:
            print(f"  ✗ {solver_name} error: {e}")
            continue
    
    solve_time = time.time() - start_time
    
    if not solved:
        return model, {
            'status': 'Failed',
            'error': 'No available MINLP solver found. Install Ipopt: conda install -c conda-forge ipopt',
            'objective_value': None,
            'solve_time': solve_time
        }
    
    # Extract results
    results_dict = {
        'status': 'Optimal',
        'solver': solver_used,
        'objective_value': pyo.value(model.obj),
        'solve_time': solve_time,
        'areas': {},
        'selections': {}
    }
    
    for f in farms:
        for c in foods:
            key = f"{f}_{c}"
            results_dict['areas'][key] = pyo.value(model.A[f, c])
            results_dict['selections'][key] = pyo.value(model.Y[f, c])
    
    return model, results_dict

def solve_with_dwave(cqm, token):
    """Solve with DWave and return sampleset."""
    sampler = LeapHybridCQMSampler(token=token)
    
    print("Submitting to DWave Leap hybrid solver...")
    start_time = time.time()
    sampleset = sampler.sample_cqm(cqm, label="Food Optimization - Professional Run")
    solve_time = time.time() - start_time
    
    return sampleset, solve_time

def solve_with_dwave_charnes_cooper(farms, foods, food_groups, config, token):
    """
    Solve fractional program with D-Wave using Charnes-Cooper transformation.
    
    Charnes-Cooper transformation for fractional program max f(x)/g(x):
    1. Introduce new variables: z = x/g(x), t = 1/g(x)
    2. Reformulate as: max t*f(z/t) subject to t*g(z/t) = 1, t > 0
    3. For linear f,g: max f(z) subject to g(z) = 1, t > 0, z = t*x
    
    Note: This is an approximation as CQM doesn't fully support all transformations.
    We'll use a piecewise linear approximation of the fractional objective.
    """
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    print("  Building Charnes-Cooper transformed CQM...")
    
    
    
    # Create CQM with transformed variables
    cqm = ConstrainedQuadraticModel()
    
    # Original variables: A[f,c] and Y[f,c]
    A = {}
    Y = {}
    
    for farm in farms:
        for food in foods:
            A[(farm, food)] = Real(f"A_{farm}_{food}", lower_bound=0, upper_bound=land_availability[farm])
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
    
    # Auxiliary variable: t = 1 / sum(A)
    # We'll approximate the fractional objective using constraints
    # For simplicity, we'll normalize by adding a constraint that sum(A) is bounded
    
    # Build numerator (same as before)
    numerator = 0
    for farm in farms:
        for food in foods:
            coeff = (
                weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) +
                weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) -
                weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) +
                weights.get('affordability', 0) * foods[food].get('affordability', 0) +
                weights.get('sustainability', 0) * foods[food].get('sustainability', 0)
            )
            numerator += coeff * A[(farm, food)]
    
    # Denominator
    denominator = sum(A[(farm, food)] for farm in farms for food in foods)
    
    # Charnes-Cooper approximation: Add constraint that denominator = constant
    # This converts the fractional program to a linear program
    # We'll use a target normalization value based on total available land
    total_land = sum(land_availability[farm] for farm in farms)
    normalization_target = total_land * 0.5  # Use 50% of total land as normalization
    
    cqm.add_constraint(
        denominator - normalization_target == 0,
        label="CharnesCooper_Normalization"
    )
    
    # Now objective is just the numerator (since denominator is fixed)
    cqm.set_objective(-numerator)
    
    # Add all other constraints (same as original CQM)
    for farm in farms:
        cqm.add_constraint(
            sum(A[(farm, food)] for food in foods) - land_availability[farm] <= 0,
            label=f"Land_Availability_{farm}"
        )
    
    for farm in farms:
        for food in foods:
            A_min = min_planting_area.get(food, 0)
            cqm.add_constraint(
                A[(farm, food)] - A_min * Y[(farm, food)] >= 0,
                label=f"Min_Area_If_Selected_{farm}_{food}"
            )
            cqm.add_constraint(
                A[(farm, food)] - land_availability[farm] * Y[(farm, food)] <= 0,
                label=f"Max_Area_If_Selected_{farm}_{food}"
            )
    
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
                    if 'max_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) - constraints['max_foods'] <= 0,
                            label=f"Food_Group_Max_{group}_{farm}"
                        )
    
    # Solve with D-Wave
    sampler = LeapHybridCQMSampler(token=token)
    print("  Submitting Charnes-Cooper CQM to D-Wave...")
    sampleset = sampler.sample_cqm(cqm, label="Food Optimization - Charnes-Cooper")
    
    solve_time = sampleset.info.get('charge_time', 0) / 1e6
    
    # Extract results
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    
    if len(feasible_sampleset) > 0:
        best = feasible_sampleset.first
        
        # Calculate the actual fractional objective
        numerator_val = sum([
            (weights.get('nutritional_value', 0) * foods[c].get('nutritional_value', 0) +
             weights.get('nutrient_density', 0) * foods[c].get('nutrient_density', 0) -
             weights.get('environmental_impact', 0) * foods[c].get('environmental_impact', 0) +
             weights.get('affordability', 0) * foods[c].get('affordability', 0) +
             weights.get('sustainability', 0) * foods[c].get('sustainability', 0)) * best.sample[f"A_{f}_{c}"]
            for f in farms for c in foods
        ])
        
        denominator_val = sum([best.sample[f"A_{f}_{c}"] for f in farms for c in foods])
        
        if denominator_val > 0:
            objective_value = numerator_val / denominator_val
        else:
            objective_value = 0
        
        results = {
            'status': 'Optimal',
            'objective_value': objective_value,
            'solve_time': solve_time,
            'feasible_count': len(feasible_sampleset),
            'total_count': len(sampleset),
            'areas': {},
            'selections': {}
        }
        
        for f in farms:
            for c in foods:
                key = f"{f}_{c}"
                results['areas'][key] = best.sample.get(f"A_{f}_{c}", 0.0)
                results['selections'][key] = best.sample.get(f"Y_{f}_{c}", 0.0)
    else:
        results = {
            'status': 'No Feasible Solution',
            'objective_value': None,
            'solve_time': solve_time,
            'feasible_count': 0,
            'total_count': len(sampleset),
            'areas': {},
            'selections': {}
        }
    
    return sampleset, results

def main(scenario='simple'):
    """Main execution function."""
    print("=" * 80)
    print("PROFESSIONAL SOLVER RUNNER")
    print("=" * 80)
    
    # Create output directories
    os.makedirs('PuLP_Results', exist_ok=True)
    os.makedirs('DWave_Results', exist_ok=True)
    os.makedirs('CQM_Models', exist_ok=True)
    os.makedirs('Constraints', exist_ok=True)
    
    # Load scenario
    print(f"\nLoading '{scenario}' scenario...")
    farms, foods, food_groups, config = load_food_data(scenario)
    print(f"  Farms: {len(farms)} - {farms}")
    print(f"  Foods: {len(foods)} - {list(foods.keys())}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CQM
    print("\nCreating CQM...")
    cqm, A, Y, constraint_metadata = create_cqm(farms, foods, food_groups, config)
    print(f"  Variables: {len(cqm.variables)}")
    print(f"  Constraints: {len(cqm.constraints)}")
    
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
            'land_availability': {str(k): v for k, v in constraint_metadata['land_availability'].items()},
            'min_area_if_selected': {str(k): v for k, v in constraint_metadata['min_area_if_selected'].items()},
            'max_area_if_selected': {str(k): v for k, v in constraint_metadata['max_area_if_selected'].items()},
            'food_group_min': {str(k): v for k, v in constraint_metadata['food_group_min'].items()},
            'food_group_max': {str(k): v for k, v in constraint_metadata['food_group_max'].items()}
        }
    }
    
    with open(constraints_path, 'w') as f:
        json.dump(constraints_json, f, indent=2)
    
    # Solve with PuLP (Dinkelbach's algorithm)
    print("\n" + "=" * 80)
    print("SOLVING WITH PULP (Dinkelbach's Algorithm)")
    print("=" * 80)
    pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config)
    print(f"  Status: {pulp_results['status']}")
    if pulp_results.get('objective_value') is not None:
        print(f"  Objective: {pulp_results['objective_value']:.6f}")
        print(f"  Iterations: {pulp_results.get('iterations', 'N/A')}")
    print(f"  Solve time: {pulp_results['solve_time']:.2f} seconds")
    
    # Solve with Pyomo (direct MINLP)
    print("\n" + "=" * 80)
    print("SOLVING WITH PYOMO (Direct MINLP)")
    print("=" * 80)
    pyomo_model, pyomo_results = solve_with_pyomo(farms, foods, food_groups, config)
    if pyomo_results.get('status') == 'Optimal':
        print(f"  Status: {pyomo_results['status']}")
        print(f"  Solver: {pyomo_results.get('solver', 'N/A')}")
        print(f"  Objective: {pyomo_results['objective_value']:.6f}")
        print(f"  Solve time: {pyomo_results['solve_time']:.2f} seconds")
    else:
        print(f"  Status: {pyomo_results.get('status', 'Failed')}")
        print(f"  Error: {pyomo_results.get('error', 'Unknown error')}")
        print(f"  Solve time: {pyomo_results['solve_time']:.2f} seconds")
    
    # Save PuLP results
    pulp_path = f'PuLP_Results/pulp_{scenario}_{timestamp}.json'
    print(f"\nSaving PuLP results to {pulp_path}...")
    with open(pulp_path, 'w') as f:
        json.dump(pulp_results, f, indent=2)
    
    # Save Pyomo results
    os.makedirs('Pyomo_Results', exist_ok=True)
    pyomo_path = f'Pyomo_Results/pyomo_{scenario}_{timestamp}.json'
    print(f"Saving Pyomo results to {pyomo_path}...")
    with open(pyomo_path, 'w') as f:
        json.dump(pyomo_results, f, indent=2)
    
    # Solve with DWave
    print("\n" + "=" * 80)
    print("SOLVING WITH DWAVE")
    print("=" * 80)
    token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
    sampleset, dwave_solve_time = solve_with_dwave(cqm, token)
    
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    print(f"  Feasible solutions: {len(feasible_sampleset)} of {len(sampleset)}")
    print(f"  Solve time: {dwave_solve_time:.2f} seconds")
    
    if feasible_sampleset:
        best = feasible_sampleset.first
        print(f"  Best energy: {best.energy:.6f}")
    
    # Save DWave results
    dwave_path = f'DWave_Results/dwave_{scenario}_{timestamp}.pickle'
    print(f"\nSaving DWave results to {dwave_path}...")
    with open(dwave_path, 'wb') as f:
        pickle.dump(sampleset, f)
    
    # Create run manifest
    manifest = {
        'scenario': scenario,
        'timestamp': timestamp,
        'cqm_path': cqm_path,
        'constraints_path': constraints_path,
        'pulp_path': pulp_path,
        'pyomo_path': pyomo_path,
        'dwave_path': dwave_path,
        'farms': farms,
        'foods': list(foods.keys()),
        'pulp_status': pulp_results['status'],
        'pulp_objective': pulp_results.get('objective_value'),
        'pulp_iterations': pulp_results.get('iterations'),
        'pyomo_status': pyomo_results.get('status'),
        'pyomo_objective': pyomo_results.get('objective_value'),
        'pyomo_solver': pyomo_results.get('solver'),
        'dwave_feasible_count': len(feasible_sampleset),
        'dwave_total_count': len(sampleset)
    }
    
    manifest_path = f'run_manifest_{scenario}_{timestamp}.json'
    print(f"\nSaving run manifest to {manifest_path}...")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SOLVER RUN COMPLETE")
    print("=" * 80)
    print(f"Manifest file: {manifest_path}")
    print("\nRun the verifier script with this manifest to check results:")
    print(f"  python verifier.py {manifest_path}")
    
    return manifest_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run solvers on a food optimization scenario')
    parser.add_argument('--scenario', type=str, default='simple', 
                       choices=['simple', 'intermediate', 'full', 'custom', 'full_family'],
                       help='Scenario to solve (default: simple)')
    
    args = parser.parse_args()
    
    main(scenario=args.scenario)
