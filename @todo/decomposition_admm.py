"""
ADMM (Alternating Direction Method of Multipliers) Decomposition

Splits the problem into A-subproblem and Y-subproblem with dual variable updates.
Converges through iterative consensus between continuous and binary variables.
"""
import time
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple
import numpy as np

from result_formatter import format_admm_result, validate_solution_constraints


def solve_with_admm(
    farms: Dict[str, float],
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    max_iterations: int = 100,
    rho: float = 1.0,
    tolerance: float = 1e-3,
    time_limit: float = 300.0
) -> Dict:
    """
    Solve farm allocation problem using ADMM.
    
    ADMM splits the problem:
    - A-subproblem: Optimize continuous allocations
    - Y-subproblem: Optimize binary selections
    - Consensus: Enforce A-Y compatibility via dual updates
    
    Args:
        farms: Dictionary of farm names to land availability
        foods: List of food names
        food_groups: Dictionary of food group constraints
        config: Configuration dictionary
        max_iterations: Maximum ADMM iterations
        rho: ADMM penalty parameter
        tolerance: Primal/dual residual tolerance
        time_limit: Maximum total solve time
    
    Returns:
        Formatted result dictionary
    """
    start_time = time.time()
    
    # Extract parameters
    params = config.get('parameters', {})
    min_planting_area = params.get('minimum_planting_area', {})
    max_planting_area = params.get('maximum_planting_area', {})
    benefits = config.get('benefits', {})
    
    # Initialize variables
    A = {(f, c): 0.0 for f in farms for c in foods}
    Y = {(f, c): 0.0 for f in farms for c in foods}
    Z = {(f, c): 0.0 for f in farms for c in foods}  # Consensus variable
    U = {(f, c): 0.0 for f in farms for c in foods}  # Scaled dual variable
    
    admm_iterations = []
    
    for iteration in range(1, max_iterations + 1):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            print(f"ADMM time limit reached at iteration {iteration}")
            break
        
        print(f"\nADMM Iteration {iteration}")
        
        # Step 1: Update A (continuous subproblem)
        A = solve_A_subproblem(farms, foods, Y, Z, U, rho, benefits, min_planting_area, max_planting_area)
        
        # Step 2: Update Y (binary subproblem)
        Y = solve_Y_subproblem(farms, foods, food_groups, A, Z, U, rho, benefits, config)
        
        # Step 3: Update Z (consensus variable)
        Z_old = Z.copy()
        for key in Z:
            Z[key] = (A[key] + Y[key]) / 2.0
        
        # Step 4: Update dual variable U
        for key in U:
            U[key] = U[key] + (A[key] - Y[key])
        
        # Calculate residuals
        primal_residual = np.sqrt(sum((A[key] - Y[key])**2 for key in A))
        dual_residual = np.sqrt(sum((Z[key] - Z_old[key])**2 for key in Z))
        
        # Calculate objective (benefit per hectare, normalized by total area)
        total_area = sum(farms.values())
        obj_value = sum(A[(f, c)] * benefits.get(c, 1.0) for f in farms for c in foods) / total_area
        
        print(f"  Objective: {obj_value:.4f}")
        print(f"  Primal Residual: {primal_residual:.6f}")
        print(f"  Dual Residual: {dual_residual:.6f}")
        
        admm_iterations.append({
            'iteration': iteration,
            'objective': obj_value,
            'primal_residual': primal_residual,
            'dual_residual': dual_residual,
            'time': elapsed
        })
        
        # Check convergence
        if primal_residual < tolerance and dual_residual < tolerance:
            print(f"  ADMM Converged at iteration {iteration}")
            break
    
    total_time = time.time() - start_time
    
    # Build final solution (use Y for binary, A for continuous)
    final_solution = {
        **{f"A_{f}_{c}": A[(f, c)] for f, c in A},
        **{f"Y_{f}_{c}": 1.0 if Y[(f, c)] > 0.5 else 0.0 for f, c in Y}
    }
    
    total_area = sum(farms.values())
    final_obj = sum(A[(f, c)] * benefits.get(c, 1.0) for f in farms for c in foods) / total_area
    
    # Validate solution
    validation = validate_solution_constraints(
        final_solution, farms, foods, food_groups, farms, config, 'farm'
    )
    
    # Format result
    result = format_admm_result(
        iterations=admm_iterations,
        final_solution=final_solution,
        objective_value=final_obj,
        total_time=total_time,
        scenario_type='farm',
        n_units=len(farms),
        n_foods=len(foods),
        total_area=sum(farms.values()),
        is_feasible=validation['is_feasible'],
        validation_results=validation,
        num_variables=2 * len(farms) * len(foods),
        num_constraints=len(farms) + len(food_groups) * 2,
        rho=rho,
        tolerance=tolerance
    )
    
    return result


def solve_A_subproblem(
    farms: Dict,
    foods: List[str],
    Y: Dict,
    Z: Dict,
    U: Dict,
    rho: float,
    benefits: Dict,
    min_planting_area: Dict,
    max_planting_area: Dict
) -> Dict:
    """Solve A-subproblem (continuous allocation)."""
    
    model = gp.Model("ADMM_A_Subproblem")
    model.setParam('OutputFlag', 0)
    
    # Variables
    A_vars = {}
    for farm in farms:
        for food in foods:
            A_vars[(farm, food)] = model.addVar(lb=0.0, name=f"A_{farm}_{food}")
    
    # Objective: benefit per hectare + ADMM penalty term (normalized by total area)
    total_area = sum(farms.values())
    obj_expr = gp.quicksum(
        A_vars[(f, c)] * benefits.get(c, 1.0) for f in farms for c in foods
    ) / total_area
    # Add ADMM augmented Lagrangian term: -U*(A-Y) - (rho/2)*||A-Y||^2
    # Simplified to: -U*A + (rho/2)*(A-Y)^2
    penalty = gp.quicksum(
        (rho / 2.0) * (A_vars[(f, c)] - Y[(f, c)])**2 - U[(f, c)] * A_vars[(f, c)]
        for f in farms for c in foods
    )
    
    model.setObjective(obj_expr - penalty, GRB.MAXIMIZE)
    
    # Constraints: Land availability
    for farm, capacity in farms.items():
        model.addConstr(
            gp.quicksum(A_vars[(farm, food)] for food in foods) <= capacity,
            name=f"Land_{farm}"
        )
    
    # Min/max area constraints
    for farm in farms:
        for food in foods:
            y_val = Y.get((farm, food), 0.0)
            if y_val > 0.5:
                min_area = min_planting_area.get(food, 0.0001)
                max_area = max_planting_area.get(food, farms[farm])
                model.addConstr(A_vars[(farm, food)] >= min_area, name=f"MinA_{farm}_{food}")
                model.addConstr(A_vars[(farm, food)] <= max_area, name=f"MaxA_{farm}_{food}")
            else:
                model.addConstr(A_vars[(farm, food)] == 0.0, name=f"ZeroA_{farm}_{food}")
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return {key: var.X for key, var in A_vars.items()}
    else:
        # Return previous values if infeasible
        return {(f, c): 0.0 for f in farms for c in foods}


def solve_Y_subproblem(
    farms: Dict,
    foods: List[str],
    food_groups: Dict,
    A: Dict,
    Z: Dict,
    U: Dict,
    rho: float,
    benefits: Dict,
    config: Dict
) -> Dict:
    """Solve Y-subproblem (binary selection)."""
    
    model = gp.Model("ADMM_Y_Subproblem")
    model.setParam('OutputFlag', 0)
    
    # Variables
    Y_vars = {}
    for farm in farms:
        for food in foods:
            Y_vars[(farm, food)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{farm}_{food}")
    
    # Objective: benefit + ADMM penalty term
    # Penalty: -U*(A-Y) - (rho/2)*||A-Y||^2
    # Collect to Y terms: U*Y - (rho/2)*(A^2 - 2AY + Y^2)
    obj_expr = gp.quicksum(
        U[(f, c)] * Y_vars[(f, c)] - (rho / 2.0) * (Y_vars[(f, c)] - A[(f, c)])**2
        for f in farms for c in foods
    )
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Food group constraints
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(foods_in_group) * len(farms))
            
            if min_foods > 0:
                model.addConstr(
                    gp.quicksum(Y_vars[(f, c)] for f in farms for c in foods_in_group) >= min_foods,
                    name=f"FG_Min_{group_name}"
                )
            
            if max_foods < float('inf'):
                model.addConstr(
                    gp.quicksum(Y_vars[(f, c)] for f in farms for c in foods_in_group) <= max_foods,
                    name=f"FG_Max_{group_name}"
                )
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return {key: var.X for key, var in Y_vars.items()}
    else:
        # Return relaxed solution
        return {(f, c): 0.0 for f in farms for c in foods}
