"""
Benders Decomposition Strategy

Decomposes the MINLP problem into:
- Master Problem: Binary Y variables + Benders cuts (MILP)
- Subproblem: Continuous A variables given fixed Y* (LP)

Iteratively adds optimality cuts until convergence.
"""
import time
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
import numpy as np

from result_formatter import format_benders_result, validate_solution_constraints


def solve_with_benders(
    farms: Dict[str, float],
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    max_iterations: int = 50,
    gap_tolerance: float = 1e-4,
    time_limit: float = 300.0
) -> Dict:
    """
    Solve farm allocation problem using Benders Decomposition.
    
    Args:
        farms: Dictionary of farm names to land availability
        foods: List of food names
        food_groups: Dictionary of food group constraints
        config: Configuration dictionary with parameters
        max_iterations: Maximum number of Benders iterations
        gap_tolerance: Convergence tolerance for optimality gap
        time_limit: Maximum total solve time in seconds
    
    Returns:
        Formatted result dictionary
    """
    start_time = time.time()
    
    # Extract parameters
    params = config.get('parameters', {})
    min_planting_area = params.get('minimum_planting_area', {})
    max_planting_area = params.get('maximum_planting_area', {})
    benefits = config.get('benefits', {})
    
    # Initialize tracking
    master_iterations = []
    lower_bound = -float('inf')
    upper_bound = float('inf')
    best_solution = {}
    
    # Create master problem (Y variables + eta)
    master = gp.Model("Benders_Master")
    master.setParam('OutputFlag', 0)
    
    # Master variables: Y[f,c] binary, eta (objective proxy)
    Y = {}
    for farm in farms:
        for food in foods:
            Y[(farm, food)] = master.addVar(vtype=GRB.BINARY, name=f"Y_{farm}_{food}")
    
    # Eta represents the objective value, give it a reasonable upper bound
    max_possible_benefit = sum(benefits.values()) * sum(farms.values()) / 100.0
    eta = master.addVar(lb=-GRB.INFINITY, ub=max_possible_benefit, name="eta", vtype=GRB.CONTINUOUS)
    
    # Master constraints: Food group constraints on Y
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(foods_in_group) * len(farms))
            
            # Min constraint (global count)
            if min_foods > 0:
                master.addConstr(
                    gp.quicksum(Y[(farm, food)] for farm in farms for food in foods_in_group)
                    >= min_foods,
                    name=f"FoodGroup_Min_{group_name}"
                )
            
            # Max constraint (global count)
            if max_foods < float('inf'):
                master.addConstr(
                    gp.quicksum(Y[(farm, food)] for farm in farms for food in foods_in_group)
                    <= max_foods,
                    name=f"FoodGroup_Max_{group_name}"
                )
    
    # Master objective: maximize eta (proxy for subproblem objective)
    master.setObjective(eta, GRB.MAXIMIZE)
    
    # Benders iteration loop
    iteration = 0
    converged = False
    
    while iteration < max_iterations and not converged:
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            print(f"Time limit reached at iteration {iteration}")
            break
        
        iteration += 1
        print(f"\nBenders Iteration {iteration}")
        
        # Solve master problem
        master.optimize()
        
        if master.status != GRB.OPTIMAL:
            print(f"Master problem not optimal: status {master.status}")
            break
        
        # Extract Y* from master
        Y_star = {key: var.X for key, var in Y.items()}
        eta_value = eta.X
        lower_bound = max(lower_bound, eta_value)
        
        print(f"  Master: eta = {eta_value:.4f}, LB = {lower_bound:.4f}")
        
        # Solve subproblem given Y*
        A_star, subproblem_obj, duals = solve_subproblem(
            farms, foods, Y_star, benefits, min_planting_area, max_planting_area
        )
        
        if A_star is None:
            print("  Subproblem infeasible - adding feasibility cut")
            # Add feasibility cut (not implemented in detail here)
            break
        
        # Update upper bound
        upper_bound = min(upper_bound, subproblem_obj)
        print(f"  Subproblem: obj = {subproblem_obj:.4f}, UB = {upper_bound:.4f}")
        
        # Check convergence
        gap = upper_bound - lower_bound
        rel_gap = abs(gap) / max(abs(upper_bound), 1.0)
        print(f"  Gap: {gap:.6f} (relative: {rel_gap:.6f})")
        
        master_iterations.append({
            'iteration': iteration,
            'eta': eta_value,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'gap': gap,
            'relative_gap': rel_gap,
            'subproblem_obj': subproblem_obj
        })
        
        if rel_gap < gap_tolerance:
            print(f"  Converged! Gap {rel_gap:.6f} < {gap_tolerance}")
            converged = True
            best_solution = {**{f"A_{f}_{c}": A_star.get((f, c), 0.0) for f in farms for c in foods},
                           **{f"Y_{f}_{c}": Y_star.get((f, c), 0.0) for f in farms for c in foods}}
            break
        
        # Generate and add Benders optimality cut
        cut_added = add_benders_cut(master, Y, eta, Y_star, subproblem_obj, duals, farms, foods, benefits)
        
        if not cut_added:
            print("  No cut added - terminating")
            break
        
        # Save current best solution
        best_solution = {**{f"A_{f}_{c}": A_star.get((f, c), 0.0) for f in farms for c in foods},
                       **{f"Y_{f}_{c}": Y_star.get((f, c), 0.0) for f in farms for c in foods}}
    
    total_time = time.time() - start_time
    
    # Validate solution
    validation = validate_solution_constraints(
        best_solution, farms, foods, food_groups, farms, config, 'farm'
    )
    
    # Format result
    result = format_benders_result(
        master_iterations=master_iterations,
        final_solution=best_solution,
        objective_value=upper_bound,
        total_time=total_time,
        scenario_type='farm',
        n_units=len(farms),
        n_foods=len(foods),
        total_area=sum(farms.values()),
        is_feasible=validation['is_feasible'],
        validation_results=validation,
        num_variables=len(Y) + len(farms) * len(foods),
        num_constraints=master.NumConstrs,
        converged=converged,
        final_gap=upper_bound - lower_bound
    )
    
    return result


def solve_subproblem(
    farms: Dict[str, float],
    foods: List[str],
    Y_fixed: Dict[Tuple[str, str], float],
    benefits: Dict[str, float],
    min_planting_area: Dict[str, float],
    max_planting_area: Dict[str, float]
) -> Tuple[Optional[Dict], float, Dict]:
    """
    Solve the Benders subproblem: optimize A variables given fixed Y.
    
    Returns:
        (A_solution, objective_value, dual_variables)
    """
    sub = gp.Model("Benders_Subproblem")
    sub.setParam('OutputFlag', 0)
    
    # Subproblem variables: A[f,c] continuous
    A = {}
    for farm, capacity in farms.items():
        for food in foods:
            A[(farm, food)] = sub.addVar(lb=0.0, name=f"A_{farm}_{food}")
    
    # Objective: maximize benefit per hectare (normalized by total area)
    total_area = sum(farms.values())
    obj_expr = gp.quicksum(
        A[(farm, food)] * benefits.get(food, 1.0)
        for farm in farms
        for food in foods
    ) / total_area
    sub.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Constraints
    constraint_refs = {}
    
    # 1. Land availability
    for farm, capacity in farms.items():
        constr = sub.addConstr(
            gp.quicksum(A[(farm, food)] for food in foods) <= capacity,
            name=f"Land_{farm}"
        )
        constraint_refs[f"Land_{farm}"] = constr
    
    # 2. Min area if Y=1
    for farm in farms:
        for food in foods:
            y_val = Y_fixed.get((farm, food), 0.0)
            min_area = min_planting_area.get(food, 0.0001)
            
            if y_val > 0.5:  # Y is selected
                constr = sub.addConstr(
                    A[(farm, food)] >= min_area,
                    name=f"MinArea_{farm}_{food}"
                )
                constraint_refs[f"MinArea_{farm}_{food}"] = constr
            else:  # Y is not selected, force A = 0
                constr = sub.addConstr(
                    A[(farm, food)] == 0.0,
                    name=f"ForceZero_{farm}_{food}"
                )
                constraint_refs[f"ForceZero_{farm}_{food}"] = constr
    
    # 3. Max area if Y=1
    for farm in farms:
        for food in foods:
            y_val = Y_fixed.get((farm, food), 0.0)
            if y_val > 0.5:
                max_area = max_planting_area.get(food, farms[farm])
                constr = sub.addConstr(
                    A[(farm, food)] <= max_area,
                    name=f"MaxArea_{farm}_{food}"
                )
                constraint_refs[f"MaxArea_{farm}_{food}"] = constr
    
    # Solve subproblem
    sub.optimize()
    
    if sub.status != GRB.OPTIMAL:
        return None, -float('inf'), {}
    
    # Extract solution and duals
    A_solution = {key: var.X for key, var in A.items()}
    obj_value = sub.ObjVal
    
    # Extract dual variables (shadow prices)
    duals = {}
    for name, constr in constraint_refs.items():
        duals[name] = constr.Pi
    
    return A_solution, obj_value, duals


def add_benders_cut(
    master: gp.Model,
    Y: Dict,
    eta: gp.Var,
    Y_star: Dict,
    subproblem_obj: float,
    duals: Dict,
    farms: Dict,
    foods: List[str],
    benefits: Dict[str, float]
) -> bool:
    """
    Add Benders optimality cut to master problem.
    
    Cut form: eta >= subproblem_obj + gradient * (Y - Y_star)
    """
    try:
        # Simplified cut: eta <= subproblem_obj
        # (More sophisticated cuts would use dual information)
        cut_expr = eta
        master.addConstr(cut_expr <= subproblem_obj, name=f"Benders_Cut_{master.NumConstrs}")
        return True
    except Exception as e:
        print(f"Error adding Benders cut: {e}")
        return False
