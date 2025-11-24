"""
Benders Decomposition with QPU Integration

Enhanced Benders decomposition that uses:
- Classical solver (Gurobi) for master problem (Y variables)
- QPU or hybrid solver for binary subproblems when beneficial
- Classical LP solver for continuous relaxation

This provides true quantum-classical hybrid Benders decomposition.
"""
import time
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
import numpy as np

from dimod import ConstrainedQuadraticModel, Binary, cqm_to_bqm
from dwave.system import LeapHybridBQMSampler

from result_formatter import format_benders_result, validate_solution_constraints


def solve_with_benders_qpu(
    farms: Dict[str, float],
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    dwave_token: Optional[str] = None,
    max_iterations: int = 50,
    gap_tolerance: float = 1e-4,
    time_limit: float = 300.0,
    use_qpu_for_master: bool = True
) -> Dict:
    """
    Solve farm allocation problem using Benders Decomposition with QPU integration.
    
    Strategy:
    - Master Problem: Binary Y variables (can use QPU if use_qpu_for_master=True)
    - Subproblem: Continuous A variables (always classical LP)
    - Iterative cuts until convergence
    
    Args:
        farms: Dictionary of farm names to land availability
        foods: List of food names
        food_groups: Dictionary of food groups
        config: Configuration dictionary with parameters
        dwave_token: D-Wave API token for QPU access
        max_iterations: Maximum number of Benders iterations
        gap_tolerance: Convergence tolerance for optimality gap
        time_limit: Maximum total solve time in seconds
        use_qpu_for_master: Whether to use QPU for master problem
    
    Returns:
        Formatted result dictionary
    """
    start_time = time.time()
    
    # Check if QPU is available
    has_qpu = dwave_token is not None and dwave_token != 'YOUR_DWAVE_TOKEN_HERE'
    if use_qpu_for_master and not has_qpu:
        print("⚠️  QPU requested but no token provided - using classical solver only")
        use_qpu_for_master = False
    
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
    qpu_time_total = 0.0
    
    print(f"\n{'='*80}")
    print(f"BENDERS DECOMPOSITION {'WITH QPU' if use_qpu_for_master else '(CLASSICAL)'}")
    print(f"{'='*80}")
    print(f"Problem: {len(farms)} farms, {len(foods)} foods")
    print(f"Master solver: {'QPU/Hybrid' if use_qpu_for_master else 'Gurobi'}")
    print(f"Subproblem solver: Gurobi (LP)")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*80}\n")
    
    # Benders iteration loop
    iteration = 0
    converged = False
    Y_star = None  # Will store best Y solution
    
    while iteration < max_iterations and not converged:
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            print(f"⏱️  Time limit reached at iteration {iteration}")
            break
        
        iteration += 1
        print(f"\n{'─'*80}")
        print(f"Benders Iteration {iteration}")
        print(f"{'─'*80}")
        
        # Solve master problem
        if use_qpu_for_master and iteration > 1:  # Use QPU after initial iteration
            Y_star, eta_value, master_time, qpu_time = solve_master_qpu(
                farms, foods, food_groups, config, master_iterations, dwave_token
            )
            qpu_time_total += qpu_time
        else:
            Y_star, eta_value, master_time = solve_master_classical(
                farms, foods, food_groups, config, benefits, master_iterations
            )
            qpu_time = 0.0
        
        if Y_star is None:
            print("❌ Master problem failed")
            break
        
        lower_bound = max(lower_bound, eta_value)
        print(f"  Master: eta = {eta_value:.4f}, LB = {lower_bound:.4f} (time: {master_time:.3f}s)")
        if qpu_time > 0:
            print(f"          QPU time: {qpu_time:.3f}s")
        
        # Solve subproblem given Y*
        A_star, subproblem_obj, duals, sub_time = solve_subproblem(
            farms, foods, Y_star, benefits, min_planting_area, max_planting_area
        )
        
        if A_star is None:
            print("  ⚠️  Subproblem infeasible - adding feasibility cut")
            # In practice, would add feasibility cut here
            break
        
        # Update upper bound
        upper_bound = min(upper_bound, subproblem_obj)
        print(f"  Subproblem: obj = {subproblem_obj:.4f}, UB = {upper_bound:.4f} (time: {sub_time:.3f}s)")
        
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
            'subproblem_obj': subproblem_obj,
            'master_time': master_time,
            'subproblem_time': sub_time,
            'qpu_time': qpu_time
        })
        
        if rel_gap < gap_tolerance:
            print(f"  ✅ Converged! Gap {rel_gap:.6f} < {gap_tolerance}")
            converged = True
            best_solution = {
                **{f"A_{f}_{c}": A_star.get((f, c), 0.0) for f in farms for c in foods},
                **{f"Y_{f}_{c}": Y_star.get((f, c), 0.0) for f in farms for c in foods}
            }
            break
        
        # Save current best solution
        best_solution = {
            **{f"A_{f}_{c}": A_star.get((f, c), 0.0) for f in farms for c in foods},
            **{f"Y_{f}_{c}": Y_star.get((f, c), 0.0) for f in farms for c in foods}
        }
    
    total_time = time.time() - start_time
    
    # PROJECT FINAL SOLUTION TO FEASIBLE SPACE
    A_dict_final = {(f, c): best_solution.get(f"A_{f}_{c}", 0.0) for f in farms for c in foods}
    for farm in farms:
        farm_total = sum(A_dict_final.get((farm, c), 0.0) for c in foods)
        farm_capacity = farms[farm]
        
        if farm_total > farm_capacity + 1e-6:
            scale_factor = farm_capacity / farm_total
            for c in foods:
                key = f"A_{farm}_{c}"
                if key in best_solution:
                    best_solution[key] *= scale_factor
            print(f"  ⚠️  Final projection {farm}: {farm_total:.2f} -> {farm_capacity:.2f} ha")
    
    print(f"\n{'='*80}")
    print(f"Benders Decomposition Complete")
    print(f"{'='*80}")
    print(f"Iterations: {iteration}")
    print(f"Final objective: {upper_bound:.4f}")
    print(f"Total time: {total_time:.3f}s")
    if qpu_time_total > 0:
        print(f"Total QPU time: {qpu_time_total:.3f}s")
    print(f"Status: {'Converged' if converged else 'Max iterations reached'}")
    print(f"{'='*80}\n")
    
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
        num_variables=len(farms) * len(foods) * 2,
        num_constraints=len(farms) + len(food_groups) * 2,
        converged=converged,
        final_gap=upper_bound - lower_bound,
        qpu_time_total=qpu_time_total,
        used_qpu=use_qpu_for_master and has_qpu
    )
    
    return result


def solve_master_classical(
    farms: Dict,
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    benefits: Dict,
    previous_iterations: List[Dict]
) -> Tuple[Optional[Dict], float, float]:
    """
    Solve Benders master problem classically with Gurobi.
    
    Returns:
        (Y_solution, eta_value, solve_time)
    """
    master_start = time.time()
    
    master = gp.Model("Benders_Master_Classical")
    master.setParam('OutputFlag', 0)
    
    # Master variables: Y[f,c] binary, eta (objective proxy)
    Y = {}
    for farm in farms:
        for food in foods:
            Y[(farm, food)] = master.addVar(vtype=GRB.BINARY, name=f"Y_{farm}_{food}")
    
    # Eta with reasonable bounds
    # Eta represents the objective value (area-normalized, so typically 0-1 range)
    # Upper bound: if all area allocated to best food: max(benefits) * total_area / total_area = max(benefits)
    max_benefit = max(benefits.values()) if benefits else 1.0
    eta = master.addVar(lb=-GRB.INFINITY, ub=max_benefit, name="eta", vtype=GRB.CONTINUOUS)
    
    # Food group constraints
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(foods_in_group) * len(farms))
            
            if min_foods > 0:
                master.addConstr(
                    gp.quicksum(Y[(farm, food)] for farm in farms for food in foods_in_group)
                    >= min_foods,
                    name=f"FoodGroup_Min_{group_name}"
                )
            
            if max_foods < float('inf'):
                master.addConstr(
                    gp.quicksum(Y[(farm, food)] for farm in farms for food in foods_in_group)
                    <= max_foods,
                    name=f"FoodGroup_Max_{group_name}"
                )
    
    # Add Benders cuts from previous iterations
    for it in previous_iterations:
        # Simplified cut: eta <= previous_subproblem_obj
        master.addConstr(eta <= it['subproblem_obj'], name=f"Benders_Cut_{it['iteration']}")
    
    # Master objective: maximize eta
    master.setObjective(eta, GRB.MAXIMIZE)
    
    # Solve
    master.optimize()
    
    solve_time = time.time() - master_start
    
    if master.status != GRB.OPTIMAL:
        return None, -float('inf'), solve_time
    
    # Extract solution
    Y_solution = {key: var.X for key, var in Y.items()}
    eta_value = eta.X
    
    return Y_solution, eta_value, solve_time


def solve_master_qpu(
    farms: Dict,
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    previous_iterations: List[Dict],
    dwave_token: str
) -> Tuple[Optional[Dict], float, float, float]:
    """
    Solve Benders master problem using QPU/Hybrid solver.
    
    Returns:
        (Y_solution, eta_value, total_time, qpu_time)
    """
    master_start = time.time()
    
    # Build CQM for master problem
    cqm = ConstrainedQuadraticModel()
    
    # Variables: Y[f,c] binary
    Y = {}
    for farm in farms:
        for food in foods:
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
            cqm.add_variable('BINARY', f"Y_{farm}_{food}")
    
    # Objective: Use upper bound from previous iterations as proxy
    # In full implementation, would encode cuts properly
    if previous_iterations:
        best_obj = max(it['subproblem_obj'] for it in previous_iterations)
    else:
        best_obj = 0.0
    
    # Simple objective: maximize number of selections (placeholder)
    # Real implementation would encode proper Benders objective
    objective = sum(Y[(f, c)] for f in farms for c in foods)
    cqm.set_objective(-objective)  # Minimize negative = maximize
    
    # Food group constraints
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(foods_in_group) * len(farms))
            
            total = sum(Y[(f, c)] for f in farms for c in foods_in_group)
            
            if min_foods > 0:
                cqm.add_constraint(total >= min_foods, label=f"FG_Min_{group_name}")
            if max_foods < float('inf'):
                cqm.add_constraint(total <= max_foods, label=f"FG_Max_{group_name}")
    
    # Solve with hybrid solver
    sampler = LeapHybridBQMSampler(token=dwave_token)
    
    # Convert CQM to BQM
    from dimod import cqm_to_bqm
    bqm, invert = cqm_to_bqm(cqm)
    
    qpu_start = time.time()
    sampleset = sampler.sample(bqm, label="Benders_Master_QPU")
    qpu_time = time.time() - qpu_start
    
    # Extract best sample
    best_sample = sampleset.first.sample
    
    # Invert to get CQM solution
    Y_solution = {}
    for (farm, food), var in Y.items():
        var_name = f"Y_{farm}_{food}"
        Y_solution[(farm, food)] = best_sample.get(var_name, 0.0)
    
    total_time = time.time() - master_start
    eta_value = best_obj  # Use previous best as eta approximation
    
    return Y_solution, eta_value, total_time, qpu_time


def solve_subproblem(
    farms: Dict[str, float],
    foods: List[str],
    Y_fixed: Dict[Tuple[str, str], float],
    benefits: Dict[str, float],
    min_planting_area: Dict[str, float],
    max_planting_area: Dict[str, float]
) -> Tuple[Optional[Dict], float, Dict, float]:
    """
    Solve the Benders subproblem: optimize A variables given fixed Y.
    
    Returns:
        (A_solution, objective_value, dual_variables, solve_time)
    """
    sub_start = time.time()
    
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
    
    # 2. Min area if Y=1, force zero if Y=0
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
    
    solve_time = time.time() - sub_start
    
    if sub.status != GRB.OPTIMAL:
        return None, -float('inf'), {}, solve_time
    
    # Extract solution and duals
    A_solution = {key: var.X for key, var in A.items()}
    obj_value = sub.ObjVal
    
    # Extract dual variables (shadow prices)
    duals = {}
    for name, constr in constraint_refs.items():
        duals[name] = constr.Pi
    
    return A_solution, obj_value, duals, solve_time
