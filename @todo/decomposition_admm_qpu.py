"""
ADMM Decomposition with QPU Integration

Enhanced ADMM that can use QPU/Hybrid solver for binary Y subproblem.
Classical A subproblem remains with Gurobi LP.
"""
import time
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
import numpy as np

from dimod import ConstrainedQuadraticModel, Binary, cqm_to_bqm, BinaryQuadraticModel
from dwave.system import LeapHybridBQMSampler
import neal  # SimulatedAnnealing fallback

from result_formatter import format_admm_result, validate_solution_constraints


def solve_with_admm_qpu(
    farms: Dict[str, float],
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    dwave_token: Optional[str] = None,
    max_iterations: int = 10,
    rho: float = 1.0,
    tolerance: float = 1e-4,
    use_qpu_for_y: bool = True
) -> Dict:
    """
    Solve farm allocation using ADMM with optional QPU for Y subproblem.
    
    ADMM alternates between:
    - A-subproblem: Continuous allocation (always classical LP)
    - Y-subproblem: Binary selection (QPU or classical)
    - Dual update: Consensus enforcement
    
    Args:
        farms: Dictionary of farm names to land availability
        foods: List of food names  
        food_groups: Dictionary of food groups
        config: Configuration dictionary
        dwave_token: D-Wave API token
        max_iterations: Maximum ADMM iterations
        rho: Aug Lagrangian penalty parameter
        tolerance: Convergence tolerance
        use_qpu_for_y: Whether to use QPU for Y subproblem
    
    Returns:
        Formatted result dictionary
    """
    start_time = time.time()
    
    # Check QPU availability
    has_qpu = dwave_token is not None and dwave_token != 'YOUR_DWAVE_TOKEN_HERE'
    use_simulated_annealing = use_qpu_for_y and not has_qpu
    if use_simulated_annealing:
        print("⚠️  QPU requested but no token provided - using SimulatedAnnealing fallback")
    
    # Extract parameters
    params = config.get('parameters', {})
    min_planting_area = params.get('minimum_planting_area', {})
    max_planting_area = params.get('maximum_planting_area', {})
    benefits = config.get('benefits', {})
    
    # Initialize variables
    A = {(f, c): 0.0 for f in farms for c in foods}
    Y = {(f, c): 0.0 for f in farms for c in foods}
    U = {(f, c): 0.0 for f in farms for c in foods}  # Dual variables
    
    iterations = []
    qpu_time_total = 0.0
    
    print(f"\n{'='*80}")
    print(f"ADMM DECOMPOSITION {'WITH QPU' if use_qpu_for_y else '(CLASSICAL)'}")
    print(f"{'='*80}")
    print(f"Problem: {len(farms)} farms, {len(foods)} foods")
    print(f"A-subproblem: Gurobi LP")
    print(f"Y-subproblem: {'QPU/Hybrid' if use_qpu_for_y else 'Gurobi MILP'}")
    print(f"Penalty ρ: {rho}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*80}\n")
    
    # ADMM loop
    for iteration in range(1, max_iterations + 1):
        print(f"\nADMM Iteration {iteration}")
        
        # Step 1: A-subproblem (always classical)
        A, a_time = solve_a_subproblem(
            farms, foods, Y, U, rho, benefits, min_planting_area, max_planting_area
        )
        
        # Step 2: Y-subproblem (QPU or classical)
        if has_qpu and use_qpu_for_y:
            Y, y_time, qpu_time = solve_y_subproblem_qpu(
                farms, foods, food_groups, config, A, U, rho, dwave_token
            )
            qpu_time_total += qpu_time
        elif use_simulated_annealing:
            Y, y_time, sa_time = solve_y_subproblem_sa(
                farms, foods, food_groups, config, A, U, rho
            )
            qpu_time_total += sa_time
            qpu_time = sa_time
        else:
            Y, y_time = solve_y_subproblem_classical(
                farms, foods, food_groups, config, A, U, rho
            )
            qpu_time = 0.0
        
        # Step 3: Dual update
        for key in U:
            U[key] += rho * (A[key] - Y[key])
        
        # Calculate residuals
        primal_residual = np.sqrt(sum((A[key] - Y[key])**2 for key in A))
        if iterations:
            # Calculate dual residual using previous Y
            Y_prev = iterations[-1]['Y_raw']  # Store raw for residual calculation
            dual_residual = np.sqrt(sum((rho * (Y[key] - Y_prev[key]))**2 for key in Y))
        else:
            dual_residual = 0.0
        
        # Calculate objective (benefit per hectare, normalized by total area)
        total_area = sum(farms.values())
        obj = sum(A[key] * benefits.get(key[1], 1.0) for key in A) / total_area
        
        print(f"  Objective: {obj:.4f}")
        print(f"  Primal Residual: {primal_residual:.6f}")
        print(f"  Dual Residual: {dual_residual:.6f}")
        if qpu_time > 0:
            print(f"  QPU Time: {qpu_time:.3f}s")
        
        # Convert tuple keys to strings for JSON serialization
        A_str = {f"{f}_{c}": v for (f, c), v in A.items()}
        Y_str = {f"{f}_{c}": v for (f, c), v in Y.items()}
        U_str = {f"{f}_{c}": v for (f, c), v in U.items()}
        
        iterations.append({
            'iteration': iteration,
            'objective': obj,
            'primal_residual': primal_residual,
            'dual_residual': dual_residual,
            'A': A_str,
            'Y': Y_str,
            'U': U_str,
            'Y_raw': Y.copy(),  # Keep for residual calculation (not serialized to JSON)
            'a_time': a_time,
            'y_time': y_time,
            'qpu_time': qpu_time
        })
        
        # Check convergence
        if primal_residual < tolerance and dual_residual < tolerance:
            print(f"  ADMM Converged at iteration {iteration}")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"ADMM Complete")
    print(f"{'='*80}")
    print(f"Iterations: {len(iterations)}")
    print(f"Final objective: {obj:.4f}")
    print(f"Total time: {total_time:.3f}s")
    if qpu_time_total > 0:
        print(f"Total QPU time: {qpu_time_total:.3f}s")
    print(f"{'='*80}\n")
    
    # Build solution with proper linking constraint enforcement
    # First, binarize Y values
    Y_binary = {key: 1.0 if val > 0.5 else 0.0 for key, val in Y.items()}
    
    # ENFORCE FOOD GROUP MINIMUM CONSTRAINTS (SA/QPU may not satisfy these)
    # NOTE: min_foods constraint counts UNIQUE foods selected (across all farms)
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            
            if min_foods > 0:
                # Count UNIQUE foods selected (a food counts if selected on ANY farm)
                unique_foods_selected = set()
                for c in foods_in_group:
                    if any(Y_binary.get((f, c), 0) > 0.5 for f in farms):
                        unique_foods_selected.add(c)
                
                current_count = len(unique_foods_selected)
                
                if current_count < min_foods:
                    # Need to add NEW unique foods to meet minimum
                    shortfall = min_foods - current_count
                    print(f"  ⚠️  Food group {group_name}: {current_count}/{min_foods} unique foods, adding {shortfall}")
                    
                    # Find foods in group NOT yet selected anywhere
                    unselected_foods = [c for c in foods_in_group if c not in unique_foods_selected]
                    
                    # Sort by benefit (descending)
                    unselected_foods.sort(key=lambda c: benefits.get(c, 1.0), reverse=True)
                    
                    # For each missing food, select it on the farm with most remaining capacity
                    for c in unselected_foods[:shortfall]:
                        # Find farm with most remaining capacity
                        farm_capacities = []
                        for f in farms:
                            current_usage = sum(A.get((f, food), 0.0) for food in foods)
                            remaining = farms[f] - current_usage
                            farm_capacities.append((f, remaining))
                        farm_capacities.sort(key=lambda x: x[1], reverse=True)
                        
                        best_farm = farm_capacities[0][0]
                        Y_binary[(best_farm, c)] = 1.0
                        min_area = min_planting_area.get(c, 0.0001)
                        A[(best_farm, c)] = max(A[(best_farm, c)], min_area)
                        print(f"    + Added Y_{best_farm}_{c} with A={A[(best_farm, c)]:.4f}")
    
    # ENFORCE LINKING CONSTRAINTS (critical for feasibility)
    # If Y=0, force A=0; if Y=1, ensure A >= min_area
    A_linked = {}
    for key in A:
        farm, food = key
        y_val = Y_binary[key]
        a_val = A[key]
        
        if y_val < 0.5:
            # Y=0, force A=0
            A_linked[key] = 0.0
        else:
            # Y=1, ensure A >= min_area
            min_area = min_planting_area.get(food, 0.0001)
            A_linked[key] = max(a_val, min_area)
    
    final_solution = {
        **{f"A_{f}_{c}": A_linked[(f, c)] for f, c in A_linked},
        **{f"Y_{f}_{c}": Y_binary[(f, c)] for f, c in Y_binary}
    }
    
    # PROJECT TO FEASIBLE SPACE (land capacity)
    for farm in farms:
        farm_total = sum(A_linked.get((farm, c), 0.0) for c in foods)
        farm_capacity = farms[farm]
        
        if farm_total > farm_capacity + 1e-6:
            scale_factor = farm_capacity / farm_total
            for c in foods:
                key = f"A_{farm}_{c}"
                if key in final_solution:
                    final_solution[key] *= scale_factor
                    A_linked[(farm, c)] *= scale_factor
            print(f"  ⚠️  Projected {farm}: {farm_total:.2f} -> {farm_capacity:.2f} ha")
    
    # RE-CHECK MIN_AREA CONSTRAINT AFTER PROJECTION
    # If scaling dropped A below min_area for selected crops, fix it
    for key in A_linked:
        farm, food = key
        y_val = Y_binary.get(key, 0.0)
        a_val = A_linked[key]
        
        if y_val > 0.5:  # Crop is selected
            min_area = min_planting_area.get(food, 0.0001)
            if a_val < min_area - 1e-6:
                # Projection dropped below min_area - enforce min_area
                A_linked[key] = min_area
                final_solution[f"A_{farm}_{food}"] = min_area
                print(f"  ⚠️  Fixed min_area for {farm}_{food}: {a_val:.4f} -> {min_area:.4f}")
    
    # Recalculate objective with corrected A values
    total_area = sum(farms.values())
    final_obj = sum(A_linked[(f, c)] * benefits.get(c, 1.0) for f in farms for c in foods) / total_area
    
    # Validate
    validation = validate_solution_constraints(
        final_solution, farms, foods, food_groups, farms, config, 'farm'
    )
    
    # Remove Y_raw from iterations before serialization (it has tuple keys)
    iterations_clean = []
    for it in iterations:
        it_clean = {k: v for k, v in it.items() if k != 'Y_raw'}
        iterations_clean.append(it_clean)
    
    # Format result
    result = format_admm_result(
        iterations=iterations_clean,
        final_solution=final_solution,
        objective_value=final_obj,
        total_time=total_time,
        scenario_type='farm',
        n_units=len(farms),
        n_foods=len(foods),
        total_area=sum(farms.values()),
        is_feasible=validation['is_feasible'],
        validation_results=validation,
        num_variables=len(A) + len(Y),
        num_constraints=len(farms) + len(food_groups) * 2,
        converged=primal_residual < tolerance and dual_residual < tolerance,
        rho=rho,
        qpu_time_total=qpu_time_total,
        used_qpu=use_qpu_for_y and has_qpu
    )
    
    return result


def solve_a_subproblem(
    farms: Dict,
    foods: List[str],
    Y: Dict,
    U: Dict,
    rho: float,
    benefits: Dict,
    min_planting_area: Dict,
    max_planting_area: Dict
) -> Tuple[Dict, float]:
    """Solve A-subproblem (continuous allocation)."""
    a_start = time.time()
    
    model = gp.Model("A_Subproblem")
    model.setParam('OutputFlag', 0)
    
    # Variables
    A_vars = {}
    for farm in farms:
        for food in foods:
            A_vars[(farm, food)] = model.addVar(lb=0.0, name=f"A_{farm}_{food}")
    
    # Objective: benefit per hectare + ADMM penalty (normalized by total area)
    total_area = sum(farms.values())
    obj_expr = gp.quicksum(
        A_vars[key] * benefits.get(key[1], 1.0) for key in A_vars
    ) / total_area
    
    # ADMM penalty: (rho/2) ||A - Y + U||^2
    penalty = gp.quicksum(
        (rho / 2) * (A_vars[key] - Y[key] + U[key])**2 for key in A_vars
    )
    
    model.setObjective(obj_expr - penalty, GRB.MAXIMIZE)
    
    # Constraints
    for farm, capacity in farms.items():
        model.addConstr(
            gp.quicksum(A_vars[(farm, food)] for food in foods) <= capacity,
            name=f"Land_{farm}"
        )
    
    # Min/max area constraints
    for key in A_vars:
        farm, food = key
        min_area = min_planting_area.get(food, 0.0001)
        max_area = max_planting_area.get(food, farms[farm])
        
        # If Y suggests selection, enforce min area
        if Y[key] > 0.5:
            model.addConstr(A_vars[key] >= min_area, name=f"MinA_{farm}_{food}")
        
        model.addConstr(A_vars[key] <= max_area, name=f"MaxA_{farm}_{food}")
    
    model.optimize()
    
    a_time = time.time() - a_start
    
    if model.status == GRB.OPTIMAL:
        A_solution = {key: var.X for key, var in A_vars.items()}
    else:
        A_solution = {key: 0.0 for key in A_vars}
    
    return A_solution, a_time


def solve_y_subproblem_classical(
    farms: Dict,
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    A: Dict,
    U: Dict,
    rho: float
) -> Tuple[Dict, float]:
    """Solve Y-subproblem classically (binary selection)."""
    y_start = time.time()
    
    model = gp.Model("Y_Subproblem_Classical")
    model.setParam('OutputFlag', 0)
    
    # Variables
    Y_vars = {}
    for farm in farms:
        for food in foods:
            Y_vars[(farm, food)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{farm}_{food}")
    
    # Objective: ADMM penalty (rho/2) ||A - Y + U||^2
    penalty = gp.quicksum(
        (rho / 2) * (A[key] - Y_vars[key] + U[key])**2 for key in Y_vars
    )
    
    model.setObjective(penalty, GRB.MINIMIZE)
    
    # Food group constraints
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(foods_in_group) * len(farms))
            
            total = gp.quicksum(Y_vars[(f, c)] for f in farms for c in foods_in_group)
            
            if min_foods > 0:
                model.addConstr(total >= min_foods, name=f"FG_Min_{group_name}")
            if max_foods < float('inf'):
                model.addConstr(total <= max_foods, name=f"FG_Max_{group_name}")
    
    model.optimize()
    
    y_time = time.time() - y_start
    
    if model.status == GRB.OPTIMAL:
        Y_solution = {key: var.X for key, var in Y_vars.items()}
    else:
        Y_solution = {key: 0.0 for key in Y_vars}
    
    return Y_solution, y_time


def solve_y_subproblem_sa(
    farms: Dict,
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    A: Dict,
    U: Dict,
    rho: float
) -> Tuple[Dict, float, float]:
    """Solve Y-subproblem using SimulatedAnnealing (fallback)."""
    y_start = time.time()
    
    # Build CQM
    cqm = ConstrainedQuadraticModel()
    
    # Variables
    Y_vars = {}
    for farm in farms:
        for food in foods:
            var_name = f"Y_{farm}_{food}"
            Y_vars[(farm, food)] = Binary(var_name)
            cqm.add_variable('BINARY', var_name)
    
    # Objective: ADMM penalty
    objective = sum(
        (A[key] - Y_vars[key] + U[key])**2 for key in Y_vars
    )
    cqm.set_objective(objective)  # Minimize penalty
    
    # Food group constraints
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(foods_in_group) * len(farms))
            
            total = sum(Y_vars[(f, c)] for f in farms for c in foods_in_group)
            
            if min_foods > 0:
                cqm.add_constraint(total >= min_foods, label=f"FG_Min_{group_name}")
            if max_foods < float('inf'):
                cqm.add_constraint(total <= max_foods, label=f"FG_Max_{group_name}")
    
    # Solve with SimulatedAnnealing
    bqm, invert = cqm_to_bqm(cqm)
    
    sampler = neal.SimulatedAnnealingSampler()
    
    sa_start = time.time()
    sampleset = sampler.sample(bqm, num_reads=100, num_sweeps=1000)
    sa_time = time.time() - sa_start
    
    # Extract solution
    best_sample = sampleset.first.sample
    
    Y_solution = {}
    for (farm, food), var in Y_vars.items():
        var_name = f"Y_{farm}_{food}"
        Y_solution[(farm, food)] = best_sample.get(var_name, 0.0)
    
    total_time = time.time() - y_start
    
    return Y_solution, total_time, sa_time


def solve_y_subproblem_qpu(
    farms: Dict,
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    A: Dict,
    U: Dict,
    rho: float,
    dwave_token: str
) -> Tuple[Dict, float, float]:
    """Solve Y-subproblem using QPU (binary selection)."""
    y_start = time.time()
    
    # Build CQM
    cqm = ConstrainedQuadraticModel()
    
    # Variables
    Y_vars = {}
    for farm in farms:
        for food in foods:
            var_name = f"Y_{farm}_{food}"
            Y_vars[(farm, food)] = Binary(var_name)
            cqm.add_variable('BINARY', var_name)
    
    # Objective: ADMM penalty (simplified for QPU)
    objective = sum(
        (A[key] - Y_vars[key] + U[key])**2 for key in Y_vars
    )
    cqm.set_objective(objective)  # Minimize penalty
    
    # Food group constraints
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(foods_in_group) * len(farms))
            
            total = sum(Y_vars[(f, c)] for f in farms for c in foods_in_group)
            
            if min_foods > 0:
                cqm.add_constraint(total >= min_foods, label=f"FG_Min_{group_name}")
            if max_foods < float('inf'):
                cqm.add_constraint(total <= max_foods, label=f"FG_Max_{group_name}")
    
    # Solve with Hybrid BQM solver (minimum 3s)
    sampler = LeapHybridBQMSampler(token=dwave_token)
    
    # Convert CQM to BQM
    bqm, invert = cqm_to_bqm(cqm)
    
    qpu_start = time.time()
    sampleset = sampler.sample(bqm, time_limit=3, label="ADMM_Y_Subproblem_QPU")
    wall_time = time.time() - qpu_start
    
    # Extract actual QPU access time from timing info
    timing_info = sampleset.info.get('timing', {})
    qpu_access_time_us = timing_info.get('qpu_access_time', 0)  # microseconds
    qpu_time = qpu_access_time_us / 1_000_000  # Convert to seconds
    
    print(f"          [Hybrid] Wall: {wall_time:.3f}s, QPU access: {qpu_time*1000:.2f}ms")
    
    # Extract solution
    best_sample = sampleset.first.sample
    
    Y_solution = {}
    for (farm, food), var in Y_vars.items():
        var_name = f"Y_{farm}_{food}"
        Y_solution[(farm, food)] = best_sample.get(var_name, 0.0)
    
    total_time = time.time() - y_start
    
    return Y_solution, total_time, qpu_time
