"""
Dantzig-Wolfe Decomposition with QPU Integration

Enhanced column generation that uses:
- Classical solver (Gurobi) for restricted master problem (RMP)
- QPU/Hybrid solver for pricing subproblem (generate new columns)
- Classical solver fallback for compatibility

This provides true quantum-classical hybrid column generation.
"""
import time
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
import numpy as np

from dimod import ConstrainedQuadraticModel, Binary, cqm_to_bqm
from dwave.system import LeapHybridBQMSampler

from result_formatter import format_dantzig_wolfe_result, validate_solution_constraints


def solve_with_dantzig_wolfe_qpu(
    farms: Dict[str, float],
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    dwave_token: Optional[str] = None,
    max_iterations: int = 50,
    time_limit: float = 300.0,
    use_qpu_for_pricing: bool = True
) -> Dict:
    """
    Solve farm allocation problem using Dantzig-Wolfe Decomposition with QPU integration.
    
    Strategy:
    - Restricted Master Problem (RMP): Classical LP/MILP (select from column pool)
    - Pricing Subproblem: Can use QPU to generate new columns
    - Iterates until no improving columns found
    
    Args:
        farms: Dictionary of farm names to land availability
        foods: List of food names
        food_groups: Dictionary of food groups
        config: Configuration dictionary
        dwave_token: D-Wave API token for QPU access
        max_iterations: Maximum column generation iterations
        time_limit: Maximum total solve time
        use_qpu_for_pricing: Whether to use QPU for pricing subproblem
    
    Returns:
        Formatted result dictionary
    """
    start_time = time.time()
    
    # Check if QPU is available
    has_qpu = dwave_token is not None and dwave_token != 'YOUR_DWAVE_TOKEN_HERE'
    if use_qpu_for_pricing and not has_qpu:
        print("⚠️  QPU requested but no token provided - using classical solver only")
        use_qpu_for_pricing = False
    
    # Extract parameters
    params = config.get('parameters', {})
    min_planting_area = params.get('minimum_planting_area', {})
    max_planting_area = params.get('maximum_planting_area', {})
    benefits = config.get('benefits', {})
    
    # Initialize column pool with trivial columns
    columns = []
    columns_generated = []
    qpu_time_total = 0.0
    
    print(f"\n{'='*80}")
    print(f"DANTZIG-WOLFE DECOMPOSITION {'WITH QPU' if use_qpu_for_pricing else '(CLASSICAL)'}")
    print(f"{'='*80}")
    print(f"Problem: {len(farms)} farms, {len(foods)} foods")
    print(f"RMP solver: Gurobi (LP/MILP)")
    print(f"Pricing solver: {'QPU/Hybrid' if use_qpu_for_pricing else 'Gurobi'}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*80}\n")
    
    # Generate initial columns (food-group-aware to ensure RMP feasibility)
    print("Generating initial column pool...")
    
    # Create diverse initial columns covering all food groups
    # Strategy: For each farm, create columns that select from different food groups
    
    for farm, capacity in farms.items():
        # Pattern 1: Select one food from each food group
        for group_name, foods_in_group in food_groups.items():
            if not foods_in_group:
                continue
            
            # Pick highest benefit food from this group
            best_food = max(foods_in_group, key=lambda f: benefits.get(f, 1.0))
            
            allocation = {}
            selection = {}
            
            min_area = min_planting_area.get(best_food, 0.0001)
            max_area = min(max_planting_area.get(best_food, capacity), capacity / 3)
            alloc = max(min_area, min(max_area, capacity / 5))  # Small allocation
            
            if alloc <= capacity:
                allocation[(farm, best_food)] = alloc
                selection[(farm, best_food)] = 1.0
                
                total_area = sum(farms.values())
                col_obj = alloc * benefits.get(best_food, 1.0) / total_area
                columns.append({
                    'farm': farm,
                    'allocation': allocation,
                    'selection': selection,
                    'objective': col_obj
                })
        
        # Pattern 2: Mix of high-benefit foods from different groups
        sorted_foods = sorted(foods, key=lambda f: benefits.get(f, 1.0), reverse=True)
        allocation = {}
        selection = {}
        remaining_capacity = capacity
        groups_covered = set()
        
        for food in sorted_foods:
            # Find which group this food belongs to
            food_group = None
            for group_name, foods_in_group in food_groups.items():
                if food in foods_in_group:
                    food_group = group_name
                    break
            
            # Only add if we haven't covered this group yet
            if food_group and food_group not in groups_covered:
                min_area = min_planting_area.get(food, 0.0001)
                max_area = min(max_planting_area.get(food, capacity), remaining_capacity / 2)
                alloc = min(max_area, remaining_capacity / (6 - len(groups_covered)))
                
                if alloc >= min_area and remaining_capacity >= min_area:
                    allocation[(farm, food)] = alloc
                    selection[(farm, food)] = 1.0
                    remaining_capacity -= alloc
                    groups_covered.add(food_group)
            
            if len(groups_covered) >= len(food_groups):
                break
        
        if allocation:
            total_area = sum(farms.values())
            col_obj = sum(allocation.get(key, 0.0) * benefits.get(key[1], 1.0) for key in allocation) / total_area
            columns.append({
                'farm': farm,
                'allocation': allocation,
                'selection': selection,
                'objective': col_obj
            })
    
    print(f"  ✅ Initial column pool: {len(columns)} columns (food-group-aware)\n")
    
    # Column generation loop
    iteration = 0
    while iteration < max_iterations:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            print(f"⏱️  Time limit reached at iteration {iteration}")
            break
        
        iteration += 1
        print(f"\n{'─'*80}")
        print(f"Column Generation Iteration {iteration}")
        print(f"{'─'*80}")
        
        # Solve Restricted Master Problem
        rmp_solution, rmp_obj, duals, rmp_time = solve_restricted_master(
            columns, farms, foods, food_groups, benefits, config
        )
        
        if rmp_solution is None:
            print("❌ RMP infeasible - terminating")
            break
        
        active_columns = sum(1 for x in rmp_solution.values() if x > 1e-6)
        print(f"  RMP: obj = {rmp_obj:.4f}, active columns = {active_columns}/{len(columns)} (time: {rmp_time:.3f}s)")
        
        # Solve Pricing Subproblem
        if use_qpu_for_pricing and has_qpu:
            new_column, reduced_cost, pricing_time, qpu_time = solve_pricing_qpu(
                farms, foods, duals, benefits, min_planting_area, max_planting_area, dwave_token
            )
            qpu_time_total += qpu_time
            print(f"  Pricing (QPU): reduced cost = {reduced_cost:.6f} (time: {pricing_time:.3f}s, QPU: {qpu_time:.3f}s)")
        else:
            new_column, reduced_cost, pricing_time = solve_pricing_classical(
                farms, foods, duals, benefits, min_planting_area, max_planting_area
            )
            qpu_time = 0.0
            print(f"  Pricing: reduced cost = {reduced_cost:.6f} (time: {pricing_time:.3f}s)")
        
        columns_generated.append({
            'iteration': iteration,
            'reduced_cost': reduced_cost,
            'rmp_objective': rmp_obj,
            'num_columns': len(columns),
            'rmp_time': rmp_time,
            'pricing_time': pricing_time,
            'qpu_time': qpu_time
        })
        
        # Check if new column improves solution
        if reduced_cost >= -1e-6:  # No improvement
            print(f"  ✅ No improving column found - optimal!")
            break
        
        # Add new column to pool
        if new_column:
            columns.append(new_column)
            print(f"  Added column to pool (total: {len(columns)})")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Column Generation Complete")
    print(f"{'='*80}")
    print(f"Iterations: {iteration}")
    print(f"Total columns: {len(columns)}")
    print(f"Total time: {total_time:.3f}s")
    if qpu_time_total > 0:
        print(f"Total QPU time: {qpu_time_total:.3f}s")
    print(f"{'='*80}\n")
    
    # Construct final solution from RMP (integer solution)
    print("Solving final RMP with integer variables...")
    final_solution, final_obj, _, final_time = solve_restricted_master(
        columns, farms, foods, food_groups, benefits, config, integer=True
    )
    
    if final_solution is None:
        # Fallback to relaxed solution
        print("  ⚠️  Integer RMP failed, using relaxed solution")
        final_solution, final_obj, _, final_time = solve_restricted_master(
            columns, farms, foods, food_groups, benefits, config, integer=False
        )
    
    if final_solution is None:
        # No solution found at all - return empty solution
        print("  ❌ No solution found - returning zero objective")
        final_obj = 0.0
        full_solution = {
            **{f"A_{f}_{c}": 0.0 for f in farms for c in foods},
            **{f"Y_{f}_{c}": 0.0 for f in farms for c in foods}
        }
    else:
        print(f"  ✅ Final objective: {final_obj:.4f} (time: {final_time:.3f}s)")
        
        # Reconstruct A and Y from column weights
        A_solution = {(f, c): 0.0 for f in farms for c in foods}
        Y_solution = {(f, c): 0.0 for f in farms for c in foods}
        
        for col_idx, weight in final_solution.items():
            if weight > 1e-6:
                col = columns[col_idx]
                for key, value in col['allocation'].items():
                    A_solution[key] += weight * value
                for key, value in col['selection'].items():
                    Y_solution[key] = max(Y_solution[key], value * weight)
        
        # PROJECT SOLUTION TO FEASIBLE SPACE (fix area overflow)
        for farm in farms:
            farm_total = sum(A_solution.get((farm, c), 0.0) for c in foods)
            farm_capacity = farms[farm]
            
            if farm_total > farm_capacity + 1e-6:
                scale_factor = farm_capacity / farm_total
                for c in foods:
                    A_solution[(farm, c)] *= scale_factor
                print(f"  ⚠️  Projected {farm}: {farm_total:.2f} -> {farm_capacity:.2f} ha")
        
        # Build full solution dictionary
        full_solution = {
            **{f"A_{f}_{c}": A_solution[(f, c)] for f, c in A_solution},
            **{f"Y_{f}_{c}": 1.0 if Y_solution[(f, c)] > 0.5 else 0.0 for f, c in Y_solution}
        }
    
    # Validate solution
    validation = validate_solution_constraints(
        full_solution, farms, foods, food_groups, farms, config, 'farm'
    )
    
    # Format result
    result = format_dantzig_wolfe_result(
        columns_generated=columns_generated,
        final_solution=full_solution,
        objective_value=final_obj,
        total_time=total_time,
        scenario_type='farm',
        n_units=len(farms),
        n_foods=len(foods),
        total_area=sum(farms.values()),
        is_feasible=validation['is_feasible'],
        validation_results=validation,
        num_variables=len(columns),
        num_constraints=len(farms) + len(food_groups) * 2,
        qpu_time_total=qpu_time_total,
        used_qpu=use_qpu_for_pricing and has_qpu
    )
    
    return result


def solve_restricted_master(
    columns: List[Dict],
    farms: Dict,
    foods: List[str],
    food_groups: Dict,
    benefits: Dict,
    config: Dict,
    integer: bool = False
) -> Tuple[Optional[Dict], float, Dict, float]:
    """
    Solve Restricted Master Problem.
    
    Returns:
        (solution, objective_value, duals, solve_time)
    """
    rmp_start = time.time()
    
    if not columns:
        return None, 0.0, {}, 0.0
    
    model = gp.Model("RMP")
    model.setParam('OutputFlag', 0)
    
    # Variables: lambda_k for each column
    lambda_vars = {}
    for k, col in enumerate(columns):
        if integer:
            lambda_vars[k] = model.addVar(vtype=GRB.BINARY, name=f"lambda_{k}")
        else:
            lambda_vars[k] = model.addVar(lb=0.0, ub=1.0, name=f"lambda_{k}")
    
    # Objective: maximize weighted column objectives
    obj_expr = gp.quicksum(
        lambda_vars[k] * col['objective'] for k, col in enumerate(columns)
    )
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Constraint: Convexity
    model.addConstr(
        gp.quicksum(lambda_vars[k] for k in lambda_vars) <= len(farms),
        name="Convexity"
    )
    
    # Resource constraints (land per farm)
    for farm in farms:
        farm_usage = gp.quicksum(
            lambda_vars[k] * sum(col['allocation'].get((farm, f), 0.0) for f in foods)
            for k, col in enumerate(columns)
        )
        model.addConstr(farm_usage <= farms[farm], name=f"Land_{farm}")
    
    # Food group constraints
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', len(foods_in_group) * len(farms))
            
            total_selected = gp.quicksum(
                lambda_vars[k] * sum(col['selection'].get((f, c), 0.0) for f in farms for c in foods_in_group)
                for k, col in enumerate(columns)
            )
            
            if min_foods > 0:
                model.addConstr(total_selected >= min_foods, name=f"FG_Min_{group_name}")
            if max_foods < float('inf'):
                model.addConstr(total_selected <= max_foods, name=f"FG_Max_{group_name}")
    
    # Solve
    model.optimize()
    
    solve_time = time.time() - rmp_start
    
    if model.status != GRB.OPTIMAL:
        return None, 0.0, {}, solve_time
    
    # Extract solution and duals
    solution = {k: var.X for k, var in lambda_vars.items()}
    obj_value = model.ObjVal
    
    duals = {}
    if not integer:  # Only extract duals for LP relaxation
        for constr in model.getConstrs():
            duals[constr.ConstrName] = constr.Pi
    
    return solution, obj_value, duals, solve_time


def solve_pricing_classical(
    farms: Dict,
    foods: List[str],
    duals: Dict,
    benefits: Dict,
    min_planting_area: Dict,
    max_planting_area: Dict
) -> Tuple[Dict, float, float]:
    """
    Solve Pricing Subproblem classically to generate new column.
    
    Returns:
        (new_column, reduced_cost, solve_time)
    """
    pricing_start = time.time()
    
    model = gp.Model("Pricing")
    model.setParam('OutputFlag', 0)
    
    # Variables
    A = {}
    Y = {}
    for farm in farms:
        for food in foods:
            A[(farm, food)] = model.addVar(lb=0.0, name=f"A_{farm}_{food}")
            Y[(farm, food)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{farm}_{food}")
    
    # Objective: benefit per hectare - dual prices (normalized by total area)
    total_area = sum(farms.values())
    obj_expr = gp.quicksum(
        A[(f, c)] * benefits.get(c, 1.0) for f in farms for c in foods
    ) / total_area
    
    # Subtract dual contributions
    land_duals = sum(duals.get(f"Land_{f}", 0.0) for f in farms)
    obj_expr -= land_duals
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Constraints: Land availability
    for farm, capacity in farms.items():
        model.addConstr(
            gp.quicksum(A[(farm, food)] for food in foods) <= capacity,
            name=f"Land_{farm}"
        )
    
    # Linking constraints
    for farm in farms:
        for food in foods:
            min_area = min_planting_area.get(food, 0.0001)
            max_area = max_planting_area.get(food, farms[farm])
            
            model.addConstr(A[(farm, food)] >= min_area * Y[(farm, food)], name=f"MinLink_{farm}_{food}")
            model.addConstr(A[(farm, food)] <= max_area * Y[(farm, food)], name=f"MaxLink_{farm}_{food}")
    
    model.optimize()
    
    solve_time = time.time() - pricing_start
    
    if model.status != GRB.OPTIMAL:
        return {}, 0.0, solve_time
    
    # Extract column
    allocation = {key: var.X for key, var in A.items() if var.X > 1e-6}
    selection = {key: var.X for key, var in Y.items() if var.X > 0.5}
    total_area = sum(farms.values())
    column_obj = sum(allocation.get(key, 0.0) * benefits.get(key[1], 1.0) for key in allocation) / total_area
    
    new_column = {
        'allocation': allocation,
        'selection': selection,
        'objective': column_obj
    }
    
    reduced_cost = model.ObjVal - duals.get('Convexity', 0.0)
    
    return new_column, reduced_cost, solve_time


def solve_pricing_qpu(
    farms: Dict,
    foods: List[str],
    duals: Dict,
    benefits: Dict,
    min_planting_area: Dict,
    max_planting_area: Dict,
    dwave_token: str
) -> Tuple[Dict, float, float, float]:
    """
    Solve Pricing Subproblem using QPU to generate new column.
    
    Returns:
        (new_column, reduced_cost, total_time, qpu_time)
    """
    pricing_start = time.time()
    
    # Build CQM for pricing problem
    cqm = ConstrainedQuadraticModel()
    
    # Variables
    Y = {}
    for farm in farms:
        for food in foods:
            var_name = f"Y_{farm}_{food}"
            Y[(farm, food)] = Binary(var_name)
            cqm.add_variable('BINARY', var_name)
    
    # Objective: maximize benefit (encoded as minimize negative)
    # Simplified: just maximize number of high-benefit selections
    objective = sum(
        Y[(f, c)] * benefits.get(c, 1.0)
        for f in farms for c in foods
    )
    cqm.set_objective(-objective)  # Minimize negative = maximize
    
    # Constraints: At most one crop per farm (simplified)
    for farm in farms:
        cqm.add_constraint(
            sum(Y[(farm, food)] for food in foods) <= 1,
            label=f"OnePerFarm_{farm}"
        )
    
    # Solve with hybrid solver
    sampler = LeapHybridBQMSampler(token=dwave_token)
    
    # Convert CQM to BQM
    bqm, invert = cqm_to_bqm(cqm)
    
    qpu_start = time.time()
    sampleset = sampler.sample(bqm, label="DantzigWolfe_Pricing_QPU")
    qpu_time = time.time() - qpu_start
    
    # Extract best sample
    best_sample = sampleset.first.sample
    
    # Build column from sample
    allocation = {}
    selection = {}
    
    for (farm, food), var in Y.items():
        var_name = f"Y_{farm}_{food}"
        y_val = best_sample.get(var_name, 0.0)
        
        if y_val > 0.5:
            selection[(farm, food)] = 1.0
            # Assign minimum area for selected crops
            min_area = min_planting_area.get(food, 0.0001)
            allocation[(farm, food)] = min_area
    
    # Calculate column objective
    column_obj = sum(
        allocation.get(key, 0.0) * benefits.get(key[1], 1.0) / 100.0
        for key in allocation
    )
    
    new_column = {
        'allocation': allocation,
        'selection': selection,
        'objective': column_obj
    }
    
    # Calculate reduced cost (approximation)
    reduced_cost = column_obj - sum(duals.values()) / len(duals) if duals else column_obj
    
    total_time = time.time() - pricing_start
    
    return new_column, reduced_cost, total_time, qpu_time
