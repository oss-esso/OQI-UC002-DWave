"""
Dantzig-Wolfe Decomposition Strategy

Column generation approach:
- Restricted Master Problem (RMP): Select from column pool
- Pricing Subproblem: Generate new columns (allocation patterns)

Iteratively generates columns until no improvement possible.
"""
import time
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
import numpy as np

from result_formatter import format_dantzig_wolfe_result, validate_solution_constraints


def solve_with_dantzig_wolfe(
    farms: Dict[str, float],
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    max_iterations: int = 50,
    time_limit: float = 300.0
) -> Dict:
    """
    Solve farm allocation problem using Dantzig-Wolfe Decomposition.
    
    Uses column generation to build solutions incrementally.
    
    Args:
        farms: Dictionary of farm names to land availability
        foods: List of food names
        food_groups: Dictionary of food group constraints
        config: Configuration dictionary
        max_iterations: Maximum column generation iterations
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
    
    # Initialize column pool with trivial columns
    columns = []
    columns_generated = []
    
    # Generate initial columns (food-group-aware to ensure RMP feasibility)
    # Strategy: Create columns that cover all food groups
    
    for farm, capacity in farms.items():
        # Pattern 1: One food from each food group
        for group_name, foods_in_group in food_groups.items():
            if not foods_in_group:
                continue
            
            best_food = max(foods_in_group, key=lambda f: benefits.get(f, 1.0))
            
            allocation = {}
            selection = {}
            
            min_area = min_planting_area.get(best_food, 0.0001)
            max_area = min(max_planting_area.get(best_food, capacity), capacity / 3)
            alloc = max(min_area, min(max_area, capacity / 5))
            
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
        
        # Pattern 2: Diverse mix covering all groups
        sorted_foods = sorted(foods, key=lambda f: benefits.get(f, 1.0), reverse=True)
        allocation = {}
        selection = {}
        remaining_capacity = capacity
        groups_covered = set()
        
        for food in sorted_foods:
            food_group = None
            for group_name, foods_in_group in food_groups.items():
                if food in foods_in_group:
                    food_group = group_name
                    break
            
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
    
    print(f"Initial column pool: {len(columns)} columns (food-group-aware)")
    
    # Column generation loop
    iteration = 0
    while iteration < max_iterations:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            print(f"Time limit reached at iteration {iteration}")
            break
        
        iteration += 1
        print(f"\nDantzig-Wolfe Iteration {iteration}")
        
        # Solve Restricted Master Problem
        rmp_solution, rmp_obj, duals = solve_restricted_master(
            columns, farms, foods, food_groups, benefits, config
        )
        
        if rmp_solution is None:
            print("RMP infeasible - terminating")
            break
        
        print(f"  RMP Objective: {rmp_obj:.4f}")
        print(f"  Active columns: {sum(1 for x in rmp_solution.values() if x > 1e-6)}/{len(columns)}")
        
        # Solve Pricing Subproblem
        new_column, reduced_cost = solve_pricing_subproblem(
            farms, foods, duals, benefits, min_planting_area, max_planting_area
        )
        
        print(f"  Pricing: Reduced cost = {reduced_cost:.6f}")
        
        columns_generated.append({
            'iteration': iteration,
            'reduced_cost': reduced_cost,
            'rmp_objective': rmp_obj,
            'num_columns': len(columns)
        })
        
        # Check if new column improves solution
        if reduced_cost >= -1e-6:  # No improvement
            print(f"  No improving column found - optimal!")
            break
        
        # Add new column to pool
        columns.append(new_column)
        print(f"  Added column to pool (total: {len(columns)})")
    
    total_time = time.time() - start_time
    
    # Construct final solution from RMP
    final_solution, final_obj, _ = solve_restricted_master(
        columns, farms, foods, food_groups, benefits, config, integer=True
    )
    
    if final_solution is None:
        # Fallback to relaxed solution
        final_solution, final_obj, _ = solve_restricted_master(
            columns, farms, foods, food_groups, benefits, config, integer=False
        )
    
    if final_solution is None:
        # No solution found - return empty solution
        final_obj = 0.0
        full_solution = {
            **{f"A_{f}_{c}": 0.0 for f in farms for c in foods},
            **{f"Y_{f}_{c}": 0.0 for f in farms for c in foods}
        }
    else:
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
        
        # Binarize Y values
        Y_binary = {key: 1.0 if val > 0.5 else 0.0 for key, val in Y_solution.items()}
        
        # ENFORCE FOOD GROUP MINIMUM CONSTRAINTS (column solution may not satisfy these)
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
                                current_usage = sum(A_solution.get((f, food), 0.0) for food in foods)
                                remaining = farms[f] - current_usage
                                farm_capacities.append((f, remaining))
                            farm_capacities.sort(key=lambda x: x[1], reverse=True)
                            
                            best_farm = farm_capacities[0][0]
                            Y_binary[(best_farm, c)] = 1.0
                            min_area = min_planting_area.get(c, 0.0001)
                            A_solution[(best_farm, c)] = max(A_solution[(best_farm, c)], min_area)
                            print(f"    + Added Y_{best_farm}_{c} with A={A_solution[(best_farm, c)]:.4f}")
        
        # Enforce linking constraints: If Y=0, force A=0; if Y=1, ensure A >= min_area
        for key in A_solution:
            farm, food = key
            y_val = Y_binary.get(key, 0.0)
            if y_val < 0.5:
                A_solution[key] = 0.0
            else:
                min_area = min_planting_area.get(food, 0.0001)
                A_solution[key] = max(A_solution[key], min_area)
        
        # PROJECT SOLUTION TO FEASIBLE SPACE (fix area overflow)
        # For each farm, if total area exceeds capacity, scale down proportionally
        for farm in farms:
            farm_total = sum(A_solution.get((farm, c), 0.0) for c in foods)
            farm_capacity = farms[farm]
            
            if farm_total > farm_capacity + 1e-6:
                # Scale down all allocations for this farm
                scale_factor = farm_capacity / farm_total
                for c in foods:
                    A_solution[(farm, c)] *= scale_factor
                
                print(f"  ⚠️  Projected {farm}: {farm_total:.2f} -> {farm_capacity:.2f} ha (scaled by {scale_factor:.4f})")
        
        # Build full solution dictionary
        full_solution = {
            **{f"A_{f}_{c}": A_solution[(f, c)] for f, c in A_solution},
            **{f"Y_{f}_{c}": Y_binary[(f, c)] for f, c in Y_binary}
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
        num_constraints=len(farms) + len(food_groups) * 2
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
) -> Tuple[Optional[Dict], float, Dict]:
    """
    Solve Restricted Master Problem.
    
    Variables: lambda_k (weight for each column)
    Objective: maximize sum_k lambda_k * column_k_objective
    Constraints: Convexity + resource limits
    """
    if not columns:
        return None, 0.0, {}
    
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
    
    # Constraint: Convexity (sum lambda = number of active allocations)
    # Simplified: sum lambda <= number of farms
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
    
    if model.status != GRB.OPTIMAL:
        return None, 0.0, {}
    
    # Extract solution and duals
    solution = {k: var.X for k, var in lambda_vars.items()}
    obj_value = model.ObjVal
    
    duals = {}
    if not integer:  # Only extract duals for LP relaxation
        for constr in model.getConstrs():
            duals[constr.ConstrName] = constr.Pi
    
    return solution, obj_value, duals


def solve_pricing_subproblem(
    farms: Dict,
    foods: List[str],
    duals: Dict,
    benefits: Dict,
    min_planting_area: Dict,
    max_planting_area: Dict
) -> Tuple[Dict, float]:
    """
    Solve Pricing Subproblem to generate new column.
    
    Find allocation pattern with negative reduced cost.
    """
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
    
    # Subtract dual contributions (simplified)
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
    
    if model.status != GRB.OPTIMAL:
        # No improving column
        return {}, 0.0
    
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
    
    return new_column, reduced_cost
