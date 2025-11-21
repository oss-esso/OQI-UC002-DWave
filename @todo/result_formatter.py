"""
Result Formatter Module

Standardizes JSON output format across all decomposition strategies
to match the reference format from benchmark results.
"""
from typing import Dict, List, Optional, Any
import json


def format_decomposition_result(
    strategy_name: str,
    scenario_type: str,
    n_units: int,
    n_foods: int,
    total_area: float,
    objective_value: float,
    solution: Dict[str, float],
    solve_time: float,
    num_iterations: int = 1,
    is_feasible: bool = True,
    validation_results: Optional[Dict] = None,
    num_variables: int = 0,
    num_constraints: int = 0,
    **kwargs
) -> Dict:
    """
    Format decomposition strategy results into standardized JSON structure.
    
    Args:
        strategy_name: Name of decomposition strategy (e.g., 'benders', 'admm')
        scenario_type: 'farm' or 'patch'
        n_units: Number of farms/patches
        n_foods: Number of food types
        total_area: Total available land area
        objective_value: Final objective function value
        solution: Dictionary of variable assignments
        solve_time: Total solution time in seconds
        num_iterations: Number of iterations for iterative methods
        is_feasible: Whether solution is feasible
        validation_results: Constraint validation results
        num_variables: Total number of decision variables
        num_constraints: Total number of constraints
        **kwargs: Additional strategy-specific metadata
    
    Returns:
        Formatted result dictionary matching reference JSON structure
    """
    
    # Extract Y variables (binary selection indicators)
    solution_plantations = {}
    total_covered_area = 0.0
    
    for var_name, value in solution.items():
        if var_name.startswith('Y_') or var_name.startswith('Y['):
            # Parse variable name to extract farm/patch and food
            solution_plantations[var_name] = value
            
        # Calculate total covered area from A variables
        if var_name.startswith('A_') or var_name.startswith('A['):
            total_covered_area += value
    
    # Build validation section
    validation_section = validation_results or {
        'is_feasible': is_feasible,
        'n_violations': 0,
        'violations': [],
        'constraint_checks': {},
        'summary': 'No validation performed' if validation_results is None else 'Validation complete'
    }
    
    # Build result dictionary
    result = {
        'metadata': {
            'decomposition_strategy': strategy_name,
            'scenario_type': scenario_type,
            'n_units': n_units,
            'n_foods': n_foods,
            'timestamp': kwargs.get('timestamp', None)
        },
        'problem_info': {
            'n_variables': num_variables,
            'n_constraints': num_constraints,
            'total_area': total_area,
            'problem_size': n_units * n_foods
        },
        'solver_info': {
            'strategy': strategy_name,
            'num_iterations': num_iterations,
            'solve_time': solve_time,
            'success': is_feasible,
            'status': 'Optimal' if is_feasible else 'Infeasible'
        },
        'solution': {
            'objective_value': objective_value,
            'is_feasible': is_feasible,
            'total_covered_area': total_covered_area,
            'solution_plantations': solution_plantations,
            'full_solution': solution
        },
        'validation': validation_section,
        'additional_info': {k: v for k, v in kwargs.items() if k != 'timestamp'}
    }
    
    return result


def format_benders_result(
    master_iterations: List[Dict],
    final_solution: Dict[str, float],
    objective_value: float,
    total_time: float,
    **kwargs
) -> Dict:
    """Format Benders decomposition specific results."""
    
    base_result = format_decomposition_result(
        strategy_name='benders',
        objective_value=objective_value,
        solution=final_solution,
        solve_time=total_time,
        num_iterations=len(master_iterations),
        **kwargs
    )
    
    # Add Benders-specific information
    base_result['benders_info'] = {
        'master_iterations': master_iterations,
        'num_cuts_added': sum(1 for it in master_iterations if 'cut_added' in it),
        'convergence_gap': master_iterations[-1].get('gap', 0.0) if master_iterations else 0.0
    }
    
    return base_result


def format_dantzig_wolfe_result(
    columns_generated: List[Dict],
    final_solution: Dict[str, float],
    objective_value: float,
    total_time: float,
    **kwargs
) -> Dict:
    """Format Dantzig-Wolfe decomposition specific results."""
    
    base_result = format_decomposition_result(
        strategy_name='dantzig_wolfe',
        objective_value=objective_value,
        solution=final_solution,
        solve_time=total_time,
        num_iterations=len(columns_generated),
        **kwargs
    )
    
    # Add Dantzig-Wolfe-specific information
    base_result['dantzig_wolfe_info'] = {
        'columns_generated': len(columns_generated),
        'column_details': columns_generated,
        'reduced_cost_final': columns_generated[-1].get('reduced_cost', 0.0) if columns_generated else 0.0
    }
    
    return base_result


def format_admm_result(
    admm_iterations: List[Dict],
    final_solution: Dict[str, float],
    objective_value: float,
    total_time: float,
    **kwargs
) -> Dict:
    """Format ADMM decomposition specific results."""
    
    base_result = format_decomposition_result(
        strategy_name='admm',
        objective_value=objective_value,
        solution=final_solution,
        solve_time=total_time,
        num_iterations=len(admm_iterations),
        **kwargs
    )
    
    # Add ADMM-specific information
    base_result['admm_info'] = {
        'iterations': admm_iterations,
        'primal_residual': admm_iterations[-1].get('primal_residual', 0.0) if admm_iterations else 0.0,
        'dual_residual': admm_iterations[-1].get('dual_residual', 0.0) if admm_iterations else 0.0,
        'rho': kwargs.get('rho', 1.0)
    }
    
    return base_result


def save_result_json(result: Dict, filepath: str):
    """Save formatted result to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def validate_solution_constraints(
    solution: Dict[str, float],
    farms_or_patches: Dict,
    foods: List[str],
    food_groups: Dict,
    land_availability: Dict,
    config: Dict,
    scenario_type: str = 'farm'
) -> Dict:
    """
    Validate solution against all problem constraints.
    
    Returns:
        Dictionary with is_feasible, n_violations, violations list, constraint_checks, summary
    """
    violations = []
    constraint_checks = {}
    
    # Helper to parse variable names
    def parse_var(var_name):
        """Parse variable name like 'A_Farm1_Beef' or 'Y[Farm1,Beef]'."""
        var_name = var_name.replace('[', '_').replace(']', '').replace(',', '_')
        parts = var_name.split('_')
        if len(parts) >= 3:
            return parts[1], '_'.join(parts[2:])  # farm/patch, food
        return None, None
    
    # Extract A and Y solutions
    A_sol = {}
    Y_sol = {}
    for var_name, value in solution.items():
        farm, food = parse_var(var_name)
        if farm and food:
            if var_name.startswith('A'):
                A_sol[(farm, food)] = value
            elif var_name.startswith('Y'):
                Y_sol[(farm, food)] = value
    
    # Constraint 1: Land availability
    for farm in farms_or_patches:
        total_used = sum(A_sol.get((farm, food), 0.0) for food in foods)
        capacity = land_availability.get(farm, 0.0)
        check_name = f"land_availability_{farm}"
        constraint_checks[check_name] = {
            'type': 'land_capacity',
            'limit': capacity,
            'actual': total_used,
            'satisfied': total_used <= capacity + 1e-6
        }
        if total_used > capacity + 1e-6:
            violations.append({
                'constraint': check_name,
                'type': 'land_exceeded',
                'farm': farm,
                'limit': capacity,
                'actual': total_used,
                'violation': total_used - capacity
            })
    
    # Constraint 2: Linking constraints (A-Y relationship)
    min_area = config.get('parameters', {}).get('minimum_planting_area', {})
    max_area = config.get('parameters', {}).get('maximum_planting_area', {})
    
    for (farm, food) in A_sol:
        a_val = A_sol[(farm, food)]
        y_val = Y_sol.get((farm, food), 0.0)
        
        # Min area if selected
        min_val = min_area.get(food, 0.0001)
        if y_val > 0.5 and a_val < min_val - 1e-6:
            violations.append({
                'constraint': f'min_area_{farm}_{food}',
                'type': 'min_area_violation',
                'farm': farm,
                'food': food,
                'required': min_val,
                'actual': a_val,
                'y_value': y_val
            })
        
        # A should be zero if Y is zero
        if y_val < 0.5 and a_val > 1e-6:
            violations.append({
                'constraint': f'linking_{farm}_{food}',
                'type': 'linking_constraint_violation',
                'farm': farm,
                'food': food,
                'A_value': a_val,
                'Y_value': y_val,
                'message': 'A > 0 but Y = 0'
            })
    
    # Constraint 3: Food group constraints (COUNT-based, not area)
    food_group_constraints = config.get('parameters', {}).get('food_group_constraints', {})
    if food_group_constraints:
        for group_name, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            min_foods = constraints.get('min_foods', 0)
            max_foods = constraints.get('max_foods', float('inf'))
            
            # Count number of Y=1 for this group across ALL farms
            total_count = sum(
                1 if Y_sol.get((farm, food), 0.0) > 0.5 else 0
                for farm in farms_or_patches
                for food in foods_in_group
            )
            
            check_name = f"food_group_{group_name}"
            constraint_checks[check_name] = {
                'type': 'food_group_count',
                'min': min_foods,
                'max': max_foods,
                'actual': total_count,
                'satisfied': min_foods <= total_count <= max_foods
            }
            
            if total_count < min_foods:
                violations.append({
                    'constraint': check_name,
                    'type': 'food_group_min_violation',
                    'group': group_name,
                    'required_min': min_foods,
                    'actual': total_count,
                    'deficit': min_foods - total_count
                })
            
            if total_count > max_foods:
                violations.append({
                    'constraint': check_name,
                    'type': 'food_group_max_violation',
                    'group': group_name,
                    'required_max': max_foods,
                    'actual': total_count,
                    'excess': total_count - max_foods
                })
    
    # Summary
    is_feasible = len(violations) == 0
    summary = f"{'Feasible' if is_feasible else 'Infeasible'}: {len(violations)} violations found"
    
    return {
        'is_feasible': is_feasible,
        'n_violations': len(violations),
        'violations': violations,
        'constraint_checks': constraint_checks,
        'summary': summary
    }
