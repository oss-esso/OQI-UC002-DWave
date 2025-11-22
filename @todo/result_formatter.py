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
    run_number: int = 1,
    **kwargs
) -> Dict:
    """
    Format decomposition strategy results to match Pyomo template exactly.
    
    Template: Benchmarks/LQ/Pyomo/config_10_run_1.json
    """
    from datetime import datetime
    
    # Extract solution areas and selections
    solution_areas = {}
    solution_selections = {}
    
    for var_name, value in solution.items():
        if var_name.startswith('A_') or var_name.startswith('Area'):
            # Convert to Farm#_Food format
            clean_name = var_name.replace('Area_', '').replace('A_', '')
            solution_areas[clean_name] = value
        elif var_name.startswith('Y_') or var_name.startswith('Choose'):
            # Convert to Farm#_Food format
            clean_name = var_name.replace('Choose_', '').replace('Y_', '')
            solution_selections[clean_name] = value
    
    # Calculate total covered area
    total_covered_area = sum(solution_areas.values())
    
    # Build validation (use provided or create default)
    if validation_results is None:
        validation_results = {
            'is_feasible': is_feasible,
            'n_violations': 0,
            'violations': [],
            'constraint_checks': {},
            'summary': {
                'total_checks': 0,
                'total_passed': 0,
                'total_failed': 0,
                'pass_rate': 1.0
            }
        }
    
    # Match exact template structure
    result = {
        'metadata': {
            'benchmark_type': 'DECOMPOSITION',
            'solver': strategy_name.upper(),
            'n_farms': n_units,
            'run_number': run_number,
            'timestamp': kwargs.get('timestamp', datetime.now().isoformat())
        },
        'result': {
            'status': kwargs.get('status', 'ok (optimal)' if is_feasible else 'failed'),
            'objective_value': objective_value,
            'solve_time': solve_time,
            'solver_time': solve_time,
            'success': is_feasible,
            'sample_id': 0,
            'n_units': n_units,
            'total_area': total_area,
            'n_foods': n_foods,
            'n_variables': num_variables,
            'n_constraints': num_constraints,
            'solver': kwargs.get('solver_engine', 'gurobi'),
            'solution_areas': solution_areas,
            'solution_selections': solution_selections,
            'total_covered_area': total_covered_area,
            'solution_summary': {
                'total_allocated': total_covered_area,
                'total_available': total_area,
                'idle_area': total_area - total_covered_area,
                'utilization': total_covered_area / total_area if total_area > 0 else 0.0
            },
            'validation': validation_results,
            'error': None
        }
    }
    
    # Add decomposition-specific info if provided
    if 'decomposition_specific' in kwargs:
        result['decomposition_specific'] = kwargs['decomposition_specific']
    
    return result


def format_benders_result(
    master_iterations: List[Dict],
    final_solution: Dict[str, float],
    objective_value: float,
    total_time: float,
    **kwargs
) -> Dict:
    """Format Benders decomposition specific results."""
    
    # Prepare decomposition-specific details
    decomposition_specific = {
        'iterations_detail': master_iterations,
        'cuts_added': sum(1 for it in master_iterations if it.get('cut_added', False)),
        'final_gap': master_iterations[-1].get('gap', 0.0) if master_iterations else 0.0,
        'lower_bound': master_iterations[-1].get('lower_bound', 0.0) if master_iterations else 0.0,
        'upper_bound': master_iterations[-1].get('upper_bound', 0.0) if master_iterations else 0.0
    }
    
    base_result = format_decomposition_result(
        strategy_name='benders',
        objective_value=objective_value,
        solution=final_solution,
        solve_time=total_time,
        num_iterations=len(master_iterations),
        decomposition_specific=decomposition_specific,
        **kwargs
    )
    
    return base_result


def format_dantzig_wolfe_result(
    columns_generated: List[Dict],
    final_solution: Dict[str, float],
    objective_value: float,
    total_time: float,
    **kwargs
) -> Dict:
    """Format Dantzig-Wolfe decomposition specific results."""
    
    # Prepare decomposition-specific details
    decomposition_specific = {
        'iterations_detail': columns_generated,
        'columns_generated': len(columns_generated),
        'final_reduced_cost': columns_generated[-1].get('reduced_cost', 0.0) if columns_generated else 0.0,
        'active_columns': kwargs.get('active_columns', len(columns_generated)),
        'total_columns': kwargs.get('total_columns', len(columns_generated))
    }
    
    base_result = format_decomposition_result(
        strategy_name='dantzig_wolfe',
        objective_value=objective_value,
        solution=final_solution,
        solve_time=total_time,
        num_iterations=len(columns_generated),
        decomposition_specific=decomposition_specific,
        **kwargs
    )
    
    return base_result


def format_admm_result(
    iterations: List[Dict],
    final_solution: Dict[str, float],
    objective_value: float,
    total_time: float,
    **kwargs
) -> Dict:
    """Format ADMM decomposition specific results."""
    
    # Prepare convergence metrics
    final_iter = iterations[-1] if iterations else {}
    convergence = {
        'primal_residual': final_iter.get('primal_residual', 0.0),
        'dual_residual': final_iter.get('dual_residual', 0.0),
        'converged': final_iter.get('primal_residual', 1.0) < kwargs.get('tolerance', 1e-4)
    }
    
    # Prepare decomposition-specific details
    decomposition_specific = {
        'iterations_detail': iterations,
        'rho': kwargs.get('rho', 1.0),
        'tolerance': kwargs.get('tolerance', 1e-4),
        'primal_residual_history': [it.get('primal_residual', 0.0) for it in iterations],
        'dual_residual_history': [it.get('dual_residual', 0.0) for it in iterations],
        'qpu_time_total': kwargs.get('qpu_time_total', 0.0),
        'used_qpu': kwargs.get('used_qpu', False)
    }
    
    base_result = format_decomposition_result(
        strategy_name='admm',
        objective_value=objective_value,
        solution=final_solution,
        solve_time=total_time,
        num_iterations=len(iterations),
        convergence=convergence,
        decomposition_specific=decomposition_specific,
        **kwargs
    )
    
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
