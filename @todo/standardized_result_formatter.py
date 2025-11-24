"""
Standardized Result Formatter for All Decomposition Strategies

Ensures all solvers return results in the exact same format as PuLP
for consistent validation and comparison.
"""
from typing import Dict, List
from datetime import datetime


def format_standard_result(
    farms: Dict[str, float],
    foods: List[str],
    solution_A: Dict,  # Can have tuple keys or string keys
    solution_Y: Dict,  # Can have tuple keys or string keys
    objective_value: float,
    solve_time: float,
    solver_name: str,
    status: str = "Optimal",
    success: bool = True,
    num_variables: int = 0,
    num_constraints: int = 0,
    **kwargs
) -> Dict:
    """
    Format result to match PuLP output structure exactly.
    
    Args:
        farms: Dictionary of farm names to land availability
        foods: List of food names
        solution_A: Area allocations (can be dict with tuple or string keys)
        solution_Y: Binary selections (can be dict with tuple or string keys)
        objective_value: Objective value (should be NORMALIZED by total area)
        solve_time: Total solution time
        solver_name: Name of the solver used
        status: Solution status
        success: Whether solve was successful
        num_variables: Number of variables
        num_constraints: Number of constraints
        **kwargs: Additional solver-specific info
    
    Returns:
        Standardized result dictionary matching PuLP format
    """
    
    # Convert solution dictionaries to standard format
    solution_areas = {}
    solution_selections = {}
    
    for farm in farms:
        for food in foods:
            key_str = f"{farm}_{food}"
            
            # Try tuple key first, then string key
            if (farm, food) in solution_A:
                solution_areas[key_str] = float(solution_A[(farm, food)])
            elif key_str in solution_A:
                solution_areas[key_str] = float(solution_A[key_str])
            elif f"A_{key_str}" in solution_A:
                solution_areas[key_str] = float(solution_A[f"A_{key_str}"])
            else:
                solution_areas[key_str] = 0.0
            
            # Binary selections
            if (farm, food) in solution_Y:
                y_val = float(solution_Y[(farm, food)])
            elif key_str in solution_Y:
                y_val = float(solution_Y[key_str])
            elif f"Y_{key_str}" in solution_Y:
                y_val = float(solution_Y[f"Y_{key_str}"])
            else:
                y_val = 0.0
            
            # Round binary to 1.0 or 0.0 (or -0.0 for negatives in PuLP style)
            if y_val > 0.5:
                solution_selections[key_str] = 1.0
            elif y_val < -0.5:
                solution_selections[key_str] = -0.0
            else:
                solution_selections[key_str] = -0.0 if y_val < 0 else 0.0
    
    # Calculate total covered area
    total_covered_area = sum(solution_areas.values())
    total_area = sum(farms.values())
    
    # Build land_data dictionary
    land_data = {}
    for farm in farms:
        farm_allocation = sum(
            solution_areas.get(f"{farm}_{food}", 0.0)
            for food in foods
        )
        land_data[farm] = farm_allocation
    
    # Construct standardized result
    result = {
        "status": status,
        "objective_value": float(objective_value),
        "solve_time": float(solve_time),
        "solver_time": float(kwargs.get('solver_time', solve_time)),
        "success": bool(success),
        "sample_id": kwargs.get('sample_id', kwargs.get('num_iterations', 1)),
        "n_units": len(farms),
        "total_area": float(total_area),
        "n_foods": len(foods),
        "n_variables": num_variables if num_variables > 0 else len(farms) * len(foods) * 2,
        "n_constraints": num_constraints,
        "solution_areas": solution_areas,
        "solution_selections": solution_selections,
        "total_covered_area": float(total_covered_area),
        "land_data": land_data,
        "solver_name": solver_name,
        "timestamp": kwargs.get('timestamp', datetime.now().isoformat())
    }
    
    # Add optional fields if provided
    if 'validation' in kwargs:
        result['validation'] = kwargs['validation']
    
    if 'decomposition_info' in kwargs:
        result['decomposition_info'] = kwargs['decomposition_info']
    
    if 'iterations' in kwargs:
        result['num_iterations'] = kwargs['iterations']
    
    return result


def extract_solution_from_decomp_result(result: Dict) -> tuple:
    """
    Extract solution_areas and solution_selections from any decomposition result format.
    
    Returns:
        (solution_areas, solution_selections) tuple of dictionaries
    """
    solution_areas = {}
    solution_selections = {}
    
    # Try different result structures
    if 'solution_areas' in result:
        # Already in standard format
        solution_areas = result['solution_areas']
        solution_selections = result.get('solution_selections', {})
    
    elif 'solution' in result:
        solution = result['solution']
        if 'full_solution' in solution:
            full_sol = solution['full_solution']
        else:
            full_sol = solution
        
        # Parse variable names
        for var_name, value in full_sol.items():
            if var_name.startswith('A_'):
                key = var_name[2:]  # Remove 'A_'
                solution_areas[key] = float(value)
            elif var_name.startswith('Y_'):
                key = var_name[2:]  # Remove 'Y_'
                solution_selections[key] = float(value)
    
    elif 'result' in result:
        res = result['result']
        if 'solution_areas' in res:
            solution_areas = res['solution_areas']
            solution_selections = res.get('solution_selections', {})
    
    return solution_areas, solution_selections
