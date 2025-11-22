"""
Quick Compatibility Wrapper for Benchmark Scripts

This module provides backward compatibility between the new standardized
result format and existing benchmark scripts that expect the old format.

Use this until benchmark scripts are fully updated.
"""

def convert_to_old_format(new_result):
    """
    Convert new standardized format to old format expected by benchmarks.
    
    New format matches Pyomo template exactly with all data in 'result' section.
    Old format had separate 'solver_info' and 'solution' sections.
    
    Args:
        new_result: Result in new standardized format (Pyomo template)
        
    Returns:
        Result in old format compatible with existing benchmarks
    """
    # New format has everything in 'result' section
    result_section = new_result.get('result', {})
    
    # Build old format with separate solver_info and solution sections
    old_format = {
        'solver_info': {
            'status': result_section.get('status', 'Unknown'),
            'solve_time': result_section.get('solve_time', 0.0),
            'num_iterations': result_section.get('iterations', 1),
            'success': result_section.get('success', False)
        },
        'solution': {
            'objective_value': result_section.get('objective_value', 0.0),
            'is_feasible': result_section.get('validation', {}).get('is_feasible', True),
            'full_solution': {
                **result_section.get('solution_areas', {}),
                **result_section.get('solution_selections', {})
            },
            'total_covered_area': result_section.get('total_covered_area', 0.0)
        },
        'metadata': new_result.get('metadata', {}),
        'validation': result_section.get('validation', {}),
        'decomposition_specific': new_result.get('decomposition_specific', {})
    }
    
    return old_format


def wrap_solver(solve_func):
    """
    Decorator to automatically convert new format to old format.
    
    Usage:
        @wrap_solver
        def solve_with_benders(...):
            ...
            return format_benders_result(...)
    """
    def wrapper(*args, **kwargs):
        result = solve_func(*args, **kwargs)
        return convert_to_old_format(result)
    return wrapper
