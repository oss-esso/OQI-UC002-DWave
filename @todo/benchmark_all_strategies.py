#!/usr/bin/env python3
"""
Comprehensive Benchmark for All Decomposition Strategies

Tests multiple decomposition approaches:
- Current Hybrid (Gurobi relaxation + QPU)
- Benders Decomposition
- Dantzig-Wolfe Decomposition
- ADMM Decomposition

Provides unified interface for comparing all strategies.
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from decomposition_strategies import DecompositionFactory, solve_with_strategy
from benchmark_utils_decomposed import generate_farm_data, create_config
from infeasibility_detection import detect_infeasibility, check_config_feasibility
from src.scenarios import load_food_data
from standardized_result_formatter import format_standard_result, extract_solution_from_decomp_result


def convert_to_standard_format(result: Dict, strategy_name: str, farms: Dict, foods: List[str]) -> Dict:
    """Convert decomposition result to standardized PuLP-compatible format."""
    
    # Extract core information
    if 'solution' in result:
        solution_dict = result['solution']
        objective_value = solution_dict.get('objective_value', 0.0)
        is_feasible = solution_dict.get('is_feasible', False)
        full_solution = solution_dict.get('full_solution', solution_dict)
    elif 'result' in result:
        result_dict = result['result']
        objective_value = result_dict.get('objective_value', 0.0)
        is_feasible = result_dict.get('success', False)
        full_solution = result_dict.get('solution', {})
    else:
        objective_value = 0.0
        is_feasible = False
        full_solution = {}
    
    # Extract A and Y from full_solution
    A_dict = {}
    Y_dict = {}
    
    # First pass: look for A_ and Y_ prefixed variables
    for var_name, value in full_solution.items():
        if var_name.startswith('A_'):
            key = var_name[2:]  # Remove 'A_'
            parts = key.split('_', 1)
            if len(parts) == 2:
                A_dict[(parts[0], parts[1])] = value
        elif var_name.startswith('Y_'):
            key = var_name[2:]  # Remove 'Y_'
            parts = key.split('_', 1)
            if len(parts) == 2:
                Y_dict[(parts[0], parts[1])] = value
    
    # If no prefixed variables found, assume non-prefixed are Y variables (binary selections)
    # This handles old format from some decomposition solvers
    if len(A_dict) == 0 and len(Y_dict) == 0:
        for var_name, value in full_solution.items():
            if not var_name.startswith('A_') and not var_name.startswith('Y_'):
                parts = var_name.split('_', 1)
                if len(parts) == 2:
                    farm, food = parts
                    Y_dict[(farm, food)] = value
                    # Assume full land allocation if crop is selected
                    # (This is a simplification for solvers that only return Y variables)
                    if value > 0.5:
                        # Need to get land availability for this farm
                        A_dict[(farm, food)] = farms.get(farm, 0.0)
    
    # RECALCULATE OBJECTIVE FROM ACTUAL SOLUTION
    # Don't trust the objective passed from decomposition algorithms!
    total_area = sum(farms.values())
    
    # Get benefits from config
    _, _, _, base_config = load_food_data('full_family')
    weights = base_config.get('parameters', {}).get('weights', {
        'nutrition': 0.4,
        'ghg': 0.3,
        'water': 0.2,
        'sustainability': 0.1
    })
    
    benefits = {}
    for food in foods:
        benefit = sum(
            base_config.get('nutrients', {}).get(food, {}).get(attr, 0) * weight
            for attr, weight in weights.items()
        )
        benefits[food] = benefit if benefit > 0 else 100.0
    
    # Calculate actual objective from solution
    actual_objective = 0.0
    for (farm, food), area in A_dict.items():
        y_val = Y_dict.get((farm, food), 0.0)
        if y_val > 0.5:  # Food is selected
            actual_objective += area * benefits.get(food, 100.0)
    
    # Normalize by total area
    actual_objective = actual_objective / total_area if total_area > 0 else 0.0
    
    print(f"  ⚠️  Objective recalculated: {objective_value:.6f} → {actual_objective:.6f}")
    
    # Get timing and metadata
    if 'solver_info' in result:
        solve_time = result['solver_info'].get('solve_time', 0.0)
        num_iterations = result['solver_info'].get('num_iterations', 1)
        status_str = result['solver_info'].get('status', 'Unknown')
    elif 'metadata' in result:
        solve_time = result.get('solve_time', 0.0)
        num_iterations = 1
        status_str = 'Unknown'
    else:
        solve_time = 0.0
        num_iterations = 1
        status_str = 'Unknown'
    
    # Create standard format result
    standard_result = format_standard_result(
        farms=farms,
        foods=foods,
        solution_A=A_dict,
        solution_Y=Y_dict,
        objective_value=actual_objective,  # Use recalculated objective
        solve_time=solve_time,
        solver_name=strategy_name,
        status="Optimal" if is_feasible else "Infeasible",
        success=is_feasible,
        num_variables=len(farms) * len(foods) * 2,
        num_constraints=result.get('metadata', {}).get('n_constraints', 0),
        num_iterations=num_iterations,
        validation=result.get('validation', {}),
        decomposition_info={
            'strategy': strategy_name,
            'original_status': status_str,
            'original_objective': objective_value,  # Keep original for debugging
            'recalculated_objective': actual_objective,
            'feasibility_check': result.get('feasibility_check', {})
        }
    )
    
    return standard_result


def run_strategy_benchmark(
    strategy_name: str,
    n_units: int,
    dwave_token: str = None,
    total_land: float = 100.0,
    **kwargs
) -> dict:
    """
    Run benchmark for a single decomposition strategy.
    
    Args:
        strategy_name: Name of decomposition strategy
        n_units: Number of farm units
        dwave_token: D-Wave API token (for current_hybrid)
        total_land: Total land area
        **kwargs: Strategy-specific parameters
    
    Returns:
        Result dictionary
    """
    print(f"\n{'='*80}")
    print(f"STRATEGY: {strategy_name.upper()}")
    print(f"{'='*80}")
    
    # Generate problem data
    farm_data = generate_farm_data(n_units, total_land)
    foods, food_groups, config_dict = create_config(farm_data['land_data'])
    
    # Load benefits from food data
    from src.scenarios import load_food_data
    _, _, _, base_config = load_food_data('full_family')
    
    # Add benefits to config
    config_dict['benefits'] = {}
    for food in foods:
        # Use composite benefit score (weighted attributes)
        weights = config_dict.get('parameters', {}).get('weights', {})
        benefit = sum(
            base_config.get('nutrients', {}).get(food, {}).get(attr, 0) * weight
            for attr, weight in weights.items()
        )
        config_dict['benefits'][food] = benefit if benefit > 0 else 100.0  # Default benefit
    
    config_dict['food_groups'] = food_groups
    config_dict['foods'] = foods
    
    farms = farm_data['land_data']
    
    # Check feasibility before solving
    print("\nChecking problem feasibility...")
    feasibility_check = check_config_feasibility(config_dict, farms, foods)
    
    if not feasibility_check['is_feasible']:
        print(f"⚠️  Configuration warnings found: {feasibility_check['num_warnings']}")
        for warning in feasibility_check['warnings']:
            print(f"   • {warning['message']}")
    
    # Solve with strategy
    try:
        result = solve_with_strategy(
            strategy_name=strategy_name,
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config_dict,
            dwave_token=dwave_token,
            **kwargs
        )
        
        result['feasibility_check'] = feasibility_check
        
        # Convert to standard format
        result = convert_to_standard_format(result, strategy_name, farms, foods)
        
        return result
        
    except Exception as e:
        print(f"❌ Strategy failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Run infeasibility detection
        print("\nRunning infeasibility diagnostics...")
        diagnostic = detect_infeasibility(farms, foods, food_groups, config_dict)
        
        return {
            'metadata': {
                'decomposition_strategy': strategy_name,
                'scenario_type': 'farm',
                'n_units': n_units,
                'n_foods': len(foods),
                'timestamp': datetime.now().isoformat()
            },
            'solver_info': {
                'strategy': strategy_name,
                'success': False,
                'status': 'Failed',
                'error': str(e)
            },
            'infeasibility_diagnostic': diagnostic.to_dict(),
            'feasibility_check': feasibility_check
        }


def run_all_strategies(
    n_units: int,
    dwave_token: str = None,
    total_land: float = 100.0,
    strategies: List[str] = None,
    **kwargs
) -> dict:
    """
    Run benchmark for all decomposition strategies.
    
    Args:
        n_units: Number of farm units
        dwave_token: D-Wave API token
        total_land: Total land area
        strategies: List of strategy names (None = all)
        **kwargs: Strategy-specific parameters
    
    Returns:
        Dictionary with results from all strategies
    """
    if strategies is None:
        strategies = DecompositionFactory.list_strategies()
    
    print(f"\n{'='*80}")
    print(f"BENCHMARKING {len(strategies)} DECOMPOSITION STRATEGIES")
    print(f"{'='*80}")
    print(f"Problem size: {n_units} farms, 27 foods")
    print(f"Total land: {total_land:.2f}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"{'='*80}")
    
    results = {}
    
    for strategy_name in strategies:
        result = run_strategy_benchmark(
            strategy_name=strategy_name,
            n_units=n_units,
            dwave_token=dwave_token,
            total_land=total_land,
            **kwargs
        )
        
        results[strategy_name] = result
    
    return results


def find_tuple_keys(obj, path="root"):
    """Recursively find dictionaries with tuple keys for debugging."""
    problematic_paths = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, tuple):
                problematic_paths.append(f"{path}.{key}")
            problematic_paths.extend(find_tuple_keys(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            problematic_paths.extend(find_tuple_keys(item, f"{path}[{i}]"))
    
    return problematic_paths


def sanitize_dict_for_json(obj):
    """Convert tuple keys to strings and numpy types to native Python types recursively."""
    if isinstance(obj, dict):
        return {
            (f"{k[0]}_{k[1]}" if isinstance(k, tuple) else str(k)): sanitize_dict_for_json(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [sanitize_dict_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_dict_for_json(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def print_comparison_table(results: dict):
    """Print comparison table of all strategies."""
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Strategy':<25} {'Status':<12} {'Time (s)':<10} {'Objective':<12} {'Iterations':<12}")
    print("-" * 80)
    
    for strategy_name, result in results.items():
        solver_info = result.get('solver_info', {})
        status = solver_info.get('status', 'Unknown')
        solve_time = solver_info.get('solve_time', 0.0)
        
        solution = result.get('solution', {})
        objective = solution.get('objective_value', 0.0)
        
        iterations = solver_info.get('num_iterations', '-')
        
        print(f"{strategy_name:<25} {status:<12} {solve_time:<10.3f} {objective:<12.4f} {iterations!s:<12}")
    
    print("-" * 80)


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description='Benchmark All Decomposition Strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=int, default=10,
                       help='Number of farm units (default: 10)')
    parser.add_argument('--strategies', type=str, default='all',
                       help='Comma-separated list of strategies or "all" (default: all)')
    parser.add_argument('--token', type=str, default=None,
                       help='D-Wave API token (for current_hybrid strategy)')
    parser.add_argument('--output-dir', type=str, default='Benchmarks/ALL_STRATEGIES',
                       help='Output directory for results')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='Maximum iterations for iterative methods (default: 50)')
    parser.add_argument('--time-limit', type=float, default=300.0,
                       help='Time limit per strategy in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Parse strategies
    if args.strategies.lower() == 'all':
        strategies = DecompositionFactory.list_strategies()
    else:
        strategies = [s.strip() for s in args.strategies.split(',')]
    
    # Validate strategies
    available = DecompositionFactory.list_strategies()
    invalid = [s for s in strategies if s not in available]
    if invalid:
        print(f"❌ Invalid strategies: {', '.join(invalid)}")
        print(f"   Available: {', '.join(available)}")
        sys.exit(1)
    
    # Get D-Wave token
    dwave_token = args.token or os.getenv('DWAVE_API_TOKEN')
    
    # Create output directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE DECOMPOSITION STRATEGY BENCHMARK")
    print("="*80)
    print(f"Configuration: {args.config} farm units")
    print(f"Strategies: {len(strategies)}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Time Limit: {args.time_limit}s per strategy")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Run benchmarks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        results = run_all_strategies(
            n_units=args.config,
            dwave_token=dwave_token,
            total_land=100.0,
            strategies=strategies,
            max_iterations=args.max_iterations,
            time_limit=args.time_limit
        )
        
        # Add metadata
        benchmark_results = {
            'metadata': {
                'timestamp': timestamp,
                'config': args.config,
                'strategies_tested': strategies,
                'max_iterations': args.max_iterations,
                'time_limit': args.time_limit
            },
            'results': results
        }
        
        # Save results
        output_file = os.path.join(
            output_dir,
            f'all_strategies_config_{args.config}_{timestamp}.json'
        )
        
        # Check for tuple keys before saving
        tuple_key_paths = find_tuple_keys(benchmark_results)
        if tuple_key_paths:
            print(f"\n⚠️  Found {len(tuple_key_paths)} tuple keys in results:")
            for path in tuple_key_paths[:10]:  # Show first 10
                print(f"   • {path}")
        
        # Always sanitize to handle both tuple keys and numpy types
        print("\nSanitizing results for JSON compatibility...")
        benchmark_results = sanitize_dict_for_json(benchmark_results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        
        # Print comparison
        print_comparison_table(results)
        
        print(f"\n✅ Benchmark complete!")
        print(f"   Results saved: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
