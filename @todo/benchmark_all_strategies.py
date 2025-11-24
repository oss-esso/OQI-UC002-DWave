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
    """Recalculate objective for decomposition result (already in standard format)."""
    
    # The result from format_decomposition_result has NESTED structure: result['result'][...]
    # We need to extract from the right level
    if 'result' in result:
        # Nested structure from format_decomposition_result
        result_data = result['result']
        objective_value = result_data.get('objective_value', 0.0)
        solution_areas = result_data.get('solution_areas', {})
        solution_selections = result_data.get('solution_selections', {})
        print(f"  DEBUG (convert): Extracted from nested result['result'], got {len(solution_areas)} solution_areas")
    else:
        # Flat structure (shouldn't happen but handle it)
        objective_value = result.get('objective_value', 0.0)
        solution_areas = result.get('solution_areas', {})
        solution_selections = result.get('solution_selections', {})
        print(f"  DEBUG (convert): Extracted from flat result, got {len(solution_areas)} solution_areas")
    
    # Build A_dict and Y_dict from the standardized format
    A_dict = {}
    Y_dict = {}
    
    for farm in farms:
        for food in foods:
            key = f"{farm}_{food}"
            A_dict[(farm, food)] = solution_areas.get(key, 0.0)
            Y_dict[(farm, food)] = 1.0 if solution_selections.get(key, 0.0) > 0.5 else 0.0
    
    total_from_solution = sum(A_dict.values())
    print(f"  DEBUG: Solution has {sum(1 for v in A_dict.values() if v > 0.01)} nonzero allocations, total {total_from_solution:.2f} ha")
    
    # RECALCULATE OBJECTIVE FROM ACTUAL SOLUTION
    total_area = sum(farms.values())
    
    # Get benefits using correct formula
    _, foods_dict, _, base_config = load_food_data('full_family')
    weights = base_config.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15
    })
    
    benefits = {}
    for food in foods:
        food_attrs = foods_dict.get(food, {})
        benefit = (
            food_attrs.get('nutritional_value', 0) * weights.get('nutritional_value', 0) +
            food_attrs.get('nutrient_density', 0) * weights.get('nutrient_density', 0) -
            food_attrs.get('environmental_impact', 0) * weights.get('environmental_impact', 0) +
            food_attrs.get('affordability', 0) * weights.get('affordability', 0) +
            food_attrs.get('sustainability', 0) * weights.get('sustainability', 0)
        )
        benefits[food] = benefit if benefit != 0 else 0.5
    
    # Calculate actual objective from solution
    actual_objective = 0.0
    for (farm, food), area in A_dict.items():
        y_val = Y_dict.get((farm, food), 0.0)
        if y_val > 0.5:  # Food is selected
            actual_objective += area * benefits.get(food, 0.5)
    
    # Normalize by total area
    actual_objective = actual_objective / total_area if total_area > 0 else 0.0
    
    # Warn if mismatch
    if abs(actual_objective - objective_value) > 0.01:
        print(f"  ⚠️  Objective mismatch: reported={objective_value:.6f}, recalculated={actual_objective:.6f}")
    
    # UPDATE objective in the NESTED result structure
    if 'result' in result:
        result['result']['objective_value'] = actual_objective
        # Add decomposition info
        if 'decomposition_info' not in result['result']:
            result['result']['decomposition_info'] = {}
        result['result']['decomposition_info']['original_objective'] = objective_value
        result['result']['decomposition_info']['recalculated_objective'] = actual_objective
    else:
        result['objective_value'] = actual_objective
        if 'decomposition_info' not in result:
            result['decomposition_info'] = {}
        result['decomposition_info']['original_objective'] = objective_value
        result['decomposition_info']['recalculated_objective'] = actual_objective
    
    # FLATTEN the result to match PuLP format (move result['result'] to top level)
    if 'result' in result:
        flattened = result['result'].copy()
        flattened['metadata'] = result.get('metadata', {})
        if 'decomposition_specific' in result:
            flattened['decomposition_specific'] = result['decomposition_specific']
        return flattened
    
    return result


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
    _, foods_dict, _, base_config = load_food_data('full_family')
    
    # Add benefits to config
    config_dict['benefits'] = {}
    for food in foods:
        # Use composite benefit score (weighted attributes)
        weights = config_dict.get('parameters', {}).get('weights', {})
        # Use foods_dict (2nd return value) which contains the actual food attributes
        food_attrs = foods_dict.get(food, {})
        # Match PuLP formula: + nutrition + nutrient_density - environmental_impact + affordability + sustainability
        benefit = (
            food_attrs.get('nutritional_value', 0) * weights.get('nutritional_value', 0) +
            food_attrs.get('nutrient_density', 0) * weights.get('nutrient_density', 0) -
            food_attrs.get('environmental_impact', 0) * weights.get('environmental_impact', 0) +
            food_attrs.get('affordability', 0) * weights.get('affordability', 0) +
            food_attrs.get('sustainability', 0) * weights.get('sustainability', 0)
        )
        config_dict['benefits'][food] = benefit if benefit != 0 else 0.5  # Default benefit if all zero
    
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
    """Print comparison table of all strategies with validation info."""
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Strategy':<25} {'Status':<12} {'Time (s)':<10} {'Objective':<12} {'Valid':<8} {'Area Used':<15}")
    print("-" * 105)
    
    for strategy_name, result in results.items():
        # Results are now in standardized format from format_standard_result
        status = result.get('status', 'Unknown')
        solve_time = result.get('solve_time', 0.0)
        objective = result.get('objective_value', 0.0)
        
        # Get validation info
        validation = result.get('validation', {})
        is_valid = validation.get('is_feasible', True)
        n_violations = validation.get('n_violations', 0)
        
        # Check total area used
        total_area = result.get('total_area', 100.0)
        total_covered = result.get('total_covered_area', 0.0)
        area_str = f"{total_covered:.1f}/{total_area:.1f}"
        
        # Mark invalid if area overflow
        if total_covered > total_area + 0.01:
            is_valid = False
            area_str += " ⚠️"
        
        valid_str = '✅' if is_valid else '❌'
        
        print(f"{strategy_name:<25} {status:<12} {solve_time:<10.3f} {objective:<12.4f} {valid_str:<8} {area_str:<15}")
    
    print("-" * 105)
    
    # Print detailed violation info for any invalid solutions
    print("\n" + "="*80)
    print("VALIDATION DETAILS")
    print("="*80)
    
    has_issues = False
    for strategy_name, result in results.items():
        issues = []
        
        # Check area overflow
        total_area = result.get('total_area', 100.0)
        total_covered = result.get('total_covered_area', 0.0)
        if total_covered > total_area + 0.01:
            overflow = total_covered - total_area
            overflow_pct = 100.0 * overflow / total_area
            issues.append(f"AREA OVERFLOW: {overflow:.2f} ha ({overflow_pct:.1f}% over limit)")
            issues.append("  -> This violates global land availability constraint!")
            issues.append(f"  -> Objective {result.get('objective_value', 0):.4f} is INVALID (exceeds physical constraints)")
        
        # Check constraint violations from validation
        validation = result.get('validation', {})
        violations = validation.get('violations', [])
        
        if violations:
            # Group violations by type
            violation_types = {}
            for v in violations:
                v_type = v.get('type', 'unknown')
                violation_types.setdefault(v_type, []).append(v)
            
            for v_type, v_list in violation_types.items():
                issues.append(f"{v_type}: {len(v_list)} violations")
        
        if issues:
            has_issues = True
            print(f"\n❌ {strategy_name.upper()}:")
            for issue in issues:
                print(f"  {issue}")
    
    if not has_issues:
        print("\n✅ All strategies produced valid solutions")
        print("   - No constraint violations")
        print("   - No area overflow")
    
    print("="*80)


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
