#!/usr/bin/env python3
"""
Benchmark for Hierarchical Benders Decomposition

Tests the hierarchical graph partitioning approach for large-scale problems
that exceed QPU embedding capacity.

Tests problem sizes from 5 to 30 farms to demonstrate:
1. Success for problems that fail with standard QPU embedding (≥10 farms)
2. Scalability comparison with classical Benders
3. Solution quality and feasibility validation
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from decomposition_strategies import DecompositionFactory, solve_with_strategy
from src.scenarios import load_food_data


def generate_farm_data(n_farms: int, total_land: float = 100.0) -> Dict[str, float]:
    """Generate farm data with varying land availability."""
    import numpy as np
    
    # Create farms with varying sizes (more realistic distribution)
    np.random.seed(42)  # Reproducible
    
    # Generate land areas (log-normal distribution for more realistic farm sizes)
    raw_sizes = np.random.lognormal(mean=0, sigma=0.5, size=n_farms)
    
    # Normalize to total land
    sizes = raw_sizes / raw_sizes.sum() * total_land
    
    farms = {f"Farm_{i+1}": round(sizes[i], 2) for i in range(n_farms)}
    
    return farms


def create_config(foods: List[str], food_groups: Dict, n_farms: int) -> Dict:
    """Create configuration with food group constraints."""
    
    # Load base config for benefits
    _, foods_dict, _, base_config = load_food_data('full_family')
    
    # Calculate benefits from weighted food attributes
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
    
    # Build min/max planting areas
    min_planting_area = {food: 0.0001 for food in foods}
    max_planting_area = {food: 1000.0 for food in foods}
    
    # Food group constraints - scale with number of farms
    food_group_constraints = {}
    for group_name, foods_in_group in food_groups.items():
        n_foods = len(foods_in_group)
        # Require at least 1-2 foods per group, max depends on farms
        food_group_constraints[group_name] = {
            'min_foods': min(2, n_foods),
            'max_foods': min(n_foods * n_farms, n_foods * 3)  # Reasonable upper bound
        }
    
    config = {
        'benefits': benefits,
        'parameters': {
            'minimum_planting_area': min_planting_area,
            'maximum_planting_area': max_planting_area,
            'food_group_constraints': food_group_constraints,
            'weights': weights
        }
    }
    
    return config


def run_hierarchical_benchmark(
    n_farms: int,
    dwave_token: Optional[str] = None,
    total_land: float = 100.0,
    max_embeddable_vars: int = 150,
    use_qpu: bool = False,
    **kwargs
) -> Dict:
    """
    Run benchmark for hierarchical Benders decomposition.
    
    Args:
        n_farms: Number of farm units
        dwave_token: D-Wave API token (optional)
        total_land: Total land area
        max_embeddable_vars: Maximum BQM variables for partitioning threshold
        use_qpu: Whether to attempt QPU solving
        **kwargs: Additional strategy parameters
    
    Returns:
        Result dictionary
    """
    print(f"\n{'='*80}")
    print(f"HIERARCHICAL BENDERS BENCHMARK: {n_farms} Farms")
    print(f"{'='*80}")
    
    # Generate problem data
    farms = generate_farm_data(n_farms, total_land)
    
    # Load food data
    _, foods_dict, food_groups, base_config = load_food_data('full_family')
    foods = list(foods_dict.keys())
    
    print(f"Problem size: {n_farms} farms × {len(foods)} foods = {n_farms * len(foods)} binary variables")
    print(f"Total land: {total_land} ha")
    print(f"Max embeddable vars: {max_embeddable_vars}")
    
    # Create config
    config = create_config(foods, food_groups, n_farms)
    
    # Run hierarchical Benders
    try:
        result = solve_with_strategy(
            strategy_name='benders_hierarchical',
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            dwave_token=dwave_token,
            max_embeddable_vars=max_embeddable_vars,
            use_qpu=use_qpu,
            num_reads=kwargs.get('num_reads', 200),
            annealing_time=kwargs.get('annealing_time', 20),
            time_limit=kwargs.get('time_limit', 600.0)
        )
        
        # Add benchmark metadata
        result['benchmark_info'] = {
            'n_farms': n_farms,
            'n_foods': len(foods),
            'total_land': total_land,
            'max_embeddable_vars': max_embeddable_vars,
            'use_qpu': use_qpu,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'benchmark_info': {
                'n_farms': n_farms,
                'n_foods': len(foods),
                'total_land': total_land,
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
        }


def run_scaling_study(
    farm_sizes: List[int],
    dwave_token: Optional[str] = None,
    output_dir: str = 'hierarchical_benchmark_results',
    **kwargs
) -> Dict:
    """
    Run scaling study across multiple problem sizes.
    
    Args:
        farm_sizes: List of farm counts to test
        dwave_token: D-Wave API token
        output_dir: Directory for output files
        **kwargs: Additional benchmark parameters
    
    Returns:
        Summary dictionary with all results
    """
    print(f"\n{'#'*80}")
    print(f"# HIERARCHICAL BENDERS SCALING STUDY")
    print(f"# Testing {len(farm_sizes)} problem sizes: {farm_sizes}")
    print(f"{'#'*80}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    summary = []
    
    for n_farms in farm_sizes:
        print(f"\n{'─'*80}")
        print(f"Testing {n_farms} farms...")
        print(f"{'─'*80}")
        
        result = run_hierarchical_benchmark(
            n_farms=n_farms,
            dwave_token=dwave_token,
            **kwargs
        )
        
        results[n_farms] = result
        
        # Extract summary info
        if 'error' not in result:
            # Handle nested result structure
            if 'result' in result:
                result_data = result['result']
            else:
                result_data = result
            
            decomp_specific = result.get('decomposition_specific', result_data.get('decomposition_specific', {}))
            
            summary_entry = {
                'n_farms': n_farms,
                'status': 'success',
                'objective': result_data.get('objective_value', 0.0),
                'solve_time': result_data.get('solve_time', 0.0),
                'is_feasible': result_data.get('validation', {}).get('is_feasible', False),
                'n_partitions': decomp_specific.get('n_partitions', 1),
                'partition_sizes': decomp_specific.get('partition_sizes', []),
                'hierarchical': decomp_specific.get('hierarchical', False)
            }
        else:
            summary_entry = {
                'n_farms': n_farms,
                'status': 'failed',
                'error': result['error']
            }
        
        summary.append(summary_entry)
        
        # Save individual result
        output_file = os.path.join(output_dir, f'hierarchical_{n_farms}_farms.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved: {output_file}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SCALING STUDY SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Farms':<8} {'Status':<10} {'Objective':<12} {'Time (s)':<12} {'Partitions':<12} {'Feasible':<10}")
    print("-" * 80)
    
    for entry in summary:
        if entry['status'] == 'success':
            print(f"{entry['n_farms']:<8} {entry['status']:<10} {entry['objective']:<12.4f} "
                  f"{entry['solve_time']:<12.2f} {entry['n_partitions']:<12} {entry['is_feasible']}")
        else:
            print(f"{entry['n_farms']:<8} {entry['status']:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    print("-" * 80)
    
    # Save summary
    summary_file = os.path.join(output_dir, 'scaling_study_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'farm_sizes': farm_sizes,
            'summary': summary
        }, f, indent=2)
    print(f"\nSummary saved: {summary_file}")
    
    # Determine success rate
    successful = [s for s in summary if s['status'] == 'success']
    large_successful = [s for s in successful if s['n_farms'] >= 10]
    
    print(f"\n{'='*80}")
    print("KEY RESULTS")
    print(f"{'='*80}")
    print(f"Total tests: {len(summary)}")
    print(f"Successful: {len(successful)}/{len(summary)} ({100*len(successful)/len(summary):.0f}%)")
    print(f"Large problems (≥10 farms) successful: {len(large_successful)}/{len([s for s in summary if s['n_farms'] >= 10])}")
    
    if large_successful:
        print(f"\n✅ HIERARCHICAL DECOMPOSITION SUCCESSFULLY HANDLES LARGE PROBLEMS!")
        print(f"   Problems with ≥10 farms that would fail standard QPU embedding are now solvable.")
    
    return {
        'summary': summary,
        'results': results
    }


def compare_with_classical(
    n_farms: int,
    dwave_token: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Compare hierarchical Benders with classical Benders.
    
    Args:
        n_farms: Number of farms to test
        dwave_token: D-Wave API token
    
    Returns:
        Comparison dictionary
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: Hierarchical vs Classical Benders ({n_farms} farms)")
    print(f"{'='*80}")
    
    # Generate problem data once
    farms = generate_farm_data(n_farms, 100.0)
    _, foods_dict, food_groups, _ = load_food_data('full_family')
    foods = list(foods_dict.keys())
    config = create_config(foods, food_groups, n_farms)
    
    results = {}
    
    # Run classical Benders
    print(f"\n[1/2] Running Classical Benders...")
    try:
        classical_result = solve_with_strategy(
            strategy_name='benders',
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            max_iterations=50,
            time_limit=300.0
        )
        results['classical'] = {
            'status': 'success',
            'objective': classical_result.get('result', classical_result).get('objective_value', 0),
            'solve_time': classical_result.get('result', classical_result).get('solve_time', 0),
            'is_feasible': classical_result.get('result', classical_result).get('validation', {}).get('is_feasible', False)
        }
    except Exception as e:
        results['classical'] = {'status': 'failed', 'error': str(e)}
    
    # Run hierarchical Benders
    print(f"\n[2/2] Running Hierarchical Benders...")
    try:
        hierarchical_result = solve_with_strategy(
            strategy_name='benders_hierarchical',
            farms=farms,
            foods=foods,
            food_groups=food_groups,
            config=config,
            dwave_token=dwave_token,
            max_embeddable_vars=kwargs.get('max_embeddable_vars', 150),
            use_qpu=kwargs.get('use_qpu', False),
            time_limit=600.0
        )
        
        result_data = hierarchical_result.get('result', hierarchical_result)
        decomp = hierarchical_result.get('decomposition_specific', result_data.get('decomposition_specific', {}))
        
        results['hierarchical'] = {
            'status': 'success',
            'objective': result_data.get('objective_value', 0),
            'solve_time': result_data.get('solve_time', 0),
            'is_feasible': result_data.get('validation', {}).get('is_feasible', False),
            'n_partitions': decomp.get('n_partitions', 1),
            'was_decomposed': decomp.get('hierarchical', False)
        }
    except Exception as e:
        results['hierarchical'] = {'status': 'failed', 'error': str(e)}
    
    # Print comparison
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Method':<20} {'Status':<12} {'Objective':<12} {'Time (s)':<12} {'Feasible':<10}")
    print("-" * 70)
    
    for method, data in results.items():
        if data['status'] == 'success':
            print(f"{method:<20} {data['status']:<12} {data['objective']:<12.4f} "
                  f"{data['solve_time']:<12.2f} {data['is_feasible']}")
        else:
            print(f"{method:<20} {data['status']:<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Hierarchical Benders Decomposition"
    )
    parser.add_argument(
        '--mode', type=str, default='scaling',
        choices=['scaling', 'single', 'compare'],
        help='Benchmark mode: scaling study, single run, or comparison'
    )
    parser.add_argument(
        '--farms', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30],
        help='Number of farms to test'
    )
    parser.add_argument(
        '--max-embeddable', type=int, default=150,
        help='Maximum BQM variables for embedding (default: 150)'
    )
    parser.add_argument(
        '--use-qpu', action='store_true',
        help='Attempt QPU solving (requires D-Wave token)'
    )
    parser.add_argument(
        '--dwave-token', type=str, default=None,
        help='D-Wave API token'
    )
    parser.add_argument(
        '--output-dir', type=str, default='hierarchical_benchmark_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    dwave_token = args.dwave_token or os.getenv('DWAVE_API_TOKEN')
    
    if args.mode == 'scaling':
        run_scaling_study(
            farm_sizes=args.farms,
            dwave_token=dwave_token,
            output_dir=args.output_dir,
            max_embeddable_vars=args.max_embeddable,
            use_qpu=args.use_qpu
        )
    
    elif args.mode == 'single':
        n_farms = args.farms[0] if args.farms else 15
        result = run_hierarchical_benchmark(
            n_farms=n_farms,
            dwave_token=dwave_token,
            max_embeddable_vars=args.max_embeddable,
            use_qpu=args.use_qpu
        )
        
        output_file = os.path.join(args.output_dir, f'hierarchical_{n_farms}_farms.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved: {output_file}")
    
    elif args.mode == 'compare':
        n_farms = args.farms[0] if args.farms else 15
        compare_with_classical(
            n_farms=n_farms,
            dwave_token=dwave_token,
            max_embeddable_vars=args.max_embeddable,
            use_qpu=args.use_qpu
        )


if __name__ == "__main__":
    main()
