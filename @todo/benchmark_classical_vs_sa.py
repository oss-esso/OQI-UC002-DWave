#!/usr/bin/env python3
"""
Comprehensive Decomposition Benchmark: Classical vs Simulated Annealing

Benchmarks all decomposition strategies in two modes:
1. Classical: Pure Gurobi solvers
2. Simulated Annealing: Using Neal SA as QPU simulator

This allows performance comparison without actual QPU hardware.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from decomposition_strategies import DecompositionFactory, DecompositionStrategy
from benchmark_utils_decomposed import generate_farm_data, create_config
from src.scenarios import load_food_data


def run_strategy_comparison(
    n_units: int = 10,
    strategies=None,
    max_iterations: int = 10,
    time_limit: float = 60.0,
    output_dir: str = None
):
    """
    Run comprehensive comparison of all strategies in classical and SA modes.
    
    Args:
        n_units: Number of farm units
        strategies: List of strategy names (None = all)
        max_iterations: Maximum iterations per strategy
        time_limit: Time limit per strategy in seconds
        output_dir: Output directory for results
    """
    if strategies is None:
        strategies = [
            'benders',
            'benders_qpu',
            'dantzig_wolfe',
            'dantzig_wolfe_qpu',
            'admm'
        ]
    
    if output_dir is None:
        output_dir = os.path.join('Benchmarks', 'DECOMPOSITION_COMPARISON')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE DECOMPOSITION STRATEGY BENCHMARK")
    print("Classical vs Simulated Annealing Comparison")
    print("="*80)
    print(f"Configuration: {n_units} farm units")
    print(f"Strategies: {len(strategies)}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Time Limit: {time_limit}s per run")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Generate problem
    print("Generating test problem...")
    farm_data = generate_farm_data(n_units=n_units, total_land=100.0)
    foods, food_groups, config = create_config(farm_data['land_data'])
    
    # Load benefits
    _, _, _, base_config = load_food_data('full_family')
    config['benefits'] = {}
    for food in foods:
        weights = config.get('parameters', {}).get('weights', {})
        benefit = sum(
            base_config.get('nutrients', {}).get(food, {}).get(attr, 0) * weight
            for attr, weight in weights.items()
        )
        config['benefits'][food] = benefit if benefit > 0 else 100.0
    
    config['food_groups'] = food_groups
    config['foods'] = foods
    farms = farm_data['land_data']
    
    print(f"  ✅ Problem: {len(farms)} farms, {len(foods)} foods\n")
    
    # Results storage
    all_results = {
        'metadata': {
            'n_units': n_units,
            'n_foods': len(foods),
            'total_area': sum(farms.values()),
            'timestamp': datetime.now().isoformat(),
            'max_iterations': max_iterations,
            'time_limit': time_limit
        },
        'strategies': {}
    }
    
    # Run each strategy in both modes
    for strategy_name in strategies:
        print(f"\n{'='*80}")
        print(f"STRATEGY: {strategy_name.upper()}")
        print(f"{'='*80}\n")
        
        strategy_results = {
            'name': strategy_name,
            'modes': {}
        }
        
        # Mode 1: Classical
        print(f"Mode 1: CLASSICAL")
        print(f"{'-'*80}")
        
        try:
            strategy = DecompositionFactory.get_strategy(strategy_name)
            
            start_time = time.time()
            result_classical = strategy.solve(
                farms=farms,
                foods=foods,
                food_groups=food_groups,
                config=config,
                max_iterations=max_iterations,
                time_limit=time_limit,
                dwave_token=None,  # No token = classical
                use_qpu_for_master=False,
                use_qpu_for_pricing=False
            )
            classical_time = time.time() - start_time
            
            strategy_results['modes']['classical'] = {
                'status': result_classical['solver_info']['status'],
                'objective': result_classical['solution']['objective_value'],
                'time': result_classical['solver_info']['solve_time'],
                'iterations': result_classical['solver_info'].get('num_iterations', 0),
                'feasible': result_classical['solution']['is_feasible'],
                'success': True
            }
            
            print(f"  ✅ Status: {result_classical['solver_info']['status']}")
            print(f"  Objective: {result_classical['solution']['objective_value']:.4f}")
            print(f"  Time: {result_classical['solver_info']['solve_time']:.3f}s")
            print(f"  Iterations: {result_classical['solver_info'].get('num_iterations', 0)}")
            print(f"  Feasible: {result_classical['solution']['is_feasible']}\n")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}\n")
            strategy_results['modes']['classical'] = {
                'status': 'Failed',
                'error': str(e),
                'success': False
            }
        
        # Mode 2: Simulated Annealing (if QPU variant)
        if '_qpu' in strategy_name:
            print(f"Mode 2: SIMULATED ANNEALING (QPU Simulator)")
            print(f"{'-'*80}")
            
            try:
                strategy = DecompositionFactory.get_strategy(strategy_name)
                
                start_time = time.time()
                result_sa = strategy.solve(
                    farms=farms,
                    foods=foods,
                    food_groups=food_groups,
                    config=config,
                    max_iterations=max_iterations,
                    time_limit=time_limit,
                    dwave_token='SIMULATED_ANNEALING',  # Trigger SA mode
                    use_qpu_for_master='benders' in strategy_name,
                    use_qpu_for_pricing='dantzig' in strategy_name
                )
                sa_time = time.time() - start_time
                
                strategy_results['modes']['simulated_annealing'] = {
                    'status': result_sa['solver_info']['status'],
                    'objective': result_sa['solution']['objective_value'],
                    'time': result_sa['solver_info']['solve_time'],
                    'iterations': result_sa['solver_info'].get('num_iterations', 0),
                    'feasible': result_sa['solution']['is_feasible'],
                    'success': True
                }
                
                print(f"  ✅ Status: {result_sa['solver_info']['status']}")
                print(f"  Objective: {result_sa['solution']['objective_value']:.4f}")
                print(f"  Time: {result_sa['solver_info']['solve_time']:.3f}s")
                print(f"  Iterations: {result_sa['solver_info'].get('num_iterations', 0)}")
                print(f"  Feasible: {result_sa['solution']['is_feasible']}\n")
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}\n")
                strategy_results['modes']['simulated_annealing'] = {
                    'status': 'Failed',
                    'error': str(e),
                    'success': False
                }
        
        all_results['strategies'][strategy_name] = strategy_results
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    print(f"{'Strategy':<25} {'Mode':<20} {'Status':<12} {'Objective':<12} {'Time (s)':<10} {'Iters':<6}")
    print("-"*90)
    
    for strategy_name, strategy_data in all_results['strategies'].items():
        for mode_name, mode_data in strategy_data['modes'].items():
            if mode_data.get('success', False):
                print(f"{strategy_name:<25} {mode_name:<20} "
                      f"{mode_data['status']:<12} "
                      f"{mode_data['objective']:<12.4f} "
                      f"{mode_data['time']:<10.3f} "
                      f"{mode_data.get('iterations', 0):<6}")
            else:
                print(f"{strategy_name:<25} {mode_name:<20} "
                      f"{'Failed':<12} {'-':<12} {'-':<10} {'-':<6}")
    
    print("-"*90)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"comparison_config_{n_units}_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved: {output_file}")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark decomposition strategies (Classical vs SA)")
    parser.add_argument('--config', type=int, default=10, help='Number of farm units')
    parser.add_argument('--strategies', type=str, default='all',
                       help='Comma-separated list of strategies or "all"')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum iterations per strategy')
    parser.add_argument('--time-limit', type=float, default=60.0,
                       help='Time limit per strategy in seconds')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse strategies
    if args.strategies == 'all':
        strategies = [
            'benders',
            'benders_qpu',
            'dantzig_wolfe',
            'dantzig_wolfe_qpu',
            'admm'
        ]
    else:
        strategies = [s.strip() for s in args.strategies.split(',')]
    
    # Run benchmark
    run_strategy_comparison(
        n_units=args.config,
        strategies=strategies,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        output_dir=args.output_dir
    )
