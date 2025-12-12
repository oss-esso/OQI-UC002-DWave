#!/usr/bin/env python3
"""
Quick Test: Run hierarchical solver on a scenario

Usage:
    python quick_test.py                          # Small test with SA
    python quick_test.py --medium                 # Medium test with SA  
    python quick_test.py --qpu                    # Small test with QPU (uses access!)
    python quick_test.py --medium --qpu           # Medium test with QPU

Author: OQI-UC002-DWave
"""

import sys
import os
import argparse
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from hierarchical_quantum_solver import solve_hierarchical, DEFAULT_CONFIG, OUTPUT_DIR
from src.scenarios import load_food_data
import json
from datetime import datetime

def run_test(scenario: str, use_qpu: bool = False, farms_limit: int = None):
    """Run test on specified scenario."""
    
    print("="*80)
    print(f"HIERARCHICAL SOLVER TEST: {scenario}")
    print("="*80)
    print(f"Mode: {'⚡ QPU' if use_qpu else '🧪 SimulatedAnnealing (no QPU access used)'}")
    if farms_limit:
        print(f"Farms limit: {farms_limit}")
    print("="*80)
    
    # Load scenario
    print(f"\nLoading {scenario}...")
    farms, foods, food_groups, config = load_food_data(scenario)
    
    params = config.get('parameters', {})
    weights = params.get('weights', {})
    la = params.get('land_availability', {})
    
    # Optionally limit farms
    if farms_limit:
        farm_names = list(la.keys())[:farms_limit]
        la = {f: la[f] for f in farm_names}
    else:
        farm_names = list(la.keys())
    
    # Build data dict
    food_names = list(foods.keys())
    food_benefits = {}
    for food in food_names:
        food_data = foods.get(food, {})
        benefit = sum(food_data.get(attr, 0.5) * w for attr, w in weights.items())
        food_benefits[food] = benefit
    
    data = {
        'foods': foods,
        'food_names': food_names,
        'food_groups': food_groups,
        'food_benefits': food_benefits,
        'weights': weights,
        'land_availability': la,
        'farm_names': farm_names,
        'total_area': sum(la.values()),
        'n_farms': len(farm_names),
        'n_foods': len(food_names),
        'config': config,
    }
    
    print(f"  Farms: {data['n_farms']}")
    print(f"  Foods: {data['n_foods']}")
    print(f"  Variables: {data['n_farms'] * data['n_foods'] * 3:,}")
    
    # Configure solver
    solver_config = DEFAULT_CONFIG.copy()
    
    if use_qpu:
        # QPU settings (production)
        solver_config['farms_per_cluster'] = 10
        solver_config['num_iterations'] = 3
        solver_config['num_reads'] = 100
    else:
        # SA settings (testing - faster)
        solver_config['farms_per_cluster'] = 10
        solver_config['num_iterations'] = 2
        solver_config['num_reads'] = 20  # Fast SA
    
    print(f"\nSolver config:")
    print(f"  Farms per cluster: {solver_config['farms_per_cluster']}")
    print(f"  Iterations: {solver_config['num_iterations']}")
    print(f"  Reads: {solver_config['num_reads']}")
    
    # Solve
    print("\n" + "="*80)
    result = solve_hierarchical(data, solver_config, use_qpu=use_qpu, verbose=True)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "qpu" if use_qpu else "sa"
    filename = f"test_{scenario}_{mode}_{data['n_farms']}farms_{timestamp}.json"
    output_file = OUTPUT_DIR / filename
    
    # Convert for JSON
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(x) for x in obj]
        elif hasattr(obj, 'tolist'):  # numpy
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_for_json(result), f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Quick test of hierarchical solver')
    parser.add_argument('--medium', action='store_true',
                       help='Run medium test (50 farms from 250-farm scenario)')
    parser.add_argument('--large', action='store_true',
                       help='Run large test (250 farms, 27 foods)')
    parser.add_argument('--qpu', action='store_true',
                       help='Use real QPU (default: SimulatedAnnealing)')
    
    args = parser.parse_args()
    
    if args.large:
        scenario = 'rotation_250farms_27foods'
        farms_limit = None  # All 250 farms
    elif args.medium:
        scenario = 'rotation_250farms_27foods'
        farms_limit = 50  # Subset of 50 farms
    else:
        scenario = 'rotation_small_50'
        farms_limit = None  # All 10 farms
    
    if args.qpu:
        response = input("\n⚠️  This will use QPU access. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
    
    result = run_test(scenario, use_qpu=args.qpu, farms_limit=farms_limit)
    
    if result['success']:
        print("\n✅ Test completed successfully!")
        
        # Quick summary
        print("\n📊 Quick Summary:")
        print(f"   Objective: {result['objective']:.4f}")
        print(f"   Violations: {result['violations']}")
        print(f"   Unique crops: {result['diversity_stats']['total_unique_crops']}/{result['diversity_stats']['max_possible_crops']}")
        print(f"   Shannon diversity: {result['diversity_stats']['shannon_diversity']:.3f}")
        print(f"   Total time: {result['timings']['total']:.2f}s")
        
        if args.qpu:
            print(f"   QPU time: {result['timings']['level2_qpu']:.3f}s")
    else:
        print("\n❌ Test failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
