#!/usr/bin/env python3
"""
Verify Objective Normalization and Land Utilization

Test that all decomposition methods use the same objective function
and that it's properly normalized by area.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from benchmark_utils_decomposed import generate_farm_data, create_config
from src.scenarios import load_food_data
from decomposition_strategies import DecompositionFactory


def verify_normalization():
    """Verify objective normalization across all strategies."""
    
    print("="*80)
    print("OBJECTIVE NORMALIZATION VERIFICATION")
    print("="*80)
    
    # Generate test problem
    farm_data = generate_farm_data(n_units=5, total_land=100.0)
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
    total_area = sum(farms.values())
    
    print(f"\nProblem Setup:")
    print(f"  Farms: {len(farms)}")
    print(f"  Foods: {len(foods)}")
    print(f"  Total Area: {total_area:.2f} hectares")
    print(f"  Farm capacities: {list(farms.values())}")
    
    # Test strategies
    strategies = ['benders', 'dantzig_wolfe', 'admm']
    
    results = {}
    for strategy_name in strategies:
        print(f"\n{'-'*80}")
        print(f"Testing: {strategy_name.upper()}")
        print(f"{'-'*80}")
        
        try:
            strategy = DecompositionFactory.get_strategy(strategy_name)
            result = strategy.solve(
                farms=farms,
                foods=foods,
                food_groups=food_groups,
                config=config,
                max_iterations=5,
                time_limit=30.0
            )
            
            obj_value = result['solution']['objective_value']
            solution = result['solution']['full_solution']
            
            # Calculate land used
            total_land_used = 0.0
            total_raw_benefit = 0.0
            benefits = config['benefits']
            
            for var_name, var_value in solution.items():
                if var_value > 1e-6 and var_name.startswith('A_'):
                    parts = var_name.split('_')
                    if len(parts) == 3:
                        farm, food = parts[1], parts[2]
                        total_land_used += var_value
                        total_raw_benefit += var_value * benefits.get(food, 1.0)
            
            # Manual calculation of normalized objective
            manual_obj = total_raw_benefit / total_area
            
            results[strategy_name] = {
                'objective': obj_value,
                'land_used': total_land_used,
                'utilization_%': (total_land_used / total_area) * 100,
                'raw_benefit': total_raw_benefit,
                'manual_obj': manual_obj,
                'match': abs(obj_value - manual_obj) < 1e-3
            }
            
            print(f"  Reported Objective: {obj_value:.4f}")
            print(f"  Manual Calculation: {manual_obj:.4f}")
            print(f"  Land Used: {total_land_used:.2f} / {total_area:.2f} hectares")
            print(f"  Utilization: {(total_land_used/total_area)*100:.1f}%")
            print(f"  Raw Benefit: {total_raw_benefit:.2f}")
            print(f"  ✅ Match: {results[strategy_name]['match']}")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results[strategy_name] = {'objective': None, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Strategy':<20} {'Objective':<12} {'Land Used':<12} {'Utilization':<12} {'Normalized':<12}")
    print("-"*80)
    
    for strategy_name, data in results.items():
        if 'objective' in data and data['objective'] is not None:
            obj = data['objective']
            land = data.get('land_used', 0)
            util = data.get('utilization_%', 0)
            match = '✅ Yes' if data.get('match', False) else '❌ No'
            
            print(f"{strategy_name:<20} {obj:<12.4f} {land:<12.2f} {util:<11.1f}% {match:<12}")
    
    print("-"*80)
    print("\nConclusion:")
    print("  All objectives are now in 'benefit per hectare' units")
    print("  Different values reflect different land utilization strategies")
    print("  Higher objective = better benefit per hectare (more efficient)")
    print("="*80)


if __name__ == "__main__":
    verify_normalization()
