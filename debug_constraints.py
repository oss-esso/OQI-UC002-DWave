#!/usr/bin/env python3
"""
Debug tool to manually validate DWave solution against all constraints.

This script loads the JSON solution and manually checks each constraint
to identify exactly why the solution is marked as infeasible.
"""

import json
import sys
import os
import math

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.scenarios import load_food_data

def load_solution(json_path):
    """Load solution from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def debug_constraint_validation(solution_data):
    """
    Debug the constraint validation for the DWave solution.
    """
    print("="*80)
    print("DWAVE SOLUTION CONSTRAINT DEBUG ANALYSIS")
    print("="*80)
    
    # Extract solution components
    solution = solution_data['solution_areas']
    land_data = solution_data['land_data']
    n_units = solution_data['n_units']
    total_area = solution_data['total_area']
    
    # Extract ACTUAL foods from the solution (not from scenario file)
    a_vars = {k: v for k, v in solution.items() if k.startswith('A_')}
    y_vars = {k: v for k, v in solution.items() if k.startswith('Y_')}
    
    # Get unique foods from variable names
    foods = set()
    for var_name in a_vars.keys():
        # Format: A_Farm1_FoodName
        parts = var_name.split('_', 2)
        if len(parts) == 3:
            foods.add(parts[2])
    foods = sorted(list(foods))
    
    # Convert to farms list
    farms = [f"Farm{i}" for i in range(1, n_units + 1)]
    
    print(f"Problem size: {n_units} farms, {len(foods)} foods")
    print(f"Total available area: {total_area:.6f}")
    print(f"Solution total covered area: {solution_data.get('total_covered_area', 'N/A')}")
    print()
    
    # Print solution structure preview
    print("Solution structure:")
    print(f"  A variables (areas): {len(a_vars)}")
    print(f"  Y variables (binary): {len(y_vars)}")
    print(f"  Foods detected: {len(foods)}")
    print()
    
    # Create land_availability from land_data
    land_availability = land_data
    
    # Load food data only for min_planting_area and food_group_constraints
    # Try to match scenario based on number of foods
    n_foods_in_solution = len(foods)
    if n_foods_in_solution <= 6:
        scenario = 'simple'
    elif n_foods_in_solution <= 15:
        scenario = 'intermediate'
    else:
        scenario = 'custom'  # 27 foods
    
    print(f"Loading scenario config: {scenario}")
    farms_template, foods_template, food_groups, config = load_food_data(scenario)
    print()
    
    violations = []
    constraint_checks = {
        'land_availability': {'passed': 0, 'failed': 0, 'violations': []},
        'linking_constraints': {'passed': 0, 'failed': 0, 'violations': []},
        'food_group': {'passed': 0, 'failed': 0, 'violations': []}
    }
    
    tolerance = 0.001  # Use same tolerance as actual validation
    
    # Get configuration parameters from loaded scenario
    min_planting_area = config['parameters'].get('minimum_planting_area', {})
    food_group_constraints = config['parameters'].get('food_group_constraints', {})
    
    print("CONSTRAINT 1: Land availability per farm")
    print("-" * 50)
    
    # Check land availability per farm
    for farm in farms:
        farm_total = sum(a_vars.get(f"A_{farm}_{food}", 0) for food in foods)
        farm_capacity = land_availability[farm]
        
        if farm_total > farm_capacity + 0.01:
            violation = f"{farm}: {farm_total:.4f} ha allocated > {farm_capacity:.4f} ha capacity"
            violations.append(violation)
            constraint_checks['land_availability']['violations'].append(violation)
            constraint_checks['land_availability']['failed'] += 1
            print(f"  ❌ {violation}")
        else:
            constraint_checks['land_availability']['passed'] += 1
            if farm_total > 0.001:
                print(f"  ✅ {farm}: {farm_total:.4f} ≤ {farm_capacity:.4f} ha")
    
    print()
    
    print("CONSTRAINT 2: Linking constraints (A and Y must be consistent)")
    print("-" * 50)
    print("  Rule: If Y=1, then min_area <= A <= farm_capacity")
    print("  Rule: If Y=0, then A must be 0")
    print()
    
    linking_violations_found = 0
    
    # Check linking constraints: A and Y must be consistent
    for farm in farms:
        farm_capacity = land_availability[farm]
        for food in foods:
            a_val = a_vars.get(f"A_{farm}_{food}", 0)
            y_val = y_vars.get(f"Y_{farm}_{food}", 0)
            min_area = min_planting_area.get(food, 0)
            
            # If Y=1 (selected), check A is within bounds
            if y_val > 0.5:  # Selected
                if a_val < min_area - tolerance:
                    violation = f"A_{farm}_{food}={a_val:.4f} < min_area={min_area:.4f} (Y=1)"
                    violations.append(violation)
                    constraint_checks['linking_constraints']['violations'].append(violation)
                    constraint_checks['linking_constraints']['failed'] += 1
                    linking_violations_found += 1
                    print(f"  ❌ {violation}")
                elif a_val > farm_capacity + tolerance:
                    violation = f"A_{farm}_{food}={a_val:.4f} > farm_capacity={farm_capacity:.4f} (Y=1)"
                    violations.append(violation)
                    constraint_checks['linking_constraints']['violations'].append(violation)
                    constraint_checks['linking_constraints']['failed'] += 1
                    linking_violations_found += 1
                    print(f"  ❌ {violation}")
                else:
                    constraint_checks['linking_constraints']['passed'] += 1
            else:  # Y=0 (not selected) - A MUST be 0
                if a_val > tolerance:
                    violation = f"A_{farm}_{food}={a_val:.4f} but Y_{farm}_{food}={y_val:.4f} (should be 0)"
                    violations.append(violation)
                    constraint_checks['linking_constraints']['violations'].append(violation)
                    constraint_checks['linking_constraints']['failed'] += 1
                    linking_violations_found += 1
                    print(f"  ❌ {violation}")
                else:
                    constraint_checks['linking_constraints']['passed'] += 1
    
    if linking_violations_found == 0:
        print("  ✅ All linking constraints satisfied")
    
    print()
    
    print("CONSTRAINT 3: Food group constraints (global)")
    print("-" * 50)
    
    # Check food group constraints
    if food_group_constraints:
        for group_name, group_data in food_group_constraints.items():
            foods_in_group = food_groups.get(group_name, [])
            if foods_in_group:
                # Count how many crops from this group are selected (Y=1) across ALL farms
                selected_crops = set()
                for crop in foods_in_group:
                    for farm in farms:
                        y_val = y_vars.get(f"Y_{farm}_{crop}", 0)
                        if y_val > 0.5:  # Crop is selected on this farm
                            selected_crops.add(crop)
                            break  # Only need to find it selected once
                
                n_selected = len(selected_crops)
                min_foods = group_data.get('min_foods', 0)
                max_foods = group_data.get('max_foods', len(foods_in_group))
                
                if n_selected < min_foods:
                    violation = f"Group {group_name}: {n_selected} crops < min_foods={min_foods}"
                    violations.append(violation)
                    constraint_checks['food_group']['violations'].append(violation)
                    constraint_checks['food_group']['failed'] += 1
                    print(f"  ❌ {violation}")
                elif n_selected > max_foods:
                    violation = f"Group {group_name}: {n_selected} crops > max_foods={max_foods}"
                    violations.append(violation)
                    constraint_checks['food_group']['violations'].append(violation)
                    constraint_checks['food_group']['failed'] += 1
                    print(f"  ❌ {violation}")
                else:
                    constraint_checks['food_group']['passed'] += 1
                    print(f"  ✅ {group_name}: {n_selected} crops in [{min_foods}, {max_foods}]")
    else:
        print("  No food group constraints defined")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    total_violations = len(violations)
    total_checks = sum(check['passed'] + check['failed'] for check in constraint_checks.values())
    
    if total_violations == 0:
        print("🎉 SOLUTION IS FEASIBLE - All constraints satisfied!")
    else:
        print(f"❌ SOLUTION IS INFEASIBLE - {total_violations} constraint violation(s) found:")
        for i, violation in enumerate(violations, 1):
            print(f"  {i}. {violation}")
    
    print()
    print("Constraint breakdown:")
    for name, check in constraint_checks.items():
        total = check['passed'] + check['failed']
        if total > 0:
            print(f"  {name}: {check['passed']}/{total} passed ({check['passed']/total*100:.1f}%)")
            for violation in check['violations']:
                print(f"    - {violation}")
    
    return {
        'is_feasible': total_violations == 0,
        'n_violations': total_violations,
        'violations': violations,
        'constraint_checks': constraint_checks
    }

def main():
    json_path = "/Users/edoardospigarolo/Documents/OQI-UC002-DWave/Benchmarks/COMPREHENSIVE/Farm_DWave/config_25_run_1.json"
    
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return
    
    print("Loading solution from:", json_path)
    solution_data = load_solution(json_path)
    
    result = debug_constraint_validation(solution_data)
    
    print(f"\nOriginal DWave result: is_feasible = {solution_data.get('is_feasible', 'N/A')}")
    print(f"Manual validation result: is_feasible = {result['is_feasible']}")
    
    if solution_data.get('is_feasible') != str(result['is_feasible']).upper():
        print("\n⚠️  DISCREPANCY DETECTED between DWave result and manual validation!")

if __name__ == "__main__":
    main()