"""
Professional verifier script.

This script:
1. Loads the run manifest
2. Loads PuLP results, DWave results, CQM, and constraints
3. Verifies both solutions against constraints
4. Compares the two solutions
5. Generates a detailed verification report
"""

import os
import sys
import json
import pickle
import shutil
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from dimod import ConstrainedQuadraticModel

def load_manifest(manifest_path):
    """Load the run manifest."""
    with open(manifest_path, 'r') as f:
        return json.load(f)

def load_cqm(cqm_path):
    """Load the CQM model."""
    return ConstrainedQuadraticModel.from_file(cqm_path)

def load_constraints(constraints_path):
    """Load constraint metadata."""
    with open(constraints_path, 'r') as f:
        return json.load(f)

def load_pulp_results(pulp_path):
    """Load PuLP results."""
    with open(pulp_path, 'r') as f:
        return json.load(f)

def load_dwave_results(dwave_path):
    """Load DWave sampleset."""
    with open(dwave_path, 'rb') as f:
        return pickle.load(f)

def verify_land_availability(solution, constraints_data):
    """Verify land availability constraints."""
    violations = []
    farms = constraints_data['farms']
    foods = constraints_data['foods']
    land_limits = constraints_data['config']['parameters']['land_availability']
    
    for farm in farms:
        total_area = 0
        for food in foods:
            key = f"{farm}_{food}"
            area = solution.get(f"A_{key}", solution.get(key, 0))
            if isinstance(area, str):
                continue
            total_area += area
        
        max_land = land_limits[farm]
        if total_area > max_land + 0.001:
            violations.append({
                'type': 'land_availability',
                'farm': farm,
                'total_area': total_area,
                'max_land': max_land,
                'violation': total_area - max_land
            })
    
    return violations

def verify_linking_constraints(solution, constraints_data, selections=None):
    """Verify min/max area linking constraints."""
    violations = []
    farms = constraints_data['farms']
    foods = constraints_data['foods']
    min_areas = constraints_data['config']['parameters'].get('minimum_planting_area', {})
    land_limits = constraints_data['config']['parameters']['land_availability']
    
    for farm in farms:
        for food in foods:
            base_key = f"{farm}_{food}"
            a_key = f"A_{base_key}"
            y_key = f"Y_{base_key}"
            
            # Get area value
            a_val = solution.get(a_key, solution.get(base_key, 0))
            
            # Get selection value - check selections dict if provided (for PuLP)
            if selections is not None:
                y_val = selections.get(base_key, 0)
            else:
                y_val = solution.get(y_key, 0)
            
            if isinstance(a_val, str) or isinstance(y_val, str):
                continue
            
            # Infer selection from area if Y value not found
            if y_val == 0 and a_val > 0.001:
                y_val = 1.0  # If area is allocated, food must be selected
            
            # Check minimum area if selected
            if y_val > 0.5:
                min_area = min_areas.get(food, 0)
                if a_val < min_area - 0.001:
                    violations.append({
                        'type': 'min_area_if_selected',
                        'farm': farm,
                        'food': food,
                        'area': a_val,
                        'min_area': min_area,
                        'violation': min_area - a_val
                    })
            
            # Check area is 0 if not selected
            if y_val < 0.5:
                if a_val > 0.001:
                    violations.append({
                        'type': 'area_zero_if_not_selected',
                        'farm': farm,
                        'food': food,
                        'area': a_val,
                        'selection': y_val,
                        'violation': a_val
                    })
    
    return violations

def verify_food_group_constraints(solution, constraints_data, selections=None):
    """Verify food group constraints."""
    violations = []
    farms = constraints_data['farms']
    food_groups = constraints_data['food_groups']
    fg_constraints = constraints_data['config']['parameters'].get('food_group_constraints', {})
    
    for group, group_foods in food_groups.items():
        if group not in fg_constraints:
            continue
        
        constraints = fg_constraints[group]
        
        for farm in farms:
            selected_count = 0
            for food in group_foods:
                base_key = f"{farm}_{food}"
                y_key = f"Y_{base_key}"
                
                # Get selection value - check selections dict if provided (for PuLP)
                if selections is not None:
                    y_val = selections.get(base_key, 0)
                else:
                    y_val = solution.get(y_key, 0)
                
                # Infer from area if Y not found
                if y_val == 0:
                    a_val = solution.get(f"A_{base_key}", solution.get(base_key, 0))
                    if isinstance(a_val, (int, float)) and a_val > 0.001:
                        y_val = 1.0
                
                if isinstance(y_val, str):
                    continue
                if y_val > 0.5:
                    selected_count += 1
            
            # Check minimum
            if 'min_foods' in constraints:
                min_foods = constraints['min_foods']
                if selected_count < min_foods:
                    violations.append({
                        'type': 'food_group_min',
                        'farm': farm,
                        'group': group,
                        'selected': selected_count,
                        'min_required': min_foods,
                        'violation': min_foods - selected_count
                    })
            
            # Check maximum
            if 'max_foods' in constraints:
                max_foods = constraints['max_foods']
                if selected_count > max_foods:
                    violations.append({
                        'type': 'food_group_max',
                        'farm': farm,
                        'group': group,
                        'selected': selected_count,
                        'max_allowed': max_foods,
                        'violation': selected_count - max_foods
                    })
    
    return violations

def verify_solution(solution, constraints_data, solver_name, selections=None):
    """Verify a solution against all constraints."""
    print(f"\n{'=' * 80}")
    print(f"VERIFYING {solver_name} SOLUTION")
    print(f"{'=' * 80}")
    
    all_violations = []
    
    # Verify land availability
    land_violations = verify_land_availability(solution, constraints_data)
    all_violations.extend(land_violations)
    if land_violations:
        print(f"\n  Land Availability Violations: {len(land_violations)}")
        for v in land_violations:
            print(f"    {v['farm']}: {v['total_area']:.2f} > {v['max_land']:.2f} (violation: {v['violation']:.2f})")
    else:
        print(f"\n  Land Availability: PASS")
    
    # Verify linking constraints
    linking_violations = verify_linking_constraints(solution, constraints_data, selections)
    all_violations.extend(linking_violations)
    if linking_violations:
        print(f"\n  Linking Constraint Violations: {len(linking_violations)}")
        for v in linking_violations[:5]:  # Show first 5
            print(f"    {v['farm']}, {v['food']}: {v['type']}")
    else:
        print(f"\n  Linking Constraints: PASS")
    
    # Verify food group constraints
    fg_violations = verify_food_group_constraints(solution, constraints_data, selections)
    all_violations.extend(fg_violations)
    if fg_violations:
        print(f"\n  Food Group Violations: {len(fg_violations)}")
        for v in fg_violations:
            print(f"    {v['farm']}, {v['group']}: {v['selected']} selected (violation: {v['violation']})")
    else:
        print(f"\n  Food Group Constraints: PASS")
    
    # Summary
    if all_violations:
        print(f"\n  TOTAL VIOLATIONS: {len(all_violations)}")
        print(f"  STATUS: FAILED")
    else:
        print(f"\n  STATUS: ALL CONSTRAINTS SATISFIED")
    
    return all_violations

def compare_solutions(pulp_solution, dwave_solution, constraints_data):
    """Compare PuLP and DWave solutions."""
    print(f"\n{'=' * 80}")
    print("SOLUTION COMPARISON")
    print(f"{'=' * 80}")
    
    farms = constraints_data['farms']
    foods = constraints_data['foods']
    
    differences = []
    
    for farm in farms:
        print(f"\n{farm}:")
        print(f"  {'Food':<15} | {'PuLP Area':<12} | {'DWave Area':<12} | {'Match':<6}")
        print(f"  {'-' * 15}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 6}")
        
        for food in foods:
            base_key = f"{farm}_{food}"
            
            pulp_area = pulp_solution.get(base_key, 0)
            dwave_area = dwave_solution.get(f"A_{base_key}", 0)
            
            if isinstance(pulp_area, str) or isinstance(dwave_area, str):
                continue
            
            diff = abs(pulp_area - dwave_area)
            match = "YES" if diff < 0.01 else "NO"
            
            if diff > 0.01:
                differences.append({
                    'farm': farm,
                    'food': food,
                    'pulp_area': pulp_area,
                    'dwave_area': dwave_area,
                    'difference': diff
                })
            
            print(f"  {food:<15} | {pulp_area:<12.2f} | {dwave_area:<12.2f} | {match:<6}")
    
    return differences

def calculate_objective(solution, constraints_data):
    """Calculate objective value from solution."""
    farms = constraints_data['farms']
    
    # Try to get foods data from constraints file first
    foods_dict = constraints_data.get('foods_data', {})
    
    # If foods_data not in constraints, try config parameters
    if not foods_dict:
        foods_dict = constraints_data['config']['parameters'].get('foods', {})
    
    # If still no foods dict, reconstruct from scenario
    if not foods_dict:
        from src.scenarios import load_food_data
        _, foods_dict, _, _ = load_food_data(constraints_data['scenario'])
    
    weights = constraints_data['config']['parameters']['weights']
    land_limits = constraints_data['config']['parameters']['land_availability']
    total_area = sum(land_limits.values())
    
    numerator = 0
    for farm in farms:
        for food in constraints_data['foods']:
            base_key = f"{farm}_{food}"
            a_key = f"A_{base_key}"
            
            area = solution.get(a_key, solution.get(base_key, 0))
            if isinstance(area, str):
                continue
            
            food_data = foods_dict.get(food, {})
            numerator += (
                weights.get('nutritional_value', 0) * food_data.get('nutritional_value', 0) * area +
                weights.get('nutrient_density', 0) * food_data.get('nutrient_density', 0) * area -
                weights.get('environmental_impact', 0) * food_data.get('environmental_impact', 0) * area +
                weights.get('affordability', 0) * food_data.get('affordability', 0) * area +
                weights.get('sustainability', 0) * food_data.get('sustainability', 0) * area
            )
    
    return numerator / total_area

def main(manifest_path):
    """Main verification function."""
    print("=" * 80)
    print("PROFESSIONAL SOLUTION VERIFIER")
    print("=" * 80)
    
    # Load manifest
    print(f"\nLoading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    
    print(f"  Scenario: {manifest['scenario']}")
    print(f"  Timestamp: {manifest['timestamp']}")
    
    # Load all components
    print("\nLoading components...")
    constraints_data = load_constraints(manifest['constraints_path'])
    pulp_results = load_pulp_results(manifest['pulp_path'])
    dwave_sampleset = load_dwave_results(manifest['dwave_path'])
    
    print(f"  CQM: {manifest['cqm_path']}")
    print(f"  Constraints: {manifest['constraints_path']}")
    print(f"  PuLP results: {manifest['pulp_path']}")
    print(f"  DWave results: {manifest['dwave_path']}")
    
    # Extract DWave best solution
    feasible_sampleset = dwave_sampleset.filter(lambda d: d.is_feasible)
    if not feasible_sampleset:
        print("\nERROR: No feasible DWave solution found!")
        return
    
    dwave_solution = feasible_sampleset.first.sample
    
    # Display timing information
    print(f"\n{'=' * 80}")
    print("SOLVER TIMING INFORMATION")
    print(f"{'=' * 80}")
    
    # PuLP timing
    pulp_solve_time = pulp_results.get('solve_time', 0)
    print(f"\n  PuLP Solve Time: {pulp_solve_time:.3f} seconds")
    
    # DWave timing
    dwave_info = dwave_sampleset.info
    if isinstance(dwave_info, dict):
        qpu_time = dwave_info.get('qpu_access_time', 0)
        charge_time = dwave_info.get('charge_time', 0)
        run_time = dwave_info.get('run_time', 0)
        
        print(f"\n  DWave Timing:")
        print(f"    QPU Access Time: {qpu_time / 1000:.3f} ms")
        print(f"    Charge Time:     {charge_time / 1000:.3f} ms")
        print(f"    Total Run Time:  {run_time / 1000:.3f} ms ({run_time / 1000000:.3f} seconds)")
    else:
        print(f"\n  DWave Timing: Not available")
    
    # Verify PuLP solution (pass selections separately)
    pulp_violations = verify_solution(pulp_results['areas'], constraints_data, "PULP", pulp_results['selections'])
    
    # Verify DWave solution (selections included in sample)
    dwave_violations = verify_solution(dwave_solution, constraints_data, "DWAVE")
    
    # Compare solutions
    differences = compare_solutions(pulp_results['areas'], dwave_solution, constraints_data)
    
    # Compare objectives
    print(f"\n{'=' * 80}")
    print("OBJECTIVE COMPARISON")
    print(f"{'=' * 80}")
    
    pulp_obj = pulp_results['objective_value']
    dwave_obj = calculate_objective(dwave_solution, constraints_data)
    
    print(f"\n  PuLP Objective:  {pulp_obj:.6f}")
    print(f"  DWave Objective: {dwave_obj:.6f}")
    print(f"  Difference:      {abs(pulp_obj - dwave_obj):.6f}")
    
    if abs(pulp_obj - dwave_obj) < 0.0001:
        print(f"  Status: IDENTICAL")
    else:
        print(f"  Status: DIFFERENT")
    
    # Generate verification report
    dwave_timing = {}
    if isinstance(dwave_sampleset.info, dict):
        dwave_timing = {
            'qpu_access_time_ms': float(dwave_sampleset.info.get('qpu_access_time', 0) / 1000),
            'charge_time_ms': float(dwave_sampleset.info.get('charge_time', 0) / 1000),
            'run_time_ms': float(dwave_sampleset.info.get('run_time', 0) / 1000),
            'run_time_seconds': float(dwave_sampleset.info.get('run_time', 0) / 1000000)
        }
    
    report = {
        'manifest': manifest,
        'verification_timestamp': datetime.now().isoformat(),
        'timing': {
            'pulp_solve_time_seconds': float(pulp_results.get('solve_time', 0)),
            'dwave': dwave_timing
        },
        'pulp_verification': {
            'violations': pulp_violations,
            'passed': bool(len(pulp_violations) == 0),
            'objective': float(pulp_obj)
        },
        'dwave_verification': {
            'violations': dwave_violations,
            'passed': bool(len(dwave_violations) == 0),
            'objective': float(dwave_obj)
        },
        'comparison': {
            'differences': differences,
            'solutions_match': bool(len(differences) == 0),
            'objectives_match': bool(abs(pulp_obj - dwave_obj) < 0.0001)
        }
    }
    
    report_path = f"verification_report_{manifest['scenario']}_{manifest['timestamp']}.json"
    print(f"\n{'=' * 80}")
    print(f"Saving verification report to: {report_path}")
    print(f"{'=' * 80}")
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Final summary
    print(f"\nVERIFICATION SUMMARY:")
    print(f"  PuLP: {'PASS' if not pulp_violations else 'FAIL'} ({len(pulp_violations)} violations)")
    print(f"  DWave: {'PASS' if not dwave_violations else 'FAIL'} ({len(dwave_violations)} violations)")
    print(f"  Solutions match: {'YES' if len(differences) == 0 else 'NO'} ({len(differences)} differences)")
    print(f"  Objectives match: {'YES' if abs(pulp_obj - dwave_obj) < 0.0001 else 'NO'}")
    
    if not pulp_violations and not dwave_violations and len(differences) == 0:
        print(f"\n  OVERALL: PERFECT MATCH - Both solvers found the same valid solution!")
    elif not pulp_violations and not dwave_violations:
        print(f"\n  OVERALL: BOTH VALID - Solvers found different valid solutions")
    else:
        print(f"\n  OVERALL: ISSUES DETECTED - See report for details")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify solver results from a run manifest')
    parser.add_argument('manifest', type=str, help='Path to the run manifest JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.manifest):
        print(f"Error: Manifest file not found: {args.manifest}")
        sys.exit(1)
    
    main(args.manifest)
