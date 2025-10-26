#!/usr/bin/env python3
"""
Constraint Violation Simulation for BQM Analysis

This script simulates the constraint violation behavior by analyzing
the BQM structure and testing different solution scenarios to understand
why the "at most one crop per plot" constraint is being violated.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from patch_sampler import generate_farms as generate_patches
from solver_runner_PATCH import create_cqm, validate_solution_constraints
from dimod import cqm_to_bqm


def simulate_constraint_violations():
    """Simulate constraint violation scenarios to understand the BQM behavior."""
    print("🧪 CONSTRAINT VIOLATION SIMULATION")
    print("="*60)
    
    # Generate the same 50-unit scenario
    print("Generating 50-unit patch scenario...")
    land_data = generate_patches(n_farms=50, seed=42)
    
    # Create food data
    foods = {
        'Wheat': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 'affordability': 0.9, 'sustainability': 0.7},
        'Corn': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.5, 'affordability': 0.8, 'sustainability': 0.6},
        'Rice': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.7, 'affordability': 0.7, 'sustainability': 0.8},
        'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.4, 'affordability': 0.6, 'sustainability': 0.9},
        'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.8, 'affordability': 0.9, 'sustainability': 0.6},
        'Apples': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.3, 'affordability': 0.7, 'sustainability': 0.8}
    }
    food_groups = {
        'grains': ['Wheat', 'Corn', 'Rice'],
        'proteins': ['Soybeans'],
        'vegetables': ['Potatoes'],
        'fruits': ['Apples']
    }
    
    config = {
        'parameters': {
            'land_availability': land_data,
            'minimum_planting_area': {food: 0.0 for food in foods},
            'food_group_constraints': {},
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.2,
                'environmental_impact': 0.25,
                'affordability': 0.15,
                'sustainability': 0.15
            },
            'idle_penalty_lambda': 0.1
        }
    }
    
    farms = list(land_data.keys())
    
    # Create CQM and convert to BQM
    print("Creating CQM and converting to BQM...")
    cqm, (X, Y), constraint_metadata = create_cqm(farms, foods, food_groups, config)
    bqm, invert = cqm_to_bqm(cqm)
    
    print(f"✅ CQM: {len(cqm.variables)} variables, {len(cqm.constraints)} constraints")
    print(f"✅ BQM: {len(bqm.variables)} variables, {len(bqm.quadratic)} quadratic terms")
    
    # Analyze objective vs penalty ratios
    print("\\nAnalyzing objective vs penalty ratios...")
    
    # Calculate typical objective coefficient magnitude
    objective_coeffs = []
    for plot in farms:
        s_p = land_data[plot]
        for crop in foods:
            B_c = (
                config['parameters']['weights'].get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                config['parameters']['weights'].get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                config['parameters']['weights'].get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                config['parameters']['weights'].get('affordability', 0) * foods[crop].get('affordability', 0) +
                config['parameters']['weights'].get('sustainability', 0) * foods[crop].get('sustainability', 0)
            )
            objective_coeff = abs((B_c + config['parameters']['idle_penalty_lambda']) * s_p)
            objective_coeffs.append(objective_coeff)
    
    avg_objective = np.mean(objective_coeffs)
    max_objective = np.max(objective_coeffs)
    
    # Get BQM penalty magnitudes
    linear_magnitudes = [abs(bias) for bias in bqm.linear.values()]
    quad_magnitudes = [abs(bias) for bias in bqm.quadratic.values()]
    
    max_linear = max(linear_magnitudes) if linear_magnitudes else 0
    max_quad = max(quad_magnitudes) if quad_magnitudes else 0
    avg_linear = np.mean(linear_magnitudes) if linear_magnitudes else 0
    avg_quad = np.mean(quad_magnitudes) if quad_magnitudes else 0
    
    print(f"\\n📊 OBJECTIVE vs PENALTY ANALYSIS:")
    print(f"   Average objective coefficient: {avg_objective:.3f}")
    print(f"   Maximum objective coefficient: {max_objective:.3f}")
    print(f"   Average linear penalty: {avg_linear:.3f}")
    print(f"   Maximum linear penalty: {max_linear:.3f}")
    print(f"   Average quadratic penalty: {avg_quad:.3f}")
    print(f"   Maximum quadratic penalty: {max_quad:.3f}")
    
    # Calculate penalty-to-objective ratios
    linear_ratio = max_linear / max_objective if max_objective > 0 else 0
    quad_ratio = max_quad / max_objective if max_objective > 0 else 0
    
    print(f"\\n🔍 PENALTY RATIOS:")
    print(f"   Linear penalty / Objective ratio: {linear_ratio:.1f}x")
    print(f"   Quadratic penalty / Objective ratio: {quad_ratio:.1f}x")
    
    # Simulate violating vs non-violating solutions
    print("\\n🧪 SIMULATING CONSTRAINT VIOLATION SCENARIOS...")
    
    # Get all BQM variables
    bqm_variables = list(bqm.variables)
    print(f"   BQM has {len(bqm_variables)} variables")
    
    # Create solutions that include all BQM variables
    print("\\n✅ CREATING VALID SOLUTION:")
    valid_solution = create_valid_solution_bqm(bqm_variables, farms, foods)
    valid_energy = bqm.energy(valid_solution)
    valid_violations = count_plot_violations(valid_solution, farms, foods)
    
    print(f"   BQM Energy: {valid_energy:.2f}")
    print(f"   Plot violations: {valid_violations}")
    
    print("\\n❌ CREATING INVALID SOLUTION:")
    invalid_solution = create_invalid_solution_bqm(bqm_variables, farms, foods, n_violations=20)
    invalid_energy = bqm.energy(invalid_solution)
    invalid_violations = count_plot_violations(invalid_solution, farms, foods)
    
    print(f"   BQM Energy: {invalid_energy:.2f}")
    print(f"   Plot violations: {invalid_violations}")
    print(f"   BQM Energy: {invalid_energy:.2f}")
    print(f"   Plot violations: {invalid_violations}")
    
    # Energy difference analysis
    energy_diff = invalid_energy - valid_energy
    print(f"\\n⚖️  ENERGY DIFFERENCE:")
    print(f"   Invalid - Valid = {energy_diff:.2f}")
    if invalid_violations > 0:
        print(f"   Per violation penalty: {energy_diff / invalid_violations:.2f}")
    
    # Analysis
    if energy_diff > 0:
        print(f"   ✅ BQM correctly penalizes violations (+{energy_diff:.2f})")
    else:
        print(f"   ❌ BQM doesn't penalize violations enough ({energy_diff:.2f})")
        print(f"   🔴 This explains why D-Wave finds violating solutions!")
    
    # Save analysis results
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'scenario': {
            'n_patches': 50,
            'total_area': sum(land_data.values()),
            'n_foods': len(foods)
        },
        'objective_analysis': {
            'avg_objective_coeff': avg_objective,
            'max_objective_coeff': max_objective
        },
        'penalty_analysis': {
            'avg_linear_penalty': avg_linear,
            'max_linear_penalty': max_linear,
            'avg_quad_penalty': avg_quad,
            'max_quad_penalty': max_quad,
            'linear_penalty_ratio': linear_ratio,
            'quad_penalty_ratio': quad_ratio
        },
        'energy_analysis': {
            'valid_solution_energy': valid_energy,
            'invalid_solution_energy': invalid_energy,
            'energy_difference': energy_diff,
            'valid_violations': valid_violations,
            'invalid_violations': invalid_violations,
            'penalty_per_violation': energy_diff / max(invalid_violations, 1) if invalid_violations > 0 else 0
        },
        'diagnosis': {
            'penalties_sufficient': bool(energy_diff > 0),
            'root_cause': 'Insufficient penalty strength' if energy_diff <= 0 else 'Penalty strength adequate'
        }
    }
    
    # Save results
    output_file = f"constraint_violation_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\\n💾 Analysis saved to: {output_path}")
    
    return analysis_results


def create_valid_solution_bqm(bqm_variables, farms, foods):
    """Create a valid solution with at most one crop per plot for BQM variables."""
    solution = {}
    
    # Initialize all BQM variables to 0
    for var in bqm_variables:
        solution[var] = 0
    
    # Assign one crop to some plots (valid assignment)
    crops_list = list(foods.keys())
    np.random.seed(42)  # For reproducibility
    
    for i, plot in enumerate(farms):
        if np.random.random() < 0.7:  # 70% chance to assign a crop
            crop = crops_list[i % len(crops_list)]
            # Set X and Y variables if they exist in BQM
            x_var = f"X_{plot}_{crop}"
            y_var = f"Y_{crop}"
            if x_var in solution:
                solution[x_var] = 1
            if y_var in solution:
                solution[y_var] = 1
    
    return solution


def create_invalid_solution_bqm(bqm_variables, farms, foods, n_violations=20):
    """Create an invalid solution with multiple crops per plot for BQM variables."""
    solution = {}
    
    # Initialize all BQM variables to 0
    for var in bqm_variables:
        solution[var] = 0
    
    # Create valid assignments first
    crops_list = list(foods.keys())
    np.random.seed(42)
    
    for i, plot in enumerate(farms):
        if np.random.random() < 0.6:  # 60% chance to assign a crop
            crop = crops_list[i % len(crops_list)]
            x_var = f"X_{plot}_{crop}"
            y_var = f"Y_{crop}"
            if x_var in solution:
                solution[x_var] = 1
            if y_var in solution:
                solution[y_var] = 1
    
    # Add violations by assigning additional crops to some plots
    violation_plots = np.random.choice(farms, size=min(n_violations, len(farms)), replace=False)
    
    for plot in violation_plots:
        # Find a different crop to assign
        current_crops = [crop for crop in foods if solution.get(f"X_{plot}_{crop}", 0) == 1]
        available_crops = [crop for crop in foods if crop not in current_crops]
        
        if available_crops:
            additional_crop = np.random.choice(available_crops)
            x_var = f"X_{plot}_{additional_crop}"
            y_var = f"Y_{additional_crop}"
            if x_var in solution:
                solution[x_var] = 1
            if y_var in solution:
                solution[y_var] = 1
    
    return solution


def create_valid_solution(farms, foods):
    """Create a valid solution with at most one crop per plot."""
    solution = {}
    
    # Initialize all variables to 0
    for plot in farms:
        for crop in foods:
            solution[f"X_{plot}_{crop}"] = 0
    
    for crop in foods:
        solution[f"Y_{crop}"] = 0
    
    # Assign one crop to some plots (valid assignment)
    crops_list = list(foods.keys())
    np.random.seed(42)  # For reproducibility
    
    for i, plot in enumerate(farms):
        if np.random.random() < 0.7:  # 70% chance to assign a crop
            crop = crops_list[i % len(crops_list)]
            solution[f"X_{plot}_{crop}"] = 1
            solution[f"Y_{crop}"] = 1
    
    return solution


def create_invalid_solution(farms, foods, n_violations=20):
    """Create an invalid solution with multiple crops per plot."""
    solution = {}
    
    # Initialize all variables to 0
    for plot in farms:
        for crop in foods:
            solution[f"X_{plot}_{crop}"] = 0
    
    for crop in foods:
        solution[f"Y_{crop}"] = 0
    
    # Create valid assignments first
    crops_list = list(foods.keys())
    np.random.seed(42)
    
    for i, plot in enumerate(farms):
        if np.random.random() < 0.6:  # 60% chance to assign a crop
            crop = crops_list[i % len(crops_list)]
            solution[f"X_{plot}_{crop}"] = 1
            solution[f"Y_{crop}"] = 1
    
    # Add violations by assigning additional crops to some plots
    violation_plots = np.random.choice(farms, size=min(n_violations, len(farms)), replace=False)
    
    for plot in violation_plots:
        # Find a different crop to assign
        current_crops = [crop for crop in foods if solution[f"X_{plot}_{crop}"] == 1]
        available_crops = [crop for crop in foods if crop not in current_crops]
        
        if available_crops:
            additional_crop = np.random.choice(available_crops)
            solution[f"X_{plot}_{additional_crop}"] = 1
            solution[f"Y_{additional_crop}"] = 1
    
    return solution


def count_plot_violations(solution, farms, foods):
    """Count how many plots have multiple crops assigned."""
    violations = 0
    
    for plot in farms:
        crops_assigned = sum(solution.get(f"X_{plot}_{crop}", 0) for crop in foods)
        if crops_assigned > 1:
            violations += 1
    
    return violations


if __name__ == "__main__":
    print("🔬 Running Constraint Violation Simulation...")
    print("   This will analyze BQM penalty strength without needing D-Wave access")
    
    try:
        results = simulate_constraint_violations()
        
        print("\\n🎯 SUMMARY:")
        if results['diagnosis']['penalties_sufficient']:
            print("   ✅ BQM penalties are sufficient to prevent violations")
            print("   ⚠️  Issue may be in D-Wave solver behavior or annealing process")
        else:
            print("   ❌ BQM penalties are insufficient to prevent violations")
            print("   🔧 Need to increase Lagrange multipliers in CQM→BQM conversion")
            
        print("\\n✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"\\n💥 Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)