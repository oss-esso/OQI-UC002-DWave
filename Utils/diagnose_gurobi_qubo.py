#!/usr/bin/env python3
"""
Diagnostic script to investigate why Gurobi QUBO hits time limit.

This script analyzes:
1. BQM problem characteristics (size, density, coefficient ranges)
2. Lagrange multiplier impact on problem conditioning
3. Objective vs constraint penalty balance
4. Problem complexity metrics
"""

import os
import sys
import json
import time
import numpy as np

# Add project root to path for accessing Benchmark Scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Benchmark Scripts'))

from .patch_sampler import generate_farms as generate_patches
from src.scenarios import load_food_data
from solver_runner_PATCH import create_cqm, solve_with_pulp, solve_with_gurobi_qubo
from dimod import cqm_to_bqm

def analyze_bqm_characteristics(bqm):
    """Analyze BQM problem characteristics."""
    print("\n" + "="*80)
    print("BQM PROBLEM CHARACTERISTICS")
    print("="*80)
    
    # Basic size metrics
    n_vars = len(bqm.variables)
    n_linear = len(bqm.linear)
    n_quadratic = len(bqm.quadratic)
    density = (2 * n_quadratic) / (n_vars * (n_vars - 1)) if n_vars > 1 else 0
    
    print(f"Variables: {n_vars}")
    print(f"Linear terms: {n_linear}")
    print(f"Quadratic terms: {n_quadratic}")
    print(f"Density: {density:.4f} ({density*100:.2f}%)")
    print(f"Offset: {bqm.offset:.6f}")
    
    # Coefficient statistics
    linear_coeffs = [abs(c) for c in bqm.linear.values()]
    quad_coeffs = [abs(c) for c in bqm.quadratic.values()]
    
    if linear_coeffs:
        print(f"\nLinear coefficients:")
        print(f"  Min: {min(linear_coeffs):.6e}")
        print(f"  Max: {max(linear_coeffs):.6e}")
        print(f"  Mean: {np.mean(linear_coeffs):.6e}")
        print(f"  Std: {np.std(linear_coeffs):.6e}")
        print(f"  Range: {max(linear_coeffs) / min(linear_coeffs) if min(linear_coeffs) > 0 else 'inf'}x")
    
    if quad_coeffs:
        print(f"\nQuadratic coefficients:")
        print(f"  Min: {min(quad_coeffs):.6e}")
        print(f"  Max: {max(quad_coeffs):.6e}")
        print(f"  Mean: {np.mean(quad_coeffs):.6e}")
        print(f"  Std: {np.std(quad_coeffs):.6e}")
        print(f"  Range: {max(quad_coeffs) / min(quad_coeffs) if min(quad_coeffs) > 0 else 'inf'}x")
    
    # Conditioning analysis
    all_coeffs = linear_coeffs + quad_coeffs
    if all_coeffs:
        condition_number = max(all_coeffs) / min(all_coeffs) if min(all_coeffs) > 0 else float('inf')
        print(f"\nProblem Conditioning:")
        print(f"  Condition number: {condition_number:.2e}")
        if condition_number > 1e10:
            print(f"  ⚠️ VERY POORLY CONDITIONED (ratio > 1e10)")
            print(f"     This makes the problem extremely difficult for Gurobi to solve")
        elif condition_number > 1e6:
            print(f"  ⚠️ POORLY CONDITIONED (ratio > 1e6)")
            print(f"     This significantly increases solve difficulty")
        elif condition_number > 1e3:
            print(f"  ⚠️ MODERATELY CONDITIONED (ratio > 1e3)")
        else:
            print(f"  ✓ Well conditioned")
    
    return {
        'n_variables': n_vars,
        'n_linear': n_linear,
        'n_quadratic': n_quadratic,
        'density': density,
        'condition_number': condition_number if all_coeffs else None,
        'linear_coeff_range': max(linear_coeffs) / min(linear_coeffs) if linear_coeffs and min(linear_coeffs) > 0 else None,
        'quad_coeff_range': max(quad_coeffs) / min(quad_coeffs) if quad_coeffs and min(quad_coeffs) > 0 else None
    }

def compare_lagrange_multipliers(cqm, foods, land_data, config, test_multipliers):
    """Compare BQM characteristics for different Lagrange multipliers."""
    print("\n" + "="*80)
    print("LAGRANGE MULTIPLIER COMPARISON")
    print("="*80)
    
    # Calculate max objective coefficient
    max_obj_coeff = 0
    for plot in land_data:
        s_p = land_data[plot]
        for crop in foods:
            B_c = (
                config['parameters']['weights'].get('nutritional_value', 0) * foods[crop].get('nutritional_value', 0) +
                config['parameters']['weights'].get('nutrient_density', 0) * foods[crop].get('nutrient_density', 0) -
                config['parameters']['weights'].get('environmental_impact', 0) * foods[crop].get('environmental_impact', 0) +
                config['parameters']['weights'].get('affordability', 0) * foods[crop].get('affordability', 0) +
                config['parameters']['weights'].get('sustainability', 0) * foods[crop].get('sustainability', 0)
            )
            idle_penalty = config['parameters'].get('idle_penalty_lambda', 0.1)
            coeff = abs((B_c + idle_penalty) * s_p)
            max_obj_coeff = max(max_obj_coeff, coeff)
    
    print(f"\nMax objective coefficient: {max_obj_coeff:.6f}")
    print(f"\nTesting multipliers: {test_multipliers}")
    
    results = []
    for multiplier in test_multipliers:
        print(f"\n{'-'*80}")
        print(f"Lagrange multiplier: {multiplier:.2e} ({multiplier/max_obj_coeff:.1f}x max_obj)")
        print(f"{'-'*80}")
        
        try:
            bqm, _ = cqm_to_bqm(cqm, lagrange_multiplier=multiplier)
            analysis = analyze_bqm_characteristics(bqm)
            analysis['lagrange_multiplier'] = multiplier
            analysis['multiplier_ratio'] = multiplier / max_obj_coeff
            results.append(analysis)
        except Exception as e:
            print(f"❌ Failed to convert with multiplier {multiplier}: {e}")
    
    return results, max_obj_coeff

def test_small_problem():
    """Test with a small problem (10 patches) to diagnose the issue."""
    print("\n" + "="*80)
    print("TESTING SMALL PROBLEM (10 patches)")
    print("="*80)
    
    # Generate small problem
    land_data = generate_patches(n_farms=10, seed=42)
    
    # Load food data
    try:
        foods, food_groups = load_food_data()
    except:
        print("Warning: Could not load food data from Excel, using fallback")
        foods = {
            'Wheat': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 'affordability': 0.9, 'sustainability': 0.7},
            'Corn': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.5, 'affordability': 0.8, 'sustainability': 0.6},
            'Rice': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.7, 'affordability': 0.7, 'sustainability': 0.8},
            'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.4, 'affordability': 0.6, 'sustainability': 0.9},
            'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.8, 'affordability': 0.9, 'sustainability': 0.6},
            'Tomatoes': {'nutritional_value': 0.4, 'nutrient_density': 0.6, 'environmental_impact': 0.7, 'affordability': 0.7, 'sustainability': 0.5}
        }
        food_groups = {
            'grains': ['Wheat', 'Corn', 'Rice'],
            'proteins': ['Soybeans'],
            'vegetables': ['Potatoes', 'Tomatoes']
        }
    
    # Create config
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
    
    # Create CQM
    print("\nCreating CQM...")
    cqm, (X, Y), constraint_metadata = create_cqm(land_data, foods, food_groups, config)
    print(f"CQM: {len(cqm.variables)} variables, {len(cqm.constraints)} constraints")
    
    # Test different Lagrange multipliers
    # Default is 10x, we're using 10000x - let's test a range
    test_multipliers = [
        10,      # Default (10x)
        100,     # 100x
        1000,    # 1000x
        10000,   # Current (10000x)
        50000,   # Even higher
    ]
    
    results, max_obj_coeff = compare_lagrange_multipliers(cqm, foods, land_data, config, test_multipliers)
    
    # Solve with PuLP for reference
    print("\n" + "="*80)
    print("SOLVING WITH PULP (REFERENCE)")
    print("="*80)
    pulp_result = solve_with_pulp(list(land_data.keys()), foods, food_groups, config)
    print(f"PuLP Status: {pulp_result['status']}")
    print(f"PuLP Objective: {pulp_result['objective_value']:.6f}")
    print(f"PuLP Time: {pulp_result['solve_time']:.3f}s")
    
    # Try solving with Gurobi QUBO using default Lagrange multiplier
    print("\n" + "="*80)
    print("SOLVING WITH GUROBI QUBO (CURRENT LAGRANGE)")
    print("="*80)
    
    lagrange_multiplier = 10000 * max_obj_coeff
    print(f"Using Lagrange multiplier: {lagrange_multiplier:.2e}")
    
    bqm, _ = cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
    
    try:
        gurobi_result = solve_with_gurobi_qubo(
            bqm,
            farms=list(land_data.keys()),
            foods=foods,
            food_groups=food_groups,
            land_availability=land_data,
            weights=config['parameters']['weights'],
            idle_penalty=config['parameters'].get('idle_penalty_lambda', 0.1),
            config=config
        )
        print(f"Gurobi QUBO Status: {gurobi_result['status']}")
        print(f"Gurobi QUBO Objective: {gurobi_result.get('objective_value', 'N/A'):.6f}")
        print(f"Gurobi QUBO Time: {gurobi_result['solve_time']:.3f}s")
        
        # Compare objectives
        if gurobi_result.get('objective_value') is not None:
            gap = abs(pulp_result['objective_value'] - gurobi_result['objective_value']) / pulp_result['objective_value']
            print(f"\nObjective gap: {gap*100:.2f}%")
    except Exception as e:
        print(f"❌ Gurobi QUBO failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    print("\n[*] Lagrange Multiplier Analysis:")
    for r in results:
        print(f"\n  Multiplier: {r['lagrange_multiplier']:.2e} ({r['multiplier_ratio']:.1f}x)")
        print(f"    Condition number: {r['condition_number']:.2e}")
        print(f"    Variables: {r['n_variables']}, Quadratic: {r['n_quadratic']}")
        print(f"    Density: {r['density']*100:.2f}%")
    
    print("\n[*] Problem Analysis:")
    if results:
        worst_condition = max(r['condition_number'] for r in results if r['condition_number'] is not None)
        print(f"  Worst condition number: {worst_condition:.2e}")
        
        if worst_condition > 1e10:
            print("\n  [X] FORMULATION PROBLEM DETECTED:")
            print("     The Lagrange multiplier is creating an extremely poorly conditioned problem.")
            print("     The ratio between largest and smallest coefficients is > 1e10.")
            print("     This makes Gurobi struggle to find optimal solutions within time limit.")
            print("\n  [!] RECOMMENDATIONS:")
            print("     1. Use a smaller Lagrange multiplier (100x instead of 10000x)")
            print("     2. Scale the objective coefficients to be closer to 1.0")
            print("     3. Consider using a different QUBO formulation")
            print("     4. Increase time limit significantly (e.g., 1800s)")
        elif worst_condition > 1e6:
            print("\n  [!] CONDITIONING PROBLEM:")
            print("     The problem is poorly conditioned but may be solvable with more time.")
            print("\n  [!] RECOMMENDATIONS:")
            print("     1. Increase time limit (e.g., 600-1800s)")
            print("     2. Consider reducing Lagrange multiplier slightly")

if __name__ == "__main__":
    test_small_problem()
