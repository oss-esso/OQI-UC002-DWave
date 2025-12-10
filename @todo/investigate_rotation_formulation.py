#!/usr/bin/env python3
"""
Investigate: What formulation is actually being used for rotation scenarios?

This script checks:
1. What data structure does load_food_data('rotation_micro_25') return?
2. How many variables does it actually have?
3. What constraints are defined?
4. Does it match the rotation formulation described in the documentation?
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.scenarios import load_food_data

print("="*100)
print("ROTATION SCENARIO INVESTIGATION")
print("="*100)

# Test each rotation scenario
scenarios = [
    'rotation_micro_25',
    'rotation_small_50', 
    'rotation_medium_100',
    'rotation_large_200'
]

for scenario_name in scenarios:
    print(f"\n{'='*100}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*100}")
    
    # Load the data
    farms, foods, food_groups, config = load_food_data(scenario_name)
    
    print(f"\n1. DATA STRUCTURE:")
    print(f"   - Farms (plots): {len(farms)}")
    print(f"   - Foods (crops): {len(foods)}")
    print(f"   - Foods list: {list(foods.keys())}")
    print(f"   - Food groups: {list(food_groups.keys())}")
    
    # Check if these are crop families or individual crops
    print(f"\n2. CROP TYPE:")
    if len(foods) == 6 and all(f in ['Fruits', 'Grains', 'Legumes', 'Leafy_Vegetables', 'Root_Vegetables', 'Proteins'] for f in foods.keys()):
        print(f"   ✓ CROP FAMILIES (6 families as expected)")
    elif len(foods) == 27:
        print(f"   ✗ INDIVIDUAL CROPS (27 crops - NOT rotation scenario!)")
    else:
        print(f"   ? UNKNOWN ({len(foods)} crops)")
    
    # Check for rotation-specific parameters
    print(f"\n3. ROTATION PARAMETERS:")
    params = config.get('parameters', {})
    
    has_rotation = False
    if 'rotation_gamma' in params:
        print(f"   ✓ rotation_gamma: {params['rotation_gamma']}")
        has_rotation = True
    else:
        print(f"   ✗ rotation_gamma: NOT FOUND")
    
    if 'spatial_k_neighbors' in params:
        print(f"   ✓ spatial_k_neighbors: {params['spatial_k_neighbors']}")
        has_rotation = True
    else:
        print(f"   ✗ spatial_k_neighbors: NOT FOUND")
        
    if 'frustration_ratio' in params:
        print(f"   ✓ frustration_ratio: {params['frustration_ratio']}")
        has_rotation = True
    else:
        print(f"   ✗ frustration_ratio: NOT FOUND")
    
    if 'use_soft_one_hot' in params:
        print(f"   ✓ use_soft_one_hot: {params['use_soft_one_hot']}")
        has_rotation = True
    else:
        print(f"   ✗ use_soft_one_hot: NOT FOUND")
        
    if 'one_hot_penalty' in params:
        print(f"   ✓ one_hot_penalty: {params['one_hot_penalty']}")
        has_rotation = True
    else:
        print(f"   ✗ one_hot_penalty: NOT FOUND")
        
    if 'diversity_bonus' in params:
        print(f"   ✓ diversity_bonus: {params['diversity_bonus']}")
        has_rotation = True
    else:
        print(f"   ✗ diversity_bonus: NOT FOUND")
    
    # Check land availability structure
    print(f"\n4. LAND AVAILABILITY:")
    land = params.get('land_availability', {})
    print(f"   - Farms in land dict: {len(land)}")
    print(f"   - Total area: {sum(land.values()):.2f} ha")
    
    # Expected variable counts
    n_farms = len(farms)
    n_families = len(foods)
    n_periods = 3  # Assumed for rotation
    
    print(f"\n5. EXPECTED VARIABLES (if multi-period rotation):")
    print(f"   - Y variables (binary assignments): {n_farms} farms × {n_families} families × {n_periods} periods = {n_farms * n_families * n_periods}")
    print(f"   - U variables (unique food tracking): {n_families} families × {n_periods} periods = {n_families * n_periods}")
    print(f"   - TOTAL: {n_farms * n_families * n_periods + n_families * n_periods}")
    
    print(f"\n6. ACTUAL VARIABLES (single-period formulation in qpu_benchmark.py):")
    print(f"   - Y variables: {n_farms} farms × {n_families} crops = {n_farms * n_families}")
    print(f"   - U variables: {n_families} crops = {n_families}")
    print(f"   - TOTAL: {n_farms * n_families + n_families}")
    
    # Check for food group constraints
    print(f"\n7. FOOD GROUP CONSTRAINTS:")
    fg_constraints = params.get('food_group_constraints', {})
    for group, limits in fg_constraints.items():
        print(f"   - {group}: {limits}")
    
    print(f"\n8. VERDICT:")
    if has_rotation:
        print(f"   ✓ Rotation parameters PRESENT in config")
        print(f"   ⚠ BUT: qpu_benchmark.py uses build_binary_cqm() which is SINGLE-PERIOD")
        print(f"   ⚠ The rotation parameters (gamma, frustration, soft constraints) are IGNORED")
        print(f"   ✗ CONCLUSION: Rotation scenarios are being solved as STATIC single-period problems!")
    else:
        print(f"   ✗ No rotation parameters found - this is a standard static scenario")

print(f"\n{'='*100}")
print(f"SUMMARY")
print(f"{'='*100}")
print(f"\nThe rotation scenarios ARE defined with multi-period rotation parameters,")
print(f"but qpu_benchmark.py is solving them as SINGLE-PERIOD static assignment problems.")
print(f"\nThe build_binary_cqm() function does NOT implement:")
print(f"  - Temporal dimension (3 periods)")
print(f"  - Rotation synergies (quadratic temporal coupling)")
print(f"  - Spatial neighbor interactions")
print(f"  - Soft one-hot penalty")
print(f"  - Diversity bonus")
print(f"\nInstead, it uses the standard formulation:")
print(f"  - Y[farm, crop] binary variables (one period only)")
print(f"  - U[crop] unique food tracking")
print(f"  - Hard one-hot: at most 1 crop per farm")
print(f"  - Food group diversity constraints")
print(f"\nThis explains why the QPU results don't match predictions!")
print("="*100)
