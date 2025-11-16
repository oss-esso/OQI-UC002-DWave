#!/usr/bin/env python3
"""
Script to analyze and report all constraints in Farm and Patch scenarios.
"""

import os
import sys
import json

# Add project root and Benchmark Scripts to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.scenarios import load_food_data
from .farm_sampler import generate_farms as generate_farms_large
from .patch_sampler import generate_farms as generate_patches_small
from solver_runner_PATCH import create_cqm

def analyze_scenario(scenario_name, land_data, foods, food_groups, config):
    """Analyze and report constraints for a scenario."""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {scenario_name}")
    print(f"{'='*80}")
    
    # Basic info
    total_area = sum(land_data.values())
    print(f"\nBASIC INFO:")
    print(f"  Number of plots: {len(land_data)}")
    print(f"  Total area: {total_area:.4f} ha")
    print(f"  Number of foods: {len(foods)}")
    print(f"  Food names: {', '.join(foods.keys())}")
    
    # Plot sizes
    print(f"\nPLOT SIZES:")
    for plot, area in sorted(land_data.items()):
        percentage = (area / total_area) * 100
        print(f"  {plot}: {area:.4f} ha ({percentage:.1f}% of total)")
    
    # Check if any plot exceeds max percentage
    max_percentage = config['parameters'].get('max_percentage_per_crop', {})
    if max_percentage:
        print(f"\nMAX PERCENTAGE PER CROP CONSTRAINTS:")
        max_pct = list(max_percentage.values())[0] if max_percentage else 0
        max_area_allowed = max_pct * total_area
        print(f"  Max percentage: {max_pct*100:.1f}%")
        print(f"  Max area allowed per crop: {max_area_allowed:.4f} ha")
        
        print(f"\n  CRITICAL CHECK - Plots larger than max allowed:")
        problematic_plots = []
        for plot, area in land_data.items():
            if area > max_area_allowed:
                problematic_plots.append((plot, area))
                print(f"    ⚠️ {plot}: {area:.4f} ha > {max_area_allowed:.4f} ha (VIOLATION INEVITABLE!)")
        
        if not problematic_plots:
            print(f"    ✅ All plots are smaller than max allowed area")
        else:
            print(f"\n  ❌ Found {len(problematic_plots)} plots that CANNOT satisfy max area constraint!")
            print(f"     These plots will cause constraint violations no matter what penalty is used!")
    
    # Create CQM to get constraint metadata
    print(f"\nCREATING CQM...")
    cqm, (X, Y), constraint_metadata = create_cqm(land_data, foods, food_groups, config)
    
    # Report constraints
    print(f"\nCONSTRAINT SUMMARY:")
    print(f"  Total CQM constraints: {len(cqm.constraints)}")
    print(f"  Total variables: {len(cqm.variables)}")
    print(f"    - X variables (plot-crop): {len(X)}")
    print(f"    - Y variables (crop selection): {len(Y)}")
    
    print(f"\nCONSTRAINT TYPES:")
    
    # At most one per plot
    n_one_per_plot = len(constraint_metadata['at_most_one_per_plot'])
    print(f"\n  1. AT MOST ONE CROP PER PLOT: {n_one_per_plot} constraints")
    print(f"     For each plot p: sum_c X_{{p,c}} <= 1")
    for plot in list(constraint_metadata['at_most_one_per_plot'].keys())[:3]:
        print(f"       Example: {plot}")
    if n_one_per_plot > 3:
        print(f"       ... and {n_one_per_plot - 3} more")
    
    # X-Y linking
    n_xy_link = len(constraint_metadata['x_y_linking'])
    print(f"\n  2. X-Y LINKING: {n_xy_link} constraints")
    print(f"     For each plot p, crop c: X_{{p,c}} <= Y_c")
    examples = list(constraint_metadata['x_y_linking'].keys())[:3]
    for plot, crop in examples:
        print(f"       Example: X_{{{plot},{crop}}} <= Y_{crop}")
    if n_xy_link > 3:
        print(f"       ... and {n_xy_link - 3} more")
    
    # Y activation
    n_y_activation = len(constraint_metadata['y_activation'])
    print(f"\n  3. Y ACTIVATION: {n_y_activation} constraints")
    print(f"     For each crop c: Y_c <= sum_p X_{{p,c}}")
    for crop in list(foods.keys())[:3]:
        print(f"       Example: Y_{crop} <= sum_p X_{{p,{crop}}}")
    if n_y_activation > 3:
        print(f"       ... and {n_y_activation - 3} more")
    
    # Area bounds - minimum
    n_min_area = len(constraint_metadata['area_bounds_min'])
    print(f"\n  4. MINIMUM AREA BOUNDS: {n_min_area} constraints")
    if n_min_area > 0:
        print(f"     For each crop c with minimum: sum_p (area_p * X_{{p,c}}) >= min_area_c")
        for crop, meta in list(constraint_metadata['area_bounds_min'].items())[:3]:
            print(f"       {crop}: >= {meta['min_area']:.4f} ha")
        if n_min_area > 3:
            print(f"       ... and {n_min_area - 3} more")
    else:
        print(f"     (No minimum area constraints)")
    
    # Area bounds - maximum
    n_max_area = len(constraint_metadata['area_bounds_max'])
    print(f"\n  5. MAXIMUM AREA BOUNDS: {n_max_area} constraints")
    if n_max_area > 0:
        print(f"     For each crop c with maximum: sum_p (area_p * X_{{p,c}}) <= max_area_c")
        for crop, meta in constraint_metadata['area_bounds_max'].items():
            max_area = meta['max_area']
            print(f"       {crop}: <= {max_area:.4f} ha")
            
            # Check if any single plot exceeds this
            violating_plots = [p for p, a in land_data.items() if a > max_area]
            if violating_plots:
                print(f"         ⚠️ WARNING: {len(violating_plots)} plots exceed this limit!")
                for p in violating_plots[:2]:
                    print(f"           - {p} ({land_data[p]:.4f} ha) > {max_area:.4f} ha")
    else:
        print(f"     (No maximum area constraints)")
    
    # Food group constraints
    n_fg_min = len(constraint_metadata['food_group_min'])
    n_fg_max = len(constraint_metadata['food_group_max'])
    print(f"\n  6. FOOD GROUP CONSTRAINTS:")
    print(f"     Minimum constraints: {n_fg_min}")
    print(f"     Maximum constraints: {n_fg_max}")
    if n_fg_min > 0 or n_fg_max > 0:
        for group, meta in constraint_metadata['food_group_min'].items():
            print(f"       {group}: >= {meta['min_foods']} crops from {meta['foods_in_group']}")
        for group, meta in constraint_metadata['food_group_max'].items():
            print(f"       {group}: <= {meta['max_foods']} crops from {meta['foods_in_group']}")
    else:
        print(f"     (No food group constraints)")
    
    return {
        'scenario': scenario_name,
        'n_plots': len(land_data),
        'total_area': total_area,
        'n_foods': len(foods),
        'n_constraints': len(cqm.constraints),
        'n_variables': len(cqm.variables),
        'constraint_counts': {
            'at_most_one_per_plot': n_one_per_plot,
            'x_y_linking': n_xy_link,
            'y_activation': n_y_activation,
            'min_area': n_min_area,
            'max_area': n_max_area,
            'food_group_min': n_fg_min,
            'food_group_max': n_fg_max
        }
    }

def main():
    """Main execution."""
    print("="*80)
    print("CONSTRAINT ANALYSIS REPORT")
    print("="*80)
    
    # Load base food data
    try:
        food_list, foods, food_groups, _ = load_food_data('simple')
    except Exception as e:
        print(f"Warning: Food data loading failed ({e}), using fallback")
        # Fallback foods
        foods = {
            'Wheat': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 'affordability': 0.9, 'sustainability': 0.7},
            'Corn': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.5, 'affordability': 0.8, 'sustainability': 0.6},
            'Rice': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.7, 'affordability': 0.7, 'sustainability': 0.8},
            'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.4, 'affordability': 0.6, 'sustainability': 0.9},
            'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.8, 'affordability': 0.9, 'sustainability': 0.6},
            'Apples': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.6, 'affordability': 0.7, 'sustainability': 0.7}
        }
        food_groups = {
            'grains': ['Wheat', 'Corn', 'Rice'],
            'proteins': ['Soybeans'],
            'other': ['Potatoes', 'Apples']
        }
    
    # Configuration
    def create_config(land_data):
        total_land = sum(land_data.values())
        return {
            'parameters': {
                'land_availability': land_data,
                'minimum_planting_area': {food: 0.0 for food in foods},
                'max_percentage_per_crop': {food: 0.4 for food in foods},  # 40% max
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
    
    # Generate test data
    print("\nGenerating test samples...")
    
    # Farm scenario (5 farms)
    farms = generate_farms_large(n_farms=5, seed=42)
    farm_config = create_config(farms)
    
    # Patch scenario (5 patches)
    patches = generate_patches_small(n_farms=5, seed=42)
    patch_config = create_config(patches)
    
    # Analyze both scenarios
    results = []
    
    farm_result = analyze_scenario("FARM SCENARIO (5 farms)", farms, foods, food_groups, farm_config)
    results.append(farm_result)
    
    patch_result = analyze_scenario("PATCH SCENARIO (5 patches)", patches, foods, food_groups, patch_config)
    results.append(patch_result)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<30} {'Farm':<20} {'Patch':<20}")
    print("-" * 70)
    print(f"{'Number of plots':<30} {farm_result['n_plots']:<20} {patch_result['n_plots']:<20}")
    print(f"{'Total area (ha)':<30} {farm_result['total_area']:<20.4f} {patch_result['total_area']:<20.4f}")
    print(f"{'Number of foods':<30} {farm_result['n_foods']:<20} {patch_result['n_foods']:<20}")
    print(f"{'Total constraints':<30} {farm_result['n_constraints']:<20} {patch_result['n_constraints']:<20}")
    print(f"{'Total variables':<30} {farm_result['n_variables']:<20} {patch_result['n_variables']:<20}")
    
    print(f"\nConstraint breakdown:")
    for constraint_type in ['at_most_one_per_plot', 'x_y_linking', 'y_activation', 'min_area', 'max_area']:
        farm_count = farm_result['constraint_counts'][constraint_type]
        patch_count = patch_result['constraint_counts'][constraint_type]
        print(f"  {constraint_type:<28} {farm_count:<20} {patch_count:<20}")
    
    # Save report
    output_file = "CONSTRAINT_ANALYSIS_REPORT.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed report saved to: {output_file}")

if __name__ == "__main__":
    main()
