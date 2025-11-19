#!/usr/bin/env python3
"""
Quick debug script to compare farm vs plot formulations
"""
import os
import sys

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

try:
    import solver_runner_BINARY as srb
    from Utils import patch_sampler
    from Utils.farm_sampler import generate_farms as generate_farms_continuous
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Simple scenario: 10 units, 5 foods
total_land = 100.0
n_units = 50

# Generate farm and patch data
farm_land = generate_farms_continuous(n_farms=n_units, seed=42)
# Scale to match total land
current_total = sum(farm_land.values())
scale_factor = total_land / current_total
farm_land = {k: v * scale_factor for k, v in farm_land.items()}

patch_land = patch_sampler.generate_grid(n_farms=n_units, area=total_land, seed=42)

# Simple food data
foods = {
    'Wheat': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.3, 'affordability': 0.8, 'sustainability': 0.7},
    'Corn': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.4, 'affordability': 0.9, 'sustainability': 0.6},
    'Rice': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 'affordability': 0.7, 'sustainability': 0.5},
}
food_groups = {'Grains': ['Wheat', 'Corn', 'Rice']}

# Simple config - REMOVE food group constraints to isolate the issue
config = {
    'parameters': {
        'land_availability': farm_land,  # Will be replaced for plots
        'minimum_planting_area': {food: 0.01 for food in foods},
        'food_group_constraints': {},  # NO food group constraints
        'weights': {'nutritional_value': 0.25, 'nutrient_density': 0.25, 'environmental_impact': 0.25, 'affordability': 0.125, 'sustainability': 0.125}
    }
}

print("="*60)
print("DEBUGGING FARM vs PLOT FORMULATIONS")
print("="*60)
print(f"Units: {n_units}")
print(f"Total Land: {total_land}")
print(f"Foods: {list(foods.keys())}")

# Test farm formulation
print(f"\n{'-'*30}")
print("FARM FORMULATION (continuous areas + binary choices)")
print(f"{'-'*30}")
config['parameters']['land_availability'] = farm_land
print(f"Farm sizes: {[f'{k}: {v:.2f}' for k, v in list(farm_land.items())[:3]]}")

import time
start = time.time()
model_farm, results_farm = srb.solve_with_pulp_farm(list(farm_land.keys()), foods, food_groups, config)
time_farm = time.time() - start

print(f"Status: {results_farm.get('status', 'Unknown')}")
print(f"Objective: {results_farm.get('objective_value', None):.6f}")
print(f"Time: {time_farm:.3f}s")

# Test plot formulation  
print(f"\n{'-'*30}")
print("PLOT FORMULATION (binary assignments only)")
print(f"{'-'*30}")
config['parameters']['land_availability'] = patch_land
plot_area = list(patch_land.values())[0]
print(f"Plot size: {plot_area:.4f} ha each")

start = time.time()
model_plot, results_plot = srb.solve_with_pulp_plots(list(patch_land.keys()), foods, food_groups, config)
time_plot = time.time() - start

print(f"Status: {results_plot.get('status', 'Unknown')}")
print(f"Objective: {results_plot.get('objective_value', None):.6f}")
print(f"Time: {time_plot:.3f}s")

# Compare
if results_farm.get('objective_value') and results_plot.get('objective_value'):
    obj_farm = results_farm['objective_value']
    obj_plot = results_plot['objective_value']
    gap = ((obj_farm - obj_plot) / obj_farm) * 100 if obj_farm > 0 else 0
    print(f"\n{'-'*30}")
    print("COMPARISON")
    print(f"{'-'*30}")
    print(f"Farm Objective:  {obj_farm:.6f}")
    print(f"Plot Objective:  {obj_plot:.6f}")
    print(f"Gap: {gap:.2f}%")
    print(f"Time Ratio: {time_plot/time_farm:.2f}x")

    # Check solutions
    print(f"\nFarm solution summary:")
    areas = results_farm.get('areas', {})
    choices = results_farm.get('selections', {})
    total_area_used = 0
    for crop in foods:
        crop_area = sum(areas.get((f, crop), 0) for f in farm_land.keys())
        crop_choices = sum(choices.get((f, crop), 0) for f in farm_land.keys())
        total_area_used += crop_area
        print(f"  {crop}: {crop_area:.2f} ha ({crop_choices} farm selections)")
    print(f"  Total used: {total_area_used:.2f} ha / {total_land:.2f} ha ({total_area_used/total_land*100:.1f}%)")
    
    print(f"\nPlot solution summary:")
    plantations = results_plot.get('plantations', {})
    total_assigned_plots = 0
    for crop in foods:
        crop_plots = sum(plantations.get((f, crop), 0) for f in patch_land.keys())
        crop_area = crop_plots * plot_area
        total_assigned_plots += crop_plots
        print(f"  {crop}: {crop_area:.2f} ha ({crop_plots} plot assignments)")
    print(f"  Total assigned plots: {total_assigned_plots}/{len(patch_land.keys())} ({total_assigned_plots/len(patch_land.keys())*100:.1f}%)")
    
    # Analyze why the objectives are different
    print(f"\nObjective analysis:")
    print("Farm formulation allows:")
    print("  - Multiple crops per farm (fractional areas)")
    print("  - Optimal continuous allocation")
    print("Plot formulation restricts:")
    print("  - At most one crop per plot (binary choice)")
    print("  - Must use entire plot area when assigning")
    
    if total_area_used == 0 and total_assigned_plots == 0:
        print("\nBoth solutions plant nothing, yet have different objectives.")
        print("This suggests the objective calculation handles the 'no planting' case differently.")