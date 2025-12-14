#!/usr/bin/env python3
"""
Generate scenarios by replicating the hard land distribution pattern
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Utils.farm_sampler import generate_farms
import numpy as np

# Generate the base hard scenario (20 farms, seed=10001)
print("="*80)
print("REPLICATING HARD LAND DISTRIBUTION PATTERN")
print("="*80)

# Base scenario (20 farms with seed=10001 - known to be hard)
base_farms = generate_farms(n_farms=20, total_area=100.0, seed=10001)
base_areas = list(base_farms.values())

print(f"\nBase distribution (20 farms, seed=10001, 100 ha total):")
print(f"  Mean: {np.mean(base_areas):.2f} ha")
print(f"  Std: {np.std(base_areas):.2f} ha")
print(f"  Min: {min(base_areas):.2f} ha")
print(f"  Max: {max(base_areas):.2f} ha")
print(f"  Areas: {[f'{a:.2f}' for a in sorted(base_areas)]}")

# Strategy: Replicate this pattern for larger farm counts
def replicate_pattern(base_areas, target_n_farms, target_total_area):
    """Replicate the base pattern to create larger scenarios"""
    # Repeat the pattern as many times as needed
    n_base = len(base_areas)
    n_repeats = (target_n_farms + n_base - 1) // n_base
    
    replicated = base_areas * n_repeats
    replicated = replicated[:target_n_farms]  # Trim to exact size
    
    # Scale to target total area
    current_total = sum(replicated)
    scale_factor = target_total_area / current_total
    replicated = [a * scale_factor for a in replicated]
    
    return replicated

# Test different sizes
for n_farms, target_area in [(20, 100), (50, 250), (90, 450), (225, 1125)]:
    areas = replicate_pattern(base_areas, n_farms, target_area)
    
    print(f"\n{n_farms} farms (target {target_area} ha):")
    print(f"  Actual total: {sum(areas):.2f} ha")
    print(f"  Mean: {np.mean(areas):.2f} ha")
    print(f"  Std: {np.std(areas):.2f} ha")
    print(f"  Min: {min(areas):.2f} ha")
    print(f"  Max: {max(areas):.2f} ha")
    print(f"  CoV: {np.std(areas)/np.mean(areas):.3f} (same variability)")

print(f"\n{'='*80}")
print("KEY INSIGHT:")
print("By replicating the hard distribution pattern,")
print("we maintain the same coefficient of variation (variability)")
print("across all problem sizes, ensuring consistent hardness!")
print("="*80)
