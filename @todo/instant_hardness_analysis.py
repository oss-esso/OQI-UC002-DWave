#!/usr/bin/env python3
"""
Analyze problem characteristics WITHOUT solving (instant results)
Then correlate with known hard/easy instances
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from src.scenarios import load_food_data

N_PERIODS = 3
N_FAMILIES = 6

# Test configurations with KNOWN hardness from previous investigation
TESTS = [
    # Known HARD instances (from 100s timeout tests)
    {'n_farms': 5, 'scenario': 'rotation_micro_25', 'known': 'TIMEOUT'},
    {'n_farms': 15, 'scenario': 'rotation_medium_100', 'known': 'TIMEOUT'},
    {'n_farms': 20, 'scenario': 'rotation_medium_100', 'known': 'TIMEOUT'},
    {'n_farms': 25, 'scenario': 'rotation_large_200', 'known': 'TIMEOUT'},
    {'n_farms': 40, 'scenario': 'rotation_large_200', 'known': 'TIMEOUT'},
    
    # Known EASY instances (from 100s timeout tests)
    {'n_farms': 3, 'scenario': 'rotation_micro_25', 'known': 'FAST'},
    {'n_farms': 8, 'scenario': 'rotation_small_50', 'known': 'FAST'},
    {'n_farms': 30, 'scenario': 'rotation_large_200', 'known': 'FAST'},
    {'n_farms': 50, 'scenario': 'rotation_large_200', 'known': 'FAST'},
    {'n_farms': 100, 'scenario': 'rotation_large_200', 'known': 'FAST'},
]

print("="*100)
print("INSTANT HARDNESS ANALYSIS: Problem Characteristics vs Known Hardness")
print("="*100)

results = []

for test in TESTS:
    n_farms = test['n_farms']
    scenario = test['scenario']
    known = test['known']
    
    farms, foods, food_groups, config = load_food_data(scenario)
    params = config.get('parameters', {})
    land_full = params.get('land_availability', {})
    weights = params.get('weights', {})
    
    # Get subset
    all_farms = list(land_full.keys())[:n_farms]
    land_avail = {f: land_full[f] for f in all_farms}
    
    # Analyze characteristics
    total_area = sum(land_avail.values())
    areas = list(land_avail.values())
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)
    cv_area = std_area / mean_area if mean_area > 0 else 0
    
    n_vars = n_farms * N_FAMILIES * N_PERIODS
    farms_per_food = n_farms / N_FAMILIES
    area_per_farm = total_area / n_farms
    area_per_var = total_area / n_vars
    
    # Constraint analysis
    n_onehot = n_farms * N_PERIODS  # One constraint per farm-period
    n_rotation_quad = n_farms * (N_PERIODS - 1) * N_FAMILIES * N_FAMILIES  # Temporal quadratics
    k_neighbors = params.get('spatial_k_neighbors', 4)
    n_spatial_edges = (n_farms * min(k_neighbors, n_farms-1)) // 2
    n_spatial_quad = n_spatial_edges * N_PERIODS * N_FAMILIES * N_FAMILIES  # Spatial quadratics
    
    total_quad = n_rotation_quad + n_spatial_quad
    quad_per_var = total_quad / n_vars if n_vars > 0 else 0
    onehot_per_var = n_onehot / n_vars
    
    # Tightness metrics
    min_planting = params.get('minimum_planting_area', {})
    if min_planting:
        avg_min_plant = np.mean(list(min_planting.values()))
        tightness = avg_min_plant / mean_area if mean_area > 0 else 0
    else:
        avg_min_plant = 0
        tightness = 0
    
    results.append({
        'n_farms': n_farms,
        'n_vars': n_vars,
        'known': known,
        'total_area': total_area,
        'mean_area': mean_area,
        'cv_area': cv_area,
        'min_area': min_area,
        'max_area': max_area,
        'farms_per_food': farms_per_food,
        'area_per_farm': area_per_farm,
        'area_per_var': area_per_var,
        'quad_per_var': quad_per_var,
        'onehot_per_var': onehot_per_var,
        'tightness': tightness,
    })

# Separate by known hardness
hard = [r for r in results if r['known'] == 'TIMEOUT']
easy = [r for r in results if r['known'] == 'FAST']

print(f"\nAnalyzed {len(results)} instances:")
print(f"  HARD (timeout): {len(hard)}")
print(f"  EASY (fast): {len(easy)}")

# Show all data
print(f"\n{'='*100}")
print("RAW DATA")
print("="*100)
print(f"\n{'Farms':<7} {'Vars':<7} {'Status':<10} {'TotalArea':<10} {'CV':<8} {'F/Food':<8} {'Area/Var':<10}")
print("-"*100)

for r in sorted(results, key=lambda x: (x['known'], x['n_farms'])):
    print(f"{r['n_farms']:<7} {r['n_vars']:<7} {r['known']:<10} {r['total_area']:<10.1f} "
          f"{r['cv_area']:<8.3f} {r['farms_per_food']:<8.2f} {r['area_per_var']:<10.3f}")

# Comparative analysis
print(f"\n{'='*100}")
print("COMPARATIVE ANALYSIS: HARD vs EASY")
print("="*100)

metrics = [
    ('Total Area (ha)', 'total_area'),
    ('Mean Area/Farm (ha)', 'mean_area'),
    ('CV (variability)', 'cv_area'),
    ('Min Area (ha)', 'min_area'),
    ('Max Area (ha)', 'max_area'),
    ('Farms per Food', 'farms_per_food'),
    ('Area per Farm (ha)', 'area_per_farm'),
    ('Area per Variable', 'area_per_var'),
    ('Quadratics per Var', 'quad_per_var'),
    ('OneHot per Var', 'onehot_per_var'),
]

print(f"\n{'Metric':<25} {'HARD (avg)':<15} {'EASY (avg)':<15} {'Ratio':<10} {'Insight'}")
print("-"*100)

insights = []

for label, key in metrics:
    hard_avg = np.mean([r[key] for r in hard])
    easy_avg = np.mean([r[key] for r in easy])
    ratio = hard_avg / easy_avg if easy_avg != 0 else 0
    
    # Determine insight
    if ratio < 0.5:
        insight = "HARD << EASY"
        insights.append((label, 'LOWER', ratio))
    elif ratio > 2.0:
        insight = "HARD >> EASY"
        insights.append((label, 'HIGHER', ratio))
    else:
        insight = "Similar"
    
    print(f"{label:<25} {hard_avg:<15.3f} {easy_avg:<15.3f} {ratio:<10.2f}x  {insight}")

# Key findings
print(f"\n{'='*100}")
print("KEY FINDINGS")
print("="*100)

# Sort insights by magnitude of difference
insights.sort(key=lambda x: abs(x[2] - 1.0), reverse=True)

print(f"\nTop factors differentiating HARD from EASY problems:")
for i, (metric, direction, ratio) in enumerate(insights[:5], 1):
    if direction == 'LOWER':
        print(f"  {i}. {metric}: HARD has {ratio:.2f}x LOWER ({1/ratio:.2f}x less)")
    else:
        print(f"  {i}. {metric}: HARD has {ratio:.2f}x HIGHER")

# Specific ranges
print(f"\n{'='*100}")
print("HARDNESS THRESHOLDS")
print("="*100)

print(f"\nTotal Area:")
hard_areas = [r['total_area'] for r in hard]
easy_areas = [r['total_area'] for r in easy]
print(f"  HARD: {min(hard_areas):.1f} - {max(hard_areas):.1f} ha (mean={np.mean(hard_areas):.1f})")
print(f"  EASY: {min(easy_areas):.1f} - {max(easy_areas):.1f} ha (mean={np.mean(easy_areas):.1f})")

if max(hard_areas) < min(easy_areas):
    print(f"  >>> CLEAR SEPARATION: Hard problems all have area < {max(hard_areas):.1f} ha")
elif min(hard_areas) < np.percentile(easy_areas, 25):
    threshold = np.percentile(easy_areas, 25)
    print(f"  >>> PATTERN: Hard problems tend to have area < {threshold:.1f} ha")

print(f"\nCoefficient of Variation (CV):")
hard_cv = [r['cv_area'] for r in hard]
easy_cv = [r['cv_area'] for r in easy]
print(f"  HARD: {min(hard_cv):.3f} - {max(hard_cv):.3f} (mean={np.mean(hard_cv):.3f})")
print(f"  EASY: {min(easy_cv):.3f} - {max(easy_cv):.3f} (mean={np.mean(easy_cv):.3f})")

print(f"\nFarms per Food:")
hard_ratio = [r['farms_per_food'] for r in hard]
easy_ratio = [r['farms_per_food'] for r in easy]
print(f"  HARD: {min(hard_ratio):.2f} - {max(hard_ratio):.2f} (mean={np.mean(hard_ratio):.2f})")
print(f"  EASY: {min(easy_ratio):.2f} - {max(easy_ratio):.2f} (mean={np.mean(easy_ratio):.2f})")

print(f"\nArea per Variable:")
hard_apv = [r['area_per_var'] for r in hard]
easy_apv = [r['area_per_var'] for r in easy]
print(f"  HARD: {min(hard_apv):.3f} - {max(hard_apv):.3f} (mean={np.mean(hard_apv):.3f})")
print(f"  EASY: {min(easy_apv):.3f} - {max(easy_apv):.3f} (mean={np.mean(easy_apv):.3f})")

# Conclusion
print(f"\n{'='*100}")
print("CONCLUSION")
print("="*100)

print(f"\nBased on {len(hard)} HARD and {len(easy)} EASY instances:")

# Primary factor
area_ratio = np.mean(hard_areas) / np.mean(easy_areas)
if area_ratio < 0.6:
    print(f"\n1. PRIMARY FACTOR: TOTAL AREA")
    print(f"   HARD problems have {area_ratio:.2f}x LESS area than EASY problems")
    print(f"   HARD: {np.mean(hard_areas):.1f} ha average")
    print(f"   EASY: {np.mean(easy_areas):.1f} ha average")
    print(f"   → Small total area creates tight constraints!")

# Secondary factor
cv_ratio = np.mean(hard_cv) / np.mean(easy_cv)
if cv_ratio > 1.2 or cv_ratio < 0.8:
    print(f"\n2. SECONDARY FACTOR: LAND VARIABILITY (CV)")
    if cv_ratio > 1:
        print(f"   HARD problems have {cv_ratio:.2f}x HIGHER variability")
    else:
        print(f"   HARD problems have {cv_ratio:.2f}x LOWER variability")
    print(f"   HARD: CV={np.mean(hard_cv):.3f}")
    print(f"   EASY: CV={np.mean(easy_cv):.3f}")

# Constraint factor
farms_ratio = np.mean(hard_ratio) / np.mean(easy_ratio)
if farms_ratio < 0.7:
    print(f"\n3. CONSTRAINT FACTOR: FARMS/FOOD RATIO")
    print(f"   HARD problems have {farms_ratio:.2f}x FEWER farms per food")
    print(f"   HARD: {np.mean(hard_ratio):.2f} farms/food")
    print(f"   EASY: {np.mean(easy_ratio):.2f} farms/food")
    print(f"   → Fewer farms means tighter one-hot constraints!")

print(f"\n{'='*100}")
print("RECOMMENDATION FOR QUANTUM ADVANTAGE:")
print(f"Target problems with:")
print(f"  - Total area < {np.percentile(hard_areas, 75):.1f} ha")
print(f"  - CV around {np.mean(hard_cv):.2f}")
print(f"  - Farms/Food ratio < {np.percentile(hard_ratio, 75):.1f}")
print("="*100)
