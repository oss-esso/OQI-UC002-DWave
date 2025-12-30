#!/usr/bin/env python3
"""
Debug script for hierarchical quantum solver.

Uses SimulatedAnnealing (no QPU) to save resources.
Tests on smallest scenario to identify root cause of:
1. Negative objectives (~-18 to -43)
2. 999 violations marker (catastrophic failure)

Author: Debug session Dec 29, 2025
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("HIERARCHICAL SOLVER DEBUG")
print("="*80)
print()

# ============================================================================
# STEP 1: Load smallest scenario
# ============================================================================
print("[STEP 1] Loading smallest 27-food scenario...")

from src.scenarios import load_food_data
from data_loader_utils import load_food_data_as_dict

# Load the same scenario that fails in benchmarks
data = load_food_data_as_dict('rotation_250farms_27foods')

# Use smallest subset (5 farms like benchmark's smallest hierarchical test)
n_farms = 5
data['farm_names'] = data['farm_names'][:n_farms]
data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
data['total_area'] = sum(data['land_availability'].values())

print(f"  Farms: {len(data['farm_names'])}")
print(f"  Foods: {len(data['food_names'])}")
print(f"  Total area: {data['total_area']:.2f}")
print()

# ============================================================================
# STEP 2: Run hierarchical solver with SA
# ============================================================================
print("[STEP 2] Running hierarchical solver with SimulatedAnnealing...")

from hierarchical_quantum_solver import solve_hierarchical, DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config['farms_per_cluster'] = 3  # Small clusters
config['num_reads'] = 50
config['num_iterations'] = 2

result = solve_hierarchical(
    data=data,
    config=config,
    use_qpu=False,  # Use SA, NOT QPU!
    verbose=True
)

print()
print("="*80)
print("DEBUG ANALYSIS")
print("="*80)

# ============================================================================
# STEP 3: Analyze the result structure
# ============================================================================
print("\n[STEP 3] Analyzing result structure...")

print(f"\nResult keys: {list(result.keys())}")
print(f"Objective: {result.get('objective')}")
print(f"Violations: {result.get('violations')}")
print(f"Success: {result.get('success')}")

# Check family_solution
family_sol = result.get('family_solution', {})
print(f"\nFamily solution type: {type(family_sol)}")
print(f"Family solution length: {len(family_sol)}")
if family_sol:
    print(f"Sample keys (first 5): {list(family_sol.keys())[:5]}")
    print(f"Sample values (first 5): {list(family_sol.values())[:5]}")

# Check crop_solution
crop_sol = result.get('crop_solution', {})
print(f"\nCrop solution type: {type(crop_sol)}")
print(f"Crop solution length: {len(crop_sol)}")
if crop_sol:
    print(f"Sample keys (first 5): {list(crop_sol.keys())[:5]}")
    print(f"Sample values (first 5): {list(crop_sol.values())[:5]}")

# ============================================================================
# STEP 4: Check what the benchmark validation expects vs what we have
# ============================================================================
print("\n[STEP 4] Checking solution format mismatch...")

print("\nBenchmark validate_solution expects variables like:")
print("  Y_f{farm_idx}_c{food_idx}_t{period}")
print("  Example: Y_f0_c0_t1, Y_f0_c1_t1, ...")

print("\nHierarchical solver returns:")
if family_sol:
    sample_key = list(family_sol.keys())[0]
    print(f"  family_solution key example: {sample_key}")
    print(f"  Key type: {type(sample_key)}")

if crop_sol:
    sample_key = list(crop_sol.keys())[0]
    print(f"  crop_solution key example: {sample_key}")
    print(f"  Key type: {type(sample_key)}")

# ============================================================================
# STEP 5: Calculate objective step by step to find where it goes negative
# ============================================================================
print("\n[STEP 5] Tracing objective calculation...")

from food_grouping import (
    aggregate_foods_to_families,
    create_family_rotation_matrix,
    FAMILY_ORDER,
)

# Get family data as the solver would
family_data = aggregate_foods_to_families(data)
families = family_data['food_names']
farm_names = family_data['farm_names']
food_benefits = family_data['food_benefits']
land_availability = family_data['land_availability']
total_area = family_data['total_area']

n_periods = 3
rotation_gamma = config.get('rotation_gamma', 0.25)
diversity_bonus = config.get('diversity_bonus', 0.15)
one_hot_penalty = config.get('one_hot_penalty', 3.0)

R = create_family_rotation_matrix(seed=42)

print(f"\nRotation matrix R shape: {R.shape}")
print(f"R min: {R.min():.4f}, R max: {R.max():.4f}, R mean: {R.mean():.4f}")
print(f"R diagonal: {np.diag(R)}")

# Calculate components
obj_benefit = 0.0
obj_rotation = 0.0
obj_diversity = 0.0
obj_penalty = 0.0

for farm in farm_names:
    area_frac = land_availability.get(farm, 1.0) / total_area
    
    # Benefit
    for c_idx, family in enumerate(families):
        for t in range(1, n_periods + 1):
            val = family_sol.get((farm, family, t), 0)
            if val:
                benefit_contrib = food_benefits.get(family, 0.5) * area_frac
                obj_benefit += benefit_contrib
    
    # Rotation synergies
    for t in range(2, n_periods + 1):
        for c1_idx, fam1 in enumerate(families):
            for c2_idx, fam2 in enumerate(families):
                v1 = family_sol.get((farm, fam1, t-1), 0)
                v2 = family_sol.get((farm, fam2, t), 0)
                if v1 and v2:
                    rot_contrib = rotation_gamma * R[c1_idx, c2_idx] * area_frac
                    obj_rotation += rot_contrib
    
    # Diversity
    for family in families:
        used = any(family_sol.get((farm, family, t), 0) for t in range(1, n_periods + 1))
        if used:
            obj_diversity += diversity_bonus * area_frac
    
    # One-hot penalty
    for t in range(1, n_periods + 1):
        count = sum(family_sol.get((farm, fam, t), 0) for fam in families)
        if count != 1:
            obj_penalty += one_hot_penalty * (count - 1) ** 2 * area_frac

print(f"\nObjective breakdown:")
print(f"  Base benefit:      {obj_benefit:+.4f}")
print(f"  Rotation synergy:  {obj_rotation:+.4f}")
print(f"  Diversity bonus:   {obj_diversity:+.4f}")
print(f"  One-hot penalty:   {-obj_penalty:+.4f}")
print(f"  ----------------------------")
print(f"  TOTAL:             {obj_benefit + obj_rotation + obj_diversity - obj_penalty:.4f}")
print(f"  Solver reported:   {result.get('objective'):.4f}")

# ============================================================================
# STEP 6: Check one-hot constraint satisfaction
# ============================================================================
print("\n[STEP 6] Checking one-hot constraint satisfaction...")

violation_details = []
for farm in farm_names:
    for t in range(1, n_periods + 1):
        count = sum(family_sol.get((farm, fam, t), 0) for fam in families)
        if count != 1:
            violation_details.append({
                'farm': farm,
                'period': t,
                'count': count,
                'expected': 1
            })

print(f"\nOne-hot violations: {len(violation_details)}")
if violation_details:
    print("First 10 violations:")
    for v in violation_details[:10]:
        print(f"  Farm {v['farm']}, Period {v['period']}: count={v['count']} (expected 1)")
else:
    print("  No one-hot violations! ✓")

# ============================================================================
# STEP 7: Analyze assignment distribution
# ============================================================================
print("\n[STEP 7] Assignment distribution analysis...")

assignments_per_farm_period = {}
for farm in farm_names:
    for t in range(1, n_periods + 1):
        key = (farm, t)
        count = sum(1 for fam in families if family_sol.get((farm, fam, t), 0) == 1)
        assignments_per_farm_period[key] = count

counts = list(assignments_per_farm_period.values())
print(f"Assignments per farm-period:")
print(f"  Min: {min(counts)}")
print(f"  Max: {max(counts)}")
print(f"  Mean: {np.mean(counts):.2f}")
print(f"  Distribution: {dict(zip(*np.unique(counts, return_counts=True)))}")

# ============================================================================
# STEP 8: Summary and recommendations
# ============================================================================
print("\n" + "="*80)
print("DEBUG SUMMARY")
print("="*80)

issues_found = []

if result.get('objective', 0) < 0:
    issues_found.append("CRITICAL: Objective is NEGATIVE")

if len(violation_details) > 0:
    issues_found.append(f"One-hot constraint violated: {len(violation_details)} violations")

if max(counts) > 2:
    issues_found.append(f"Too many assignments: max={max(counts)} per farm-period")

if min(counts) == 0:
    issues_found.append(f"Missing assignments: {counts.count(0)} farm-periods have no assignment")

if issues_found:
    print("\nISSUES FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✓ No obvious issues found - check objective calculation in BQM builder")

print("\n" + "="*80)
