"""
Test script to validate constraints on a single configuration.
"""
import os
import sys
from benchmark_scalability_PATCH import (
    load_full_family_with_n_patches, 
    calculate_objective_from_bqm_sample
)
from solver_runner_PATCH import create_cqm
from dimod import cqm_to_bqm
from constraint_validator import validate_bqm_patch_constraints, print_validation_report

# Test configuration
n_patches = 5
dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')

print("="*80)
print(f"CONSTRAINT VALIDATION TEST: {n_patches} Patches")
print("="*80)

# Load scenario
patches, foods, food_groups, config = load_full_family_with_n_patches(n_patches, seed=42 + 1)

print(f"\nLoaded scenario:")
print(f"  Patches: {len(patches)}")
print(f"  Foods: {len(foods)}")
print(f"  Food Groups: {len(food_groups)}")

# Create CQM and convert to BQM
print(f"\nCreating CQM and converting to BQM...")
cqm, (X, Y), constraint_metadata = create_cqm(patches, foods, food_groups, config)
bqm, invert = cqm_to_bqm(cqm)

print(f"  CQM Variables: {len(cqm.variables)}")
print(f"  CQM Constraints: {len(cqm.constraints)}")
print(f"  BQM Variables: {len(bqm.variables)}")

# Solve with DWave
print(f"\n" + "="*80)
print("SOLVING WITH DWAVE")
print("="*80)

from dwave.system import LeapHybridBQMSampler
sampler = LeapHybridBQMSampler(token=dwave_token)

import time
start = time.time()
sampleset = sampler.sample(bqm, label="Constraint Validation Test")
solve_time = time.time() - start

print(f"\nSolved in {solve_time:.2f}s")
print(f"Samples: {len(sampleset)}")

if len(sampleset) > 0:
    best = sampleset.first
    
    # Calculate objective
    objective = calculate_objective_from_bqm_sample(best.sample, invert, patches, foods, config)
    
    print(f"\nBest Solution:")
    print(f"  BQM Energy: {best.energy:.6f}")
    print(f"  Objective Value: {objective:.6f}")
    
    # Validate constraints
    print(f"\n" + "="*80)
    print("VALIDATING CONSTRAINTS")
    print("="*80)
    
    validation = validate_bqm_patch_constraints(
        best.sample, invert, patches, foods, food_groups, config
    )
    
    all_satisfied = print_validation_report(validation, verbose=True)
    
    if all_satisfied:
        print("\n" + "="*80)
        print("✅ SUCCESS: All constraints are satisfied!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ FAILURE: Some constraints are violated!")
        print("="*80)
        print("\nThis suggests the BQM conversion or penalty terms need adjustment.")
else:
    print("\n❌ No solutions returned!")
