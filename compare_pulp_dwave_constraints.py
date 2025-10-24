"""
Compare constraint satisfaction between PuLP and DWave solutions.
"""
import os
from benchmark_scalability_PATCH import (
    load_full_family_with_n_patches, 
    calculate_objective_from_bqm_sample
)
from solver_runner_PATCH import create_cqm, solve_with_pulp
from dimod import cqm_to_bqm
from constraint_validator import validate_bqm_patch_constraints, validate_pulp_patch_constraints, print_validation_report

# Test configuration
n_patches = 5
dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')

print("="*80)
print(f"CONSTRAINT COMPARISON: PuLP vs DWave ({n_patches} Patches)")
print("="*80)

# Load scenario
patches, foods, food_groups, config = load_full_family_with_n_patches(n_patches, seed=42 + 1)

print(f"\nScenario:")
print(f"  Patches: {len(patches)}")
print(f"  Foods: {len(foods)}")
print(f"  Food Groups: {len(food_groups)}")

# Create CQM
print(f"\nCreating CQM...")
cqm, (X, Y), constraint_metadata = create_cqm(patches, foods, food_groups, config)
print(f"  Variables: {len(cqm.variables)}")
print(f"  Constraints: {len(cqm.constraints)}")

# Solve with PuLP
print(f"\n" + "="*80)
print("SOLVING WITH PULP")
print("="*80)

import time
pulp_start = time.time()
pulp_model, pulp_results = solve_with_pulp(patches, foods, food_groups, config)
pulp_time = time.time() - pulp_start

print(f"\nSolved in {pulp_time:.2f}s")
print(f"Status: {pulp_results['status']}")
print(f"Objective: {pulp_results.get('objective_value', 'N/A')}")

# Validate PuLP constraints
print(f"\n" + "="*80)
print("VALIDATING PULP CONSTRAINTS")
print("="*80)

pulp_validation = validate_pulp_patch_constraints(
    pulp_results.get('X_variables', {}),
    pulp_results.get('Y_variables', {}),
    patches, foods, food_groups, config
)
pulp_satisfied = print_validation_report(pulp_validation, verbose=True)

# Convert to BQM and solve with DWave
print(f"\n" + "="*80)
print("SOLVING WITH DWAVE")
print("="*80)

print(f"\nConverting CQM to BQM...")
bqm, invert = cqm_to_bqm(cqm)
print(f"  BQM Variables: {len(bqm.variables)}")

from dwave.system import LeapHybridBQMSampler
sampler = LeapHybridBQMSampler(token=dwave_token)

dwave_start = time.time()
sampleset = sampler.sample(bqm, label="PuLP vs DWave Comparison")
dwave_time = time.time() - dwave_start

print(f"\nSolved in {dwave_time:.2f}s")
print(f"Samples: {len(sampleset)}")

if len(sampleset) > 0:
    best = sampleset.first
    objective = calculate_objective_from_bqm_sample(best.sample, invert, patches, foods, config)
    
    print(f"BQM Energy: {best.energy:.6f}")
    print(f"Objective: {objective:.6f}")
    
    # Validate DWave constraints
    print(f"\n" + "="*80)
    print("VALIDATING DWAVE CONSTRAINTS")
    print("="*80)
    
    dwave_validation = validate_bqm_patch_constraints(
        best.sample, invert, patches, foods, food_groups, config
    )
    dwave_satisfied = print_validation_report(dwave_validation, verbose=True)
    
    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<35} {'PuLP':<20} {'DWave':<20}")
    print("-"*80)
    
    # Objectives
    pulp_obj = pulp_results.get('objective_value', 0)
    dwave_obj = objective
    obj_diff = abs(pulp_obj - dwave_obj)
    obj_gap = (obj_diff / pulp_obj * 100) if pulp_obj else 0
    
    print(f"{'Objective Value':<35} {pulp_obj:<20.6f} {dwave_obj:<20.6f}")
    print(f"{'Objective Gap':<35} {'-':<20} {obj_gap:<20.2f}%")
    
    # Solve times
    print(f"{'Solve Time (s)':<35} {pulp_time:<20.4f} {dwave_time:<20.4f}")
    
    # Constraint satisfaction
    pulp_violations = pulp_validation['summary']['total_violations']
    dwave_violations = dwave_validation['summary']['total_violations']
    pulp_satisfaction = ((pulp_validation['summary']['total_constraints'] - pulp_violations) / 
                        pulp_validation['summary']['total_constraints'] * 100)
    dwave_satisfaction = ((dwave_validation['summary']['total_constraints'] - dwave_violations) / 
                         dwave_validation['summary']['total_constraints'] * 100)
    
    print(f"{'Total Constraints':<35} {pulp_validation['summary']['total_constraints']:<20} {dwave_validation['summary']['total_constraints']:<20}")
    print(f"{'Violations':<35} {pulp_violations:<20} {dwave_violations:<20}")
    print(f"{'Satisfaction %':<35} {pulp_satisfaction:<20.2f}% {dwave_satisfaction:<20.2f}%")
    
    # Area utilization
    pulp_util = pulp_validation['summary']['area_utilization']
    dwave_util = dwave_validation['summary']['area_utilization']
    print(f"{'Area Utilization':<35} {pulp_util:<20.2f}% {dwave_util:<20.2f}%")
    
    # Crops selected
    print(f"{'Crops Selected':<35} {pulp_validation['summary']['crops_selected']:<20} {dwave_validation['summary']['crops_selected']:<20}")
    
    # Detailed comparison of violations
    print("\n" + "-"*80)
    print("CONSTRAINT BREAKDOWN")
    print("-"*80)
    
    constraint_types = [
        ('at_most_one_per_plot', 'At Most One Crop Per Plot'),
        ('x_y_linking', 'X-Y Linking'),
        ('y_activation', 'Y Activation'),
        ('area_bounds', 'Area Bounds'),
        ('food_group_diversity', 'Food Group Diversity')
    ]
    
    print(f"\n{'Constraint Type':<35} {'PuLP Violations':<20} {'DWave Violations':<20}")
    print("-"*80)
    
    for key, name in constraint_types:
        pulp_v = pulp_validation['constraint_details'][key]['num_violations']
        dwave_v = dwave_validation['constraint_details'][key]['num_violations']
        
        pulp_status = "✅" if pulp_v == 0 else f"❌ {pulp_v}"
        dwave_status = "✅" if dwave_v == 0 else f"❌ {dwave_v}"
        
        print(f"{name:<35} {pulp_status:<20} {dwave_status:<20}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if pulp_satisfied and dwave_satisfied:
        print("\n✅ EXCELLENT: Both solvers produce fully feasible solutions!")
    elif pulp_satisfied and not dwave_satisfied:
        print(f"\n⚠️  GOOD: PuLP is fully feasible, DWave has {dwave_violations} minor violations")
        print("   This is expected due to BQM discretization - violations are likely small")
    elif not pulp_satisfied and dwave_satisfied:
        print("\n🎉 SURPRISING: DWave is fully feasible while PuLP has violations!")
    else:
        print(f"\n⚠️  BOTH HAVE VIOLATIONS: PuLP={pulp_violations}, DWave={dwave_violations}")
        print("   Check problem formulation and constraints")
    
    print("\n" + "="*80)
else:
    print("\n❌ DWave returned no solutions!")
