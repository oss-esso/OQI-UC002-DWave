"""
Check constraint satisfaction for all cached DWave results.
"""
import json
from pathlib import Path
import os
import sys

# Add Benchmark Scripts to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

from benchmark_scalability_PATCH import (
    load_full_family_with_n_patches, 
    calculate_objective_from_bqm_sample
)
from solver_runner_PATCH import create_cqm
from dimod import cqm_to_bqm
from .constraint_validator import validate_bqm_patch_constraints
from dwave.system import LeapHybridBQMSampler
import os

configs = [5, 10, 15, 25]
dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')

print("\n" + "="*120)
print("CONSTRAINT SATISFACTION CHECK FOR ALL CONFIGURATIONS")
print("="*120)
print(f"{'Config':<10} {'Objective':<15} {'Total Const':<12} {'Violations':<12} {'Status':<20} {'Details'}")
print("-"*120)

for n_patches in configs:
    try:
        # Load scenario
        patches, foods, food_groups, config = load_full_family_with_n_patches(n_patches, seed=42 + 1)
        
        # Create CQM and BQM
        cqm, (X, Y), _ = create_cqm(patches, foods, food_groups, config)
        bqm, invert = cqm_to_bqm(cqm)
        
        # Solve with DWave
        sampler = LeapHybridBQMSampler(token=dwave_token)
        sampleset = sampler.sample(bqm, label=f"Constraint Check {n_patches}")
        
        if len(sampleset) > 0:
            best = sampleset.first
            
            # Calculate objective
            objective = calculate_objective_from_bqm_sample(best.sample, invert, patches, foods, config)
            
            # Validate constraints
            validation = validate_bqm_patch_constraints(
                best.sample, invert, patches, foods, food_groups, config
            )
            
            summary = validation['summary']
            total_const = summary['total_constraints']
            total_viol = summary['total_violations']
            
            status = "✅ ALL SATISFIED" if validation['all_satisfied'] else "❌ VIOLATIONS"
            
            # Get violation details
            details = []
            for const_type, const_data in validation['constraint_details'].items():
                if const_data['num_violations'] > 0:
                    details.append(f"{const_type}:{const_data['num_violations']}")
            
            details_str = ", ".join(details) if details else "-"
            
            print(f"{n_patches:<10} {objective:<15.6f} {total_const:<12} {total_viol:<12} {status:<20} {details_str}")
        else:
            print(f"{n_patches:<10} {'N/A':<15} {'N/A':<12} {'N/A':<12} {'❌ NO SOLUTION':<20} -")
            
    except Exception as e:
        print(f"{n_patches:<10} {'ERROR':<15} {'ERROR':<12} {'ERROR':<12} {'❌ EXCEPTION':<20} {str(e)[:40]}")

print("="*120)
print("\nKEY INSIGHTS:")
print("- Small violations in area bounds are expected due to BQM discretization")
print("- Penalty weights in cqm_to_bqm() can be tuned to reduce violations")
print("- Most important: structural constraints (X-Y linking, one per plot) should be satisfied")
print("="*120)
