#!/usr/bin/env python3
"""
Targeted Test for Patch DWave CQM Constraint Violations

This script:
1. Builds a Patch CQM scenario (10 plots) exactly as done in comprehensive_benchmark
2. Inspects all constraints in detail
3. Solves with LeapHybridCQMSampler (CQM direct, not BQM conversion)
4. Validates the solution and reports constraint violations

Purpose: Diagnose why Patch DWave CQM is violating "at most one crop per plot" constraints
despite having them explicitly defined in the CQM.
"""

import os
import sys
import json
from dimod import ConstrainedQuadraticModel, Binary
from dwave.system import LeapHybridCQMSampler

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from patch_sampler import generate_farms as generate_patches_small
from src.scenarios import load_food_data
import solver_runner_BINARY as solver_runner

def main():
    print("=" * 80)
    print("TARGETED TEST: Patch DWave CQM Constraint Validation")
    print("=" * 80)
    
    # Configuration
    n_plots = 10
    fixed_total_land = 100.0
    seed = 42
    
    # 1. Generate patch data (even grid)
    print(f"\n1. Generating patch data...")
    print(f"   N plots: {n_plots}")
    print(f"   Total land: {fixed_total_land} ha")
    print(f"   Seed: {seed}")
    
    patches_unscaled = generate_patches_small(n_farms=n_plots, seed=seed)
    patches_total = sum(patches_unscaled.values())
    patch_scale_factor = fixed_total_land / patches_total if patches_total > 0 else 0
    patches_scaled = {k: v * patch_scale_factor for k, v in patches_unscaled.items()}
    
    plot_area = fixed_total_land / n_plots
    print(f"   ✓ Generated {len(patches_scaled)} plots")
    print(f"   Area per plot: {plot_area:.4f} ha")
    print(f"   Total area: {sum(patches_scaled.values()):.4f} ha")
    
    # 2. Load food data
    print(f"\n2. Loading food data...")
    food_list, foods, food_groups, _ = load_food_data('full_family')
    print(f"   ✓ Loaded {len(foods)} foods")
    print(f"   ✓ Loaded {len(food_groups)} food groups")
    
    # 3. Create configuration
    print(f"\n3. Creating configuration...")
    config = {
        'parameters': {
            'land_availability': patches_scaled,
            'minimum_planting_area': {food: 0.0001 for food in foods},
            'food_group_constraints': {
                group: {'min_foods': 1, 'max_foods': len(food_list)}
                for group, food_list in food_groups.items()
            },
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.2,
                'environmental_impact': 0.25,
                'affordability': 0.15,
                'sustainability': 0.15
            },
            'idle_penalty_lambda': 0.0
        }
    }
    print(f"   ✓ Configuration created")
    
    # 4. Build CQM using create_cqm_plots (binary formulation)
    print(f"\n4. Building CQM (binary formulation)...")
    plots_list = list(patches_scaled.keys())
    
    cqm, Y, constraint_metadata = solver_runner.create_cqm_plots(
        plots_list, foods, food_groups, config
    )
    
    print(f"   ✓ CQM built successfully")
    print(f"   Variables: {len(cqm.variables)} (all binary)")
    print(f"   Constraints: {len(cqm.constraints)}")
    
    # 5. Inspect constraints in detail
    print(f"\n5. Inspecting CQM constraints in detail...")
    print(f"\n   Constraint breakdown:")
    
    # Group constraints by type
    constraint_types = {}
    for label, constraint in cqm.constraints.items():
        if label.startswith('Max_Assignment_'):
            constraint_types.setdefault('plot_assignment', []).append((label, constraint))
        elif label.startswith('Min_Plots_'):
            constraint_types.setdefault('min_plots', []).append((label, constraint))
        elif label.startswith('Max_Plots_'):
            constraint_types.setdefault('max_plots', []).append((label, constraint))
        elif label.startswith('Food_Group_'):
            constraint_types.setdefault('food_group', []).append((label, constraint))
        else:
            constraint_types.setdefault('other', []).append((label, constraint))
    
    for ctype, constraints in constraint_types.items():
        print(f"   - {ctype}: {len(constraints)} constraints")
    
    # Examine plot assignment constraints (the critical ones)
    print(f"\n   Detailed inspection of plot assignment constraints:")
    plot_constraints = constraint_types.get('plot_assignment', [])
    
    for i, (label, constraint) in enumerate(plot_constraints[:3]):  # Show first 3
        print(f"\n   Constraint {i+1}: {label}")
        print(f"     Type: {constraint.sense}")
        print(f"     RHS: {constraint.rhs}")
        
        # Get the linear part of the constraint
        linear = constraint.lhs.linear
        print(f"     Number of variables: {len(linear)}")
        print(f"     Variables involved:")
        for var, coeff in list(linear.items())[:5]:  # Show first 5 variables
            print(f"       {var}: coefficient={coeff}")
        if len(linear) > 5:
            print(f"       ... and {len(linear) - 5} more variables")
    
    # 6. Save CQM for inspection
    print(f"\n6. Saving CQM to file...")
    cqm_path = 'test_patch_cqm_10plots.cqm'
    with open(cqm_path, 'wb') as f:
        import shutil
        shutil.copyfileobj(cqm.to_file(), f)
    print(f"   ✓ Saved to: {cqm_path}")
    
    # 7. Solve with DWave CQM (direct, not BQM conversion)
    print(f"\n7. Solving with DWave LeapHybridCQMSampler...")
    
    dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-7b81782896495d7c6a061bda257a9d9b03b082cd')
    
    print(f"   Submitting to DWave Leap...")
    sampler = LeapHybridCQMSampler(token=dwave_token)
    
    sampleset = sampler.sample_cqm(
        cqm, 
        label="Patch CQM Constraint Test - Direct CQM Solve"
    )
    
    # Extract timing
    # Extract timing from sampleset.info
    timing_info = sampleset.info.get('timing', {})
    
    # Hybrid solve time (total time including QPU)
    hybrid_time = (timing_info.get('run_time') or 
                  sampleset.info.get('run_time') or
                  timing_info.get('charge_time') or
                  sampleset.info.get('charge_time'))
    
    if hybrid_time is not None:
        hybrid_time = hybrid_time / 1e6  # Convert from microseconds to seconds
    
    # QPU access time
    qpu_time = (timing_info.get('qpu_access_time') or
               sampleset.info.get('qpu_access_time'))
    
    if qpu_time is not None:
        qpu_time = qpu_time / 1e6  # Convert from microseconds to seconds
    
    print(f"   ✓ Solved in {hybrid_time:.3f}s")
    print(f"   QPU time: {qpu_time:.4f}s")
    print(f"   Total samples: {len(sampleset)}")
    
    # 8. Extract and analyze best solution
    print(f"\n8. Analyzing best solution...")
    
    if len(sampleset) > 0:
        best = sampleset.first
        is_feasible = best.is_feasible
        cqm_sample = dict(best.sample)
        energy = best.energy
        
        print(f"   Feasibility: {is_feasible}")
        print(f"   Energy: {energy:.6f}")
        
        # Count selected crops per plot
        print(f"\n   Crops per plot:")
        plot_crops = {}
        for var_name, value in cqm_sample.items():
            if value > 0.5:  # Binary variable selected
                # Variable format: Y_PlotX_FoodName
                parts = var_name.split('_', 2)  # Split into ['Y', 'PlotX', 'FoodName']
                if len(parts) == 3 and parts[0] == 'Y':
                    plot = parts[1]
                    food = parts[2]
                    plot_crops.setdefault(plot, []).append(food)
        
        for plot in sorted(plots_list):
            crops = plot_crops.get(plot, [])
            status = "✓" if len(crops) <= 1 else "❌"
            print(f"     {status} {plot}: {len(crops)} crops - {crops[:3]}")
            if len(crops) > 3:
                print(f"        ... and {len(crops) - 3} more")
        
        # Check constraint violations
        print(f"\n   Checking CQM constraint violations...")
        violated_constraints = []
        
        # Specifically check plot assignment constraints
        print(f"\n   Detailed check of plot assignment constraints:")
        for label, constraint in cqm.constraints.items():
            if label.startswith('Max_Assignment_'):
                lhs_value = constraint.lhs.energy(cqm_sample)
                rhs_value = constraint.rhs
                sense = constraint.sense
                
                # The constraint is: sum(Y) - 1 <= 0, which means sum(Y) <= 1
                # With offset=-1 and RHS=0, we evaluate: (sum of selected Y values) - 1 <= 0
                plot_name = label.replace('Max_Assignment_', '')
                n_crops = sum(1 for v in plot_crops.get(plot_name, []))
                
                is_violated = lhs_value > rhs_value + 1e-6
                status = "❌ VIOLATED" if is_violated else "✓ OK"
                
                print(f"     {status} {label}: LHS={lhs_value:.4f}, RHS={rhs_value:.4f}, crops={n_crops}")
                
                if is_violated:
                    violated_constraints.append({
                        'label': label,
                        'sense': sense,
                        'lhs': lhs_value,
                        'rhs': rhs_value,
                        'violation': abs(lhs_value - rhs_value),
                        'n_crops': n_crops
                    })
        
        # Check all other constraints
        for label, constraint in cqm.constraints.items():
            if not label.startswith('Max_Assignment_'):
                # Evaluate constraint with the solution
                lhs_value = constraint.lhs.energy(cqm_sample)
                rhs_value = constraint.rhs
                sense = constraint.sense
                
                # Check if constraint is violated
                is_violated = False
                if sense == '<=':
                    is_violated = lhs_value > rhs_value + 1e-6
                elif sense == '>=':
                    is_violated = lhs_value < rhs_value - 1e-6
                elif sense == '==':
                    is_violated = abs(lhs_value - rhs_value) > 1e-6
                
                if is_violated:
                    violated_constraints.append({
                        'label': label,
                        'sense': sense,
                        'lhs': lhs_value,
                        'rhs': rhs_value,
                        'violation': abs(lhs_value - rhs_value)
                    })
        
        print(f"\n   Total constraint violations: {len(violated_constraints)}")
        
        if violated_constraints:
            print(f"   ❌ Found {len(violated_constraints)} violated constraints:")
            for v in violated_constraints[:10]:  # Show first 10
                n_crops_info = f" (n_crops={v['n_crops']})" if 'n_crops' in v else ""
                print(f"     - {v['label']}: {v['lhs']:.4f} {v['sense']} {v['rhs']:.4f} (violation: {v['violation']:.4f}){n_crops_info}")
            if len(violated_constraints) > 10:
                print(f"     ... and {len(violated_constraints) - 10} more")
        else:
            print(f"   ✓ All CQM constraints satisfied!")
        
        # 9. Validate with solver_runner validation function
        print(f"\n9. Validating with solver_runner validation...")
        validation_result = solver_runner.validate_solution_constraints(
            cqm_sample, plots_list, foods, food_groups, patches_scaled, config
        )
        
        print(f"   Feasible: {validation_result['is_feasible']}")
        print(f"   Violations: {validation_result['n_violations']}")
        
        if validation_result['n_violations'] > 0:
            print(f"\n   Violations detected:")
            for violation in validation_result['violations'][:10]:
                print(f"     - {violation}")
            if len(validation_result['violations']) > 10:
                print(f"     ... and {len(validation_result['violations']) - 10} more")
        
        # 10. Save results
        print(f"\n10. Saving results...")
        
        results = {
            'status': 'Feasible' if is_feasible else 'Infeasible',
            'is_feasible': bool(is_feasible),
            'energy': float(energy),
            'hybrid_time': float(hybrid_time),
            'qpu_time': float(qpu_time),
            'n_variables': len(cqm.variables),
            'n_constraints': len(cqm.constraints),
            'n_plots': n_plots,
            'n_foods': len(foods),
            'total_area': float(sum(patches_scaled.values())),
            'cqm_violated_constraints': len(violated_constraints),
            'validation_violations': validation_result['n_violations'],
            'validation_result': validation_result,
            'solution_plantations': {k: float(v) for k, v in cqm_sample.items()}
        }
        
        output_path = 'test_patch_cqm_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✓ Results saved to: {output_path}")
        
        # Summary
        print(f"\n{'=' * 80}")
        print(f"SUMMARY")
        print(f"{'=' * 80}")
        print(f"CQM Feasibility: {is_feasible}")
        print(f"CQM Constraint Violations: {len(violated_constraints)}")
        print(f"Validation Violations: {validation_result['n_violations']}")
        
        if validation_result['n_violations'] > 0:
            print(f"\n⚠️  ISSUE CONFIRMED: Solution violates 'at most one crop per plot' constraint")
            print(f"    despite CQM having explicit constraints for this.")
        else:
            print(f"\n✓ No violations found - solution is valid!")
    
    else:
        print(f"   ❌ No solutions returned by DWave!")
    
    print(f"\n{'=' * 80}")

if __name__ == '__main__':
    main()
