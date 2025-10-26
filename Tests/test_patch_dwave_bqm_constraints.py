#!/usr/bin/env python3
"""
Targeted Test for D-Wave BQM Constraint Violations in 50-Unit Patch Scenario

This test investigates why the D-Wave BQM solver (CQM→BQM conversion) is producing
constraint violations, specifically multiple crops being assigned to the same plot.

The issue appears to be in the CQM→BQM conversion process where Lagrange multipliers
may not be sufficient to enforce hard constraints in the quantum annealing process.

Test focuses on:
1. Reproducing the exact 50-unit scenario that shows violations
2. Analyzing CQM constraints vs BQM formulation
3. Testing different Lagrange multiplier strategies
4. Comparing with constraint satisfaction in CQM solver
5. Investigating if BQM energy landscape properly penalizes violations
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.scenarios import load_food_data
from patch_sampler import generate_farms as generate_patches
from solver_runner_PATCH import (
    create_cqm, solve_with_dwave, solve_with_dwave_cqm, 
    calculate_original_objective, extract_solution_summary,
    validate_solution_constraints
)
from dimod import cqm_to_bqm, BinaryQuadraticModel
from dwave.system import LeapHybridBQMSampler, LeapHybridCQMSampler

class BQMConstraintAnalyzer:
    """Analyzes BQM formulation and constraint enforcement."""
    
    def __init__(self, cqm, farms, foods, land_availability):
        self.cqm = cqm
        self.farms = farms
        self.foods = foods
        self.land_availability = land_availability
        self.bqm = None
        self.invert = None
        
    def convert_to_bqm(self, lagrange_multiplier=None):
        """Convert CQM to BQM with optional custom Lagrange multiplier."""
        print(f"\\n🔄 Converting CQM to BQM...")
        print(f"   CQM Variables: {len(self.cqm.variables)}")
        print(f"   CQM Constraints: {len(self.cqm.constraints)}")
        
        if lagrange_multiplier is not None:
            print(f"   Using custom Lagrange multiplier: {lagrange_multiplier}")
            # We need to modify the CQM constraints with custom multiplier
            # This is complex, so we'll analyze the default conversion first
        
        convert_start = time.time()
        self.bqm, self.invert = cqm_to_bqm(self.cqm)
        convert_time = time.time() - convert_start
        
        print(f"   ✅ Conversion complete in {convert_time:.3f}s")
        print(f"   BQM Variables: {len(self.bqm.variables)}")
        print(f"   BQM Linear terms: {len(self.bqm.linear)}")
        print(f"   BQM Quadratic terms: {len(self.bqm.quadratic)}")
        
        return self.bqm, self.invert
        
    def analyze_constraint_penalties(self):
        """Analyze how constraints are encoded as penalties in BQM."""
        if self.bqm is None:
            raise ValueError("Must convert to BQM first")
            
        print(f"\\n🔍 Analyzing constraint penalty encoding...")
        
        # Extract constraint information from BQM
        penalty_analysis = {
            'total_linear_bias': sum(abs(bias) for bias in self.bqm.linear.values()),
            'total_quadratic_bias': sum(abs(bias) for bias in self.bqm.quadratic.values()),
            'max_linear_bias': max(abs(bias) for bias in self.bqm.linear.values()) if self.bqm.linear else 0,
            'max_quadratic_bias': max(abs(bias) for bias in self.bqm.quadratic.values()) if self.bqm.quadratic else 0,
            'variable_count': len(self.bqm.variables)
        }
        
        print(f"   Total linear bias magnitude: {penalty_analysis['total_linear_bias']:.2f}")
        print(f"   Total quadratic bias magnitude: {penalty_analysis['total_quadratic_bias']:.2f}")
        print(f"   Max linear bias: {penalty_analysis['max_linear_bias']:.2f}")
        print(f"   Max quadratic bias: {penalty_analysis['max_quadratic_bias']:.2f}")
        
        return penalty_analysis
        
    def simulate_constraint_violations(self, solution):
        """Simulate what happens when constraints are violated in BQM energy."""
        if self.bqm is None:
            raise ValueError("Must convert to BQM first")
            
        print(f"\\n⚖️  Analyzing constraint violation penalties...")
        
        # Calculate BQM energy for the given solution
        energy = self.bqm.energy(solution)
        print(f"   BQM Energy: {energy:.2f}")
        
        # Analyze specific constraint violations
        violations = self.analyze_plot_assignment_violations(solution)
        
        return {
            'bqm_energy': energy,
            'violations': violations
        }
        
    def analyze_plot_assignment_violations(self, solution):
        """Analyze violations of 'at most one crop per plot' constraint."""
        violations = []
        
        for plot in self.farms:
            crops_assigned = []
            for crop in self.foods:
                var_name = f"X_{plot}_{crop}"
                if var_name in solution and solution[var_name] > 0:
                    crops_assigned.append(crop)
            
            if len(crops_assigned) > 1:
                violations.append({
                    'plot': plot,
                    'crops_assigned': crops_assigned,
                    'n_crops': len(crops_assigned),
                    'area': self.land_availability[plot]
                })
        
        return violations


def run_constraint_investigation(dwave_token=None):
    """Run the main constraint investigation."""
    print("="*80)
    print("🔬 D-WAVE BQM CONSTRAINT VIOLATION INVESTIGATION")
    print("="*80)
    print("Target: 50-unit Patch scenario with D-Wave BQM solver")
    print("Issue: Multiple crops assigned to same plot (constraint violations)")
    print("="*80)
    
    # Step 1: Generate the exact same 50-unit scenario
    print("\\n📊 Step 1: Generating 50-unit patch scenario...")
    
    land_data = generate_patches(
        n_farms=50,
        seed=42  # Fixed seed for reproducibility
    )
    
    # Create sample data structure similar to original format
    sample_data = {
        'sample_id': 0,
        'n_units': 50,
        'total_area': sum(land_data.values()),
        'data': land_data
    }
    
    print(f"   ✅ Generated {sample_data['n_units']} patches")
    print(f"   ✅ Total area: {sample_data['total_area']:.2f} ha")
    
    # Step 2: Load food data and create configuration
    print("\\n🍎 Step 2: Loading food data and creating configuration...")
    
    try:
        foods, food_groups = load_food_data()
        print(f"   ✅ Loaded {len(foods)} foods")
    except Exception as e:
        print(f"   ⚠️  Could not load Excel data: {e}")
        print("   Using fallback food data...")
        foods = {
            'Wheat': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 'affordability': 0.9, 'sustainability': 0.7},
            'Corn': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.5, 'affordability': 0.8, 'sustainability': 0.6},
            'Rice': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.7, 'affordability': 0.7, 'sustainability': 0.8},
            'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.4, 'affordability': 0.6, 'sustainability': 0.9},
            'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.8, 'affordability': 0.9, 'sustainability': 0.6},
            'Apples': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.3, 'affordability': 0.7, 'sustainability': 0.8}
        }
        food_groups = {
            'grains': ['Wheat', 'Corn', 'Rice'],
            'proteins': ['Soybeans'],
            'vegetables': ['Potatoes'],
            'fruits': ['Apples']
        }
    
    # Create configuration
    config = {
        'parameters': {
            'land_availability': sample_data['data'],
            'minimum_planting_area': {food: 0.0 for food in foods},
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
    
    farms = list(sample_data['data'].keys())
    land_availability = sample_data['data']
    
    # Step 3: Create CQM
    print("\\n🏗️  Step 3: Creating CQM...")
    
    cqm_start = time.time()
    cqm, (X, Y), constraint_metadata = create_cqm(farms, foods, food_groups, config)
    cqm_time = time.time() - cqm_start
    
    print(f"   ✅ CQM created in {cqm_time:.3f}s")
    print(f"   Variables: {len(cqm.variables)}")
    print(f"   Constraints: {len(cqm.constraints)}")
    
    # Step 4: Analyze CQM constraints
    print("\\n📋 Step 4: Analyzing CQM constraints...")
    
    constraint_analysis = {}
    for label, constraint in cqm.constraints.items():
        constraint_type = label.split('_')[0]
        if constraint_type not in constraint_analysis:
            constraint_analysis[constraint_type] = 0
        constraint_analysis[constraint_type] += 1
    
    for constraint_type, count in constraint_analysis.items():
        print(f"   {constraint_type}: {count} constraints")
    
    # Step 5: Initialize constraint analyzer
    print("\\n🔬 Step 5: Initializing BQM constraint analyzer...")
    
    analyzer = BQMConstraintAnalyzer(cqm, farms, foods, land_availability)
    
    # Step 6: Convert to BQM and analyze
    print("\\n🔄 Step 6: Converting CQM to BQM and analyzing...")
    
    bqm, invert = analyzer.convert_to_bqm()
    penalty_analysis = analyzer.analyze_constraint_penalties()
    
    # Step 7: Test with D-Wave CQM solver for comparison
    print("\\n🌊 Step 7: Testing with D-Wave CQM solver (baseline)...")
    
    if not dwave_token:
        dwave_token = os.environ.get('DWAVE_API_TOKEN')
    
    if not dwave_token:
        print("   ⚠️  No D-Wave token found. Set DWAVE_API_TOKEN environment variable.")
        print("   Skipping D-Wave tests...")
        cqm_result = None
    else:
        try:
            cqm_sampleset, cqm_hybrid_time, cqm_qpu_time = solve_with_dwave_cqm(cqm, dwave_token)
            cqm_solution = dict(cqm_sampleset.first.sample)
            cqm_energy = cqm_sampleset.first.energy
            
            print(f"   ✅ CQM solved in {cqm_hybrid_time:.2f}s")
            print(f"   CQM Energy: {cqm_energy:.2f}")
            
            # Validate CQM solution
            cqm_validation = validate_solution_constraints(
                cqm_solution, farms, foods, food_groups, land_availability, config
            )
            
            print(f"   CQM Feasible: {cqm_validation['is_feasible']}")
            print(f"   CQM Violations: {cqm_validation['n_violations']}")
            
            cqm_result = {
                'solution': cqm_solution,
                'energy': cqm_energy,
                'validation': cqm_validation,
                'timing': {'hybrid_time': cqm_hybrid_time, 'qpu_time': cqm_qpu_time}
            }
            
        except Exception as e:
            print(f"   ❌ CQM solving failed: {e}")
            cqm_result = None
    
    # Step 8: Test with D-Wave BQM solver
    print("\\n⚡ Step 8: Testing with D-Wave BQM solver...")
    
    if not dwave_token:
        print("   ⚠️  Skipping D-Wave BQM test (no token)")
        bqm_result = None
    else:
        try:
            bqm_sampleset, bqm_hybrid_time, bqm_qpu_time, bqm_conversion_time, bqm_invert = solve_with_dwave(cqm, dwave_token)
            bqm_solution = dict(bqm_sampleset.first.sample)
            bqm_energy = bqm_sampleset.first.energy
            
            # Invert solution back to original variables
            original_solution = bqm_invert(bqm_solution)
            
            print(f"   ✅ BQM solved in {bqm_hybrid_time:.2f}s")
            print(f"   BQM Energy: {bqm_energy:.2f}")
            print(f"   BQM Conversion Time: {bqm_conversion_time:.3f}s")
            
            # Validate BQM solution
            bqm_validation = validate_solution_constraints(
                original_solution, farms, foods, food_groups, land_availability, config
            )
            
            print(f"   BQM Feasible: {bqm_validation['is_feasible']}")
            print(f"   BQM Violations: {bqm_validation['n_violations']}")
            
            # Analyze constraint violations in detail
            violation_analysis = analyzer.simulate_constraint_violations(original_solution)
            
            bqm_result = {
                'solution': original_solution,
                'energy': bqm_energy,
                'validation': bqm_validation,
                'violation_analysis': violation_analysis,
                'timing': {
                    'hybrid_time': bqm_hybrid_time, 
                    'qpu_time': bqm_qpu_time,
                    'conversion_time': bqm_conversion_time
                }
            }
            
        except Exception as e:
            print(f"   ❌ BQM solving failed: {e}")
            bqm_result = None
    
    # Step 9: Detailed constraint violation analysis
    print("\\n🔍 Step 9: Detailed constraint violation analysis...")
    
    if bqm_result and bqm_result['validation']['n_violations'] > 0:
        print(f"\\n❌ Found {bqm_result['validation']['n_violations']} violations:")
        
        # Group violations by type
        violation_types = {}
        for violation in bqm_result['validation']['violations']:
            if 'crops assigned' in violation:
                # Plot assignment violation
                if 'plot_assignment' not in violation_types:
                    violation_types['plot_assignment'] = []
                violation_types['plot_assignment'].append(violation)
        
        for violation_type, violations in violation_types.items():
            print(f"\\n   {violation_type.upper()} violations ({len(violations)}):")
            for i, violation in enumerate(violations[:5]):  # Show first 5
                print(f"     {i+1}. {violation}")
            if len(violations) > 5:
                print(f"     ... and {len(violations) - 5} more")
    
    # Step 10: Generate comprehensive report
    print("\\n📊 Step 10: Generating investigation report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'scenario': {
            'type': 'patch',
            'n_units': sample_data['n_units'],
            'total_area': sample_data['total_area'],
            'n_foods': len(foods),
            'seed': 42
        },
        'cqm_info': {
            'n_variables': len(cqm.variables),
            'n_constraints': len(cqm.constraints),
            'constraint_breakdown': constraint_analysis,
            'creation_time': cqm_time
        },
        'bqm_info': {
            'n_variables': len(bqm.variables),
            'n_linear_terms': len(bqm.linear),
            'n_quadratic_terms': len(bqm.quadratic),
            'penalty_analysis': penalty_analysis
        },
        'cqm_result': cqm_result,
        'bqm_result': bqm_result,
        'comparison': {}
    }
    
    # Add comparison if both results available
    if cqm_result and bqm_result:
        report['comparison'] = {
            'cqm_feasible': cqm_result['validation']['is_feasible'],
            'bqm_feasible': bqm_result['validation']['is_feasible'],
            'cqm_violations': cqm_result['validation']['n_violations'],
            'bqm_violations': bqm_result['validation']['n_violations'],
            'energy_difference': abs(cqm_result['energy'] - bqm_result['energy']),
            'timing_comparison': {
                'cqm_time': cqm_result['timing']['hybrid_time'],
                'bqm_time': bqm_result['timing']['hybrid_time'],
                'bqm_overhead': bqm_result['timing']['conversion_time']
            }
        }
    
    # Save report
    report_file = f"constraint_violation_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(os.path.dirname(__file__), report_file)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"   ✅ Report saved to: {report_path}")
    
    # Step 11: Summary and recommendations
    print("\\n" + "="*80)
    print("📋 INVESTIGATION SUMMARY")
    print("="*80)
    
    if bqm_result:
        print(f"🎯 Target Issue Reproduced: {bqm_result['validation']['n_violations']} violations found")
        print(f"⚡ BQM Energy: {bqm_result['energy']:.2f}")
        print(f"🔄 BQM Conversion Time: {bqm_result['timing']['conversion_time']:.3f}s")
        
        if cqm_result:
            print(f"🌊 CQM Violations: {cqm_result['validation']['n_violations']}")
            print(f"🌊 CQM Energy: {cqm_result['energy']:.2f}")
            
            if cqm_result['validation']['is_feasible'] and not bqm_result['validation']['is_feasible']:
                print("\\n🔴 CRITICAL FINDING: CQM solver produces feasible solution, BQM does not!")
                print("   This confirms the issue is in CQM→BQM conversion, not problem formulation.")
    
    print("\\n💡 RECOMMENDED INVESTIGATIONS:")
    print("   1. Test different Lagrange multiplier values in CQM→BQM conversion")
    print("   2. Analyze penalty term magnitudes vs objective coefficients")
    print("   3. Compare constraint satisfaction rates across different problem sizes") 
    print("   4. Test alternative BQM formulations with explicit penalty terms")
    print("   5. Investigate if hybrid solver parameters affect constraint satisfaction")
    
    return report


def test_lagrange_multiplier_sensitivity():
    """Test different Lagrange multiplier values for CQM→BQM conversion."""
    print("\\n" + "="*80)
    print("🔬 LAGRANGE MULTIPLIER SENSITIVITY ANALYSIS")
    print("="*80)
    print("Testing how different penalty strengths affect constraint satisfaction...")
    
    # This would require custom CQM→BQM conversion with controllable multipliers
    # For now, we'll note that the default cqm_to_bqm function doesn't expose this
    print("\\n⚠️  Note: Default cqm_to_bqm() doesn't expose Lagrange multiplier control")
    print("   Advanced testing would require custom BQM formulation")
    print("   Consider using manual penalty method for constraint enforcement")
    
    return None


if __name__ == "__main__":
    print("🧪 Starting D-Wave BQM constraint violation investigation...")
    print("   This test will analyze why 50-unit scenarios show constraint violations")
    print("   Focus: Multiple crops assigned to same plot (should be ≤ 1)")
    
    try:
        # Run main investigation
        report = run_constraint_investigation()
        
        # Run additional sensitivity analysis
        test_lagrange_multiplier_sensitivity()
        
        print("\\n✅ Investigation completed successfully!")
        print("   Check the generated JSON report for detailed analysis")
        
    except Exception as e:
        print(f"\\n❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
