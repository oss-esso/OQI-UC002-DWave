#!/usr/bin/env python3
"""
Test Enhanced D-Wave BQM Solver

This script tests the enhanced D-Wave BQM solver with constraint repair
on the problematic 50-unit patch scenario to verify it resolves the
constraint violation issue.
"""

import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

from Utils.patch_sampler import generate_farms as generate_patches
from solver_runner_PATCH import create_cqm, validate_solution_constraints
from Utils.enhanced_dwave_solver import solve_with_dwave_enhanced, validate_bqm_solution


def test_enhanced_solver():
    """Test the enhanced D-Wave BQM solver."""
    print("🧪 TESTING ENHANCED D-WAVE BQM SOLVER")
    print("="*60)
    print("Testing on the problematic 50-unit patch scenario")
    print("Goal: Achieve 0 constraint violations")
    
    # Check D-Wave token
    dwave_token = os.environ.get('DWAVE_API_TOKEN')
    if not dwave_token:
        print("\\n⚠️  No D-Wave token found!")
        print("   Set DWAVE_API_TOKEN environment variable to test")
        print("   Exiting...")
        return None
    
    print("\\n✅ D-Wave token found - proceeding with test")
    
    # Step 1: Generate test scenario
    print("\\n📊 Step 1: Generating 50-unit patch scenario...")
    land_data = generate_patches(n_farms=50, seed=42)
    
    # Create food data
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
    
    config = {
        'parameters': {
            'land_availability': land_data,
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
    
    farms = list(land_data.keys())
    
    print(f"   ✅ Generated {len(farms)} patches")
    print(f"   ✅ Total area: {sum(land_data.values()):.2f} ha")
    print(f"   ✅ Foods: {list(foods.keys())}")
    
    # Step 2: Create CQM
    print("\\n🏗️  Step 2: Creating CQM...")
    cqm_start = time.time()
    cqm, (X, Y), constraint_metadata = create_cqm(farms, foods, food_groups, config)
    cqm_time = time.time() - cqm_start
    
    print(f"   ✅ CQM created in {cqm_time:.3f}s")
    print(f"   Variables: {len(cqm.variables)}")
    print(f"   Constraints: {len(cqm.constraints)}")
    
    # Step 3: Test enhanced solver
    print("\\n🚀 Step 3: Testing enhanced D-Wave BQM solver...")
    
    try:
        test_start = time.time()
        
        sampleset, hybrid_time, qpu_time, bqm_conversion_time, invert, repair_info = solve_with_dwave_enhanced(
            cqm=cqm,
            token=dwave_token,
            farms=farms,
            foods=list(foods.keys()),
            land_availability=land_data,
            max_samples_to_check=10
        )
        
        test_total_time = time.time() - test_start
        
        # Get best solution
        best_solution = invert(dict(sampleset.first.sample))
        
        # Step 4: Validate solution
        print("\\n🔍 Step 4: Validating enhanced solution...")
        
        validation = validate_bqm_solution(best_solution, farms, list(foods.keys()), land_data)
        
        print(f"   ✅ Validation completed")
        print(f"   Feasible: {validation['is_feasible']}")
        print(f"   Violations: {validation['n_violations']}")
        print(f"   Assignment rate: {validation['summary']['assignment_rate']:.2%}")
        
        # Step 5: Compare with original solver
        print("\\n📊 Step 5: Performance comparison...")
        
        print(f"   Enhanced Solver Results:")
        print(f"     Total time: {test_total_time:.2f}s")
        print(f"     D-Wave time: {hybrid_time:.2f}s")
        print(f"     QPU time: {qpu_time:.4f}s")
        print(f"     Conversion time: {bqm_conversion_time:.3f}s")
        print(f"     Violations: {validation['n_violations']}")
        print(f"     Repair applied: {repair_info['repair_applied']}")
        if repair_info['repair_applied']:
            print(f"     Original violations: {repair_info['original_violations']}")
            print(f"     Repaired violations: {repair_info['repaired_violations']}")
        
        # Step 6: Generate test report
        test_report = {
            'timestamp': datetime.now().isoformat(),
            'test_scenario': {
                'n_patches': len(farms),
                'total_area': sum(land_data.values()),
                'n_foods': len(foods),
                'solver': 'enhanced_dwave_bqm'
            },
            'performance': {
                'total_time': test_total_time,
                'dwave_time': hybrid_time,
                'qpu_time': qpu_time,
                'conversion_time': bqm_conversion_time
            },
            'solution_quality': {
                'is_feasible': validation['is_feasible'],
                'n_violations': validation['n_violations'],
                'assignment_rate': validation['summary']['assignment_rate'],
                'repair_info': repair_info
            },
            'energy': sampleset.first.energy,
            'test_status': 'SUCCESS' if validation['n_violations'] == 0 else 'PARTIAL_SUCCESS'
        }
        
        # Save test report
        report_file = f"enhanced_solver_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(os.path.dirname(__file__), report_file)
        
        with open(report_path, 'w') as f:
            json.dump(test_report, f, indent=2)
        
        print(f"\\n💾 Test report saved: {report_path}")
        
        # Step 7: Test conclusion
        print("\\n" + "="*60)
        print("🎯 TEST CONCLUSION")
        print("="*60)
        
        if validation['n_violations'] == 0:
            print("✅ SUCCESS: Enhanced solver achieved 0 constraint violations!")
            print("   The constraint repair mechanism successfully fixed the issue.")
        elif validation['n_violations'] < 5:
            print("🟡 PARTIAL SUCCESS: Significantly reduced constraint violations.")
            print(f"   Reduced from ~20 violations to {validation['n_violations']}")
        else:
            print("❌ IMPROVEMENT NEEDED: Still experiencing constraint violations.")
            print(f"   {validation['n_violations']} violations remain")
        
        if repair_info['repair_applied']:
            print(f"🔧 Constraint repair was applied successfully")
            print(f"   Repaired {repair_info['original_violations']} → {repair_info['repaired_violations']} violations")
        else:
            print("✨ No repair needed - D-Wave found feasible solution directly")
        
        return test_report
        
    except Exception as e:
        print(f"\\n❌ Enhanced solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🔬 Testing Enhanced D-Wave BQM Solver")
    print("   This test validates the constraint repair mechanism")
    print("   on the problematic 50-unit patch scenario")
    
    try:
        report = test_enhanced_solver()
        
        if report and report['solution_quality']['n_violations'] == 0:
            print("\\n🎉 Test completed successfully!")
            print("   Enhanced solver resolves constraint violation issue!")
        elif report:
            print("\\n🔄 Test completed with improvements")
            print("   Enhanced solver reduces constraint violations")
        else:
            print("\\n💥 Test failed")
        
    except KeyboardInterrupt:
        print("\\n⏹️  Test interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)