#!/usr/bin/env python3
"""
Constraint Violation Investigation Runner

This script orchestrates the complete investigation of constraint violations
in the D-Wave BQM solver for 50-unit patch scenarios.

Runs:
1. Main constraint investigation (test_patch_dwave_bqm_constraints.py)
2. Advanced BQM analysis with manual constraint penalties (advanced_bqm_analysis.py)
3. Generates comprehensive report comparing approaches

The investigation focuses on understanding why the CQM→BQM conversion
produces solutions that violate the "at most one crop per plot" constraint.
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

# Import test modules
try:
    from Tests.test_patch_dwave_bqm_constraints import run_constraint_investigation
    from Tests.advanced_bqm_analysis import (
        ManualBQMConstructor, run_manual_bqm_experiment, 
        analyze_constraint_satisfaction_patterns
    )
except ImportError:
    # If running from Tests directory
    from test_patch_dwave_bqm_constraints import run_constraint_investigation
    from advanced_bqm_analysis import (
        ManualBQMConstructor, run_manual_bqm_experiment, 
        analyze_constraint_satisfaction_patterns
    )

# Import project modules
from src.scenarios import load_food_data
from Utils.patch_sampler import generate_farms as generate_patches


def setup_test_environment():
    """Set up the test environment with consistent data."""
    print("🔧 Setting up test environment...")
    
    # Generate consistent test data
    land_data = generate_patches(
        n_farms=50,
        seed=42  # Fixed seed for reproducibility
    )
    
    # Create sample data structure
    sample_data = {
        'sample_id': 0,
        'n_units': 50,
        'total_area': sum(land_data.values()),
        'data': land_data
    }
    
    # Load food data
    try:
        foods, food_groups = load_food_data()
        print(f"   ✅ Loaded {len(foods)} foods from Excel")
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
    
    return {
        'sample_data': sample_data,
        'farms': farms,
        'foods': foods,
        'food_groups': food_groups,
        'land_availability': land_availability,
        'config': config
    }


def run_comprehensive_investigation():
    """Run the complete constraint violation investigation."""
    
    print("="*90)
    print("🚀 COMPREHENSIVE CONSTRAINT VIOLATION INVESTIGATION")
    print("="*90)
    print("Problem: D-Wave BQM solver assigns multiple crops to same plot")
    print("Target: 50-unit patch scenario")  
    print("Goal: Understand root cause and test solutions")
    print("="*90)
    
    overall_start = time.time()
    
    # Check D-Wave token
    #dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
    dwave_token = None

    if not dwave_token:
        print("\\n⚠️  WARNING: No D-Wave token found!")
        print("   Set DWAVE_API_TOKEN environment variable for full testing")
        print("   Proceeding with structure analysis only...")
    else:
        print("\\n✅ D-Wave token found - full testing enabled")
    
    # Phase 1: Main investigation
    print("\\n" + "="*60)
    print("📊 PHASE 1: MAIN CONSTRAINT INVESTIGATION")
    print("="*60)
    
    phase1_start = time.time()
    
    try:
        main_report = run_constraint_investigation(dwave_token=dwave_token)
        phase1_time = time.time() - phase1_start
        print(f"\\n✅ Phase 1 completed in {phase1_time:.1f}s")
        
        # Extract key findings
        if main_report.get('bqm_result'):
            bqm_violations = main_report['bqm_result']['validation']['n_violations']
            print(f"   🎯 BQM Violations Found: {bqm_violations}")
        
        if main_report.get('cqm_result'):
            cqm_violations = main_report['cqm_result']['validation']['n_violations']
            print(f"   🌊 CQM Violations: {cqm_violations}")
            
    except Exception as e:
        print(f"\\n❌ Phase 1 failed: {e}")
        main_report = None
        import traceback
        traceback.print_exc()
    
    # Phase 2: Advanced BQM analysis (if we have D-Wave access)
    if dwave_token:
        print("\\n" + "="*60)
        print("🔬 PHASE 2: ADVANCED BQM ANALYSIS")
        print("="*60)
        
        phase2_start = time.time()
        
        try:
            # Set up test environment
            test_env = setup_test_environment()
            
            # Run manual BQM experiments with different penalty multipliers
            print("\\n🧪 Testing manual BQM construction with penalty multipliers...")
            
            penalty_multipliers = [1, 10, 50, 100, 500, 1000]
            
            manual_results = run_manual_bqm_experiment(
                farms=test_env['farms'],
                foods=list(test_env['foods'].keys()),
                land_availability=test_env['land_availability'],
                food_data=test_env['foods'],
                config=test_env['config'],
                penalty_multipliers=penalty_multipliers,
                dwave_token=dwave_token,
                food_groups=test_env['food_groups']
            )
            
            # Analyze patterns
            pattern_analysis = analyze_constraint_satisfaction_patterns(manual_results)
            
            phase2_time = time.time() - phase2_start
            print(f"\\n✅ Phase 2 completed in {phase2_time:.1f}s")
            
        except Exception as e:
            print(f"\\n❌ Phase 2 failed: {e}")
            manual_results = None
            pattern_analysis = None
            import traceback
            traceback.print_exc()
    else:
        print("\\n⏭️  Skipping Phase 2 (requires D-Wave token)")
        manual_results = None
        pattern_analysis = None
    
    # Phase 3: Comprehensive analysis and recommendations
    print("\\n" + "="*60)
    print("📋 PHASE 3: COMPREHENSIVE ANALYSIS & RECOMMENDATIONS")
    print("="*60)
    
    total_time = time.time() - overall_start
    
    # Generate combined report
    combined_report = {
        'timestamp': datetime.now().isoformat(),
        'investigation_summary': {
            'total_time': total_time,
            'phases_completed': [],
            'dwave_token_available': dwave_token is not None
        },
        'main_investigation': main_report,
        'manual_bqm_results': manual_results,
        'pattern_analysis': pattern_analysis,
        'recommendations': []
    }
    
    # Add phase completion status
    if main_report:
        combined_report['investigation_summary']['phases_completed'].append('main_investigation')
    if manual_results:
        combined_report['investigation_summary']['phases_completed'].append('advanced_bqm_analysis')
    
    # Generate recommendations based on findings
    recommendations = []
    
    if main_report and main_report.get('bqm_result'):
        bqm_violations = main_report['bqm_result']['validation']['n_violations']
        
        if bqm_violations > 0:
            recommendations.append({
                'issue': 'BQM constraint violations detected',
                'severity': 'HIGH',
                'description': f'{bqm_violations} constraint violations in BQM solver',
                'solutions': [
                    'Increase Lagrange multiplier in CQM→BQM conversion',
                    'Use manual BQM construction with explicit penalties',
                    'Switch to CQM solver for constraint-critical applications',
                    'Implement post-processing constraint repair'
                ]
            })
    
    if manual_results and pattern_analysis:
        feasible_multipliers = pattern_analysis.get('feasible_multipliers', [])
        
        if feasible_multipliers:
            min_feasible = min(feasible_multipliers)
            recommendations.append({
                'issue': 'Penalty multiplier optimization',
                'severity': 'MEDIUM', 
                'description': f'Manual BQM feasible at penalty multiplier ≥ {min_feasible}',
                'solutions': [
                    f'Use penalty multiplier of at least {min_feasible} for constraint satisfaction',
                    'Implement adaptive penalty scaling based on problem size',
                    'Consider penalty-to-objective ratio in BQM design'
                ]
            })
        else:
            recommendations.append({
                'issue': 'Manual BQM constraint enforcement failed',
                'severity': 'HIGH',
                'description': 'No penalty multiplier achieved constraint satisfaction',
                'solutions': [
                    'Revise constraint formulation approach',
                    'Investigate alternative penalty functions',
                    'Consider hybrid classical-quantum constraint handling'
                ]
            })
    
    # Performance analysis
    if main_report and main_report.get('comparison'):
        comp = main_report['comparison']
        if comp.get('cqm_feasible') and not comp.get('bqm_feasible'):
            recommendations.append({
                'issue': 'CQM vs BQM feasibility gap',
                'severity': 'CRITICAL',
                'description': 'CQM produces feasible solutions, BQM does not',
                'solutions': [
                    'Prefer CQM solver for applications requiring constraint satisfaction',
                    'Develop BQM-specific constraint enforcement strategies',
                    'Implement solution verification and repair for BQM results'
                ]
            })
    
    combined_report['recommendations'] = recommendations
    
    # Save comprehensive report
    report_file = f"comprehensive_constraint_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(os.path.dirname(__file__), report_file)
    
    with open(report_path, 'w') as f:
        json.dump(combined_report, f, indent=2, default=str)
    
    # Print summary
    print(f"\\n📊 INVESTIGATION COMPLETED")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Phases completed: {len(combined_report['investigation_summary']['phases_completed'])}")
    print(f"   Recommendations generated: {len(recommendations)}")
    print(f"   Report saved: {report_path}")
    
    # Print key findings
    print(f"\\n🔍 KEY FINDINGS:")
    
    if main_report and main_report.get('bqm_result'):
        bqm_violations = main_report['bqm_result']['validation']['n_violations']
        print(f"   • BQM solver: {bqm_violations} constraint violations")
        
        if main_report.get('cqm_result'):
            cqm_violations = main_report['cqm_result']['validation']['n_violations']
            print(f"   • CQM solver: {cqm_violations} constraint violations")
            
            if cqm_violations == 0 and bqm_violations > 0:
                print(f"   • ✅ CONFIRMED: Issue is in CQM→BQM conversion, not problem formulation")
    
    if pattern_analysis and pattern_analysis.get('feasible_multipliers'):
        min_feasible = min(pattern_analysis['feasible_multipliers'])
        print(f"   • Manual BQM feasible with penalty multiplier ≥ {min_feasible}")
    
    # Print recommendations summary
    print(f"\\n💡 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"   {i}. {rec['description']}")
        print(f"      → {rec['solutions'][0]}")
    
    print(f"\\n✅ Investigation complete! Check {report_file} for full details.")
    return combined_report


if __name__ == "__main__":
    print("🔬 Starting comprehensive constraint violation investigation...")
    print("   This will analyze why D-Wave BQM produces constraint violations")
    print("   in 50-unit patch scenarios and test potential solutions.")
    
    try:
        report = run_comprehensive_investigation()
        print("\\n🎉 Investigation completed successfully!")
        
    except KeyboardInterrupt:
        print("\\n⏹️  Investigation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\\n💥 Investigation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)