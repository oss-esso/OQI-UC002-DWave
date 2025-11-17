#!/usr/bin/env python3
"""
CQM Constraint Diagnostic Tool

This script investigates the constraint structure of CQM models to identify
potential conflicts, redundancies, or issues that might cause infeasibility.

Key features:
- Analyzes constraint relationships and dependencies
- Identifies conflicting constraints
- Checks for over-constrained variables
- Generates detailed constraint reports
- Saves CQM model for inspection
- Does NOT require D-Wave solver access

Usage:
    python diagnose_cqm_constraints.py --scenario patch --config 10
    python diagnose_cqm_constraints.py --scenario farm --config 15 --detailed
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import shutil

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Utils.farm_sampler import generate_farms
from Utils.patch_sampler import generate_farms as generate_patches
from src.scenarios import load_food_data

# Import solver to create CQM
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))
import solver_runner_BINARY as solver_runner


class CQMConstraintDiagnostic:
    """Diagnostic tool for analyzing CQM constraints."""
    
    def __init__(self, cqm, constraint_metadata, scenario_info):
        """
        Initialize diagnostic tool.
        
        Args:
            cqm: ConstrainedQuadraticModel object
            constraint_metadata: Metadata about constraints from creation
            scenario_info: Dictionary with scenario information
        """
        self.cqm = cqm
        self.metadata = constraint_metadata
        self.info = scenario_info
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'scenario': scenario_info,
            'summary': {},
            'constraints': {},
            'variables': {},
            'potential_issues': [],
            'recommendations': []
        }
    
    def analyze_all(self):
        """Run all diagnostic analyses."""
        print(f"\n{'='*80}")
        print("CQM CONSTRAINT DIAGNOSTIC")
        print(f"{'='*80}")
        print(f"Scenario: {self.info['scenario_type']}")
        print(f"Units: {self.info['n_units']}")
        print(f"Foods: {self.info['n_foods']}")
        print(f"Variables: {len(self.cqm.variables)}")
        print(f"Constraints: {len(self.cqm.constraints)}")
        
        self.analyze_constraint_types()
        self.analyze_variable_constraints()
        self.check_constraint_bounds()
        self.identify_potential_conflicts()
        self.generate_recommendations()
        
        return self.report
    
    def analyze_constraint_types(self):
        """Analyze types and distribution of constraints."""
        print(f"\n--- Analyzing Constraint Types ---")
        
        constraint_types = defaultdict(int)
        constraint_details = []
        
        for label, constraint in self.cqm.constraints.items():
            # Determine constraint type
            lhs = constraint.lhs
            rhs = constraint.rhs
            sense = constraint.sense
            
            # Count linear vs quadratic terms
            num_linear = len(lhs.linear) if hasattr(lhs, 'linear') else 0
            num_quadratic = len(lhs.quadratic) if hasattr(lhs, 'quadratic') else 0
            
            constraint_info = {
                'label': label,
                'sense': sense.name,
                'rhs': rhs,
                'num_linear_terms': num_linear,
                'num_quadratic_terms': num_quadratic,
                'num_variables': len(lhs.variables) if hasattr(lhs, 'variables') else 0
            }
            
            # Categorize constraint
            if '_Area_' in label or 'Max_Area' in label or 'MaxArea' in label:
                constraint_info['category'] = 'area_bounds'
            elif '_One_' in label or 'AtMostOne' in label:
                constraint_info['category'] = 'selection_limit'
            elif 'FoodGroup' in label:
                constraint_info['category'] = 'food_group'
            elif 'MinArea' in label:
                constraint_info['category'] = 'min_area'
            else:
                constraint_info['category'] = 'other'
            
            constraint_types[constraint_info['category']] += 1
            constraint_details.append(constraint_info)
        
        self.report['constraints'] = {
            'total': len(self.cqm.constraints),
            'by_category': dict(constraint_types),
            'details': constraint_details
        }
        
        print(f"  Total constraints: {len(self.cqm.constraints)}")
        for category, count in constraint_types.items():
            print(f"    {category}: {count}")
    
    def analyze_variable_constraints(self):
        """Analyze which variables appear in which constraints."""
        print(f"\n--- Analyzing Variable-Constraint Relationships ---")
        
        var_to_constraints = defaultdict(list)
        
        for label, constraint in self.cqm.constraints.items():
            lhs = constraint.lhs
            if hasattr(lhs, 'variables'):
                for var in lhs.variables:
                    var_to_constraints[var].append(label)
        
        # Find most and least constrained variables
        constraint_counts = {var: len(constraints) for var, constraints in var_to_constraints.items()}
        
        if constraint_counts:
            max_constrained = max(constraint_counts.values())
            min_constrained = min(constraint_counts.values())
            avg_constrained = sum(constraint_counts.values()) / len(constraint_counts)
            
            most_constrained_vars = [var for var, count in constraint_counts.items() if count == max_constrained]
            least_constrained_vars = [var for var, count in constraint_counts.items() if count == min_constrained]
            
            self.report['variables'] = {
                'total': len(self.cqm.variables),
                'max_constraints_per_var': max_constrained,
                'min_constraints_per_var': min_constrained,
                'avg_constraints_per_var': avg_constrained,
                'most_constrained_examples': most_constrained_vars[:5],
                'least_constrained_examples': least_constrained_vars[:5],
                'variable_constraint_map': {var: constraints for var, constraints in list(var_to_constraints.items())[:10]}
            }
            
            print(f"  Total variables: {len(self.cqm.variables)}")
            print(f"  Constraints per variable:")
            print(f"    Max: {max_constrained}")
            print(f"    Min: {min_constrained}")
            print(f"    Avg: {avg_constrained:.2f}")
            
            # Check for over-constrained variables
            if max_constrained > 10:
                self.report['potential_issues'].append({
                    'type': 'over_constrained_variables',
                    'severity': 'warning',
                    'message': f"Some variables have {max_constrained} constraints, which might indicate redundancy",
                    'examples': most_constrained_vars[:3]
                })
                print(f"  ⚠️  Warning: {len(most_constrained_vars)} variables have {max_constrained} constraints")
    
    def check_constraint_bounds(self):
        """Check for impossible or conflicting bounds."""
        print(f"\n--- Checking Constraint Bounds ---")
        
        issues_found = 0
        
        for label, constraint in self.cqm.constraints.items():
            sense = constraint.sense
            rhs = constraint.rhs
            
            # Check for obviously impossible constraints
            if sense.name == 'GE' and rhs < 0:
                # Greater-than-or-equal to negative might be intentional
                pass
            elif sense.name == 'LE' and rhs < 0:
                self.report['potential_issues'].append({
                    'type': 'impossible_bound',
                    'severity': 'error',
                    'constraint': label,
                    'message': f"Constraint {label} requires <= {rhs}, which might be impossible for non-negative variables"
                })
                issues_found += 1
                print(f"  ❌ Issue: {label} has impossible bound <= {rhs}")
        
        if issues_found == 0:
            print(f"  ✓ No obvious bound issues found")
        else:
            print(f"  ⚠️  Found {issues_found} potential bound issues")
    
    def identify_potential_conflicts(self):
        """Identify potential conflicts between constraints."""
        print(f"\n--- Identifying Potential Conflicts ---")
        
        # For patch scenario, check for the specific issue in the JSON result
        if self.info['scenario_type'] == 'patch':
            # Check if "at most one crop per plot" constraints exist
            at_most_one_constraints = []
            for label in self.cqm.constraints.keys():
                if 'AtMostOne' in label or '_One_' in label:
                    at_most_one_constraints.append(label)
            
            if at_most_one_constraints:
                print(f"  ✓ Found {len(at_most_one_constraints)} 'at most one' constraints")
                self.report['constraints']['at_most_one_count'] = len(at_most_one_constraints)
                
                # Verify these constraints are properly formulated
                for label in at_most_one_constraints[:3]:  # Check first 3
                    constraint = self.cqm.constraints[label]
                    lhs = constraint.lhs
                    
                    print(f"  Checking constraint: {label}")
                    print(f"    Sense: {constraint.sense.name}")
                    print(f"    RHS: {constraint.rhs}")
                    print(f"    Num variables: {len(lhs.variables) if hasattr(lhs, 'variables') else 'N/A'}")
                    
                    # For "at most one", we expect: sum(Y_plot_c for all c) <= 1
                    if constraint.sense.name != 'LE' or constraint.rhs != 1:
                        self.report['potential_issues'].append({
                            'type': 'incorrect_at_most_one',
                            'severity': 'error',
                            'constraint': label,
                            'message': f"'At most one' constraint has wrong formulation: {constraint.sense.name} {constraint.rhs}"
                        })
                        print(f"    ❌ ERROR: Expected LE 1, got {constraint.sense.name} {constraint.rhs}")
            else:
                self.report['potential_issues'].append({
                    'type': 'missing_at_most_one',
                    'severity': 'critical',
                    'message': "No 'at most one crop per plot' constraints found for patch scenario"
                })
                print(f"  ❌ CRITICAL: No 'at most one' constraints found!")
        
        # Check food group constraints
        food_group_constraints = [l for l in self.cqm.constraints.keys() if 'FoodGroup' in l]
        if food_group_constraints:
            print(f"  ✓ Found {len(food_group_constraints)} food group constraints")
            
            # Sample some to verify formulation
            for label in food_group_constraints[:2]:
                constraint = self.cqm.constraints[label]
                print(f"  Checking: {label}")
                print(f"    Sense: {constraint.sense.name}, RHS: {constraint.rhs}")
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis."""
        print(f"\n--- Recommendations ---")
        
        # Based on the config_10_run_1.json showing 9 violations
        # where plots have 2-3 crops assigned instead of at most 1
        
        if self.info['scenario_type'] == 'patch':
            # Check if the issue is in the formulation
            self.report['recommendations'].append({
                'priority': 'high',
                'area': 'constraint_formulation',
                'recommendation': 'Verify that Y_plot_crop variables are truly binary (0 or 1)',
                'reason': 'Solution shows fractional or multiple crop assignments per plot'
            })
            
            self.report['recommendations'].append({
                'priority': 'high',
                'area': 'constraint_enforcement',
                'recommendation': 'Check if CQM→BQM conversion with Lagrange multipliers is strong enough',
                'reason': 'Constraints may be too weak to prevent multiple crop assignments'
            })
            
            self.report['recommendations'].append({
                'priority': 'medium',
                'area': 'debugging',
                'recommendation': 'Test with smaller problem (3-5 plots) to isolate issue',
                'reason': 'Easier to manually verify constraint satisfaction'
            })
            
            print(f"  • Verify Y variables are strictly binary")
            print(f"  • Check Lagrange multiplier strength in BQM conversion")
            print(f"  • Test with smaller problem size first")


def create_and_diagnose_cqm(scenario_type: str, n_units: int, total_land: float = 100.0):
    """
    Create CQM and run diagnostic analysis.
    
    Args:
        scenario_type: 'farm' or 'patch'
        n_units: Number of farms or patches
        total_land: Total land area in hectares
    """
    print(f"\n{'='*80}")
    print(f"CREATING CQM FOR DIAGNOSTIC")
    print(f"{'='*80}")
    print(f"Scenario: {scenario_type}")
    print(f"Units: {n_units}")
    print(f"Total land: {total_land} ha")
    
    # Generate land data
    if scenario_type == 'patch':
        print(f"\nGenerating {n_units} patches (even grid)...")
        from Utils.patch_sampler import generate_grid
        land_availability = generate_grid(n_farms=n_units, area=total_land, seed=42)
        units_list = list(land_availability.keys())
        print(f"  ✓ Generated {len(units_list)} patches")
    else:  # farm
        print(f"\nGenerating {n_units} farms (uneven distribution)...")
        farms_unscaled = generate_farms(n_farms=n_units, seed=42)
        scale_factor = total_land / sum(farms_unscaled.values())
        land_availability = {k: v * scale_factor for k, v in farms_unscaled.items()}
        units_list = list(land_availability.keys())
        print(f"  ✓ Generated {len(units_list)} farms")
    
    # Load food data
    print(f"\nLoading food data...")
    food_list, foods, food_groups, _ = load_food_data('full_family')
    print(f"  ✓ Loaded {len(foods)} foods in {len(food_groups)} groups")
    
    # Create configuration
    config = {
        'parameters': {
            'land_availability': land_availability,
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
    
    # Create CQM
    print(f"\nCreating CQM...")
    if scenario_type == 'patch':
        cqm, Y, constraint_metadata = solver_runner.create_cqm_plots(
            units_list, foods, food_groups, config
        )
        variables_info = {'Y': 'Binary selection variables'}
    else:  # farm
        cqm, A, Y, constraint_metadata = solver_runner.create_cqm_farm(
            units_list, foods, food_groups, config
        )
        variables_info = {'A': 'Continuous area variables', 'Y': 'Binary selection variables'}
    
    print(f"  ✓ Created CQM:")
    print(f"    Variables: {len(cqm.variables)}")
    print(f"    Constraints: {len(cqm.constraints)}")
    for var_type, desc in variables_info.items():
        print(f"    {var_type}: {desc}")
    
    # Save CQM model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cqm_dir = os.path.join(project_root, 'CQM_Models', 'diagnostics')
    os.makedirs(cqm_dir, exist_ok=True)
    
    cqm_filename = f"cqm_diagnostic_{scenario_type}_{n_units}units_{timestamp}.cqm"
    cqm_path = os.path.join(cqm_dir, cqm_filename)
    
    print(f"\nSaving CQM model...")
    with open(cqm_path, 'wb') as f:
        shutil.copyfileobj(cqm.to_file(), f)
    print(f"  ✓ Saved to: {cqm_path}")
    
    # Run diagnostic
    scenario_info = {
        'scenario_type': scenario_type,
        'n_units': n_units,
        'n_foods': len(foods),
        'total_land': total_land,
        'cqm_file': cqm_path
    }
    
    diagnostic = CQMConstraintDiagnostic(cqm, constraint_metadata, scenario_info)
    report = diagnostic.analyze_all()
    
    # Save diagnostic report
    report_filename = f"diagnostic_report_{scenario_type}_{n_units}units_{timestamp}.json"
    report_path = os.path.join(cqm_dir, report_filename)
    
    print(f"\nSaving diagnostic report...")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✓ Saved to: {report_path}")
    
    # Print summary of issues
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    
    if report['potential_issues']:
        print(f"\n⚠️  Found {len(report['potential_issues'])} potential issues:")
        for issue in report['potential_issues']:
            severity_icon = '🔴' if issue['severity'] == 'critical' else '🟠' if issue['severity'] == 'error' else '🟡'
            print(f"\n{severity_icon} {issue['type'].upper()} ({issue['severity']})")
            print(f"   {issue['message']}")
    else:
        print(f"\n✅ No obvious issues detected")
    
    if report['recommendations']:
        print(f"\n📋 Recommendations ({len(report['recommendations'])}):")
        for rec in report['recommendations']:
            priority_icon = '🔥' if rec['priority'] == 'high' else '📌' if rec['priority'] == 'medium' else '💡'
            print(f"\n{priority_icon} [{rec['area'].upper()}]")
            print(f"   {rec['recommendation']}")
            print(f"   Reason: {rec['reason']}")
    
    print(f"\n{'='*80}")
    print(f"Files saved:")
    print(f"  CQM Model: {cqm_path}")
    print(f"  Report: {report_path}")
    print(f"{'='*80}\n")
    
    return cqm, report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Diagnose CQM constraint formulation without running D-Wave solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python diagnose_cqm_constraints.py --scenario patch --config 10
  python diagnose_cqm_constraints.py --scenario farm --config 15
  python diagnose_cqm_constraints.py --scenario patch --config 5 --land 50
        '''
    )
    
    parser.add_argument('--scenario', type=str, required=True, choices=['farm', 'patch'],
                       help='Scenario type: farm (continuous) or patch (binary)')
    parser.add_argument('--config', type=int, required=True,
                       help='Number of units (farms or patches)')
    parser.add_argument('--land', type=float, default=100.0,
                       help='Total land area in hectares (default: 100.0)')
    
    args = parser.parse_args()
    
    # Run diagnostic
    try:
        cqm, report = create_and_diagnose_cqm(
            scenario_type=args.scenario,
            n_units=args.config,
            total_land=args.land
        )
        
        print(f"\n✅ Diagnostic complete!")
        print(f"\nNo D-Wave solver access was used.")
        print(f"Review the saved files for detailed analysis.")
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
