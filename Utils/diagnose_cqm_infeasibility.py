"""
Diagnose why CQM is returning infeasible solutions.

This script helps identify which constraints are causing infeasibility
and suggests potential fixes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dimod import ConstrainedQuadraticModel


def analyze_cqm_constraints(cqm: ConstrainedQuadraticModel):
    """
    Analyze CQM constraints to identify potential infeasibility sources.
    
    Args:
        cqm: ConstrainedQuadraticModel to analyze
    """
    print("="*80)
    print("CQM CONSTRAINT ANALYSIS")
    print("="*80)
    
    # Count constraint types
    constraint_types = {
        'LE': 0,  # Less than or equal
        'GE': 0,  # Greater than or equal
        'EQ': 0   # Equal
    }
    
    constraints_by_category = {
        'at_most_one': [],
        'min_plots': [],
        'max_plots': [],
        'food_group': [],
        'other': []
    }
    
    # Analyze each constraint
    for label, constraint in cqm.constraints.items():
        sense = constraint.sense.name
        if sense in constraint_types:
            constraint_types[sense] += 1
        
        # Categorize by label
        if any(kw in label for kw in ['Max_Assignment', 'AtMostOne', 'Max_Area']):
            constraints_by_category['at_most_one'].append((label, constraint))
        elif 'Min_Plots' in label or 'MinPlots' in label:
            constraints_by_category['min_plots'].append((label, constraint))
        elif 'Max_Plots' in label or 'MaxPlots' in label:
            constraints_by_category['max_plots'].append((label, constraint))
        elif 'FoodGroup' in label or 'Food_Group' in label:
            constraints_by_category['food_group'].append((label, constraint))
        else:
            constraints_by_category['other'].append((label, constraint))
    
    # Print summary
    print(f"\nTotal Constraints: {len(cqm.constraints)}")
    print(f"Total Variables: {len(cqm.variables)}")
    print(f"\nConstraint Senses:")
    for sense, count in constraint_types.items():
        print(f"  {sense}: {count}")
    
    print(f"\nConstraint Categories:")
    for category, constraints in constraints_by_category.items():
        print(f"  {category}: {len(constraints)}")
    
    # Analyze "at most one" constraints
    print("\n" + "="*80)
    print("AT MOST ONE CONSTRAINTS ANALYSIS")
    print("="*80)
    
    if constraints_by_category['at_most_one']:
        sample_constraint = constraints_by_category['at_most_one'][0]
        label, constraint = sample_constraint
        
        print(f"\nSample constraint: {label}")
        print(f"  Sense: {constraint.sense.name}")
        print(f"  RHS: {constraint.rhs}")
        print(f"  Number of variables: {len(constraint.lhs.variables)}")
        
        # Check if all coefficients are 1
        if hasattr(constraint.lhs, 'linear'):
            coeffs = list(constraint.lhs.linear.values())
            print(f"  Linear coefficients: min={min(coeffs):.2f}, max={max(coeffs):.2f}")
            if not all(abs(c - 1.0) < 1e-6 for c in coeffs):
                print(f"  ⚠️  WARNING: Not all coefficients are 1.0!")
        
        # Check offset
        if hasattr(constraint.lhs, 'offset'):
            print(f"  LHS offset: {constraint.lhs.offset}")
    
    # Analyze minimum plot constraints
    print("\n" + "="*80)
    print("MINIMUM PLOTS CONSTRAINTS ANALYSIS")
    print("="*80)
    
    if constraints_by_category['min_plots']:
        print(f"\nTotal minimum plot constraints: {len(constraints_by_category['min_plots'])}")
        
        # Check for conflicts
        total_plots = len(constraints_by_category['at_most_one'])
        total_min_plots_required = 0
        
        for label, constraint in constraints_by_category['min_plots']:
            rhs = constraint.rhs
            total_min_plots_required += rhs
        
        print(f"  Available plots: {total_plots}")
        print(f"  Minimum plots required (sum): {total_min_plots_required}")
        
        if total_min_plots_required > total_plots:
            print(f"  ❌ INFEASIBILITY DETECTED:")
            print(f"     Total minimum plots required ({total_min_plots_required}) > available plots ({total_plots})")
            print(f"     This makes the problem INFEASIBLE!")
    
    # Analyze food group constraints
    print("\n" + "="*80)
    print("FOOD GROUP CONSTRAINTS ANALYSIS")
    print("="*80)
    
    if constraints_by_category['food_group']:
        print(f"\nTotal food group constraints: {len(constraints_by_category['food_group'])}")
        
        # Sample a few
        for label, constraint in constraints_by_category['food_group'][:3]:
            print(f"\n  {label}:")
            print(f"    Sense: {constraint.sense.name}")
            print(f"    RHS: {constraint.rhs}")
            print(f"    Variables involved: {len(constraint.lhs.variables)}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Check for over-constrained problem
    if total_min_plots_required > total_plots:
        recommendations.append(
            "🔴 CRITICAL: Reduce minimum planting area requirements - "
            "current minimums exceed available land"
        )
    
    # Check number of constraints vs variables
    constraint_to_var_ratio = len(cqm.constraints) / len(cqm.variables)
    if constraint_to_var_ratio > 0.5:
        recommendations.append(
            f"⚠️  High constraint-to-variable ratio ({constraint_to_var_ratio:.2f}) - "
            "problem may be over-constrained"
        )
    
    # CQM solver recommendations
    recommendations.append(
        "✓ Try increasing solver time_limit parameter (default is often too short)"
    )
    recommendations.append(
        "✓ Consider relaxing food group diversity constraints if problem is infeasible"
    )
    recommendations.append(
        "✓ Check if PuLP/Gurobi finds the problem infeasible too (validates formulation)"
    )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
    
    print("\n" + "="*80)


def check_solution_against_constraints(cqm: ConstrainedQuadraticModel, solution: dict):
    """
    Check a solution against all CQM constraints.
    
    Args:
        cqm: ConstrainedQuadraticModel
        solution: Dictionary mapping variable names to values
    """
    print("\n" + "="*80)
    print("SOLUTION VALIDATION")
    print("="*80)
    
    violations = []
    
    for label, constraint in cqm.constraints.items():
        # Evaluate LHS
        lhs_value = 0.0
        
        if hasattr(constraint.lhs, 'linear'):
            for var, coeff in constraint.lhs.linear.items():
                var_name = str(var)
                var_value = solution.get(var_name, 0.0)
                lhs_value += coeff * var_value
        
        if hasattr(constraint.lhs, 'offset'):
            lhs_value += constraint.lhs.offset
        
        # Compare with RHS based on sense
        sense = constraint.sense.name
        rhs = constraint.rhs
        
        violated = False
        if sense == 'LE' and lhs_value > rhs + 1e-6:
            violated = True
            violation_amount = lhs_value - rhs
        elif sense == 'GE' and lhs_value < rhs - 1e-6:
            violated = True
            violation_amount = rhs - lhs_value
        elif sense == 'EQ' and abs(lhs_value - rhs) > 1e-6:
            violated = True
            violation_amount = abs(lhs_value - rhs)
        
        if violated:
            violations.append({
                'label': label,
                'sense': sense,
                'lhs': lhs_value,
                'rhs': rhs,
                'violation': violation_amount
            })
    
    print(f"\nTotal constraints: {len(cqm.constraints)}")
    print(f"Violated constraints: {len(violations)}")
    print(f"Feasibility: {'INFEASIBLE ❌' if violations else 'FEASIBLE ✓'}")
    
    if violations:
        print(f"\nViolations (showing first 10):")
        for v in violations[:10]:
            print(f"  • {v['label']}")
            print(f"    {v['lhs']:.4f} {v['sense']} {v['rhs']:.4f}")
            print(f"    Violation amount: {v['violation']:.4f}")
    
    return violations


if __name__ == "__main__":
    print("CQM Infeasibility Diagnostic Tool")
    print("="*80)
    print("\nUsage:")
    print("  from Utils.diagnose_cqm_infeasibility import analyze_cqm_constraints")
    print("  analyze_cqm_constraints(cqm)")
    print("\n  from Utils.diagnose_cqm_infeasibility import check_solution_against_constraints")
    print("  violations = check_solution_against_constraints(cqm, solution)")
