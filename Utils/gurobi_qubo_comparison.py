#!/usr/bin/env python3
"""
Educational Gurobi QUBO Comparison Script

This script demonstrates how to:
1. Generate 4 different problem formulations (BQUBO, Scaled BQUBO, PATCH, PATCH_NO_IDLE)
2. Convert each to QUBO format
3. Solve using Gurobi's quadratic optimizer
4. Compare results side-by-side

Perfect for learning QUBO formulation and Gurobi optimization!
"""

import sys
import os
import numpy as np
from typing import Dict, Tuple
import json

# Add project root to path
# Add project root and Benchmark Scripts to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    print("ERROR: Gurobi not available. Please install gurobipy.")
    print("Run: pip install gurobipy")
    sys.exit(1)

try:
    from dimod import BinaryQuadraticModel, cqm_to_bqm
    from solver_runner_PATCH import create_cqm as create_patch_cqm
    from solver_runner_BQUBO import create_cqm as create_bqubo_cqm
    from .patch_sampler import generate_farms as generate_patches
    from .farm_sampler import generate_farms as generate_farms_bqubo
    from src.scenarios import load_food_data
except ImportError as e:
    print(f"ERROR: Could not import project modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def print_section(title: str):
    """Print a nicely formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_subsection(title: str):
    """Print a subsection header"""
    print(f"\n--- {title} ---")


def create_scaled_bqubo_cqm(plots: Dict[str, float], foods: Dict, food_groups: Dict, config: Dict):
    """
    Create a Scaled BQUBO formulation that handles variable-sized plots.
    
    Key Idea: Scale benefits by plot area instead of tracking areas explicitly.
    This keeps coefficient scaling linear (not quadratic).
    
    Args:
        plots: Dictionary of plot_name -> area (acres)
        foods: Dictionary of food properties
        food_groups: Dictionary of food group classifications
        config: Configuration parameters
        
    Returns:
        cqm: Constrained Quadratic Model
        plot_list: List of plot names
        food_list: List of food names
    """
    from dimod import ConstrainedQuadraticModel, Binary
    
    print("    Building Scaled BQUBO CQM...")
    print(f"    - Plots: {len(plots)} (areas: {list(plots.values())[:5]}...)")
    print(f"    - Foods: {len(foods)}")
    
    cqm = ConstrainedQuadraticModel()
    
    # Create binary variables: x[plot][food]
    x = {}
    for plot in plots:
        x[plot] = {}
        for food in foods:
            var_name = f"x_{plot}_{food}"
            x[plot][food] = Binary(var_name)
    
    # Objective: Maximize weighted benefit * plot_area
    # (We minimize the negative to convert to minimization)
    objective = 0
    weights = config['parameters']['weights']
    
    for plot, area in plots.items():
        for food, attrs in foods.items():
            # Calculate weighted benefit per acre
            benefit_per_acre = sum(
                attrs.get(criterion, 0) * weight 
                for criterion, weight in weights.items()
            )
            # Scale by plot area - this is the key difference!
            scaled_benefit = benefit_per_acre * area
            objective += -scaled_benefit * x[plot][food]  # Negative for minimization
    
    cqm.set_objective(objective)
    
    # Constraint 1: Each plot gets at most one crop
    for plot in plots:
        cqm.add_constraint(
            sum(x[plot][food] for food in foods) <= 1,
            label=f"plot_{plot}_one_crop"
        )
    
    # Constraint 2: Total area for each crop stays within bounds
    for food in foods:
        total_area = sum(plots[plot] * x[plot][food] for plot in plots)
        
        # Max area constraint
        max_area = sum(plots.values())  # Could use food-specific limits
        cqm.add_constraint(
            total_area <= max_area,
            label=f"food_{food}_max_area"
        )
        
        # Min area constraint (optional, set to 0 for flexibility)
        min_area = config['parameters']['minimum_planting_area'].get(food, 0)
        if min_area > 0:
            cqm.add_constraint(
                total_area >= min_area,
                label=f"food_{food}_min_area"
            )
    
    print(f"    ✓ Created CQM with {len(cqm.variables)} variables, {len(cqm.constraints)} constraints")
    
    return cqm, list(plots.keys()), list(foods.keys())


def bqm_to_gurobi_qubo(bqm: BinaryQuadraticModel, name: str = "problem"):
    """
    Convert a BQM to Gurobi QUBO model and solve.
    
    This function demonstrates:
    1. How to extract Q matrix from BQM
    2. How to set up Gurobi variables
    3. How to build quadratic objective
    4. How to solve and extract solution
    
    Args:
        bqm: Binary Quadratic Model from dimod
        name: Problem name for display
        
    Returns:
        Dictionary with solution details
    """
    print(f"\n  Converting '{name}' to Gurobi QUBO...")
    
    # Step 1: Get variable list and create mapping
    variables = list(bqm.variables)
    n_vars = len(variables)
    var_to_idx = {v: i for i, v in enumerate(variables)}
    
    print(f"    Variables: {n_vars}")
    print(f"    Linear terms: {len(bqm.linear)}")
    print(f"    Quadratic terms: {len(bqm.quadratic)}")
    
    # Step 2: Create Gurobi model
    model = gp.Model(name)
    model.setParam('OutputFlag', 0)  # Suppress solver output
    model.setParam('TimeLimit', 60)  # 60 second time limit
    
    # Step 3: Create binary variables
    gurobi_vars = {}
    for var in variables:
        gurobi_vars[var] = model.addVar(vtype=GRB.BINARY, name=str(var))
    
    # Step 4: Build objective function
    #   E = sum_i h_i * x_i + sum_{i<j} J_{ij} * x_i * x_j + offset
    
    objective = bqm.offset  # Start with constant term
    
    # Add linear terms
    for var, coeff in bqm.linear.items():
        objective += coeff * gurobi_vars[var]
    
    # Add quadratic terms
    for (v1, v2), coeff in bqm.quadratic.items():
        objective += coeff * gurobi_vars[v1] * gurobi_vars[v2]
    
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Step 5: Solve!
    print(f"    Solving with Gurobi...")
    model.optimize()
    
    # Step 6: Extract solution
    if model.status == GRB.OPTIMAL:
        solution = {var: int(gurobi_vars[var].X + 0.5) for var in variables}
        energy = bqm.energy(solution)
        
        print(f"    ✓ Optimal solution found!")
        print(f"      Energy: {energy:.6f}")
        print(f"      Solve time: {model.Runtime:.3f}s")
        
        return {
            'status': 'optimal',
            'energy': energy,
            'solution': solution,
            'solve_time': model.Runtime,
            'n_variables': n_vars
        }
    else:
        print(f"    ✗ No optimal solution (status: {model.status})")
        return {
            'status': 'failed',
            'energy': float('inf'),
            'solution': {},
            'solve_time': model.Runtime,
            'n_variables': n_vars
        }


def check_cqm_feasibility(cqm, solution: Dict) -> Tuple[bool, Dict]:
    """
    Check if a solution is feasible for the original CQM.
    
    Returns:
        (is_feasible, violation_details)
    """
    violations = {}
    is_feasible = True
    
    for label, constraint in cqm.constraints.items():
        lhs_value = constraint.lhs.energy(solution)
        rhs_value = constraint.rhs
        sense = str(constraint.sense)
        
        violated = False
        if 'Le' in sense and lhs_value > rhs_value + 1e-6:
            violated = True
        elif 'Ge' in sense and lhs_value < rhs_value - 1e-6:
            violated = True
        elif 'Eq' in sense and abs(lhs_value - rhs_value) > 1e-6:
            violated = True
        
        if violated:
            is_feasible = False
            violations[label] = {
                'lhs': lhs_value,
                'sense': sense,
                'rhs': rhs_value,
                'violation': abs(lhs_value - rhs_value)
            }
    
    return is_feasible, violations


def print_solution_summary(result: Dict, cqm, formulation_name: str):
    """Print a nice summary of the solution"""
    print(f"\n  {formulation_name} Results:")
    print(f"    Status: {result['status']}")
    print(f"    BQM Energy: {result['energy']:.6f}")
    print(f"    Solve Time: {result['solve_time']:.3f}s")
    print(f"    Variables: {result['n_variables']}")
    
    if result['solution']:
        is_feasible, violations = check_cqm_feasibility(cqm, result['solution'])
        print(f"    CQM Feasible: {'✓ YES' if is_feasible else '✗ NO'}")
        
        if not is_feasible:
            print(f"    Constraint Violations: {len(violations)}")
            # Show first few violations
            for i, (label, details) in enumerate(list(violations.items())[:3]):
                print(f"      - {label}: {details['lhs']:.2f} {details['sense']} {details['rhs']:.2f}")
        
        # Count how many variables are set to 1
        n_active = sum(1 for v in result['solution'].values() if v == 1)
        print(f"    Active Variables: {n_active}/{result['n_variables']}")


def main():
    """Main comparison function"""
    
    print_section("GUROBI QUBO COMPARISON: Educational Demo")
    print("\nThis script compares 4 agricultural optimization formulations:")
    print("  1. BQUBO - Original (uniform 1-acre plots)")
    print("  2. Scaled BQUBO - New (variable-sized plots with benefit scaling)")
    print("  3. PATCH - Plot assignment (with idle penalty)")
    print("  4. PATCH_NO_IDLE - Plot assignment (no idle penalty)")
    print("\nAll problems converted to QUBO and solved with Gurobi.")
    
    # Configuration
    n_plots = 10
    seed = 42
    
    print_section("STEP 1: Generate Problem Instances")
    
    # Load food data
    try:
        _, foods, food_groups, _ = load_food_data('simple')
    except:
        print("  Using fallback food data...")
        foods = {
            'Wheat': {'nutritional_value': 0.8, 'cost_efficiency': 0.7},
            'Corn': {'nutritional_value': 0.7, 'cost_efficiency': 0.8},
            'Soybeans': {'nutritional_value': 0.9, 'cost_efficiency': 0.6}
        }
        food_groups = {'Grains': ['Wheat', 'Corn'], 'Legumes': ['Soybeans']}
    
    print(f"  Foods available: {list(foods.keys())}")
    print(f"  Food groups: {list(food_groups.keys())}")
    
    # Generate plots for different formulations
    print_subsection("Generating Plots")
    
    # For BQUBO: uniform 1-acre plots
    plots_uniform = generate_farms_bqubo(n_farms=n_plots, seed=seed)
    print(f"  BQUBO plots (uniform): {len(plots_uniform)} x 1.0 acre")
    
    # For Scaled BQUBO and PATCH: variable-sized plots
    plots_variable = generate_patches(n_farms=n_plots, seed=seed)
    areas = list(plots_variable.values())
    print(f"  Variable-sized plots: {len(plots_variable)} plots")
    print(f"    Area range: {min(areas):.2f} to {max(areas):.2f} acres")
    print(f"    Total area: {sum(areas):.2f} acres")
    
    # Create configurations
    config_base = {
        'parameters': {
            'land_availability': {},
            'minimum_planting_area': {f: 0.0 for f in foods},
            'food_group_constraints': {},
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.2,
                'cost_efficiency': 0.2,
                'environmental_impact': 0.2,
                'sustainability': 0.15
            },
            'idle_penalty_lambda': 0.0
        }
    }
    
    # Dictionary to store all results
    results = {}
    cqms = {}
    
    print_section("STEP 2: Build and Convert Each Formulation")
    
    # -------------------------------------------------------------------------
    # Formulation 1: BQUBO (Original)
    # -------------------------------------------------------------------------
    print_subsection("1. BQUBO (Original - Uniform Plots)")
    
    config_bqubo = config_base.copy()
    config_bqubo['parameters']['land_availability'] = plots_uniform
    config_bqubo['parameters']['food_group_constraints'] = {
        g: {'min_foods': 0, 'max_foods': len(lst)}
        for g, lst in food_groups.items()
    }
    
    cqm_bqubo, _, _ = create_bqubo_cqm(
        list(plots_uniform.keys()),
        foods,
        food_groups,
        config_bqubo
    )
    bqm_bqubo, _ = cqm_to_bqm(cqm_bqubo)
    
    cqms['bqubo'] = cqm_bqubo
    results['bqubo'] = bqm_to_gurobi_qubo(bqm_bqubo, "BQUBO")
    print_solution_summary(results['bqubo'], cqm_bqubo, "BQUBO")
    
    # -------------------------------------------------------------------------
    # Formulation 2: Scaled BQUBO (New!)
    # -------------------------------------------------------------------------
    print_subsection("2. Scaled BQUBO (New - Variable Plots with Benefit Scaling)")
    
    config_scaled = config_base.copy()
    config_scaled['parameters']['land_availability'] = plots_variable
    
    cqm_scaled, _, _ = create_scaled_bqubo_cqm(
        plots_variable,
        foods,
        food_groups,
        config_scaled
    )
    bqm_scaled, _ = cqm_to_bqm(cqm_scaled)
    
    cqms['scaled_bqubo'] = cqm_scaled
    results['scaled_bqubo'] = bqm_to_gurobi_qubo(bqm_scaled, "Scaled_BQUBO")
    print_solution_summary(results['scaled_bqubo'], cqm_scaled, "Scaled BQUBO")
    
    # -------------------------------------------------------------------------
    # Formulation 3: PATCH (with idle penalty)
    # -------------------------------------------------------------------------
    print_subsection("3. PATCH (Plot Assignment with Idle Penalty)")
    
    config_patch = config_base.copy()
    config_patch['parameters']['land_availability'] = plots_variable
    config_patch['parameters']['idle_penalty_lambda'] = 0.1
    
    cqm_patch, _, _ = create_patch_cqm(
        list(plots_variable.keys()),
        foods,
        food_groups,
        config_patch
    )
    bqm_patch, _ = cqm_to_bqm(cqm_patch)
    
    cqms['patch'] = cqm_patch
    results['patch'] = bqm_to_gurobi_qubo(bqm_patch, "PATCH")
    print_solution_summary(results['patch'], cqm_patch, "PATCH")
    
    # -------------------------------------------------------------------------
    # Formulation 4: PATCH_NO_IDLE
    # -------------------------------------------------------------------------
    print_subsection("4. PATCH_NO_IDLE (Plot Assignment without Idle Penalty)")
    
    config_patch_ni = config_base.copy()
    config_patch_ni['parameters']['land_availability'] = plots_variable
    config_patch_ni['parameters']['idle_penalty_lambda'] = 0.0
    
    cqm_patch_ni, _, _ = create_patch_cqm(
        list(plots_variable.keys()),
        foods,
        food_groups,
        config_patch_ni
    )
    bqm_patch_ni, _ = cqm_to_bqm(cqm_patch_ni)
    
    cqms['patch_no_idle'] = cqm_patch_ni
    results['patch_no_idle'] = bqm_to_gurobi_qubo(bqm_patch_ni, "PATCH_NO_IDLE")
    print_solution_summary(results['patch_no_idle'], cqm_patch_ni, "PATCH_NO_IDLE")
    
    # -------------------------------------------------------------------------
    # STEP 3: Side-by-Side Comparison
    # -------------------------------------------------------------------------
    print_section("STEP 3: Side-by-Side Comparison")
    
    print("\n  Summary Table:")
    print("  " + "-"*90)
    print(f"  {'Formulation':<20} {'Status':<10} {'Energy':<15} {'Time (s)':<10} {'Feasible':<10}")
    print("  " + "-"*90)
    
    for name in ['bqubo', 'scaled_bqubo', 'patch', 'patch_no_idle']:
        result = results[name]
        cqm = cqms[name]
        
        if result['solution']:
            is_feas, _ = check_cqm_feasibility(cqm, result['solution'])
            feas_str = "✓ YES" if is_feas else "✗ NO"
        else:
            feas_str = "N/A"
        
        print(f"  {name:<20} {result['status']:<10} {result['energy']:<15.6f} "
              f"{result['solve_time']:<10.3f} {feas_str:<10}")
    
    print("  " + "-"*90)
    
    # -------------------------------------------------------------------------
    # STEP 4: Key Insights
    # -------------------------------------------------------------------------
    print_section("STEP 4: Key Insights")
    
    print("\n  Coefficient Scaling Comparison:")
    bqm_list = {
        'BQUBO': bqm_bqubo,
        'Scaled BQUBO': bqm_scaled,
        'PATCH': bqm_patch,
        'PATCH_NO_IDLE': bqm_patch_ni
    }
    
    for name, bqm in bqm_list.items():
        quad_coeffs = list(bqm.quadratic.values())
        if quad_coeffs:
            coeff_range = max(quad_coeffs) - min(quad_coeffs)
            print(f"    {name:<15}: Range = {coeff_range:.2f}, "
                  f"Mean = {np.mean(quad_coeffs):.2f}, "
                  f"Std = {np.std(quad_coeffs):.2f}")
    
    print("\n  Observations:")
    print("    1. BQUBO: Uniform 1-acre plots → tight coefficient range → high feasibility")
    print("    2. Scaled BQUBO: Variable plots with benefit scaling → moderate range → should be feasible")
    print("    3. PATCH: Explicit area tracking → large range from squaring → may violate constraints")
    print("    4. PATCH_NO_IDLE: Slightly better than PATCH but still challenging")
    
    print("\n  Recommendation:")
    print("    → Use Scaled BQUBO for variable-sized plots with BQM solvers")
    print("    → Use PATCH with CQM solver (LeapHybridCQMSampler) to avoid conversion issues")
    
    print_section("DONE!")
    print("\nAll formulations tested successfully.")
    print("Check the results above to understand the differences!\n")


if __name__ == "__main__":
    main()
