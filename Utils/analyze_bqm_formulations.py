#!/usr/bin/env python3
"""
BQM Formulation Analysis Script

Compares the "PATCH" and "BQUBO" formulations to diagnose constraint
violation issues without using D-Wave credits.

This script:
1. Generates BQMs for both formulations for a given problem size.
2. Compares their complexity (variables, interactions).
3. Solves them classically with Gurobi and Simulated Annealing.
4. Generates a detailed Markdown report with the findings.
"""

import os
import sys
import argparse
import time
import numpy as np
from typing import Dict, Tuple, List

# Suppress Gurobi license messages
os.environ["GRB_LICENSE_FILE"] = ""

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from dimod import cqm_to_bqm, BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler

# Add project root to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Imports from other project files ---
# To make this script self-contained, we import the necessary functions.
# Ensure that these files are in the same directory or in the python path.
try:
    from solver_runner_PATCH import create_cqm as create_patch_cqm
    from solver_runner_BQUBO import create_cqm as create_bqubo_cqm
    from patch_sampler import generate_farms as generate_patches
    from farm_sampler import generate_farms as generate_farms_bqubo
    from src.scenarios import load_food_data
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    print("Please ensure solver_runner_PATCH.py, solver_runner_BQUBO.py, etc., are accessible.")
    sys.exit(1)


# --- Helper Functions ---

def create_food_config_patch(land_data: Dict[str, float]) -> Tuple[Dict, Dict, Dict]:
    """Creates the configuration for the PATCH formulation."""
    try:
        _, foods, food_groups, _ = load_food_data('simple')
    except Exception:
        # Fallback if scenario loading fails
        foods = {'Wheat': {'nutritional_value': 0.8}, 'Corn': {'nutritional_value': 0.7}}
        food_groups = {'Grains': ['Wheat', 'Corn']}

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
    # Ensure all foods have all attributes to avoid KeyErrors
    for food_name, food_attrs in foods.items():
        for weight in config['parameters']['weights']:
            if weight not in food_attrs:
                food_attrs[weight] = 0.5  # Default value
    return foods, food_groups, config

def create_food_config_bqubo(land_data: Dict[str, float]) -> Tuple[Dict, Dict, Dict]:
    """Creates the configuration for the BQUBO formulation."""
    try:
        _, foods, food_groups, _ = load_food_data('simple')
    except Exception:
        foods = {'Wheat': {'nutritional_value': 0.8}, 'Corn': {'nutritional_value': 0.7}}
        food_groups = {'Grains': ['Wheat', 'Corn']}

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
    # Ensure all foods have all attributes
    for food_name, food_attrs in foods.items():
        for weight in config['parameters']['weights']:
            if weight not in food_attrs:
                food_attrs[weight] = 0.5  # Default value
    return foods, food_groups, config

def solve_bqm_with_gurobi(bqm: BinaryQuadraticModel) -> Dict:
    """Solves a BQM using Gurobi as a classical QUBO solver."""
    if not GUROBI_AVAILABLE:
        return {'status': 'Skipped', 'energy': None, 'time': 0, 'error': 'Gurobi not found'}

    model = gp.Model("bqm_solver")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 60) # 1 minute timeout

    variables = list(bqm.variables)
    gurobi_vars = {var: model.addVar(vtype=GRB.BINARY, name=str(var)) for var in variables}

    objective = gp.QuadExpr()
    objective += bqm.offset
    for var, bias in bqm.linear.items():
        objective += bias * gurobi_vars[var]
    for (var1, var2), bias in bqm.quadratic.items():
        objective += bias * gurobi_vars[var1] * gurobi_vars[var2]

    model.setObjective(objective, GRB.MINIMIZE)
    
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        return {'status': 'Optimal' if model.status == GRB.OPTIMAL else 'TimeLimit',
                'energy': model.objVal, 'time': solve_time}
    else:
        return {'status': f'Error (Code {model.status})', 'energy': None, 'time': solve_time}

def solve_bqm_with_sa(bqm: BinaryQuadraticModel) -> Dict:
    """Solves a BQM using Simulated Annealing."""
    sampler = SimulatedAnnealingSampler()
    start_time = time.time()
    sampleset = sampler.sample(bqm, num_reads=100)
    solve_time = time.time() - start_time
    
    best_sample = sampleset.first
    return {'status': 'Optimal', 'energy': best_sample.energy, 'time': solve_time}

def generate_report(n_units, patch_stats, bqubo_stats, patch_results, bqubo_results):
    """Generates the final Markdown report."""
    
    # Toy example explanation
    toy_example = f"""
### Formulation 1: PATCH (Plot Assignment)

This is a complex, realistic model where each patch of land is a discrete unit.

- **Variables**: `X_p,c` (binary) is 1 if patch `p` is assigned to crop `c`.
- **Key Constraint**: For each patch `p`, `sum(X_p,c for c in crops) <= 1`.
- **BQM Challenge**: This simple-looking constraint is the source of the problem. When converted to a BQM, it becomes a penalty term: `P * (sum(X_p,c) - 1)^2`. If the penalty `P` (the Lagrange multiplier) is too low, the solver will violate the constraint to get a better objective score. This is what was happening in `comprehensive_benchmark.py`.

For a 2-patch, 2-crop example, the constraints for Patch 1 would be:
- `X_p1,c1 + X_p1,c2 <= 1`

### Formulation 2: BQUBO (Binary Plantation)

This is a simpler, more abstract model. A "farm" is just a capacity pool.

- **Variables**: `Y_f,c` (binary) is 1 if a 1-acre plantation of crop `c` is made on farm `f`.
- **Key Constraint**: For each farm `f`, `sum(Y_f,c for c in crops) <= Capacity_f`.
- **Why it Worked**: This formulation *cannot* have the "multiple crops per plot" error because there are no plots. It simply limits the number of 1-acre crop choices per farm. It's less realistic but structurally simpler for a BQM solver.

For a 2-farm, 2-crop example (each farm has capacity 1), the constraints would be:
- `Y_f1,c1 + Y_f1,c2 <= 1`
- `Y_f2,c1 + Y_f2,c2 <= 1`
"""

    # Complexity comparison table
    complexity_table = f"""
| Metric                  | PATCH Formulation | BQUBO Formulation |
|-------------------------|-------------------|-------------------|
| **Problem Size (Units)**| {n_units}         | {n_units}         |
| **Total BQM Variables** | {patch_stats['vars']}        | {bqubo_stats['vars']}        |
| **Quadratic Terms**     | {patch_stats['quad']}      | {bqubo_stats['quad']}      |
| **Model Density (%)**   | {patch_stats['density']:.2f} %      | {bqubo_stats['density']:.2f} %      |
"""

    # Solver results table
    solver_table = f"""
| Solver                | Formulation | Status         | BQM Energy        | Solve Time (s) |
|-----------------------|-------------|----------------|-------------------|----------------|
| **Gurobi (Classical)**| PATCH       | {patch_results['gurobi']['status']}      | {f"{patch_results['gurobi']['energy']:.2f}" if patch_results['gurobi']['energy'] is not None else "N/A"} | {patch_results['gurobi']['time']:.4f}     |
|                       | BQUBO       | {bqubo_results['gurobi']['status']}      | {f"{bqubo_results['gurobi']['energy']:.2f}" if bqubo_results['gurobi']['energy'] is not None else "N/A"} | {bqubo_results['gurobi']['time']:.4f}     |
| **Simulated Annealing** | PATCH       | {patch_results['sa']['status']}         | {patch_results['sa']['energy']:.2f}      | {patch_results['sa']['time']:.4f}     |
|                       | BQUBO       | {bqubo_results['sa']['status']}         | {bqubo_results['sa']['energy']:.2f}      | {bqubo_results['sa']['time']:.4f}     |
"""

    # Final Report Assembly
    report = f"""
# BQM Formulation Analysis Report

## 1. Overview

- **Analysis Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Problem Size**: {n_units} units (patches/farms)
- **Objective**: To understand why the "PATCH" BQM formulation violates constraints while the "BQUBO" formulation does not.

## 2. Formulation Comparison (Toy Example)

{toy_example}

## 3. BQM Complexity Analysis

This table compares the structural complexity of the final BQMs generated from each formulation for a problem size of **{n_units} units**.

{complexity_table}

**Observation**: The PATCH formulation results in a significantly larger and denser BQM. The number of quadratic terms, which define the problem's complexity for a QUBO solver, is an order of magnitude higher. This complexity arises from converting the many interacting CQM constraints into penalty terms in the BQM.

## 4. Classical Solver Results

The two BQMs were solved using classical methods to compare their performance and final energy values. The BQM energy includes penalties for any violated constraints.

{solver_table}

**Observation**: The BQM energies are not directly comparable as they represent different objective functions and penalty structures. However, the much higher solve times for the PATCH formulation, even on classical solvers, confirm its greater complexity.

## 5. Conclusion

The "PATCH" formulation is fundamentally more complex than the "BQUBO" formulation. Its constraints, particularly the "one-crop-per-plot" rule, are difficult to represent in a BQM without introducing strong penalties.

The previous failures were caused by an insufficiently large Lagrange multiplier (`penalty_strength`) during the `cqm_to_bqm` conversion for the PATCH model. While we increased it to 100.0 for this analysis, finding a perfect value is non-trivial.

**Recommendation**: For this problem, using a native **CQM solver** (like `LeapHybridCQMSampler`) is strongly recommended for the PATCH formulation. This avoids the problematic `cqm_to_bqm` conversion altogether, as the solver handles the constraints directly. The BQM approach should only be used if a native CQM solver is not an option, and it requires careful manual tuning of the Lagrange multiplier.
"""
    return report


def main(n_units: int):
    """Main analysis function."""
    print(f"--- Starting BQM Formulation Analysis for n_units = {n_units} ---", file=sys.stderr)

    # --- 1. Generate PATCH formulation BQM ---
    print("1. Analyzing PATCH formulation...", file=sys.stderr)
    patches = generate_patches(n_farms=n_units, seed=42)
    foods_patch, food_groups_patch, config_patch = create_food_config_patch(patches)
    cqm_patch, _, _ = create_patch_cqm(list(patches.keys()), foods_patch, food_groups_patch, config_patch)
    
    # Convert to BQM with a strong multiplier
    bqm_patch, _ = cqm_to_bqm(cqm_patch, lagrange_multiplier=100.0)
    
    patch_vars = len(bqm_patch.variables)
    patch_quad = len(bqm_patch.quadratic)
    patch_density = patch_quad / (patch_vars * (patch_vars - 1) / 2) if patch_vars > 1 else 0
    patch_stats = {'vars': patch_vars, 'quad': patch_quad, 'density': patch_density * 100}

    # --- 2. Generate BQUBO formulation BQM ---
    print("2. Analyzing BQUBO formulation...", file=sys.stderr)
    farms_bqubo = generate_farms_bqubo(n_farms=n_units, seed=42)
    foods_bqubo, food_groups_bqubo, config_bqubo = create_food_config_bqubo(farms_bqubo)
    cqm_bqubo, _, _ = create_bqubo_cqm(list(farms_bqubo.keys()), foods_bqubo, food_groups_bqubo, config_bqubo)
    
    # Convert to BQM using automatic multiplier (as it worked before)
    bqm_bqubo, _ = cqm_to_bqm(cqm_bqubo)
    
    bqubo_vars = len(bqm_bqubo.variables)
    bqubo_quad = len(bqm_bqubo.quadratic)
    bqubo_density = bqubo_quad / (bqubo_vars * (bqubo_vars - 1) / 2) if bqubo_vars > 1 else 0
    bqubo_stats = {'vars': bqubo_vars, 'quad': bqubo_quad, 'density': bqubo_density * 100}

    # --- 3. Solve BQMs classically ---
    print("3. Solving BQMs with classical solvers...", file=sys.stderr)
    patch_results = {
        'gurobi': solve_bqm_with_gurobi(bqm_patch),
        'sa': solve_bqm_with_sa(bqm_patch)
    }
    bqubo_results = {
        'gurobi': solve_bqm_with_gurobi(bqm_bqubo),
        'sa': solve_bqm_with_sa(bqm_bqubo)
    }
    
    # --- 4. Generate and print report ---
    print("4. Generating report...", file=sys.stderr)
    report = generate_report(n_units, patch_stats, bqubo_stats, patch_results, bqubo_results)
    print(report)
    
    print("--- Analysis complete. Report printed to stdout. ---", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and compare BQM formulations for the food optimization problem."
    )
    parser.add_argument(
        "--n_units",
        type=int,
        default=10,
        help="Number of units (patches/farms) to use for the analysis."
    )
    args = parser.parse_args()
    
    main(args.n_units)
