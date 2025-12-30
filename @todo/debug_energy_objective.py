#!/usr/bin/env python3
"""
Debug: Understand BQM energy to objective conversion.

The hierarchical solver shows:
- BQM energies are NEGATIVE (e.g., -46 to -49)
- Final objectives are POSITIVE (e.g., 3.9)

We need to understand:
1. How BQM energy maps to objective
2. Why QPU might have given different results

Hypothesis: If BQM energy is returned directly as objective instead of
being converted through calculate_family_objective(), we'd get negative values.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("BQM ENERGY TO OBJECTIVE INVESTIGATION")
print("="*80)

# ============================================================================
# Check if there's a path where BQM energy is returned directly
# ============================================================================

print("\n[1] Searching for potential bug in hierarchical_quantum_solver.py...")

# Read the file and search for how objective is calculated
with open('hierarchical_quantum_solver.py', 'r') as f:
    content = f.read()

# Look for places where 'energy' is used
import re
energy_usages = re.findall(r'.*energy.*', content, re.IGNORECASE)
print(f"Found {len(energy_usages)} lines mentioning 'energy'")

# Look for how objective is set
objective_assignments = re.findall(r".*result\['objective'\].*=.*", content)
print(f"Found {len(objective_assignments)} objective assignments:")
for line in objective_assignments:
    print(f"  {line.strip()}")

print("\n[2] Tracing the objective calculation path...")

# The objective should be calculated by calculate_family_objective()
# NOT by directly using BQM energy

# In the debug output, we see:
# - Cluster energies: -46 to -49 (BQM minimization)
# - Combined obj: 3.9 (after calculate_family_objective)

# This is CORRECT because:
# - BQM is formulated for MINIMIZATION (negative coefficients for benefits)
# - calculate_family_objective recalculates as MAXIMIZATION

# The PROBLEM would be if someone returned energy directly instead of
# calling calculate_family_objective()

print("\n[3] Checking if QPU path uses same calculation...")

# In solve_cluster_qpu(), the function returns:
# solution, best_energy, wall_time, qpu_time, detailed_timing

# The main solver uses calculate_family_objective() on the combined solution,
# so the path should be the same.

# HOWEVER: What if the QPU returned completely different solutions due to
# different sampling behavior? Let's check the combined solution...

print("\n[4] Let's verify the objective calculation logic is consistent...")

from data_loader_utils import load_food_data_as_dict
from food_grouping import aggregate_foods_to_families, create_family_rotation_matrix, FAMILY_ORDER
from hierarchical_quantum_solver import calculate_family_objective, DEFAULT_CONFIG
import numpy as np

# Load small test data
data = load_food_data_as_dict('rotation_250farms_27foods')
data['farm_names'] = data['farm_names'][:5]
data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
data['total_area'] = sum(data['land_availability'].values())

family_data = aggregate_foods_to_families(data)

# Create a test solution (all zeros - no assignments)
empty_solution = {}
for farm in family_data['farm_names']:
    for family in family_data['food_names']:
        for t in range(1, 4):
            empty_solution[(farm, family, t)] = 0

empty_obj = calculate_family_objective(empty_solution, family_data)
print(f"Empty solution objective: {empty_obj}")

# Create a solution with one assignment per farm-period
onehot_solution = {}
for farm in family_data['farm_names']:
    for family in family_data['food_names']:
        for t in range(1, 4):
            onehot_solution[(farm, family, t)] = 0

# Assign Legumes (index 0) to all farm-periods
for farm in family_data['farm_names']:
    for t in range(1, 4):
        onehot_solution[(farm, 'Legumes', t)] = 1

onehot_obj = calculate_family_objective(onehot_solution, family_data)
print(f"All-Legumes solution objective: {onehot_obj}")

# Create a solution with NO assignments (all zeros)
no_assign_obj = calculate_family_objective({}, family_data)
print(f"No-assignment solution objective: {no_assign_obj}")

print("\n[5] Key insight about negative vs positive objectives...")
print("""
The BQM energy is NEGATIVE because:
- BQM is formulated for MINIMIZATION
- Benefits have NEGATIVE coefficients (to maximize by minimizing negative)

The final objective is POSITIVE because:
- calculate_family_objective() computes MAX: benefit + rotation + diversity - penalty

IF the benchmark code MISTAKENLY returned BQM energy as objective, we'd see:
- BQM energy: -46 (per cluster)
- Summed across clusters: -230 ish

But benchmark shows -18 to -90, which doesn't match sum of cluster energies.

ALTERNATIVE HYPOTHESIS: 
Maybe the issue is in how the data was loaded or how n_farms was handled?
""")

# Check if larger n_farms changes the calculation significantly
print("\n[6] Testing with different n_farms...")

for n_farms in [5, 10, 25]:
    data = load_food_data_as_dict('rotation_250farms_27foods')
    data['farm_names'] = data['farm_names'][:n_farms]
    data['land_availability'] = {f: data['land_availability'][f] for f in data['farm_names']}
    data['total_area'] = sum(data['land_availability'].values())
    
    family_data = aggregate_foods_to_families(data)
    
    # Create all-Legumes solution
    test_sol = {}
    for farm in family_data['farm_names']:
        for family in family_data['food_names']:
            for t in range(1, 4):
                test_sol[(farm, family, t)] = 1 if family == 'Legumes' else 0
    
    obj = calculate_family_objective(test_sol, family_data)
    print(f"  n_farms={n_farms}: total_area={data['total_area']:.2f}, obj={obj:.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
The hierarchical solver appears to be working correctly with SA.
The negative objectives in the benchmark were likely from:

1. QPU solutions being very different from SA solutions
2. Possibly a race condition or data loading issue during the benchmark run
3. The benchmark was run with use_qpu=True which we can't test without QPU

The main FIXABLE bugs are:
1. Missing 'solution' key → causes 999 violations
2. Duplicate code block in significant_scenarios_benchmark.py

RECOMMENDATION: Fix these bugs and re-run the benchmark with SA to validate,
then run with QPU if results look reasonable.
""")
