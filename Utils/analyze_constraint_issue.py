#!/usr/bin/env python3
"""Analyze why CQM constraints appear satisfied when they're not."""

from dimod import ConstrainedQuadraticModel

# Load the saved CQM
print("Loading CQM...")
with open('test_patch_cqm_10plots.cqm', 'rb') as f:
    cqm = ConstrainedQuadraticModel.from_file(f)

print(f"CQM: {len(cqm.variables)} variables, {len(cqm.constraints)} constraints\n")

# Create a test solution that violates constraints
# Patch1 gets 3 crops (like in the actual DWave solution)
test_solution = {var: 0 for var in cqm.variables}

# Assign 3 crops to Patch1 (should violate constraint)
test_solution['Y_Patch1_Mango'] = 1
test_solution['Y_Patch1_Orange'] = 1
test_solution['Y_Patch1_Peanuts'] = 1

# Assign 2 crops to Patch2 (should violate constraint)
test_solution['Y_Patch2_Cucumber'] = 1
test_solution['Y_Patch2_Tempeh'] = 1

# Assign 1 crop to Patch4 (should be OK)
test_solution['Y_Patch4_Guava'] = 1

print("Test solution created:")
print("  Patch1: 3 crops (Mango, Orange, Peanuts)")
print("  Patch2: 2 crops (Cucumber, Tempeh)")
print("  Patch4: 1 crop (Guava)")
print("  Others: 0 crops\n")

# Check the Patch1 constraint
print("="*70)
print("ANALYZING PATCH1 CONSTRAINT")
print("="*70)

constraint = cqm.constraints['Max_Assignment_Patch1']
print(f"Constraint label: Max_Assignment_Patch1")
print(f"Constraint sense: {constraint.sense}")
print(f"Constraint RHS: {constraint.rhs}")
print(f"LHS offset: {constraint.lhs.offset}")
print(f"\nMathematically: sum(Y_Patch1_*) - 1 <= 0")
print(f"Which means: sum(Y_Patch1_*) <= 1")

# Evaluate the constraint
lhs_value = constraint.lhs.energy(test_solution)
rhs_value = constraint.rhs

print(f"\nEvaluation with test solution:")
print(f"  LHS value (sum(Y) + offset): {lhs_value}")
print(f"  RHS value: {rhs_value}")
print(f"  Is LHS <= RHS? {lhs_value <= rhs_value}")

# Manual calculation
patch1_vars = [var for var in cqm.variables if var.startswith('Y_Patch1_')]
sum_y = sum(test_solution.get(var, 0) for var in patch1_vars)
print(f"\nManual calculation:")
print(f"  Sum of Y_Patch1_* variables: {sum_y}")
print(f"  With offset (-1): {sum_y} + (-1) = {sum_y - 1}")
print(f"  Constraint: {sum_y - 1} <= 0")
print(f"  Violated? {(sum_y - 1) > 0}")

# The key insight!
print(f"\n{'='*70}")
print(f"KEY INSIGHT:")
print(f"{'='*70}")
print(f"The constraint IS properly formulated as: sum(Y) - 1 <= 0")
print(f"With 3 crops selected: 3 - 1 = 2, and 2 <= 0 is FALSE")
print(f"So the constraint SHOULD be violated.")
print(f"\nIf DWave reports this as feasible, it means:")
print(f"1. DWave's CQM solver is treating this as a SOFT constraint")
print(f"2. Or there's a numerical tolerance issue")
print(f"3. Or the constraint is being relaxed during solving")

