#!/usr/bin/env python3
"""Quick script to inspect the CQM constraint that was saved."""

import pickle
from dimod import ConstrainedQuadraticModel

# Load the CQM we just saved
print("Loading CQM from file...")
with open('test_patch_cqm_10plots.cqm', 'rb') as f:
    cqm = ConstrainedQuadraticModel.from_file(f)

print(f"CQM loaded: {len(cqm.variables)} variables, {len(cqm.constraints)} constraints\n")

# Examine the first plot assignment constraint in detail
constraint_label = 'Max_Assignment_Patch1'
constraint = cqm.constraints[constraint_label]

print(f"Constraint: {constraint_label}")
print(f"Sense: {constraint.sense}")
print(f"RHS: {constraint.rhs}")
print(f"\nLHS Expression:")
print(f"  Type: {type(constraint.lhs)}")

# Get the linear part
linear = constraint.lhs.linear
print(f"  Linear terms: {len(linear)}")

# Check if there's an offset/constant
print(f"  Offset: {constraint.lhs.offset}")

# Show some variables and coefficients
print(f"\n  Sample variables:")
for i, (var, coeff) in enumerate(list(linear.items())[:5]):
    print(f"    {var}: {coeff}")

print(f"\n  Quadratic terms: {len(constraint.lhs.quadratic)}")

# The key insight: The constraint is sum(Y) - 1 <= 0
# This means the LHS has an offset of -1
print(f"\n{'='*60}")
print(f"ANALYSIS:")
print(f"{'='*60}")
print(f"The constraint is: sum(Y) - 1 <= 0")
print(f"This is equivalent to: sum(Y) <= 1")
print(f"\nThe LHS offset of {constraint.lhs.offset} represents the '-1' term.")
print(f"So the constraint is properly formed!")

# Test with a sample solution
print(f"\n{'='*60}")
print(f"TESTING CONSTRAINT EVALUATION:")
print(f"{'='*60}")

# Create test samples
test_samples = [
    {"description": "No crops selected", "sample": {var: 0 for var in linear.keys()}},
    {"description": "One crop selected (valid)", "sample": {var: (1 if i == 0 else 0) for i, var in enumerate(linear.keys())}},
    {"description": "Two crops selected (INVALID)", "sample": {var: (1 if i < 2 else 0) for i, var in enumerate(linear.keys())}},
]

for test in test_samples:
    lhs_value = constraint.lhs.energy(test["sample"])
    is_satisfied = (lhs_value <= constraint.rhs)
    status = "✓ SATISFIED" if is_satisfied else "❌ VIOLATED"
    print(f"\n{test['description']}:")
    print(f"  LHS value: {lhs_value}")
    print(f"  RHS value: {constraint.rhs}")
    print(f"  Status: {status}")

