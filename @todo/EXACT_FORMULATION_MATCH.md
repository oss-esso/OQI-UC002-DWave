# EXACT FORMULATION MATCH REQUIRED

## Problem

The `hierarchical_statistical_test.py` Gurobi formulation does NOT match `statistical_comparison_test.py`.

## Current (WRONG)

```python
# Simple rotation matrix from create_family_rotation_matrix()
# Hard ==1 constraints
# No spatial neighbors
# No frustration
```

## Required (from statistical_comparison_test.py lines 243-445)

```python
# Frustration-based rotation matrix with seed 42
# Spatial neighbor graph (k=4 nearest)
# Soft constraints (≤2, not ==1)
# Quadratic penalties
# Full spatial synergies
```

## Action Required

Replace the entire `solve_gurobi_ground_truth()` function in hierarchical_statistical_test.py with the EXACT code from statistical_comparison_test.py, only adding:
- Line 1: Aggregate 27→6
- Use family_data instead of data for variables

The rest must be IDENTICAL byte-for-byte.
