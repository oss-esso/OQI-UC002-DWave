# Constraint Comparison Files - Quick Reference

## Overview

When CQM validation runs, it automatically generates a detailed constraint comparison file showing all PuLP vs CQM constraints side-by-side.

## File Location

```
Benchmarks/COMPREHENSIVE/constraint_comparisons/
└── constraint_comparison_{scenario}_{N}units_{timestamp}.txt
```

**Examples:**
- `constraint_comparison_patch_10units_20251117_161023.txt`
- `constraint_comparison_farm_15units_20251117_161145.txt`

## File Structure

The comparison file has 4 main sections:

### Section 1: PuLP Constraints
Lists all constraints from the PuLP (Gurobi) model, grouped by type:
- `max_area`: Maximum area constraints
- `min_area`: Minimum area constraints
- `at_most_one`: At most one crop per plot
- `food_group`: Food group diversity
- `coupling`: Variable coupling constraints
- `other`: Uncategorized

For each constraint:
- Name
- Full representation (truncated if too long)

### Section 2: CQM Constraints
Lists all constraints from the CQM model, grouped by type.

For each constraint:
- Name
- Sense (LE, GE, EQ)
- RHS (right-hand side value)
- Number of variables involved

### Section 3: Comparison Summary
Highlights differences:
- Constraint types only in PuLP
- Constraint types only in CQM
- Constraint types in both (with count comparison)

Example:
```
❌ Only in PuLP: ['at_most_one']
   - at_most_one: 10 constraints

⚠️  Only in CQM: ['coupling']
   - coupling: 270 constraints

✓ In both: ['food_group']
   ❌ food_group: PuLP=10, CQM=0  ← COUNT MISMATCH!
```

### Section 4: Validation Discrepancies
Lists all critical issues found during validation with details.

## Example Output

```
================================================================================
CONSTRAINT COMPARISON: PuLP vs CQM - PATCH SCENARIO
================================================================================
Generated: 2025-11-17T16:10:23
Configuration: 10 units
Total foods: 27

================================================================================
SECTION 1: PuLP CONSTRAINTS (Total: 20)
================================================================================

--- MAX_AREA (10 constraints) ---

1. Max_Area_Patch1
   sum([Y_Patch1_Wheat, Y_Patch1_Rice, Y_Patch1_Maize, ...]) <= 1

2. Max_Area_Patch2
   sum([Y_Patch2_Wheat, Y_Patch2_Rice, Y_Patch2_Maize, ...]) <= 1

--- FOOD_GROUP (10 constraints) ---

1. MinFoodGroup_Grains
   Y_Patch1_Wheat + Y_Patch2_Wheat + ... >= 1

2. MaxFoodGroup_Grains
   Y_Patch1_Wheat + Y_Patch2_Wheat + ... <= 10

================================================================================
SECTION 2: CQM CONSTRAINTS (Total: 47)
================================================================================

--- COUPLING (27 constraints) ---

1. Coupling_Patch1_Wheat
   Sense: EQ
   RHS: 0
   Variables: 2

--- OTHER (20 constraints) ---

1. MinPlotUtilization_Patch1
   Sense: GE
   RHS: 0.0001
   Variables: 27

================================================================================
SECTION 3: COMPARISON SUMMARY
================================================================================

Constraint Type Comparison:
  PuLP types: ['food_group', 'max_area']
  CQM types: ['coupling', 'other']

  ❌ Only in PuLP: ['food_group', 'max_area']
     - food_group: 10 constraints
     - max_area: 10 constraints  ← MISSING IN CQM!

  ⚠️  Only in CQM: ['coupling', 'other']
     - coupling: 27 constraints
     - other: 20 constraints

================================================================================
SECTION 4: VALIDATION DISCREPANCIES (3)
================================================================================

1. MISSING_AT_MOST_ONE_CONSTRAINTS [critical]
   Expected 10 'at most one' constraints, found 0
   expected: 10
   found: 0

2. FOOD_GROUP_COUNT_MISMATCH [error]
   Food group constraint count mismatch: PuLP=10, CQM=0
   pulp_count: 10
   cqm_count: 0
```

## How to Use

1. **Run benchmark** - Comparison file is generated automatically
2. **Check console output** - Shows path to comparison file
3. **Open comparison file** - Review detailed constraint differences
4. **Identify missing constraints** - Look at "Only in PuLP" section
5. **Fix formulation** - Update `solver_runner_BINARY.py` to add missing constraints
6. **Re-run validation** - New comparison file shows if issues are fixed

## Common Issues to Look For

### Missing "At Most One" Constraints
**In PuLP:**
```
Max_Area_Patch1: sum(Y_Patch1_*) <= 1
```

**In CQM:**
```
(Missing or named differently)
```

**Fix:** Add constraint in `create_cqm_plots()`:
```python
for plot in plots_list:
    cqm.add_constraint(
        sum(Y[plot, crop] for crop in foods) <= 1,
        label=f"AtMostOne_{plot}"
    )
```

### Missing Food Group Constraints
**In PuLP:**
```
MinFoodGroup_Grains: Y_*_Wheat + Y_*_Rice + ... >= 1
MaxFoodGroup_Grains: Y_*_Wheat + Y_*_Rice + ... <= 10
```

**In CQM:**
```
(Missing)
```

**Fix:** Add food group constraints in `create_cqm_plots()`.

### Different Constraint Names
Sometimes constraints exist but have different names. The comparison file helps identify if they're truly missing or just renamed.

## Troubleshooting

### File Not Created
- Check console output for error messages
- Verify `Benchmarks/COMPREHENSIVE/constraint_comparisons/` directory exists
- Check file permissions

### Can't Read File
- File is plain text - open with any text editor
- On Mac: `cat path/to/constraint_comparison_*.txt`
- On Windows: `type path\to\constraint_comparison_*.txt`

### Too Long to Read
- Search for "SECTION 3" to jump to comparison summary
- Search for "SECTION 4" to jump to discrepancies
- Use grep/find to search for specific constraint types

## Integration

The comparison file is generated automatically as part of the validation process:

```python
from Utils.validate_cqm_vs_pulp import validate_before_dwave_submission

is_valid = validate_before_dwave_submission(
    cqm=cqm,
    pulp_model=pulp_model,
    scenario_type='patch',
    scenario_info={'n_units': 10, 'n_foods': 27}
)

# Comparison file is automatically created in:
# Benchmarks/COMPREHENSIVE/constraint_comparisons/
```

## Summary

✅ **Automatic generation** during validation
✅ **Detailed constraint listing** for both PuLP and CQM
✅ **Side-by-side comparison** with highlighted differences
✅ **Saved permanently** for future reference
✅ **Plain text format** easy to read and search
✅ **Separate files** for Farm vs Patch scenarios
