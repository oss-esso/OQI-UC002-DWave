# Patch CQM Constraint Bug Report

## Issue Summary

Both Patch DWave CQM and Patch DWave BQM are returning infeasible solutions that violate the "at most one crop per plot" constraint. The root cause is **conflicting constraints** that make the problem infeasible.

## Root Cause Analysis

### The Problem

The binary formulation in `solver_runner_BINARY.py` creates food group constraints **PER PLOT**:

```python
# Food group constraints - WRONG: Applied per plot!
for farm in farms:
    if 'min_foods' in constraints:
        cqm.add_constraint(
            sum(Y[(farm, food)] for food in foods_in_group) >= min_foods,
            label=f"Food_Group_Min_{group}_{farm}"
        )
```

With the current configuration:
- 5 food groups
- `min_foods = 1` for each group
- 10 plots

This creates constraints saying: "**Each plot must have at least 1 food from each of the 5 groups**"

But we also have:
```python
# Plot assignment constraint
cqm.add_constraint(
    sum(Y[(farm, food)] for food in foods) <= 1,
    label=f"Max_Assignment_{farm}"
)
```

### The Conflict

- Plot assignment constraint: "At most 1 crop per plot"
- Food group constraints: "At least 5 crops per plot" (1 from each of 5 groups)

**These constraints are mathematically incompatible!**

### Test Results

From our targeted test (`test_patch_cqm_constraints.py`):

```
Crops per plot:
  ❌ Patch1: 3 crops - ['Mango', 'Orange', 'Peanuts']
  ❌ Patch2: 2 crops - ['Cucumber', 'Tempeh']
  ✓ Patch4: 1 crops - ['Guava']
  ... (9 out of 10 plots violated)

CQM Status: Infeasible (is_feasible=False)
Validation Violations: 9
```

The DWave CQM solver **correctly identifies the solution as infeasible**, but returns the best infeasible solution it can find (which violates constraints).

## The Fix

Food group constraints should be **GLOBAL** across all plots, not per-plot:

### Current (Wrong):
```python
# Per-plot: Each plot needs 1 food from each group
for farm in farms:
    for group, constraints in food_group_constraints.items():
        cqm.add_constraint(
            sum(Y[(farm, food)] for food in foods_in_group) >= min_foods,
            label=f"Food_Group_Min_{group}_{farm}"
        )
```

### Corrected (Right):
```python
# Global: Across ALL plots, we need at least min_foods from each group
for group, constraints in food_group_constraints.items():
    foods_in_group = food_groups.get(group, [])
    if 'min_foods' in constraints:
        cqm.add_constraint(
            sum(Y[(farm, food)] for farm in farms for food in foods_in_group) >= min_foods,
            label=f"Food_Group_Min_{group}_Global"
        )
```

This would mean: "Across all 10 plots, at least 1 plot must grow something from each food group" - which is feasible!

## Impact

This bug affects:
1. **Patch DWave CQM** - Returns infeasible solutions
2. **Patch DWave BQM** - Returns infeasible solutions (same constraints converted to penalties)
3. **Benchmark results** - All Patch DWave results show constraint violations

## Recommended Actions

1. **Fix `create_cqm_plots()` in `solver_runner_BINARY.py`**
   - Change food group constraints from per-plot to global

2. **Update `comprehensive_benchmark.py`** 
   - Consider filtering out infeasible solutions or marking them clearly

3. **Re-run benchmarks** after fix to get valid results

4. **Review continuous formulation** in `solver_runner.py`
   - Check if food group constraints have the same issue there

## Files Affected

- `solver_runner_BINARY.py` - Lines 473-507 (food group constraints)
- `comprehensive_benchmark.py` - May need to handle infeasible solutions better
- All benchmark results in `Benchmarks/COMPREHENSIVE/Patch_DWave*/` - Invalid due to this bug

