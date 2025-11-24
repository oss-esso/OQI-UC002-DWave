# PATCH Scenario Infeasibility Fix - Summary

**Date:** November 24, 2025  
**Issue:** PATCH scenario reported "Infeasible" by Gurobi  
**Status:** ✅ RESOLVED

## Problem Identification

### Root Cause
The PATCH scenario (binary formulation) was infeasible due to **unconditional minimum plot constraints**:

- **27 crops** in the full_family scenario
- Each crop had `min_planting_area = 0.01 ha`
- With `plot_area = 4.0 ha`, this required `ceil(0.01/4.0) = 1 plot minimum` per crop
- **Total requirement:** 27 crops × 1 plot = **27 plots**
- **Available plots:** Only **25 plots**
- **Result:** 27 > 25 = **INFEASIBLE**

### Why FARM Scenario Worked
The FARM scenario uses **continuous variables** with **conditional constraints**:
- Constraints: `IF Y_{farm,crop}=1 THEN A_{farm,crop} >= min_area`
- Multiple farms can share the burden of minimum requirements
- More flexible allocation across distributed land units

## Solution Implementation

### Conditional Minimum/Maximum Constraints

Implemented **conditional constraints** for the PATCH scenario (binary formulation):

**Before (Unconditional):**
```python
# Every crop MUST have at least min_plots, even if not planted
for crop in foods:
    if min_planting_area[crop] > 0:
        min_plots = ceil(min_planting_area[crop] / plot_area)
        sum(Y[patch, crop] for patch in patches) >= min_plots  # INFEASIBLE!
```

**After (Conditional):**
```python
# IF a crop is planted ANYWHERE, THEN it must use at least min_plots
for crop in foods:
    if min_planting_area[crop] > 0:
        min_plots = ceil(min_planting_area[crop] / plot_area)
        for patch in patches:
            # If Y[patch,crop]=1, then total >= min_plots
            sum(Y[p, crop] for p in patches) >= min_plots * Y[patch, crop]
```

**Key Insight:**
- If crop is **not planted** anywhere: `sum(Y) = 0` → constraint not triggered ✓
- If crop **is planted** on any patch: `sum(Y) >= min_plots` → ensures minimum ✓

## Results

### Before Fix
```
PATCH Scenario:
  Variables: 675
  Constraints: 89
  Status: Infeasible ❌
```

### After Fix
```
PATCH Scenario:
  Variables: 675
  Constraints: 737
  Status: Optimal ✅
  Objective: 0.305130
  Solve Time: 0.084s
```

### Constraint Verification
Testing with 25 patches confirmed:
- **5 crops planted:** All respect `min_plots <= actual <= max_plots` ✓
- **22 crops unplanted:** Correctly have 0 plots (no violation) ✓
- **All constraints satisfied** ✓

## Files Modified

1. **`solver_runner_CUSTOM_HYBRID.py`**
   - `create_cqm_plots()`: Added conditional min/max constraints for CQM
   - `solve_with_pulp_plots()`: Added conditional min/max constraints for PuLP
   - Updated progress bar calculations

## Technical Details

### CQM Implementation
```python
for food in foods:
    if food in min_planting_area and min_planting_area[food] > 0:
        min_plots = math.ceil(min_planting_area[food] / plot_area)
        total_assignments = sum(Y[(farm, food)] for farm in farms)
        
        for farm in farms:
            # If Y_{farm,food} = 1, then sum(Y_{*,food}) >= min_plots
            cqm.add_constraint(
                total_assignments - min_plots * Y[(farm, food)] >= 0,
                label=f"Min_Plots_If_Selected_{farm}_{food}"
            )
```

### PuLP Implementation
```python
for crop in foods:
    if crop in min_planting_area and min_planting_area[crop] > 0:
        min_plots = math.ceil(min_planting_area[crop] / plot_area)
        total_crop_assignments = pl.lpSum([X_pulp[(f, crop)] for f in farms])
        
        for f in farms:
            model += total_crop_assignments >= min_plots * X_pulp[(f, crop)], \
                     f"Min_Plots_If_{f}_{crop}"
```

## Verification

Created `verify_conditional_constraints.py` to validate:
- ✅ Unplanted crops have 0 plots (no min violation)
- ✅ Planted crops satisfy min/max bounds
- ✅ Solution is optimal and feasible

## Impact

- **PATCH scenario** now produces feasible, optimal solutions
- **No degradation** to FARM scenario performance
- **Proper semantics**: Minimum requirements only apply when crop is selected
- **Scalable**: Works for any number of patches and crops

## Conclusion

The fix successfully resolves the infeasibility by making minimum/maximum plot constraints **conditional** on crop selection. This aligns the binary (PATCH) formulation with the continuous (FARM) formulation's logical behavior: constraints only apply when crops are actually planted.
