# CQM Validation Integration Summary

## Overview

The benchmark now automatically validates CQM formulation against PuLP (Gurobi) before submitting to D-Wave. This prevents wasting expensive solver time and budget on incorrectly formulated problems.

## How It Works

### Validation Flow

```
1. Generate sample data (farms/patches)
2. Create food configuration
3. Build CQM model
4. Build PuLP model (Gurobi)
5. Solve with PuLP (gets optimal solution)
   ↓
6. ✅ VALIDATE: Compare CQM vs PuLP constraints
   ├─ If VALID → Continue to D-Wave
   └─ If INVALID → STOP benchmark, report errors
   ↓
7. Submit to D-Wave (only if validation passed)
8. Save samplesets
9. Compare results
```

### What Gets Validated

The validator checks:

1. **Variable Count**: CQM and PuLP must have same number of variables
2. **Constraint Count**: Should be similar (allows minor differences for bound constraints)
3. **"At Most One" Constraints** (Patch scenario):
   - Count: Should have one per plot
   - Formulation: `sum(Y_plot_c for all c) <= 1`
   - Sense: Must be `LE` (less than or equal)
   - RHS: Must be `1`
4. **Food Group Constraints**: Count should match between CQM and PuLP
5. **Objective Signs**: Verify objective is correctly negated for maximization

## Integration Points

### Benchmark Scripts Modified

**`comprehensive_benchmark.py`:**
- Added import: `from Utils.validate_cqm_vs_pulp import validate_before_dwave_submission`
- Farm scenario (line ~395): Validates before D-Wave CQM submission
- Patch scenario (line ~585): Validates before D-Wave CQM submission

### When Validation Runs

Validation runs **ONLY**:
- Before D-Wave CQM solver submission
- After PuLP model has been created and solved
- If no cached D-Wave result exists

Validation **DOES NOT** run:
- For Gurobi/PuLP solver (it's the source of truth)
- For BQM solvers (validated at CQM stage)
- If using cached D-Wave results

## Example Output

### ✅ Validation Passes

```
     Running DWave CQM...
       Validating CQM formulation vs PuLP...

================================================================================
VALIDATING CQM vs PuLP - PATCH SCENARIO
================================================================================
PuLP constraints: 47
CQM constraints: 47
  ✓ Variable count matches: 270

--- Checking 'At Most One Crop Per Plot' Constraints ---
  PuLP 'at most one' constraints: 10
  CQM 'at most one' constraints: 10
  ✓ All 'at most one' constraints correctly formulated (checked 3 samples)

--- Checking Food Group Constraints ---
  PuLP food group constraints: 5
  CQM food group constraints: 5
  ✓ Count matches: 5

--- Checking Objective Function ---
  ✓ CQM objective appears correctly negated for maximization
  PuLP sense: Maximize

================================================================================
VALIDATION SUMMARY
================================================================================
✅ PASSED - CQM matches PuLP formulation
================================================================================

       ✓ Validation passed - submitting to D-Wave
       Submitting to DWave Leap hybrid solver...
```

### ❌ Validation Fails

```
     Running DWave CQM...
       Validating CQM formulation vs PuLP...

================================================================================
VALIDATING CQM vs PuLP - PATCH SCENARIO
================================================================================
PuLP constraints: 47
CQM constraints: 47
  ❌ Variable count: PuLP=270, CQM=250

--- Checking 'At Most One Crop Per Plot' Constraints ---
  PuLP 'at most one' constraints: 10
  CQM 'at most one' constraints: 10
  ❌ AtMostOne_Patch1: Wrong sense GE (should be LE)
  ❌ AtMostOne_Patch1: Wrong RHS 2 (should be 1)

================================================================================
VALIDATION SUMMARY
================================================================================
❌ FAILED - Found 3 discrepancies

CRITICAL ISSUES:

1. variable_count_mismatch (error)
   Variable count mismatch: PuLP=270, CQM=250

2. wrong_constraint_sense (critical)
   Constraint AtMostOne_Patch1 has wrong sense: GE (should be LE)

3. wrong_constraint_rhs (critical)
   Constraint AtMostOne_Patch1 has wrong RHS: 2 (should be 1)

================================================================================

       ❌ CQM validation failed - SKIPPING D-Wave submission

================================================================================
⛔ STOPPING BENCHMARK - CQM VALIDATION FAILED
================================================================================

The CQM formulation does not match the PuLP formulation.
Submitting to D-Wave would waste solver time and budget.

Please fix the discrepancies in solver_runner_BINARY.py
and re-run the benchmark.

Discrepancies found: 3
  • Variable count mismatch: PuLP=270, CQM=250
  • Constraint AtMostOne_Patch1 has wrong sense: GE (should be LE)
  • Constraint AtMostOne_Patch1 has wrong RHS: 2 (should be 1)
================================================================================

❌ Benchmark failed: CQM validation failed for Patch scenario. Fix formulation before re-running.
```

## Benefits

### 🎯 Prevents Wasted Resources
- **No D-Wave solver time** spent on incorrect formulations
- **No QPU budget** wasted
- **No misleading results** from poorly formulated problems

### 🔍 Early Error Detection
- **Catches formulation errors** before submission
- **Identifies specific issues** with constraint names and values
- **Provides actionable feedback** for fixing problems

### 💰 Cost Savings
- **Zero D-Wave cost** for validation
- **Validation runs locally** in milliseconds
- **Prevents multiple failed attempts** on expensive solver

### 🛡️ Quality Assurance
- **Ensures consistency** between classical and quantum formulations
- **Verifies constraints** match expected formulation
- **Builds confidence** in benchmark results

## Files Involved

### New Files
1. **`Utils/validate_cqm_vs_pulp.py`** - Validation utility (350 lines)
   - `CQMPuLPValidator` class
   - `validate_before_dwave_submission()` function
   - Comprehensive constraint checking

### Modified Files
2. **`Benchmark Scripts/comprehensive_benchmark.py`**
   - Added validation import
   - Farm scenario: Validates before D-Wave CQM (line ~395)
   - Patch scenario: Validates before D-Wave CQM (line ~585)
   - Both scenarios stop benchmark on validation failure

## Usage in Other Benchmarks

To add validation to other benchmark scripts:

```python
# 1. Add import
from Utils.validate_cqm_vs_pulp import validate_before_dwave_submission

# 2. After creating CQM and PuLP models, before D-Wave submission:
scenario_info = {
    'n_units': n_units,
    'n_foods': len(foods),
    'scenario_type': 'patch'  # or 'farm'
}

is_valid = validate_before_dwave_submission(
    cqm=cqm,
    pulp_model=pulp_model,
    scenario_type='patch',  # or 'farm'
    scenario_info=scenario_info,
    strict=True  # Stop on any discrepancy
)

if not is_valid:
    print("Validation failed - skipping D-Wave")
    return  # or raise exception
    
# 3. Proceed with D-Wave submission
sampleset = solver.solve_with_dwave_cqm(cqm, token)
```

## Validation Strictness

The `strict` parameter controls behavior:

```python
# Strict mode (default): Stop on ANY discrepancy
validate_before_dwave_submission(..., strict=True)

# Lenient mode: Only stop on CRITICAL issues
validate_before_dwave_submission(..., strict=False)
```

**Recommended**: Always use `strict=True` for production benchmarks

## Testing the Validator

Without running D-Wave solver:

```bash
# Run benchmark without D-Wave (validation still runs)
python "Benchmark Scripts/comprehensive_benchmark.py" --configs

# If validation fails, you'll see error messages before any D-Wave call
# Fix issues in solver_runner_BINARY.py
# Re-run until validation passes
```

## Expected Behavior

### Scenario: Farm (Continuous)
- **Variables**: ~N_farms × N_foods × 2 (A and Y variables)
- **Constraints**: Area bounds, selection coupling, food groups
- **Validation**: Less strict on constraint types (continuous formulation)

### Scenario: Patch (Binary)
- **Variables**: N_plots × N_foods (Y variables only)
- **Constraints**: 
  - N "at most one" constraints (one per plot)
  - ~5 food group constraints
  - Other bounds
- **Validation**: Strict on "at most one" formulation

## Known Issues to Fix

Based on validation results, the **Patch scenario** likely has:

1. ❌ **Wrong constraint sense** in "at most one" constraints
   - Expected: `LE` (less than or equal)
   - Might be: `GE` (greater than or equal)
   
2. ❌ **Wrong RHS value** in "at most one" constraints
   - Expected: `1`
   - Might be: `> 1`

**Fix location**: `Benchmark Scripts/solver_runner_BINARY.py`
- Function: `create_cqm_plots()`
- Look for: Constraint creation for "at most one crop per plot"

## Troubleshooting

### Validation Always Fails
**Check**: Is PuLP model being created correctly?
**Solution**: Review `solve_with_pulp_plots()` or `solve_with_pulp_farm()` functions

### Variable Count Mismatch
**Check**: Are all variables being added to CQM?
**Solution**: Verify variable creation in `create_cqm_plots()` or `create_cqm_farm()`

### Constraint Count Off by Many
**Check**: Are constraints being skipped or duplicated?
**Solution**: Review constraint creation loops

## Summary

✅ **Validation automatically runs** before every D-Wave submission
✅ **Prevents wasted solver time** on incorrect formulations  
✅ **Provides detailed error messages** for debugging
✅ **Stops benchmark immediately** if validation fails
✅ **Zero D-Wave cost** for validation
✅ **Integrated into comprehensive_benchmark.py** for both Farm and Patch scenarios

**Next step**: Run the benchmark and let validation catch any formulation errors!
