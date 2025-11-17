# Summary of D-Wave Sampleset Saving and CQM Diagnostic Tools

## Issues Addressed

### 1. ❌ Empty Samplesets Directory
**Problem**: Samplesets were not being saved because `to_pandas_dataframe()` raised an error:
```
Expected a 1D array, got an array with shape (65, 47)
```

**Root Cause**: The `to_pandas_dataframe()` method can return constraint satisfaction data that causes shape issues when trying to add metadata columns.

**Solution**: Updated `Utils/save_dwave_sampleset.py` with error handling:
- Try `to_pandas_dataframe()` first
- If it fails, manually extract data from sampleset
- Robustly add metadata columns regardless of DataFrame structure

### 2. ⚠️ Constraint Violations in D-Wave Solutions
**Problem**: D-Wave CQM solver returning infeasible solutions with violations like:
```
"Plot Patch1: 2.000 crops assigned (should be ≤ 1)"
"Plot Patch2: 3.000 crops assigned (should be ≤ 1)"
```

**Solution**: Created comprehensive diagnostic tool to investigate constraints without using D-Wave solver.

## Files Created/Modified

### ✅ Updated Files

1. **`Utils/save_dwave_sampleset.py`**
   - Added try-except block for robust DataFrame conversion
   - Manual data extraction fallback
   - Better metadata column insertion logic

2. **`Benchmark Scripts/comprehensive_benchmark.py`**
   - Import statement for `save_dwave_sampleset`
   - Added sampleset saving after each D-Wave solver call
   - Error handling for sampleset saving

### ✅ New Files Created

3. **`Utils/diagnose_cqm_constraints.py`** (490 lines)
   - Comprehensive CQM constraint analysis tool
   - No D-Wave solver access required
   - Analyzes constraint types, relationships, and potential conflicts
   - Saves CQM model and diagnostic report
   - Provides actionable recommendations

4. **`CQM_DIAGNOSTIC_GUIDE.md`**
   - Complete usage guide for diagnostic tool
   - Example workflows
   - Troubleshooting tips
   - Integration instructions

5. **`SAMPLESET_STORAGE_GUIDE.md`** (previously created)
   - Usage guide for sampleset saving utility
   - File naming conventions
   - Analysis examples

6. **`OUTPUT_PATH_FIXES_SUMMARY.md`** (previously created)
   - Documentation of output path corrections

## Key Features Implemented

### Sampleset Saving Utility

**Features:**
- ✅ Automatic directory structure creation
- ✅ Standardized filename format
- ✅ Metadata embedding (benchmark, scenario, solver, config, run, timestamp)
- ✅ Robust error handling
- ✅ Saves ALL solutions, not just the best one
- ✅ CSV format for easy analysis

**Usage:**
```python
from Utils.save_dwave_sampleset import save_sampleset_to_dataframe

filepath = save_sampleset_to_dataframe(
    sampleset=sampleset,
    benchmark_type='COMPREHENSIVE',
    scenario_type='Patch',
    solver_type='DWave',
    config_id=10,
    run_id=1
)
```

**Output:**
```
Benchmarks/COMPREHENSIVE/Patch_DWave/samplesets/
└── comprehensive_Patch_DWave_config10_run1_20251117_153045.csv
```

### CQM Constraint Diagnostic Tool

**Features:**
- ✅ Analyzes constraint types and categories
- ✅ Identifies over-constrained variables
- ✅ Checks for impossible bounds
- ✅ Detects potential constraint conflicts
- ✅ Verifies "at most one" constraints for patch scenarios
- ✅ Generates actionable recommendations
- ✅ Saves CQM model for inspection
- ✅ NO D-Wave access required

**Usage:**
```bash
python Utils/diagnose_cqm_constraints.py --scenario patch --config 10
```

**Output:**
```
CQM_Models/diagnostics/
├── cqm_diagnostic_patch_10units_20251117_153045.cqm
└── diagnostic_report_patch_10units_20251117_153045.json
```

**Console Output Example:**
```
================================================================================
CQM CONSTRAINT DIAGNOSTIC
================================================================================
Scenario: patch
Variables: 270
Constraints: 47

--- Analyzing Constraint Types ---
  selection_limit: 10
  food_group: 5

--- Identifying Potential Conflicts ---
  ✓ Found 10 'at most one' constraints
  Checking constraint: AtMostOne_Patch1
    Sense: LE
    RHS: 1
    Num variables: 27

--- Recommendations ---
  • Verify Y variables are strictly binary
  • Check Lagrange multiplier strength
```

## Directory Structure Created

```
Benchmarks/
└── {BENCHMARK_TYPE}/
    └── {Scenario}_{Solver}/
        ├── config_{N}_run_{M}.json      # Best solution (existing)
        └── samplesets/                   # NEW: All solutions
            └── {benchmark}_{scenario}_{solver}_config{N}_run{M}_{timestamp}.csv

CQM_Models/
└── diagnostics/                          # NEW: Diagnostic outputs
    ├── cqm_diagnostic_{scenario}_{N}units_{timestamp}.cqm
    └── diagnostic_report_{scenario}_{N}units_{timestamp}.json
```

## Integration Points

### In comprehensive_benchmark.py

**Farm D-Wave CQM (line ~393):**
```python
sampleset, solve_time, qpu_time = solver_runner.solve_with_dwave_cqm(cqm, dwave_token)

# Save complete sampleset to DataFrame
try:
    sampleset_path = save_sampleset_to_dataframe(
        sampleset=sampleset,
        benchmark_type='COMPREHENSIVE',
        scenario_type='Farm',
        solver_type='DWave',
        config_id=sample_data['n_units'],
        run_id=1
    )
    print(f"       ✓ Saved sampleset to: {os.path.basename(sampleset_path)}")
except Exception as e:
    print(f"       Warning: Failed to save sampleset: {e}")
```

**Patch D-Wave CQM (line ~563):** Same pattern

**Patch D-Wave BQM (line ~672):** Same pattern

### To Add to Other Benchmarks

Add these two imports to any benchmark script:
```python
from Utils.save_dwave_sampleset import save_sampleset_to_dataframe
import os  # if not already imported
```

Add after getting sampleset:
```python
try:
    save_sampleset_to_dataframe(
        sampleset=sampleset,
        benchmark_type='LQ',  # or 'ROTATION', 'NLN', etc.
        scenario_type='Farm',  # or 'Patch'
        solver_type='DWave',  # or 'DWaveBQM'
        config_id=n_units,
        run_id=1
    )
except Exception as e:
    print(f"Warning: Failed to save sampleset: {e}")
```

## Recommended Workflow for Constraint Issues

1. **Run Diagnostic** (No D-Wave access needed):
   ```bash
   python Utils/diagnose_cqm_constraints.py --scenario patch --config 10
   ```

2. **Review Report**:
   ```bash
   cat CQM_Models/diagnostics/diagnostic_report_*.json | jq '.potential_issues'
   ```

3. **Identify Issues**:
   - Check constraint formulation
   - Verify constraint sense (LE vs GE vs EQ)
   - Confirm RHS values

4. **Fix in solver_runner_BINARY.py**:
   - Update `create_cqm_plots()` or `create_cqm_farm()`
   - Correct constraint formulations

5. **Re-run Diagnostic**:
   ```bash
   python Utils/diagnose_cqm_constraints.py --scenario patch --config 10
   ```

6. **Test with Small Problem**:
   ```bash
   python Utils/diagnose_cqm_constraints.py --scenario patch --config 3
   ```

7. **Run Benchmark** (if diagnostic passes):
   ```bash
   python "Benchmark Scripts/comprehensive_benchmark.py" 3 --dwave
   ```

8. **Analyze Samplesets**:
   ```python
   import pandas as pd
   df = pd.read_csv('Benchmarks/.../samplesets/comprehensive_Patch_DWave_*.csv')
   print(df[df['is_feasible'] == True].describe())
   ```

## Known Issues to Investigate

Based on `config_10_run_1.json`:

**Issue**: Multiple crops assigned to single plots
```json
{
  "validation": {
    "n_violations": 9,
    "violations": [
      "Plot Patch1: 2.000 crops assigned (should be ≤ 1)",
      "Plot Patch2: 3.000 crops assigned (should be ≤ 1)",
      ...
    ]
  }
}
```

**Possible Root Causes**:
1. ❓ "At most one" constraints incorrectly formulated
2. ❓ Lagrange multiplier too weak in CQM→BQM conversion
3. ❓ Binary variables not properly constrained
4. ❓ Constraint conflicts causing solver to ignore some constraints

**Next Steps**:
1. ✅ Run diagnostic tool to check constraint formulation
2. ⏳ Review diagnostic report for specific issues
3. ⏳ Fix constraint formulation if needed
4. ⏳ Test Lagrange multiplier values (currently 100000.0)
5. ⏳ Verify binary variable declarations

## Documentation Files

All documentation is comprehensive and ready to use:

- **`SAMPLESET_STORAGE_GUIDE.md`** - How to save and analyze samplesets
- **`CQM_DIAGNOSTIC_GUIDE.md`** - How to diagnose constraint issues
- **`OUTPUT_PATH_FIXES_SUMMARY.md`** - Output path corrections made

## Testing Commands (No D-Wave Access Needed)

```bash
# Test sampleset utility help
python Utils/save_dwave_sampleset.py

# Test diagnostic tool on patch scenario
python Utils/diagnose_cqm_constraints.py --scenario patch --config 10

# Test diagnostic tool on farm scenario
python Utils/diagnose_cqm_constraints.py --scenario farm --config 10

# Test with smaller problem
python Utils/diagnose_cqm_constraints.py --scenario patch --config 3
```

## Summary

✅ **Sampleset Saving**: Fixed and integrated into comprehensive_benchmark.py
✅ **CQM Diagnostic Tool**: Created comprehensive analysis tool
✅ **Documentation**: Complete guides for both utilities
✅ **No D-Wave Access**: Diagnostic tool works entirely offline
⏳ **Constraint Investigation**: Ready to proceed with diagnostic workflow

**All work completed without requiring D-Wave solver access!**
