# CQM Constraint Diagnostic Tool - Usage Guide

## Overview

The `Utils/diagnose_cqm_constraints.py` tool analyzes CQM (Constrained Quadratic Model) formulations to identify potential issues, conflicts, and constraint problems **without requiring D-Wave solver access**.

## Purpose

Based on the observation that D-Wave solutions show constraint violations (e.g., "Plot Patch1: 2.000 crops assigned (should be ≤ 1)"), this tool helps diagnose:

1. **Constraint formulation errors** - Are constraints correctly formulated?
2. **Conflicting constraints** - Do constraints contradict each other?
3. **Over-constrained variables** - Are variables involved in too many constraints?
4. **Missing constraints** - Are expected constraints present?
5. **Bound issues** - Are constraint bounds reasonable?

## Usage

### Basic Usage

```bash
# Diagnose patch scenario with 10 plots
python Utils/diagnose_cqm_constraints.py --scenario patch --config 10

# Diagnose farm scenario with 15 farms
python Utils/diagnose_cqm_constraints.py --scenario farm --config 15

# Custom land area
python Utils/diagnose_cqm_constraints.py --scenario patch --config 5 --land 50
```

### Arguments

- `--scenario` (required): Either `farm` or `patch`
  - `farm`: Continuous formulation with uneven farm sizes
  - `patch`: Binary formulation with equal plot sizes
- `--config` (required): Number of units (farms or patches)
- `--land` (optional): Total land area in hectares (default: 100.0)

## Output Files

The tool creates two files in `CQM_Models/diagnostics/`:

### 1. CQM Model File
**Format**: `cqm_diagnostic_{scenario}_{N}units_{timestamp}.cqm`

**Example**: `cqm_diagnostic_patch_10units_20251117_153045.cqm`

This is the actual CQM model that can be:
- Loaded for manual inspection
- Used for testing different solving approaches
- Shared for debugging without revealing problem data

### 2. Diagnostic Report
**Format**: `diagnostic_report_{scenario}_{N}units_{timestamp}.json`

**Example**: `diagnostic_report_patch_10units_20251117_153045.json`

## Report Structure

The JSON report contains:

```json
{
  "timestamp": "2025-11-17T15:30:45",
  "scenario": {
    "scenario_type": "patch",
    "n_units": 10,
    "n_foods": 27,
    "total_land": 100.0,
    "cqm_file": "/path/to/cqm/file.cqm"
  },
  "summary": {},
  "constraints": {
    "total": 47,
    "by_category": {
      "selection_limit": 10,
      "food_group": 5,
      "area_bounds": 30,
      "other": 2
    },
    "at_most_one_count": 10,
    "details": [...]
  },
  "variables": {
    "total": 270,
    "max_constraints_per_var": 3,
    "min_constraints_per_var": 1,
    "avg_constraints_per_var": 1.74,
    "most_constrained_examples": [...],
    "variable_constraint_map": {...}
  },
  "potential_issues": [
    {
      "type": "incorrect_at_most_one",
      "severity": "error",
      "constraint": "AtMostOne_Patch1",
      "message": "..."
    }
  ],
  "recommendations": [
    {
      "priority": "high",
      "area": "constraint_formulation",
      "recommendation": "...",
      "reason": "..."
    }
  ]
}
```

## Console Output

The tool provides real-time diagnostic output:

```
================================================================================
CQM CONSTRAINT DIAGNOSTIC
================================================================================
Scenario: patch
Units: 10
Foods: 27
Variables: 270
Constraints: 47

--- Analyzing Constraint Types ---
  Total constraints: 47
    area_bounds: 0
    selection_limit: 10
    food_group: 5
    other: 32

--- Analyzing Variable-Constraint Relationships ---
  Total variables: 270
  Constraints per variable:
    Max: 3
    Min: 1
    Avg: 1.74

--- Checking Constraint Bounds ---
  ✓ No obvious bound issues found

--- Identifying Potential Conflicts ---
  ✓ Found 10 'at most one' constraints
  Checking constraint: AtMostOne_Patch1
    Sense: LE
    RHS: 1
    Num variables: 27

--- Recommendations ---
  • Verify Y variables are strictly binary
  • Check Lagrange multiplier strength in BQM conversion
  • Test with smaller problem size first

================================================================================
DIAGNOSTIC SUMMARY
================================================================================

⚠️  Found 1 potential issues:

🟠 INCORRECT_AT_MOST_ONE (error)
   'At most one' constraint has wrong formulation: GE 1

📋 Recommendations (3):

🔥 [CONSTRAINT_FORMULATION]
   Verify that Y_plot_crop variables are truly binary (0 or 1)
   Reason: Solution shows fractional or multiple crop assignments per plot

================================================================================
Files saved:
  CQM Model: /path/to/cqm_diagnostic_patch_10units_20251117_153045.cqm
  Report: /path/to/diagnostic_report_patch_10units_20251117_153045.json
================================================================================
```

## Key Diagnostic Features

### 1. Constraint Type Analysis
Categorizes constraints into:
- **selection_limit**: "At most one crop per plot" constraints
- **food_group**: Food group diversity constraints
- **area_bounds**: Land area bounds
- **min_area**: Minimum planting area constraints
- **other**: Uncategorized constraints

### 2. Variable Analysis
- Identifies most and least constrained variables
- Calculates average constraint involvement
- Flags over-constrained variables (>10 constraints)

### 3. Bound Checking
- Verifies constraint bounds are mathematically possible
- Identifies constraints that might always be violated

### 4. Conflict Detection
For patch scenarios, specifically checks:
- Presence of "at most one crop per plot" constraints
- Correct formulation (sum(Y_p_c) ≤ 1)
- Proper constraint sense and RHS

### 5. Recommendations
Provides actionable recommendations based on findings:
- **High priority**: Critical formulation issues
- **Medium priority**: Optimization suggestions
- **Low priority**: Best practices

## Interpreting Results

### Expected Patch Scenario Constraints

For a patch scenario with N plots and M foods:

1. **Selection limits**: N constraints (one per plot)
   - Form: `sum(Y_plot_c for all crops c) <= 1`
   - Ensures at most one crop per plot

2. **Food group constraints**: 1 per food group (typically 5)
   - Form: `sum(Y_p_c for all plots p, crops c in group) >= 1`
   - Ensures diversity across food groups

3. **Total**: ~N + 5 constraints for simple case

### Common Issues

**Issue**: "Plot has 2-3 crops assigned (should be ≤ 1)"

**Possible Causes**:
1. Constraint formulation error (wrong sense: GE instead of LE)
2. Weak Lagrange multipliers in CQM→BQM conversion
3. Solver finding local minima that violate soft constraints
4. Binary variables not properly enforced

**Diagnostic Checks**:
```python
# Check constraint formulation
constraint = cqm.constraints['AtMostOne_Patch1']
print(f"Sense: {constraint.sense.name}")  # Should be 'LE'
print(f"RHS: {constraint.rhs}")           # Should be 1
```

## Integration with Comprehensive Benchmark

The diagnostic tool uses the same:
- Data generation (`generate_farms`, `generate_patches`)
- Food loading (`load_food_data`)
- CQM creation (`create_cqm_plots`, `create_cqm_farm`)

This ensures the diagnosed model matches what the benchmark runs.

## Next Steps After Diagnosis

1. **Review diagnostic report JSON** - Check `potential_issues` and `recommendations`
2. **Examine specific constraints** - Look at `constraints.details` for problematic ones
3. **Fix formulation** - Update constraint creation in `solver_runner_BINARY.py`
4. **Re-run diagnostic** - Verify fixes
5. **Test with small problem** - Use 3-5 plots to manually verify
6. **Run benchmark** - Test with D-Wave solver

## Example Workflow

```bash
# 1. Diagnose current formulation
python Utils/diagnose_cqm_constraints.py --scenario patch --config 10

# 2. Review report
cat CQM_Models/diagnostics/diagnostic_report_patch_10units_*.json | jq '.potential_issues'

# 3. Fix issues in solver_runner_BINARY.py
# (edit create_cqm_plots function)

# 4. Re-diagnose
python Utils/diagnose_cqm_constraints.py --scenario patch --config 10

# 5. Compare reports
# Check if issues are resolved

# 6. Test with small problem
python Utils/diagnose_cqm_constraints.py --scenario patch --config 3

# 7. Run benchmark if diagnostic passes
python "Benchmark Scripts/comprehensive_benchmark.py" 3 --dwave
```

## No D-Wave Access Required

**Important**: This tool does NOT require:
- D-Wave API token
- Solver access
- QPU time
- Internet connection (once dependencies installed)

It only analyzes the CQM structure locally.

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the project root
cd /path/to/OQI-UC002-DWave
python Utils/diagnose_cqm_constraints.py --scenario patch --config 10
```

### Missing Dependencies
```bash
# Install required packages
conda install dimod
# or
pip install dimod
```

### Can't Find solver_runner_BINARY
```bash
# Check that solver_runner_BINARY.py exists in Benchmark Scripts/
ls "Benchmark Scripts/solver_runner_BINARY.py"
```

## Advanced Usage

### Load Saved CQM for Analysis

```python
from dimod import ConstrainedQuadraticModel

# Load saved CQM
with open('CQM_Models/diagnostics/cqm_diagnostic_patch_10units_20251117_153045.cqm', 'rb') as f:
    cqm = ConstrainedQuadraticModel.from_file(f)

# Inspect manually
print(f"Variables: {len(cqm.variables)}")
print(f"Constraints: {len(cqm.constraints)}")

# Check specific constraint
constraint = cqm.constraints['AtMostOne_Patch1']
print(f"Constraint: {constraint}")
```

### Compare Before/After Fixes

```bash
# Before fix
python Utils/diagnose_cqm_constraints.py --scenario patch --config 10
mv CQM_Models/diagnostics/diagnostic_report_*.json before_fix.json

# After fix
python Utils/diagnose_cqm_constraints.py --scenario patch --config 10
mv CQM_Models/diagnostics/diagnostic_report_*.json after_fix.json

# Compare
diff before_fix.json after_fix.json
```
