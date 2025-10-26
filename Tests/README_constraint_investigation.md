# D-Wave BQM Constraint Violation Investigation

This directory contains targeted tests to investigate constraint violations in the D-Wave BQM solver for 50-unit patch scenarios.

## Problem Description

The D-Wave BQM solver (using CQM→BQM conversion) is producing solutions that violate the "at most one crop per plot" constraint. Specifically, in 50-unit scenarios, we observe ~20 constraint violations where multiple crops are assigned to the same plot.

## Investigation Files

### Main Investigation
- **`test_patch_dwave_bqm_constraints.py`** - Primary investigation script
  - Reproduces the 50-unit scenario showing violations
  - Compares CQM vs BQM solver behavior
  - Analyzes constraint formulation and penalty encoding
  - Generates detailed violation analysis

### Advanced Analysis
- **`advanced_bqm_analysis.py`** - Manual BQM construction and analysis
  - Tests manual BQM formulation with explicit constraint penalties
  - Analyzes penalty-to-objective ratios
  - Tests different Lagrange multiplier values
  - Provides deep insight into BQM energy landscape

### Orchestration
- **`run_constraint_investigation.py`** - Comprehensive test runner
  - Runs both main investigation and advanced analysis
  - Generates combined report with recommendations
  - Provides summary of findings and next steps

## Key Findings (Expected)

1. **Root Cause**: The issue is likely in the CQM→BQM conversion process where Lagrange multipliers may not be sufficient to enforce hard constraints in the quantum annealing process.

2. **CQM vs BQM**: The CQM solver likely produces feasible solutions while the BQM solver does not, confirming the issue is in the conversion, not the problem formulation.

3. **Penalty Scaling**: Manual BQM construction may require significantly higher penalty multipliers (100x-1000x objective coefficients) to enforce constraints.

## Usage

### Quick Investigation
```bash
# Run just the main investigation
python Tests/test_patch_dwave_bqm_constraints.py
```

### Comprehensive Analysis
```bash
# Run full investigation with advanced BQM analysis
python Tests/run_constraint_investigation.py
```

### Prerequisites
- D-Wave API token (set `DWAVE_API_TOKEN` environment variable)
- All project dependencies installed
- Access to the 50-unit patch scenario data

## Expected Outputs

### Reports Generated
- `constraint_violation_investigation_YYYYMMDD_HHMMSS.json` - Main investigation results
- `comprehensive_constraint_investigation_YYYYMMDD_HHMMSS.json` - Combined analysis

### Key Metrics Analyzed
- Number of constraint violations per solver
- BQM energy values and penalty structure
- Constraint satisfaction vs penalty multiplier relationship
- Timing comparisons between solvers

## Recommendations (Anticipated)

Based on similar quantum optimization challenges, we expect to recommend:

1. **Immediate**: Use CQM solver for constraint-critical applications
2. **Short-term**: Implement solution verification and constraint repair for BQM results
3. **Medium-term**: Develop custom BQM formulation with properly scaled penalties
4. **Long-term**: Investigate hybrid classical-quantum constraint handling

## Technical Details

### Constraint Formulation
The "at most one crop per plot" constraint is formulated as:
```
∀ plot p: Σ_c X_{p,c} ≤ 1
```

In BQM, this becomes a penalty term:
```
Penalty: M * Σ_p [Σ_c X_{p,c} - 1]²₊
```

Where M is the Lagrange multiplier that must be large enough to make constraint violations energetically unfavorable.

### Investigation Methodology
1. **Reproduce** - Generate exact 50-unit scenario showing violations
2. **Compare** - Test both CQM and BQM solvers on identical problems
3. **Analyze** - Examine penalty structure and energy landscape
4. **Test** - Try different penalty multipliers and formulations
5. **Recommend** - Provide actionable solutions based on findings

## Notes

- Tests use fixed random seed (42) for reproducibility
- Results may vary based on D-Wave solver availability and performance
- Investigation focuses specifically on the 50-unit scenario but findings should generalize
- Manual BQM construction provides insights into penalty scaling requirements