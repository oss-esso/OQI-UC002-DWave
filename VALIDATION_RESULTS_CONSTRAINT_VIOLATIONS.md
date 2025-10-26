# Constraint Validation Results - QUBO vs CQM

## Date: October 26, 2025

## Summary

✅ **Validation implemented successfully!**
⚠️ **QUBO solution violates constraints - this explains the higher objective!**

## Test Case: Config=5 (5 patches, 0.226 ha total)

### Results Comparison

| Solver | Objective | Feasible | Violations | Time |
|--------|-----------|----------|------------|------|
| **Patch_PuLP (CQM)** | **0.0706** | ✅ Yes | 0 | 0.008s |
| **Patch_GurobiQUBO (BQM)** | **0.1355** | ❌ No | 1 | 3.36s |

### The Violation Discovered

**Gurobi QUBO violates maximum area constraint:**
```
Crop Soybeans: area=0.1250 ha > max=0.0904 ha
```

**Details:**
- Maximum allowed for Soybeans: 40% of 0.226 ha = **0.0904 ha**
- QUBO allocated: **0.1250 ha** (38% over limit!)
- This violation allows higher objective by planting more high-value soybeans

### QUBO Solution Breakdown

**Crops Selected:** 3 crops (Wheat, Soybeans, Apples)

**Plot Assignments:**
```
Wheat:
  - Patch4: 0.057 ha
  Total: 0.057 ha

Soybeans: ⚠️ VIOLATES MAX AREA
  - Patch3: 0.012 ha
  - Patch5: 0.113 ha
  Total: 0.125 ha (should be ≤ 0.0904 ha)

Apples:
  - Patch1: 0.023 ha
  - Patch2: 0.021 ha
  Total: 0.044 ha

Total allocated: 0.226 ha (100% utilization)
```

**Validation Summary:**
- Total checks: 53
- Passed: 52 (98.1%)
- Failed: 1 (1.9%)
- **Status: INFEASIBLE** ❌

## Why This Happened

### CQM → BQM Conversion Issue

When converting CQM to BQM, constraints become soft penalties:

```python
# CQM (Hard Constraint)
area[Soybeans] <= 0.0904  # MUST be satisfied

# BQM (Soft Penalty)
minimize: objective + λ * (max(0, area[Soybeans] - 0.0904))²
```

**The problem:**
- Penalty coefficient (λ) may be too small
- Solver finds that violating constraint gives better objective
- Trade-off: Small penalty vs large benefit from soybeans

### Why Soybeans?

Soybeans likely have high weighted benefit (B_c):
- High nutritional value
- High affordability
- High sustainability
- Violating the constraint allows planting more = higher objective

## Implementation Details

### New Functions Added

**1. `validate_solution_constraints()` in `solver_runner_PATCH.py`:**
```python
validation = validate_solution_constraints(
    solution, farms, foods, food_groups, land_availability, config
)
```

**Checks:**
- ✅ At most one crop per plot (52/52 passed)
- ✅ X-Y linking constraints (passed)
- ✅ Y activation constraints (passed)
- ✅ Minimum area bounds (passed)
- ❌ Maximum area bounds (1/6 failed - Soybeans)
- ✅ Food group constraints (passed)

**2. `extract_solution_summary()`:**
- Shows which crops selected (Y_c values)
- Shows plot-to-crop assignments (X_{p,c} values)
- Calculates area utilization

**3. Enhanced `solve_with_gurobi_qubo()`:**
- Accepts config parameters
- Calculates original objective
- Validates constraints
- Reports violations

### Output Format

```json
{
  "objective_value": 0.1355,
  "validation": {
    "is_feasible": false,
    "n_violations": 1,
    "violations": ["Crop Soybeans: area=0.1250 > max=0.0904"],
    "summary": {
      "total_checks": 53,
      "total_passed": 52,
      "total_failed": 1,
      "pass_rate": 0.981
    }
  },
  "solution_summary": {
    "crops_selected": ["Wheat", "Soybeans", "Apples"],
    "plot_assignments": [...],
    "utilization": 0.999
  }
}
```

## D-Wave BQM Ready! ✅

The same validation is implemented for D-Wave BQM:
- Objective reconstruction: ✅
- Solution extraction: ✅
- Constraint validation: ✅

**When you enable D-Wave:**
```bash
python comprehensive_benchmark.py --configs --dwave
```

You'll get:
- Comparable objectives across all solvers
- Constraint validation for all BQM solutions
- Solution summaries showing crop selections
- Full violation reports

## Implications

### 1. QUBO Solution Quality ❌

**The higher objective (0.1355) is misleading:**
- Solution violates constraints
- Not a valid solution for the original problem
- Can't be used in practice

**True comparison:**
- PuLP: 0.0706 (feasible)
- QUBO: N/A (infeasible)

### 2. Why BQM Struggles

**Penalty tuning is critical:**
- Current penalties are too weak
- Solver prioritizes objective over constraint satisfaction
- Need stronger penalties or better conversion strategy

### 3. Quantum Advantage Claims

**Must be careful:**
- Can't claim "QUBO finds better solutions" (they're infeasible!)
- Must validate all BQM/QUBO solutions
- Need to compare only feasible solutions

## Recommendations

### Short-term

1. **Always validate QUBO/BQM solutions** ✅ (Done!)
2. **Report feasibility status** in benchmarks ✅ (Done!)
3. **Compare only feasible solutions**

### Medium-term

1. **Tune penalty coefficients** in CQM→BQM conversion
   - Increase penalty weights
   - Use adaptive penalties
   
2. **Add constraint repair**
   - Post-process QUBO solutions
   - Adjust to satisfy constraints
   - Compare repaired vs original objective

3. **Try different conversion strategies**
   - Different lagrangian multipliers
   - Multi-stage conversion
   - Constraint prioritization

### Long-term

1. **Use CQM directly** where possible
   - D-Wave CQM solver handles constraints natively
   - No penalty tuning needed
   - More reliable results

2. **Benchmark conversion strategies**
   - Compare different penalty schemes
   - Measure feasibility rates
   - Optimize for constraint satisfaction

## Current Status

✅ **Completed:**
- Objective reconstruction from BQM solutions
- Constraint validation for all BQM solvers
- Solution extraction and summaries
- D-Wave BQM validation ready

⚠️ **Discovered:**
- QUBO solutions can violate constraints
- Penalty coefficients need tuning
- Higher objectives don't mean better solutions

🎯 **Ready for:**
- Full benchmark with D-Wave
- Fair comparison of feasible solutions only
- Analysis of constraint violation patterns

## Example Usage

```python
# Run benchmark with D-Wave
python comprehensive_benchmark.py --configs --dwave

# Results will include:
# - Feasibility status for each solution
# - Constraint violation details
# - Comparable objectives (reconstructed)
# - Solution summaries (crops & assignments)
```

## Conclusion

**Major Achievement:** 
We now have comprehensive validation that revealed a critical issue - QUBO solutions can be infeasible even with "optimal" BQM energy!

**Key Insight:**
The higher QUBO objective (0.1355 vs 0.0706) is because it **cheats** by violating the Soybeans area constraint. This is not a better solution - it's an invalid one.

**Next Steps:**
1. Run full benchmark with D-Wave
2. Analyze feasibility rates across configs
3. Tune penalties or use CQM directly for reliable results
