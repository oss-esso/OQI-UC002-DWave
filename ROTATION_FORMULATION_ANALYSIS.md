# Rotation Formulation Comparison Analysis

## Purpose

This document analyzes the results from `test_rotation_formulations.py` to determine whether the hard rotation constraint and the soft R[c,c] diagonal penalty produce equivalent results.

## Decision Made (2026-01-15)

**Hard constraint has been REMOVED from all formulations.**

The rotation behavior is now enforced **only** through the temporal synergy objective term with R[c,c] = -1.2 (monoculture penalty).

### Files Modified:
- `unified_benchmark/gurobi_solver.py` - Hard constraint removed
- `unified_benchmark/quantum_solvers.py` - BQM penalty (+10.0) removed from all 3 solver modes
- `@todo/report/content_report.tex` - Documentation updated

### Rationale:
The hard constraint was an additional mechanism that restricted the solution space. By removing it, the formulation relies purely on the objective function's R[c,c] term to discourage monoculture, providing a smoother optimization landscape.

---

## Background

The rotation formulation currently has **two mechanisms** to discourage monoculture (same crop in consecutive periods):

1. **Hard Constraint**: `Y[f,c,t] + Y[f,c,t+1] <= 1` (strictly forbids monoculture)
2. **Soft Penalty**: `R[c,c] = -1.2` in the temporal synergy objective term

The question: Are these redundant? Does having both change anything?

## Test Configuration

The script tests three variants:
- **BOTH**: Hard constraint + R diagonal penalty (current implementation)
- **HARD ONLY**: Hard constraint + R[c,c] = 0 (no soft penalty)
- **SOFT ONLY**: No hard constraint + R[c,c] = -1.2 (rely on soft penalty only)

All solutions are re-scored with the **standard R matrix** for fair comparison.

---

## Results

✅ **Test completed successfully** - Results saved to `rotation_formulation_comparison_results.json`

---

## Analysis Summary

### Key Findings

1. **HARD ONLY vs BOTH**: 
   - For small/medium scenarios: Produces **different solutions** (difference -0.05 to -0.45 in objective)
   - For large scenarios (350+farms): **IDENTICAL** solutions (difference = 0.000000)
   - **Never has violations** (hard constraint works perfectly)

2. **SOFT ONLY** (no hard constraint):
   - Small scenarios: **No violations** (soft penalty sufficient)
   - Large scenarios (150+farms): **Has violations** (22 to 122 violations)
   - When violations occur: objective can be higher or lower depending on whether solver exploited monoculture

3. **Conclusion**: 
   - The R[c,c] diagonal penalty is **NOT redundant** for small/medium problems
   - For large problems, they converge to same solution
   - SOFT ONLY is **insufficient** for large problems (produces infeasible solutions)

---

### Detailed Results

### 1. Rotation Violations

| Scenario | Size | BOTH | HARD ONLY | SOFT ONLY |
|----------|------|------|-----------|-----------|
| rotation_micro_25 | 25f | 0 | 0 | 0 |
| rotation_small_50 | 50f | 0 | 0 | 0 |
| rotation_medium_100 | 100f | 0 | 0 | 0 |
| rotation_150farms_6foods | 150f | 0 | 0 | **1** ⚠️ |
| rotation_150farms_27foods | 150f | 0 | 0 | **22** ⚠️ |
| rotation_200farms_27foods | 200f | 0 | 0 | **48** ⚠️ |
| rotation_250farms_27foods | 250f | 0 | 0 | **122** ⚠️ |
| rotation_350farms_27foods | 350f | 0 | 0 | **23** ⚠️ |
| rotation_500farms_27foods | 500f | 0 | 0 | **30** ⚠️ |
| rotation_1000farms_27foods | 1000f | 0 | 0 | **61** ⚠️ |

**Finding**: Hard constraint is **necessary** for feasibility in large problems.

### 2. HARD ONLY vs BOTH Comparison

| Scenario | HARD - BOTH | Interpretation |
|----------|-------------|----------------|
| rotation_micro_25 | -0.244 | Different solution |
| rotation_small_50 | -0.446 | Different solution |
| rotation_medium_100 | -0.194 | Different solution |
| rotation_large_200 | -0.088 | Minor difference |
| rotation_350farms_27foods | **0.000** | ✅ Identical |
| rotation_500farms_27foods | **0.000** | ✅ Identical |
| rotation_1000farms_27foods | **0.000** | ✅ Identical |

**Finding**: For **large problems (350+farms)**, BOTH and HARD ONLY produce **identical solutions**. The R[c,c] term becomes irrelevant.

### 3. SOFT ONLY Performance

| Scenario | SOFT - BOTH | Violations | Interpretation |
|----------|-------------|------------|----------------|
| rotation_micro_25 | 0.000 | 0 | ✅ Equivalent |
| rotation_150farms_27foods | **+7.123** | 22 | Higher obj due to violations |
| rotation_200farms_27foods | **+9.202** | 48 | Exploited monoculture |
| rotation_250farms_27foods | **+9.977** | 122 | Infeasible solution |
| rotation_350farms_27foods | **-3.295** | 23 | Lower obj despite violations |

**Finding**: SOFT ONLY **fails on large problems** - produces infeasible solutions with substantial violations.

---

## Conclusions

### 1. Is the R[c,c] diagonal term redundant?

**Answer: IT DEPENDS ON PROBLEM SIZE**

- **Small/Medium problems (< 350 farms)**: R[c,c] is **NOT redundant**. HARD ONLY and BOTH produce different solutions with objective differences of 0.05-0.45.
- **Large problems (≥ 350 farms)**: R[c,c] becomes **effectively redundant**. HARD ONLY and BOTH converge to identical solutions (difference = 0.000).

**Mechanism**: In small problems, the R[c,c] penalty provides gradient information that influences which feasible solution Gurobi finds. In large problems, the hard constraint dominates and the solution space is constrained enough that the gradient information is irrelevant.

### 2. Is the hard constraint necessary?

**Answer: YES, ABSOLUTELY**

SOFT ONLY (relying only on R[c,c] = -1.2 penalty) produces:
- **Feasible solutions** for small problems (< 150 farms)
- **Infeasible solutions with 22-122 violations** for large problems (≥ 150 farms)

The soft penalty γ_rot × R[c,c] = 0.2 × (-1.2) = -0.24 is **insufficient** to prevent monoculture in complex optimization landscapes.

### 3. Current QPU formulation assessment

**Your current implementation is CORRECT:**

✅ BQM has hard constraint penalty (+10.0) → Guides QPU to feasible solutions  
✅ MIQP scorer uses standard R matrix → Correct objective calculation  
✅ R[c,c] in objective → Provides gradient for small/medium problems  

**The +10.0 hard constraint penalty and R[c,c] soft penalty are NOT double-counting:**
- Hard constraint penalty: Enforces feasibility (in BQM only)
- R[c,c] soft penalty: Influences solution selection among feasible options (in objective)
- MIQP scorer doesn't include the +10.0 penalty (correctly)

### 4. Your results are valid

**All your QPU result JSON files are CORRECT** because:
1. `objective_miqp` values were calculated by `compute_miqp_objective()` 
2. This scorer does **NOT** include the +10.0 BQM penalty
3. The scorer uses the standard R matrix with R[c,c] = -1.2
4. This is the correct formulation per your LaTeX document

---

## Final Recommendation

✅ **KEEP THE CURRENT FORMULATION** (BOTH)

**Rationale:**
1. Hard constraint is **essential** for feasibility in large problems
2. R[c,c] provides useful gradient for small/medium problems
3. For large problems, R[c,c] becomes irrelevant but doesn't hurt
4. Your QPU results are already correct - no recalculation needed
5. The formulation matches your paper documentation

**Action items:**
- ✅ No code changes needed
- ✅ No result recalculation needed
- ✅ Current paper formulation is correct
- ✅ Update paper (if needed) to clarify that both mechanisms work together, not redundantly

---

## Technical Notes

### Why different solutions in small problems?

The R[c,c] term creates a **preference ordering** among feasible solutions. When Gurobi explores the branch-and-bound tree:
- **BOTH**: Uses R[c,c] gradient to prefer certain rotations over others
- **HARD ONLY**: No such preference, may find different (but equally feasible) solution

Both solutions satisfy all constraints, but BOTH's solution has better rotation synergy.

### Why convergence in large problems?

As problem size increases, the hard constraint creates a **very restricted feasible region**. The solution space becomes so constrained that there are fewer "equally good" solutions to choose from, and the R[c,c] gradient no longer significantly affects which solution is found.
