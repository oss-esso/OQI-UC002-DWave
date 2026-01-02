# Violation Impact Analysis Report

**Date:** January 2, 2026  
**Analysis:** Quantifying how constraint violations degrade QPU objective values

---

## Executive Summary

Analyzed **22 QPU runs** across 3 methods (Hierarchical, Native, Hybrid) to quantify the impact of constraint violations on objective values. Key finding: **Violations have a nearly perfect negative correlation (-0.998) with objective degradation**, with each violation contributing approximately **-3.53 to the penalty term**. The penalty mechanism is working as designed but reveals that violated solutions are fundamentally compromised.

---

## Methodology

### Objective Decomposition

For each QPU solution, we calculated:

1. **Reported Objective**: The value returned by the QPU, which includes:
   - True MIQP objective (nutritional benefit + rotation synergy)
   - Penalty terms for constraint violations

2. **True Objective**: Recalculated MIQP objective WITHOUT penalty terms:
   ```
   Objective = Σ (B_c × L_f × Y_{f,c,t}) / A_total
             + Σ (R_{c1,c2} × L_f × Y_{f,c1,t} × Y_{f,c2,t+1}) / A_total
   ```
   Where:
   - B_c = nutritional value - environmental impact
   - R_{c1,c2} = rotation synergy between crops
   - L_f = land availability for farm f
   - Y_{f,c,t} = binary decision variable

3. **Penalty Impact**: Difference between reported and true objective
   ```
   Penalty Impact = Reported Objective - True Objective
   ```

### Data Sources

- **Hierarchical method**: `qpu_hier_all_6family.json` (9 scenarios)
- **Native method**: `qpu_native_6family.json` (9 scenarios)
- **Hybrid method**: `qpu_hybrid_27food.json` (4 scenarios)

---

## Results Summary

### Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Runs Analyzed** | 22 |
| **Total Violations** | 241 |
| **Total Penalty Impact** | -851.41 |
| **Average Penalty per Run** | -38.70 |
| **Average Penalty per Violation** | -3.53 |
| **Correlation (violations vs penalty)** | **-0.998** (nearly perfect) |

### By Method Comparison

| Method | Avg Reported Obj | Avg True Obj | Avg Penalty Impact | Avg Violations |
|--------|------------------|--------------|-------------------|----------------|
| **Hierarchical** | -34.10 | 2.75 | -36.85 | 10.8 |
| **Native** | -0.52 | 0.31 | -0.83 | 0.1 |
| **Hybrid** | -127.84 | 0.24 | -128.08 | 35.8 |

### Detailed Per-Scenario Results

#### Hierarchical Method (9 scenarios)

| Scenario | Reported Obj | True Obj | Penalty | Violations |
|----------|--------------|----------|---------|-----------|
| rotation_micro_25 | -5.50 | 2.60 | -8.10 | 2 |
| rotation_small_50 | -3.88 | 2.70 | -6.58 | 3 |
| rotation_15farms_6foods | -5.10 | 2.73 | -7.83 | 4 |
| rotation_medium_100 | -24.89 | 2.60 | -27.49 | 8 |
| rotation_25farms_6foods | -31.15 | 2.76 | -33.91 | 8 |
| rotation_large_200 | -37.47 | 2.80 | -40.27 | 11 |
| rotation_50farms_6foods | -50.96 | 2.80 | -53.76 | 18 |
| rotation_75farms_6foods | -70.15 | 2.87 | -73.02 | 23 |
| rotation_100farms_6foods | -77.78 | 2.90 | -80.68 | 20 |

**Pattern:** As farm count increases, violations increase, penalties accumulate dramatically.

#### Native Method (9 scenarios)

| Scenario | Reported Obj | True Obj | Penalty | Violations |
|----------|--------------|----------|---------|-----------|
| rotation_micro_25 | -4.67 | 2.80 | -7.47 | 1 |
| All others (8 scenarios) | 0.00 | 0.00 | 0.00 | 0 |

**Pattern:** Only smallest scenario has violations; 8/9 scenarios returned zero solutions (likely infeasible on QPU hardware).

#### Hybrid Method (4 scenarios)

| Scenario | Reported Obj | True Obj | Penalty | Violations |
|----------|--------------|----------|---------|-----------|
| rotation_25farms_27foods | -171.34 | 0.40 | -171.74 | 46 |
| rotation_50farms_27foods | -340.03 | 0.54 | -340.57 | 97 |
| rotation_100farms_27foods | 0.00 | 0.00 | 0.00 | 0 |
| rotation_200farms_27foods | 0.00 | 0.00 | 0.00 | 0 |

**Pattern:** 27-food problems produce massive violation counts; 2/4 scenarios failed entirely.

---

## Key Findings

### 1. Violation Penalty is Linear and Severe 📉

**Regression Analysis:**
```
Penalty Impact = -3.53 × Violations + constant
Correlation: -0.998 (R² ≈ 0.996)
```

**Interpretation:**
- Each constraint violation contributes approximately **-3.53** to the objective
- This is a **design feature** of the penalty-based formulation
- The penalty must be large enough to discourage violations
- But large penalties make solutions with violations appear terrible

### 2. True Objectives Are Positive But Small 📊

**Hierarchical method:**
- Average true objective: **2.75** (positive!)
- Range: 2.60 to 2.90
- Relatively consistent across scenarios

**Problem:** The true objectives are decent, but violations make reported values deeply negative (-77.78 worst case).

### 3. Violations Dominate the Objective 🚨

**Impact Ratio:**
```
|Penalty Impact| / True Objective = 36.85 / 2.75 ≈ 13.4×
```

For hierarchical method:
- True objective contributes: 2.75 (7%)
- Penalty terms contribute: -36.85 (93%)

**Conclusion:** The reported objective is **93% penalty, 7% true objective** on average.

### 4. Method Comparison Reveals Trade-offs ⚖️

| Method | True Obj Quality | Violation Rate | Overall Usability |
|--------|------------------|----------------|-------------------|
| **Hierarchical** | Good (2.75) | High (10.8) | ❌ Penalties dominate |
| **Native** | Low (0.31) | Very low (0.1) | ❌ Mostly fails (8/9 zero) |
| **Hybrid** | Very low (0.24) | Very high (35.8) | ❌ Catastrophic penalties |

**Interpretation:**
- **Hierarchical**: Finds reasonable solutions BUT violates constraints frequently
- **Native**: Avoids violations BUT mostly can't find solutions at all
- **Hybrid**: Worst of both worlds—poor quality AND high violations

### 5. Scaling Behavior is Problematic 📈

As problem size increases (farms: 5 → 100):

| Metric | Trend |
|--------|-------|
| Violations | +10× (2 → 20) |
| Penalty Impact | +10× (-8.10 → -80.68) |
| True Objective | Stable (~2.75) |
| Reported Objective | -16× worse (-5.50 → -77.78) |

**Problem:** Violations scale linearly with problem size, making larger problems appear hopeless even if true objective is decent.

---

## Implications

### What This Means for QPU Performance

1. **Reported objectives are misleading** ❌
   - A reported objective of -77.78 looks terrible
   - But the true objective is 2.90 (reasonable)
   - The issue is violations, not objective quality

2. **Violations are the critical bottleneck** 🚧
   - Even with decent solution quality
   - Violations make solutions unacceptable
   - Need: Better constraint satisfaction, not just better objectives

3. **Penalty-based formulation is problematic** ⚠️
   - Penalty must be large (else violations not discouraged)
   - Large penalty makes violated solutions appear worthless
   - Creates false impression that QPU solutions are terrible

### Comparison to Gurobi

From prior analysis, Gurobi achieves (e.g., 100-farm scenario):
- **Gurobi objective**: 53.77
- **QPU true objective**: 2.90
- **QPU reported objective**: -77.78

**Decomposition:**
- Gap to Gurobi (true quality): 53.77 - 2.90 = **50.87** (fundamental quality gap)
- Violation penalty: -80.68 (artificial degradation)
- Total reported gap: 53.77 - (-77.78) = **131.55**

**Conclusion:**
- ~39% of reported gap is quality difference (50.87 / 131.55)
- ~61% of reported gap is violation penalties (80.68 / 131.55)

---

## Recommendations

### Immediate Actions 🎯

1. **Report BOTH objectives:**
   - True objective (without penalties): Shows actual solution quality
   - Reported objective (with penalties): Shows feasibility status
   - Violations count: Shows magnitude of infeasibility

2. **Prioritize violation reduction:**
   - Current bottleneck is constraint satisfaction, not optimization
   - Focus on: Better embeddings, tighter decomposition, repair heuristics

3. **Adjust penalty tuning:**
   - Experiment with lower penalty coefficients
   - Test adaptive penalties (increase over iterations)
   - Consider soft constraints with graduated penalties

### Alternative Approaches 🔄

1. **Hard constraint formulations:**
   - Explore formulations that enforce constraints structurally
   - Example: Fixed-charge network flow models
   - Trade-off: May be harder to embed on QPU

2. **Hybrid penalty + repair:**
   - Use penalty formulation for initial solution
   - Apply repair heuristic to fix violations
   - Report repaired true objective

3. **Multi-objective optimization:**
   - Treat feasibility and quality as separate objectives
   - Use Pareto front to balance trade-offs
   - May require custom annealing schedules

### Long-term Strategy 🎓

1. **Fundamental rethinking:**
   - Current approach: Map MIQP to QUBO with penalties
   - Alternative: Redesign problem to be QUBO-native
   - Consider: Problem reformulation, different decompositions

2. **Better baselines:**
   - Compare to "random feasible" solutions
   - Measure: Quality improvement over trivial solutions
   - Clarify: What value does QPU actually add?

3. **Production viability:**
   - Current state: QPU not viable for production
   - Required: <10% gap to classical, near-zero violations
   - Timeline: Significant research breakthroughs needed

---

## Visualizations

Generated plots in `professional_plots/`:

1. **`violation_impact_analysis.png/pdf`** - Four-panel analysis:
   - Reported vs True objective scatter
   - Penalty impact vs violations (with linear trend)
   - Distribution of penalty impacts by method
   - Method comparison bar chart

---

## Conclusions

### Key Takeaway 💡

**The QPU objective values are dominated by constraint violation penalties, not by solution quality.** With an average of 10.8 violations per scenario and -3.53 penalty per violation, the reported objective is **93% penalties** and only **7% true objective**.

### What This Changes 🔄

**Previous understanding:**
- "QPU solutions are 77.5% worse than Gurobi" ❌

**Corrected understanding:**
- "QPU solutions have ~50 objective gap to Gurobi" (quality issue)
- "Plus ~81 penalty from 20 violations" (feasibility issue)
- "Violations are the primary problem, not quality" ✅

### Action Items ✅

1. ✅ **Report true objectives** in all future analyses
2. ⚠️ **Fix violation problem** before optimizing objective
3. ⚠️ **Consider hard constraints** or better repair methods
4. ⚠️ **Adjust penalty coefficients** for better balance

---

## Technical Details

### Files Generated

- **`evaluate_violation_impact.py`** - Analysis implementation (475 lines)
- **`professional_plots/violation_impact_analysis.json`** - Detailed results
- **`professional_plots/violation_impact_analysis.png/pdf`** - Visualizations
- **`VIOLATION_IMPACT_REPORT.md`** - This report

### Penalty Calculation Formula

Based on analysis, the penalty appears to follow:
```python
penalty = -λ × n_violations
λ ≈ 3.53 (average penalty coefficient)
```

Actual penalty formulation in QUBO likely:
```python
penalty_term = Σ P_one_hot × (Σ Y_{f,c,t} - 1)²
             + Σ P_rotation × violation_count
```

Where P_one_hot and P_rotation are large constants (e.g., 100-1000).

---

**Report Generated:** 2026-01-02  
**Scenarios Analyzed:** 22 runs across 3 methods  
**Key Finding:** Violations contribute **-3.53 per violation** to objective  
**Correlation:** -0.998 (violations perfectly predict penalty impact)
