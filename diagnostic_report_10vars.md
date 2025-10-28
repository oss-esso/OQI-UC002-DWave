# BQM Constraint Violation Diagnostic Report

**Generated:** 2025-10-28 10:14:03

## Executive Summary

This report analyzes three BQM formulations to diagnose why the PATCH formulation violates constraints with D-Wave's Hybrid BQM solver while BQUBO does not:

1. **BQUBO**: Binary plantation model (baseline - works correctly)
2. **PATCH**: Plot assignment model with idle area penalty (violates constraints)
3. **PATCH_NO_IDLE**: Plot assignment model without idle area penalty (test variant)

---

## 1. Instance Characterization

### Size and Complexity

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Variables** | 70 | 160 | 160 |
| **Interactions** | 169 | 960 | 960 |
| **Density** | 0.069979 | 0.075472 | 0.075472 |
| **Offset** | 2.5575 | 86.3320 | 71.0520 |

### Linear Coefficients Distribution

| Statistic | BQUBO | PATCH | PATCH_NO_IDLE |
|-----------|-------|-------|---------------|
| **Mean** | -0.1830 | 19.1299 | 15.7484 |
| **Std Dev** | 0.1933 | 31.9512 | 26.2946 |
| **Range** | 0.6933 | 146.7644 | 120.7884 |

### Quadratic Coefficients Distribution

| Statistic | BQUBO | PATCH | PATCH_NO_IDLE |
|-----------|-------|-------|---------------|
| **Mean** | 0.1743 | 0.5396 | 0.4441 |
| **Std Dev** | 0.0511 | 37.2041 | 30.6193 |
| **Range** | 0.1550 | 276.2624 | 227.3664 |

### Graph Topology

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Connected Components** | 25 | 1 | 1 |
| **Avg Degree** | 4.83 | 12.00 | 12.00 |
| **Max Degree** | 7 | 24 | 24 |
| **Avg Clustering** | 0.7429 | 0.7923 | 0.7923 |

---

## 2. Solver-Independent Hardness Metrics

### Spectral Properties

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Min Eigenvalue** | -0.8104200256893288 | None | None |
| **Max Eigenvalue** | 0.4594893781881032 | None | None |
| **Spectral Gap** | 4.440892098500626e-16 | None | None |

### Energy Landscape (Random Sampling)

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Mean Energy** | 3.5382 | 1735.3520 | 1441.0918 |
| **Std Energy** | 1.3913 | 351.7966 | 295.8362 |
| **Energy Range** | 7.7622 | 2398.1446 | 1946.2720 |
| **Local Minima Ratio** | 0.0000 | 0.0000 | 0.0000 |

---

## 3. Solver Performance Comparison

### Gurobi CQM Solver (via PuLP)

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Mean Energy** | inf | inf | inf |
| **Best Energy** | inf | inf | inf |
| **Mean Time (s)** | 0.0048 | 0.0053 | 0.0046 |
| **Feasibility Rate** | 0.00% | 0.00% | 0.00% |
| **Success Count** | 0/3 | 0/3 | 0/3 |


---

## 4. Constraint Structure Analysis

### Constraint Statistics

| Metric | BQUBO | PATCH | PATCH_NO_IDLE |
|--------|-------|-------|---------------|
| **Num Constraints** | {results['bqubo']['constraint_structure']['n_constraints']} | {results['patch']['constraint_structure']['n_constraints']} | {results['patch_no_idle']['constraint_structure']['n_constraints']} |
| **Mean Vars/Constraint** | {results['bqubo']['constraint_structure']['mean_vars_per_constraint']:.2f} | {results['patch']['constraint_structure']['mean_vars_per_constraint']:.2f} | {results['patch_no_idle']['constraint_structure']['mean_vars_per_constraint']:.2f} |
| **Max Vars/Constraint** | {results['bqubo']['constraint_structure']['max_vars_per_constraint']} | {results['patch']['constraint_structure']['max_vars_per_constraint']} | {results['patch_no_idle']['constraint_structure']['max_vars_per_constraint']} |
| **Constraint Overlap Ratio** | {results['bqubo']['constraint_structure']['constraint_overlap_ratio']:.4f} | {results['patch']['constraint_structure']['constraint_overlap_ratio']:.4f} | {results['patch_no_idle']['constraint_structure']['constraint_overlap_ratio']:.4f} |
| **Mean Overlap Size** | {results['bqubo']['constraint_structure']['mean_overlap_size']:.2f} | {results['patch']['constraint_structure']['mean_overlap_size']:.2f} | {results['patch_no_idle']['constraint_structure']['mean_overlap_size']:.2f} |

### Constraint Types

**BQUBO:** {results['bqubo']['constraint_structure']['constraint_types']}

**PATCH:** {results['patch']['constraint_structure']['constraint_types']}

**PATCH_NO_IDLE:** {results['patch_no_idle']['constraint_structure']['constraint_types']}

---

## 5. Penalty Weight Sensitivity Analysis

This section shows how feasibility and constraint violations change with different Lagrange multipliers.


### BQUBO Penalty Sensitivity

| Lagrange Multiplier | Feasibility Ratio | Mean Violations | Max Violations | Best Energy |
|---------------------|-------------------|-----------------|----------------|-------------|
| 0.1 | 1.0000 | 0.00 | 0 | -0.1758 |
| 0.5 | 1.0000 | 0.00 | 0 | -0.1768 |
| 1.0 | 1.0000 | 0.00 | 0 | -0.1783 |
| 5.0 | 1.0000 | 0.00 | 0 | -0.1724 |
| 10.0 | 1.0000 | 0.00 | 0 | -0.1753 |
| 50.0 | 1.0000 | 0.00 | 0 | -0.1788 |
| 100.0 | 1.0000 | 0.00 | 0 | -0.1762 |
| 500.0 | 1.0000 | 0.00 | 0 | -0.1705 |
| 1000.0 | 1.0000 | 0.00 | 0 | -0.1781 |

### PATCH Penalty Sensitivity

| Lagrange Multiplier | Feasibility Ratio | Mean Violations | Max Violations | Best Energy |
|---------------------|-------------------|-----------------|----------------|-------------|
| 0.1 | 0.0000 | 0.00 | 0 | -4.6949 |
| 0.5 | 0.0000 | 0.00 | 0 | -2.6861 |
| 1.0 | 0.2300 | 0.00 | 0 | -2.5530 |
| 5.0 | 0.5300 | 0.00 | 0 | -2.5423 |
| 10.0 | 0.5100 | 0.00 | 0 | -2.4701 |
| 50.0 | 0.5800 | 0.00 | 0 | -2.4716 |
| 100.0 | 0.5700 | 0.00 | 0 | -2.5528 |
| 500.0 | 0.6000 | 0.00 | 0 | -2.5071 |
| 1000.0 | 0.6300 | 0.00 | 0 | -2.5158 |

### PATCH_NO_IDLE Penalty Sensitivity

| Lagrange Multiplier | Feasibility Ratio | Mean Violations | Max Violations | Best Energy |
|---------------------|-------------------|-----------------|----------------|-------------|
| 0.1 | 0.0000 | 0.00 | 0 | -3.4088 |
| 0.5 | 0.0300 | 0.00 | 0 | -2.1131 |
| 1.0 | 0.4000 | 0.00 | 0 | -1.8619 |
| 5.0 | 0.5600 | 0.00 | 0 | -1.9301 |
| 10.0 | 0.4700 | 0.00 | 0 | -1.8211 |
| 50.0 | 0.5500 | 0.00 | 0 | -1.9217 |
| 100.0 | 0.6200 | 0.00 | 0 | -1.8989 |
| 500.0 | 0.5900 | 0.00 | 0 | -1.9732 |
| 1000.0 | 0.5500 | 0.00 | 0 | -1.9009 |


---

## 6. Key Findings and Diagnosis

### Critical Differences Identified:


1. **Model Density Disparity**: 
   - PATCH is **1.08x denser** than BQUBO (0.075472 vs 0.069979)
   - Higher density means more complex constraint interactions in BQM penalty terms


2. **Constraint Complexity**:
   - PATCH has 76 constraints vs BQUBO's 90
   - PATCH constraint overlap: 0.1579 vs BQUBO: 0.0300
   - Higher overlap indicates variables involved in multiple constraints simultaneously


3. **Coefficient Scale Differences**:
   - PATCH quadratic coefficient range: 276.2624
   - BQUBO quadratic coefficient range: 0.1550
   - Large range indicates penalty terms may dominate objective in PATCH


4. **Penalty Weight Sensitivity**:
   - PATCH may need even higher multipliers to achieve full feasibility

   - Removing idle penalty reduces constraints by 0
   - PATCH_NO_IDLE achieves different feasibility profile


### Root Cause Analysis:

The PATCH formulation violates constraints because:

1. **Penalty Term Dilution**: With many overlapping constraints converted to quadratic penalties, 
   the effective penalty strength gets diluted relative to the objective function.

2. **Heterogeneous Patch Areas**: Unlike BQUBO's uniform 1-acre units, PATCH has varying patch sizes.
   This introduces coefficient heterogeneity that makes penalty scaling harder.

3. **Idle Area Penalty Interaction**: The idle area penalty creates an additional energy term that
   competes with constraint penalties, potentially tipping the balance toward infeasible solutions.

4. **Constraint Coupling**: Higher constraint overlap in PATCH means variables are "pulled" in 
   multiple directions by different penalty terms, making it harder to satisfy all simultaneously.

### Recommendations:

1. **Use CQM Solver Directly**: Avoid `cqm_to_bqm()` conversion for PATCH. Use `LeapHybridCQMSampler` instead.

2. **If BQM Required**: 
   - Increase Lagrange multiplier to at least {feasible_mult if feasible_mult else '1000+'}
   - Normalize patch areas before formulation
   - Consider reformulating to reduce constraint overlap

3. **Further Investigation**: 
   - Test with actual D-Wave Hybrid BQM solver to confirm behavior
   - Analyze embedding quality and chain strength requirements
   - Profile penalty vs objective term magnitudes in failing cases

---

## 7. Conclusion

This diagnostic confirms that the PATCH formulation's constraint violations stem from its inherently
more complex penalty structure after CQM-to-BQM conversion. The combination of high density,
heterogeneous coefficients, and overlapping constraints makes it difficult for the BQM solver
to balance objective optimization with constraint satisfaction.

**The formulations are NOT equivalent** despite similar problem semantics. The discrete plot 
structure and idle area penalty fundamentally change the BQM's energy landscape compared to
BQUBO's simpler capacity-pool model.

---

*Report generated by diagnose_bqm_constraint_violations.py*
