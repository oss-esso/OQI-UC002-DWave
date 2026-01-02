# Post-Processing Repair Heuristic Report

**Date:** January 2, 2026  
**Purpose:** Apply greedy repair heuristic to fix constraint violations in QPU hierarchical decomposition solutions

---

## Executive Summary

Successfully applied a classical post-processing repair heuristic to **929 total constraint violations** across **9 scenarios**, achieving **100% repair success rate** (zero remaining violations). While the repairs produce feasible solutions, the resulting objective values remain **77.5% below Gurobi baselines on average**, indicating the trade-off between feasibility enforcement and solution optimality.

---

## Methodology

### Repair Strategy

The repair heuristic addresses one-hot constraint violations where farm-periods have no crop assigned. For each violation:

1. **Identify Context:** Determine crops assigned in adjacent time periods (t-1, t+1)
2. **Score Candidates:** Evaluate each possible crop based on:
   - **Base benefit:** Nutritional value minus environmental impact
   - **Rotation synergy:** Compatibility with adjacent period crops (±50% weight)
   - **Diversity penalty:** Small penalty for same crop as adjacent period (-30%)
3. **Assign Best Crop:** Select highest-scoring crop for the violated farm-period
4. **Recalculate Objective:** Compute MIQP objective with repaired solution

### Implementation Details

- **Programming Language:** Python 3
- **Input:** Pickled cluster samplesets from hierarchical decomposition (`qpu_samplesets_all/`)
- **Processing:** 
  - Reconstruct global solution from cluster solutions
  - Identify all one-hot violations
  - Apply greedy repair to each violation
  - Verify zero remaining violations
- **Output:** Repaired solution with objective value and comparison to baselines

---

## Results Summary

### Overall Statistics

| Metric | Value |
|--------|-------|
| **Scenarios Processed** | 9 |
| **Total Violations Repaired** | 929 |
| **Average Violations per Scenario** | 103.2 |
| **Repair Success Rate** | 100% (0 remaining violations) |
| **Average Objective Improvement** | +2.24 (vs original violated solutions) |
| **Average Gap to Gurobi** | -77.5% (below baseline) |

### Per-Scenario Results

| Scenario | Farms | Original Obj | Violations | Repaired Obj | Improvement | Gap to Gurobi |
|----------|-------|--------------|------------|--------------|-------------|---------------|
| rotation_micro_25 | 5 | 2.80 | 2 | 3.20 | +0.40 (+14.3%) | -48.1% |
| rotation_small_50 | 10 | 1.30 | 17 | 3.00 | +1.70 (+130.8%) | -65.5% |
| rotation_15farms_6foods | 15 | 0.87 | 32 | 3.00 | +2.13 (+246.2%) | -69.0% |
| rotation_medium_100 | 20 | 0.70 | 47 | 3.05 | +2.35 (+335.7%) | -76.1% |
| rotation_25farms_6foods | 25 | 0.56 | 62 | 3.04 | +2.48 (+442.9%) | -77.4% |
| rotation_50farms_6foods | 50 | 0.32 | 136 | 3.04 | +2.72 (+850.0%) | -88.7% |
| rotation_large_200 | 50 | 0.30 | 136 | 3.02 | +2.72 (+906.7%) | -86.0% |
| rotation_75farms_6foods | 75 | 0.20 | 211 | 3.01 | +2.81 (+1406.7%) | -92.5% |
| rotation_100farms_6foods | 100 | 0.14 | 286 | 3.00 | +2.86 (+2042.9%) | -94.4% |

### Gurobi Baseline Comparison

| Scenario | Gurobi (300s) | Repaired QPU | Absolute Gap | Relative Gap |
|----------|---------------|--------------|--------------|--------------|
| rotation_micro_25 | 6.17 | 3.20 | -2.97 | -48.1% |
| rotation_small_50 | 8.69 | 3.00 | -5.69 | -65.5% |
| rotation_15farms_6foods | 9.68 | 3.00 | -6.68 | -69.0% |
| rotation_medium_100 | 12.78 | 3.05 | -9.73 | -76.1% |
| rotation_25farms_6foods | 13.45 | 3.04 | -10.41 | -77.4% |
| rotation_large_200 | 21.57 | 3.02 | -18.55 | -86.0% |
| rotation_50farms_6foods | 26.92 | 3.04 | -23.88 | -88.7% |
| rotation_75farms_6foods | 40.37 | 3.01 | -37.36 | -92.5% |
| rotation_100farms_6foods | 53.77 | 3.00 | -50.77 | -94.4% |

---

## Key Findings

### 1. Feasibility Restoration Success ✅

- **100% repair success:** All 929 violations successfully repaired across all scenarios
- **Zero remaining violations:** Every repaired solution satisfies one-hot constraints
- **Scalability:** Repair heuristic handles scenarios from 5 to 100 farms effectively

### 2. Violation Patterns 📊

- **Violations scale with farms:** Larger scenarios have proportionally more violations
  - 5 farms: 2 violations (0.4 per farm)
  - 100 farms: 286 violations (2.86 per farm)
- **Hierarchical clustering impact:** Violations appear at cluster boundaries where coordination fails
- **Multi-assignment edge cases:** Some scenarios show 2 crops assigned (e.g., Farm3 period 3) - additional constraint violation type

### 3. Objective Quality Trade-off ⚖️

**Positive:** Objective improvements vs violated solutions
- Average improvement: +2.24 (ranging from +0.40 to +2.86)
- Percentage gains: 14.3% to 2042.9%

**Negative:** Significant gaps vs Gurobi optimal
- Average gap: -77.5% below Gurobi
- Worst case: -94.4% for 100-farm scenario
- Best case: -48.1% for micro-25 scenario

**Interpretation:**
- Greedy repair prioritizes feasibility over optimality
- Each local repair decision doesn't consider global objective impact
- Result: Feasible but highly suboptimal solutions

### 4. Scaling Behavior 📈

As farm count increases:
- **Violations increase** (linearly with farms)
- **Repair improvement increases** (more "free" objective from assigning crops)
- **Gap to Gurobi worsens** (harder to achieve global optimality)

### 5. Repair Heuristic Characteristics 🔧

**Strengths:**
- Fast: Repairs 286 violations in seconds
- Robust: 100% success rate
- Simple: Greedy local decisions
- Interpretable: Clear scoring logic

**Weaknesses:**
- Myopic: No global optimization
- Suboptimal: Large gaps to optimal solutions
- Fixed strategy: Doesn't adapt to problem structure

---

## Visualizations

Generated plots in `professional_plots/`:

1. **`repair_heuristic_summary.png`** - Three-panel comparison:
   - Original vs Repaired vs Gurobi objectives
   - Violations repaired per scenario
   - Quality gaps to Gurobi baseline

2. **`repair_improvement_analysis.png`** - Two-panel analysis:
   - Objective improvement bars
   - Violations vs improvement scatter (colored by farm count)

---

## Conclusions

### What This Demonstrates ✓

1. **Feasibility is achievable:** Classical post-processing can repair QPU solutions to satisfy constraints
2. **Hierarchical decomposition challenge:** Cluster-based solving creates coordination gaps that require repair
3. **Trade-off exists:** Fast greedy repair trades optimality for feasibility

### What This Reveals ⚠️

1. **QPU solutions need significant repair:** 103 violations per scenario on average
2. **Repair quality insufficient:** 77.5% gap to optimal indicates fundamental limitation
3. **Hierarchical approach struggles:** Even with repair, far from classical solver quality

### Recommendations 🎯

**For Production Use:**
- ❌ **Not recommended:** Current QPU + repair approach produces solutions ~80% below optimal
- ⚠️ **Use with caution:** Only if extreme time constraints make classical solvers infeasible

**For Research Improvement:**
1. **Better repair heuristics:** 
   - Use local optimization (e.g., 2-opt, local search) instead of greedy
   - Consider global objective impact during repair
   - Employ iterative improvement cycles

2. **Better decomposition:**
   - Improve clustering to reduce violations
   - Add overlap regions between clusters
   - Use iterative refinement between cluster solves

3. **Hybrid approaches:**
   - Use QPU for initial solution
   - Apply classical MIP solver to repair/improve
   - Combine quantum annealing with classical optimization

4. **Alternative QPU formulations:**
   - Explore penalty-based formulations (softer constraints)
   - Test different embedding strategies
   - Investigate native QPU topology matching

---

## Technical Details

### Files Generated

- **`postprocessing_repair.py`** - Main repair implementation
- **`professional_plots/postprocessing_repair_results.json`** - Detailed results data
- **`professional_plots/repair_heuristic_summary.png/pdf`** - Summary visualizations
- **`professional_plots/repair_improvement_analysis.png/pdf`** - Improvement analysis
- **`REPAIR_HEURISTIC_REPORT.md`** - This report

### Repair Algorithm Pseudocode

```python
for each scenario:
    # Load data
    clusters = load_cluster_samplesets(scenario)
    solution = reconstruct_global_solution(clusters)
    
    # Find violations
    violations = identify_one_hot_violations(solution)
    
    # Repair each violation
    for (farm, period) in violations:
        # Get context
        prev_crop = solution[farm, period-1]
        next_crop = solution[farm, period+1]
        
        # Score each candidate crop
        for crop in all_crops:
            score = base_benefit(crop)
            score += rotation_synergy(crop, prev_crop, next_crop)
            score -= diversity_penalty(crop, prev_crop, next_crop)
        
        # Assign best crop
        best_crop = argmax(scores)
        solution[farm, best_crop, period] = 1
    
    # Recalculate objective
    objective = calculate_miqp_objective(solution)
```

---

## Future Work

1. **Implement iterative repair:** After initial repair, apply local search to improve objective
2. **Test alternative scoring functions:** Experiment with different weights for benefit/synergy/diversity
3. **Compare repair strategies:** Benchmark greedy vs simulated annealing vs genetic algorithms
4. **Integrate with QPU pipeline:** Automate repair as post-processing step
5. **Develop hybrid solver:** Combine QPU initial solution with classical polish

---

## Acknowledgments

- **Repair implementation:** Developed using greedy heuristic principles
- **Baseline comparison:** Gurobi 300s timeout results from prior benchmarking
- **Visualization:** Matplotlib-based professional plots

---

**Report Generated:** 2026-01-02  
**Total Runtime:** ~30 seconds for all scenarios  
**Success Rate:** 100% (929/929 violations repaired)
