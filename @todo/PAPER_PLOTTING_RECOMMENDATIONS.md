# Paper Plotting Recommendations: Quantum Annealing for Crop Rotation

**Analysis Date:** December 15, 2025  
**Based on:** Filtered ANALYSIS_REPORT.md (186 unique result files)

---

## Executive Summary: No Genuine Quantum Advantage Found

After comprehensive analysis of your experimental results, **I cannot recommend presenting this work as demonstrating "quantum advantage."** The results show methodological issues that undermine claims of superiority over classical methods.

However, there ARE valuable contributions that can be published honestly. Below I outline what to include and exclude.

---

## ❌ DO NOT PLOT OR INCLUDE

### 1. **Aggregated 27→6 Formulation Results**
**Files to exclude:**
- `scaling_test_*.csv` (rows with `27→6 Aggregated`)
- Any comparison showing Gurobi struggling on aggregated formulations

**Why exclude:**
- Your own `formulation_comparison.tex` admits this deliberately handicaps Gurobi
- "Smoothed benefits → weaker bounds → slower convergence"
- This is formulation engineering, not genuine advantage
- Reviewers will immediately identify this as unfair comparison

### 2. **Gurobi "Timeout" Results with Quality Gaps >5%**
**Files to exclude:**
- `benchmark_results_20251214_205508.csv`:
  - rotation_micro_25: 14.4% gap
  - rotation_small_50: 19.2% gap
- `significant_scenarios_comparison.csv`:
  - scale_small_5farms: 8.2% gap
  - cliff_hard_15farms: 9.7% gap
  - scale_medium_20farms: 12.9% gap

**Why exclude:**
- Claiming "speedup" while accepting 10-20% worse quality is misleading
- Real advantage requires being BOTH faster AND equal/better quality
- Cannot compare 80% quality solution at 30s vs 100% quality at 300s and call it "speedup"

### 3. **Inconsistent Gurobi Results**
**Conflicting files - investigate before using ANY of these:**
- `gurobi_timeout_test_20251214_184357.csv`: Shows timeouts at 300s
- `gurobi_timeout_test_20251214_180751.csv`: Shows optimal in 0.01-0.18s for SAME scenarios
- Objective values differ by 2x (e.g., 6.17 vs 11.0)

**Why exclude:**
- Indicates different problem formulations or misconfiguration
- Cannot publish without understanding why results differ
- Suggests experimental design flaws

### 4. **Hierarchical QPU "Results" with 0.0 Objective**
**Files to exclude:**
- `significant_scenarios_comparison.csv`:
  - scale_large_25farms: QPU obj=0.0
  - scale_xlarge_50farms: QPU obj=0.0
  - scale_xxlarge_100farms: QPU obj=0.0

**Why exclude:**
- These are FAILED runs, not successes
- obj=0.0 indicates no valid solution found
- Cannot claim advantage when method fails

---

## ✅ CAN INCLUDE (With Honest Framing)

### 1. **Gurobi Scaling Analysis**
**File to use:** `hardness_analysis_results/combined_all_results.csv`

**What to plot:**
```
X-axis: Number of farms (3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100)
Y-axis: Gurobi solve time (seconds)
```

**Key findings to highlight:**
- Small problems (3-10 farms, <180 vars): <1 second
- Medium problems (15-40 farms, 270-720 vars): 1-70 seconds  
- Large problems (50-100 farms, 900 vars): 140-270 seconds
- All problems reached OPTIMAL status with <1% gap

**Honest caption:**
> "Gurobi performance on rotation problems with 6 food families. Classical solver scales polynomially, finding optimal solutions for all problem sizes. Timeout-based comparisons are therefore not indicative of fundamental computational limits."

### 2. **Decomposition Strategy Effectiveness**
**Files to use:** 
- `qpu_results_integrated.csv` (QPU decomposition methods)
- `hardness_analysis_results.csv` (classical baseline)

**What to plot:**
Compare three QPU decomposition approaches:
1. Direct QPU (fails >100 vars)
2. Clique decomposition (works up to ~500 vars)
3. Spatial-temporal decomposition (works up to 900 vars)

**X-axis:** Problem size (variables)
**Y-axis:** Success rate or solution quality

**Honest caption:**
> "QPU decomposition strategies enable quantum annealing of problems beyond direct embedding limits. However, decomposed solutions show 8-15% quality degradation compared to classical optimal solutions."

### 3. **Quality-Time Tradeoff Analysis** 
**File to use:** `significant_scenarios_comparison.csv`

**What to plot:**
Scatter plot with:
- X-axis: Solution time (seconds)
- Y-axis: Solution quality (objective value)
- Points colored by method (Gurobi vs QPU decomposition)

**Include only scenarios where:**
- Both methods succeeded
- Quality gap is documented
- Times are comparable (not comparing 30s vs 300s timeout)

**Honest caption:**
> "Quality-time tradeoffs in crop rotation optimization. QPU provides approximate solutions in 20-50s (85-92% of optimal quality). Gurobi achieves 100% optimal solutions in 150-300s for complex formulations. Application-dependent tradeoff between speed and optimality."

### 4. **Problem Hardness Characterization**
**File to use:** `hardness_analysis_results.csv`

**What to plot:**
Show how problem difficulty correlates with:
- Number of farms
- Total area
- Coefficient of variation (CV) in farm sizes
- farms_per_food ratio

**Multiple subplots showing:**
1. Solve time vs n_farms
2. Solve time vs CV_area
3. Solve time vs farms_per_food

**Honest caption:**
> "Problem instance characteristics affecting computational complexity. Rotation constraints with frustrated synergy matrices create harder optimization landscapes. Well-structured problems remain tractable for classical solvers."

### 5. **Native 6-Family Formulation Performance**
**File to use:** `scaling_test_*.csv` (Native 6-Family rows only)

**What to plot:**
```
Comparison of Native 6-Family formulation:
X-axis: Problem size (360, 900, 1620, 4050 vars)
Y-axis: Solve time

Two lines:
- Gurobi: Shows optimal solutions in 0.2-3.9s
- QPU: Shows solutions in 2.9-10.3s
```

**Honest caption:**
> "Performance on well-designed Native 6-Family formulation. Classical solver dominates across all problem sizes, finding optimal solutions faster than QPU provides approximate solutions. This demonstrates importance of problem formulation in computational performance."

---

## 📊 Recommended Paper Structure

### Title Suggestions
**❌ Don't use:**
- "Quantum Advantage for Crop Rotation Optimization"
- "Quantum Speedup in Agricultural Planning"
- "Outperforming Classical Solvers with Quantum Annealers"

**✅ Use instead:**
- "Decomposition Strategies for Quantum Annealing in Large-Scale Crop Rotation"
- "Quality-Time Tradeoffs in Hybrid Classical-Quantum Agricultural Optimization"
- "Scaling Quantum Annealing Beyond Embedding Limits: A Crop Rotation Case Study"

### Abstract Framework
```
We investigate quantum annealing approaches for multi-period crop rotation 
optimization, a combinatorial problem with applications in sustainable agriculture. 
While direct QPU embedding is limited to ~100 variables, we develop spatial-temporal 
decomposition strategies that enable quantum annealing of problems up to 900 variables.

We compare quantum annealing against classical MIQP solvers across multiple problem 
formulations and scales. Results show that:

1. Well-designed classical formulations remain highly tractable (optimal solutions 
   in 0.2-4s for 4000+ variables)
   
2. Quantum annealing provides approximate solutions (85-92% optimal quality) in 
   20-50s for poorly-structured classical formulations
   
3. Decomposition strategies successfully scale quantum methods but introduce 
   quality degradation

4. Problem formulation design has greater impact on solvability than choice of 
   classical vs quantum solver

Our results demonstrate the importance of fair experimental design and highlight
complementary roles for classical and quantum approaches rather than quantum 
advantage.
```

### Results Sections

**Section 1: Classical Solver Characterization**
- Plot: Gurobi scaling analysis (✅ Include #1)
- Plot: Problem hardness factors (✅ Include #4)
- Finding: Document when classical methods succeed/struggle

**Section 2: Decomposition Engineering**
- Plot: Decomposition strategy comparison (✅ Include #2)
- Finding: Successfully extend QPU applicability beyond embedding limits
- Limitation: Quality degradation from decomposition

**Section 3: Formulation Impact**
- Plot: Native vs Aggregated vs Hybrid comparisons (✅ Include #5)
- Finding: Formulation design dominates solver choice
- Discussion: Why aggregation hurts classical solvers

**Section 4: Quality-Time Tradeoffs**
- Plot: Scatter plot analysis (✅ Include #3)
- Finding: No domain where QPU is both faster AND better quality
- Discussion: Application-specific utility functions

### Discussion Points

**Be transparent about:**
1. No evidence of quantum advantage in rigorous sense
2. Classical solvers can be much faster with good formulations
3. Decomposition introduces quality loss
4. Timeout-based comparisons may reflect configuration issues
5. Problem scale alone doesn't determine hardness

**Positive contributions:**
1. Novel decomposition strategies for quantum annealing
2. Comprehensive benchmarking methodology
3. Problem hardness characterization
4. Agricultural optimization application domain
5. Honest assessment of classical vs quantum tradeoffs

---

## 🔬 Additional Experiments Needed

Before publication, address these gaps:

### 1. **Resolve Gurobi Inconsistencies**
Rerun all experiments with:
- Documented Gurobi configuration (MIPFocus, Presolve, Threads)
- Same formulation across all tests
- Document why previous results differed

### 2. **Fair Formulation Comparisons**
Test QPU and Gurobi on:
- Same problem formulation
- Same constraints (hard, not penalty-based)
- Same timeout limits (if any)
- Same quality thresholds

### 3. **Statistical Validation**
- Multiple runs per scenario (not single runs)
- Mean ± std dev for timing
- Statistical significance tests
- Confidence intervals

### 4. **Ablation Studies**
Test impact of:
- Penalty weights
- Decomposition granularity
- Number of QPU reads
- Annealing time parameters

---

## 📝 Supplementary Materials

Include full data in supplements:
1. Complete result tables (CSV files)
2. Problem formulation details
3. Solver configurations
4. Failed experiment logs
5. Statistical analysis code

**Transparency builds credibility** - showing negative results strengthens positive claims.

---

## Final Recommendation

**For a high-impact, honest publication:**

1. **Frame as:** "Hybrid Classical-Quantum Decomposition Strategies" 
2. **Emphasize:** Engineering contributions, not quantum advantage
3. **Be transparent:** About limitations and negative results  
4. **Focus on:** Valid scientific contributions (decomposition, characterization)
5. **Avoid:** Overclaiming or misleading comparisons

**Bottom line:** You have solid engineering work in decomposition strategies and problem characterization. Present it honestly as methodology development rather than quantum advantage demonstration. This will result in a more credible, citable, and impactful paper.

---

## Plots Summary Table

| Plot | Data Source | What to Show | Include? | Priority |
|------|-------------|--------------|----------|----------|
| Gurobi Scaling | combined_all_results.csv | Time vs problem size | ✅ Yes | HIGH |
| Decomposition Comparison | qpu_results_integrated.csv | Strategy success rates | ✅ Yes | HIGH |
| Quality-Time Tradeoff | significant_scenarios_comparison.csv | Pareto frontier | ✅ Yes | HIGH |
| Problem Hardness | hardness_analysis_results.csv | Difficulty factors | ✅ Yes | MEDIUM |
| Native Formulation | scaling_test_*.csv (native only) | Classical dominance | ✅ Yes | MEDIUM |
| Aggregated Formulation | scaling_test_*.csv (aggregated) | ❌ No | EXCLUDE | - |
| Timeout "Speedup" | benchmark_results_*.csv | ❌ No | EXCLUDE | - |
| Hierarchical Failed | significant_scenarios_*.csv (obj=0) | ❌ No | EXCLUDE | - |

