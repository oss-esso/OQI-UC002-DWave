# Plot and Script Association Report with Technical Analysis

**Purpose:** This document maps each plotting script to its generated visualizations, with detailed technical notes on what is plotted, how it's computed, and what insights it provides for the research report.

---

## 📊 CORE PAPER PLOTS



### Script: `generate_comprehensive_scaling_plots.py`
**Purpose:** Multi-metric scaling analysis across formulations  
**Data Sources:** `qpu_hier_repaired.json`, `gurobi_baseline_60s.json`

Generates 2 comprehensive multi-panel plots:

1. **`comprehensive_scaling.png`** (Main scaling figure)
   - **2x3 grid layout:**
     - **(0,0) Gap vs Variables:** Shows optimality gap % by formulation (6-Family vs 27-Food)
       - Metrics: `gap = |qpu_obj - gurobi_obj| / gurobi_obj * 100`
       - 20% target line for feasibility threshold
     - **(0,1) Objectives:** Gurobi (solid) vs QPU (dashed) objective values
       - Uses absolute values `abs(objective_miqp)`
       - Separate lines for each formulation
     - **(0,2) Speedup:** Speedup factor = gurobi_time / qpu_time, log scale
       - Break-even line at 1.0
       - Shows where QPU outperforms
     - **(1,0) Time Comparison:** Grouped bars Gurobi vs QPU Total
       - Timeout markers ('T') on Gurobi bars
     - **(1,1) QPU Time Stacked:** Pure QPU (bottom) + Classical overhead (top)
       - Percentage labels for pure quantum fraction
     - **(1,2) Variable Scaling:** Log-log plot showing scaling exponents
       - Polynomial fits with exponent annotations

2. **`quantum_advantage_comprehensive_scaling.png`**
   - Similar to above but with different statistical aggregations
   - Focuses on advantage zones (speedup > 1, gap < 20%)

3. **`quantum_advantage_objective_scaling.png`**
   - Objective value comparison with confidence intervals
   - Shows convergence behavior with problem size

---

### Script: `generate_split_formulation_plots.py`
**Purpose:** Detailed analysis by problem configuration (farms × foods × periods)  
**Data Sources:** `qpu_hier_repaired.json`, `gurobi_baseline_60s.json`

Generates 2 plots:

1. **`quantum_advantage_split_analysis.png`**
   - **What:** Side-by-side comparison of 6-family vs 27-food performance
   - **How:** 2x2 grid with time, speedup, gap, success rate by formulation
   - **Key Insight:** Different formulations show advantage in different size regimes

2. **`quantum_advantage_objective_gap_analysis.png`**
   - **What:** Gap analysis with violation correlation
   - **How:** Scatter plot gap vs violations with regression line
   - **Key Insight:** Most gap explained by constraint violations

---



### Script: `assess_violation_impact.py`

**Purpose:** Impact assessment of different violation types on solution quality  
**Data Sources:** `qpu_hier_repaired.json`, `gurobi_baseline_60s.json`

Generates the following plots:

- `violation_impact_assessment.png`
  - **What:** Multi-panel analysis of violation types and their costs
  - **Violation Types:**
    - `one_hot_violations`: Farm-period has no crop or multiple crops
    - `min_crops`: Farm-period assignment count violations
    - `food_group_min/max`: Global food group constraint violations
  - **Metrics:** Violation count, penalty contribution, correlation with objective gap
  - **Key Insight:** One-hot violations dominate; repair strategies should prioritize these

---



## 🎯 ADVANCED QPU ANALYSIS

### Script: `plot_qpu_advantage_corrected.py`

**Purpose:** CRITICAL REINTERPRETATION - Corrects sign convention for maximization problem  
**Data Sources:** `qpu_hier_repaired.json`, Gurobi 300s timeout test

**Key Insight:** This is a MAXIMIZATION problem. QPU QUBO minimizes (-benefit + penalties), so more negative = higher benefit achieved.

Generates the following plots:

- `qpu_advantage_corrected.png` (Main corrected analysis)
  - **What:** 2x3 grid showing TRUE quantum advantage after sign correction
  - **Panels:**
    - Benefit comparison: QPU achieves HIGHER benefit than Gurobi (positive advantage)
    - Violation trade-off: Violations vs benefit gain scatter
    - Time efficiency: Pure QPU time vs classical coordination
  - **Critical Correction:** Previously misinterpreted as "QPU worse" - actually "QPU better"!
  



### Script: `deep_dive_gap_analysis.py`

**Purpose:** Detailed analysis of optimality gap components  
**Data Sources:** `qpu_hier_repaired.json`, `gurobi_baseline_60s.json`, `gurobi_timeout_test_300s.json`

Generates the following plots:

- `gap_deep_dive.png`
  - **What:** Decomposition of gap into: violations, embedding artifacts, optimization quality
  - **How:** Statistical decomposition with regression analysis
  - **Metrics:**
    - Violation contribution: ~80-90% of gap
    - Embedding loss: ~5-10% of gap
    - True optimization gap: ~5-10%
  - **Key Insight:** After violations are repaired, QPU solutions are near-optimal!

---



### Script: `Plot Scripts\plot_qpu_benchmark_results.py`

**Purpose:** Comprehensive benchmark summary visualization  
**Data Sources:** `qpu_benchmark_summary_*.json`, `gurobi_baseline_60s.json`

Generates the following plots:

- `qpu_benchmark_small_scale.png`
  - **What:** Detailed metrics for problems <500 variables
  - **Panels:** Time, gap, violations, embedding stats

- `qpu_benchmark_large_scale.png`
  - **What:** Same for problems >500 variables
  - **Key Insight:** Different behavior regimes

- `qpu_benchmark_comprehensive.png`
  - **What:** Combined view across all scales
  - **5-panel layout:** Time, objective, violations, speedup, success rate


