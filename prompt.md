# Implementation Plan: Report Visualizations and Data Analysis
## Objective
Generate two high-quality figures (multi-panel subplots) and corresponding LaTeX tables for the      agricultural optimization report. The analysis compares classical Gurobi performance against D-Wave  hybrid and pure QPU decomposition methods.
## Data Sources
- **Classical/Hybrid Data:** `@data\comprehensive_benchmark_configs_dwave_20251130_212742.json`      (contains Gurobi, Gurobi QUBO, CQM, and BQM results).
- **Decomposition Data:** `@data\decomposition_scaling_results.json` and`@data\qpu_benchmark_*.json`.
- **Gurobi 200s Baseline:**`@data\gurobi_timeout_verification\gurobi_timeout_test_20260331_141105.json` or the unified benchmark      in `benchmark_20260329_105633.json`.
- **Theory/Structure Reference:** `@@todo/report/content_report.tex`.
## Task 1: Study 1.a - Hybrid Solver Comparison
**Goal:** Create a figure with two subplots comparing 4 solvers: Gurobi (200s), Gurobi QUBO, D-Wave  CQM, and D-Wave BQM.
### 1.1 Plotting
- **Subplot 1 (Runtime):** X-axis: Scale (number of patches/farms). Y-axis: Solve time in seconds    (log scale). Traces: 4 (Gurobi 200s, Gurobi QUBO, Hybrid CQM, Hybrid BQM).
- **Subplot 2 (Solution Quality):** X-axis: Scale. Y-axis: Objective value (maximize). Same 4 traces.   18
### 1.2 Quantitative Tables (LaTeX)
Generate LaTeX tables following the formatting logic in `@generate_tables.py`:
- **Timing Table:** For CQM and BQM, report: Scale, Solve Time, Pure QPU Time, and QPU Utilization % (calculated as `qpu_time / solve_time * 100`).
- **Violation Table:** Report the count of violations and the specific constraint types violated     (e.g., One-Hot, Food Group) for each of the 4 methods across scales. Use `analyze_violations.py` if  needed to re-evaluate the solution dictionaries found in the JSONs.
## Task 2: Study 2 - Pure QPU Decomposition Analysis
**Goal:** Create a figure with two subplots comparing Gurobi (200s) on Formulation A against 8 pure  QPU decomposition methods.
### 2.1 Plotting
- **Subplot 1 (Runtime):** X-axis: Scale. Y-axis: Solve time (log scale). Traces: Gurobi (New 200s) +      8 Decomposition Methods (PlotBased, Multilevel(5), Multilevel(10), Louvain, Spectral(10), CQM-First, Coordinated, HybridGrid).
- **Subplot 2 (Solution Quality):** X-axis: Scale. Y-axis: Objective value. Same traces.
### 2.2 Quantitative Tables (LaTeX)
- **Timing Breakdown Table:** For each decomposition method, report: Scale, Total Solve Time, Pure   QPU Time, and Classical Embedding Time. This should clearly show the overhead of `minorminer`.       
- **Violation Table:** Detailed violation count and type per decomposition method. Note which methods      (like Louvain or CQM-First) maintain zero violations vs. others.
## Task 3: Technical Integrity & Validation
- **Violation Evaluation:** If violation data is missing in the primary JSONs, use the solution      plantation dictionaries and run them through the existing logic in `analyze_violations.py` or        `assess_violation_impact.py` to ensure consistency.
- **LaTeX Formatting:** Use `booktabs`, `multirow`, and ensure the tables are wrapped in a standard  `table` environment with appropriate captions and labels as defined in `generate_tables.py`.
- **Output:** Save plots as high-resolution PDFs in `images/Plots/` and LaTeX snippets in`Benchmarks/decomposition_scaling/tables/report_tables.tex`.
## Implementation Order
1. **Research:** Map the exact keys in the JSON files to the metrics needed (objective, solve_time,  qpu_access_time).
2. **Execution (Plots):** Update or create a script (e.g., `generate_report_visuals.py`) to produce  the two 2-panel figures.
3. **Execution (Tables):** Update `generate_tables.py` or create a new one to output the specific    LaTeX tables for timing and violations.
4. **Validation:** Cross-check objective values against the Abstract in `content_report.tex` to      
   ensure the "improvement" claims match the data.