# Study 2.B Data Readiness and Report Update Plan

## Goal
Prepare complete, validated Study 2.B data before implementation, then regenerate plots/tables and update report content and conclusions using the new benchmark outcomes.

## Current State — UPDATED 2026-04-20

### ✅ Phase 1 COMPLETE — Variant B Gurobi full data filled to 2000 farms
Run at 600s timeout (`--timeout 600 --mode hard --variant-b-full-max-farms 2000`).

| n_farms | obj (MIQP) | violations (rotation) | one_hot_viols | status |
|---------|------------|----------------------|---------------|--------|
| 10      | 5.54       | 0                    | 0             | optimal |
| 25      | 12.29      | 0                    | 0             | timeout (feasible) |
| 50      | 23.49      | 0                    | 0             | optimal |
| 100     | 43.63      | 11                   | 0             | optimal |
| 200     | 86.60      | 22                   | 0             | optimal |
| 500     | 216.42     | 47                   | 0             | timeout |
| 1000    | 434.28     | 72                   | 0             | timeout |
| 2000    | 866.26     | 152                  | 0             | timeout |

Key findings:
- Gurobi always satisfies one-hot constraints (hard LP constraints in model); all violations are **rotation violations**.
- `healed_objective == objective` for all Gurobi full rows (healed correction targets one-hot violations only).
- Decomposed B (`Clique`, `SpatialTemporal(5)`) rows do not yet have healed_objective populated.

Results written to:
- `Benchmarks/decomposition_scaling/solver_comparison_results_hard.json`
- `Benchmarks/decomposition_scaling/solver_comparison_results.json` (mode-agnostic copy)
Log: `logs/benchmark_variant_b_600s.log`

### ✅ Phase 3 COMPLETE — Plots regenerated (2026-04-20)
New PDFs in `Benchmarks/decomposition_scaling/`:
- `fig_study2b_quality_decomposed_abs.pdf` — log-log absolute MIQP objective: Gurobi full vs Clique vs ST(5)
- `fig_study2b_healed_violations.pdf` — dual-panel (solve time + gap+violations), style matching QPU comprehensive reference

All 27 output PDFs generated cleanly.

### ⏳ Remaining
- Phase 2 unified comparison dataset (harmonized schema with QPU data) not yet built.
- Phase 4 table regeneration not yet done.
- Phase 5 report text update not yet done.

### Previous state (before 2026-04-20)
- Variant B in decomposition scaling had `Gurobi_full` only up to 100 farms.
- Above 100 farms, `Gurobi_full` was explicitly skipped in the benchmark script.
- Decomposed Variant B (`Clique`, `SpatialTemporal(5)`) was already present up to 2000 farms.
- Study 2.B report plots in the main report pipeline still consume `qpu_hier_repaired.json` + timeout test files from `@todo/gurobi_timeout_verification`.

## Hard References (code and data)
- Run gate that skips Variant B full above 100 farms:
  - `Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py:1524`
- Variant B benchmark max size cap:
  - `Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py:1514`
- CLI controls for benchmark timeout and decomposed mode:
  - `Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py:1557`
  - `Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py:1559`
- Current Variant B full table (ends at 100 farms):
  - `Benchmarks/decomposition_scaling/tables/table_variant_B_full.tex:27`
- Current results file shows Variant B full SKIPPED at larger sizes:
  - `Benchmarks/decomposition_scaling/solver_comparison_results.json` (entries where `variant=B`, `decomposition=none`, `status=SKIPPED`)
- Main report plotting loader for timeout files:
  - `generate_all_report_plots.py:334`
  - `generate_all_report_plots.py:507`
- Single-plot exporter that drives `scaling_qpu_breakdown_by_vars.pdf`:
  - `generate_single_plot_pdfs.py:191`
  - `generate_single_plot_pdfs.py:363`
  - `generate_single_plot_pdfs.py:2693`
- Data inventory and provenance:
  - `data_and_script_catalog.md`

## What We Are Not Doing
- No report text edits yet.
- No plot style redesign yet.
- No benchmark formulation refactor yet (only data completion and consistency first).

## Decision Tracks for Plot Line Count (requested 1/2 and 2/4 options)

### Track A (recommended first): 27-crop only, fastest to complete
- Gurobi full: 1 line
- QPU runs: 2 lines (Clique and SpatialTemporal, if available in selected QPU dataset)
- Gurobi decomposed: 2 lines (Clique and SpatialTemporal)
- Why: decomposition_scaling Variant B is currently 27-crop only, so this avoids mixing formulations.

### Track B (extended): 6-food + 27-crop split
- Gurobi full: 2 lines
- QPU runs: 4 lines
- Gurobi decomposed: 4 lines
- Requires additional data generation for decomposition_scaling in 6-food Variant B (currently not present in script defaults).

## Phase Plan

## Phase 1 - Fill Missing Variant B Gurobi Full Data
### Objective
Generate `Gurobi_full` Variant B points for farms above 100 (target: 200, 500, 1000, 2000).

### Planned implementation change (small, controlled)
- In `benchmark_solvers_comparison.py`, replace the hard skip condition with a configurable limit.
- Add CLI parameter: `--variant-b-full-max-farms` (default 100 for backward compatibility).
- Run with `--variant-b-full-max-farms 2000` for data completion.

### Execution commands (PowerShell)
- Full run (hard mode, 200s):
  - `python Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py --timeout 200 --mode hard --variant-b-full-max-farms 2000`
- Optional reproducibility run (soft mode):
  - `python Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py --timeout 200 --mode soft --variant-b-full-max-farms 2000`

### Validation checks
- `solver_comparison_results_hard.json` contains Variant B full rows for farms 200, 500, 1000, 2000 with non-SKIPPED status.
- `tables/table_variant_B_full.tex` includes those rows.

## Phase 2 - Build Unified Study 2.B Comparison Dataset
### Objective
Create a single comparison table for 3 groups:
1) Gurobi full
2) QPU runs
3) Gurobi decomposed

### Source mapping
- Gurobi full + decomposed: `Benchmarks/decomposition_scaling/solver_comparison_results_hard.json`
- QPU Study 2.B baseline: `qpu_hier_repaired.json`
- Optional Gurobi timeout companion data for QPU study context: latest `@todo/gurobi_timeout_verification/gurobi_timeout_test_*.json`

### Harmonized fields
- `n_farms`
- `n_vars`
- `group` (gurobi_full | qpu | gurobi_decomposed)
- `method` (full | clique | spatialtemporal)
- `formulation` (27-crop only in Track A; 6-food and 27-crop in Track B)
- `objective`
- `wall_time_s`

### Quality gates
- No duplicate (`group`, `method`, `n_farms`, `formulation`) keys.
- Consistent farm-size set across compared lines in each panel.

## Phase 3 - Regenerate Plots (solve time + solution quality priority)
### Objective
Regenerate Study 2.B plots emphasizing:
- Solve time ordering: `Gurobi_full` slower than QPU runs, and QPU runs slower than `Gurobi_decomposed`.
- Solution quality comparison across the same groups.

### Primary outputs
- `fig_solver_time_B_all.pdf` (decomposition_scaling report)
- `fig_solver_quality_B_all.pdf` (decomposition_scaling report)
- Study 2.B single-plot counterparts used in report workflows (including `scaling_qpu_breakdown_by_vars.pdf` lineage)

### Plot grouping policy
- Track A default: 1 (full) + 2 (qpu) + 2 (decomposed)
- Track B optional: 2 (full) + 4 (qpu) + 4 (decomposed)

### Regeneration commands (PowerShell)
- `python Benchmarks/decomposition_scaling/generate_plots.py`
- `python generate_all_report_plots.py`
- `python generate_single_plot_pdfs.py`

## Phase 4 - Regenerate Tables
### Objective
Update numerical tables from the new results.

### Targets
- `Benchmarks/decomposition_scaling/tables/table_variant_B_full.tex`
- `Benchmarks/decomposition_scaling/tables/all_tables.tex`
- Any report-facing table snapshots that consume Study 2.B values

### Quality gates
- Variant B full table includes farms 10, 25, 50, 100, 200, 500, 1000, 2000.
- No stale SKIPPED rows for those farm sizes.

## Phase 5 - Update report content and conclusions (after data is finalized)
### Objective
Update Study 2.B text so claims are strictly data-backed.

### Files to update
- `@todo/report/content_report.tex`
- If needed for consistency summaries: `IEEE_v2.tex` notes section (only if explicitly requested)

### Update scope
- Results narrative for Study 2.B
- Figure captions tied to new 3-group plots
- Table references and quantitative statements
- Conclusions section language based on measured outcomes

### Claim discipline
- Use measured metrics only.
- Keep solve time and solution quality as primary axes.
- Explicitly separate formulation-dependent behavior if Track B is enabled.

## Phase 6 - Final Verification and Handoff
### Checks
- Benchmark JSON contains complete Variant B full data above 100 farms.
- Plot PDFs regenerated without missing files.
- Table values match JSON sources.
- Report text references existing figures/tables and updated numbers.

### Deliverables
- Updated benchmark JSON files
- Updated decomposition_scaling plots and tables
- Updated Study 2.B report plots
- Updated report text and conclusions

## Risks and Mitigations
- Risk: full Variant B at 2000 farms may be long-running or memory heavy.
  - Mitigation: run in descending checkpoints (200, 500, 1000, 2000) and persist each completion.
- Risk: mixed data sources create apples-to-oranges comparisons.
  - Mitigation: enforce harmonized schema and explicit `formulation` labels before plotting.
- Risk: stale loaders pick old timeout files.
  - Mitigation: verify latest-file selection and record the exact input filenames used for each generated figure.

## Rollback Plan
- Keep previous JSON and PDFs with timestamped backups before regeneration.
- If new full-B runs fail at high sizes, keep valid partial sizes and mark unsupported points explicitly in plots/tables.

## Approval Gate Before Implementation
Proceed with Track A first (27-crop only), then optionally extend to Track B if needed.