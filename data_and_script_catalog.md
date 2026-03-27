# Data File & Generator Script Catalog

**Purpose:** Complete mapping of data files used by the three plot-generating sources,
the scripts that plot them, and the scripts that originally generated the data files.

**Sources traced:**
1. `@todo/report/content_report.tex` — main report (20 figures)
2. `Benchmarks/decomposition_scaling/decomposition_benchmark_report.tex` — decomposition study (10 figure panels)
3. `generate_single_plot_pdfs.py` — single-plot PDF exporter (imports from `generate_all_report_plots.py`)

---

## 1. Data Files

### 1.1 QPU Hierarchical Results

| Data File | Path | Description |
|-----------|------|-------------|
| `qpu_hier_repaired.json` | `./qpu_hier_repaired.json` | QPU hierarchical benchmark results (repaired), 13 rotation scenarios. Main Study 3 data. |

**Used by figures in:**
- `content_report.tex`: Figs 5–8 (comprehensive scaling, qpu_advantage_corrected, split_analysis, violation_impact_assessment)
- `generate_single_plot_pdfs.py`: Sections 1–6 (scaling, split formulation, objective gap, advantage, violation, gap deep dive)
- `paper_plots/generate_study_plots_v3.py`: study1 QPU comparison panels

**Generator script:** `unified_benchmark.py` via `unified_benchmark/core.py::save_benchmark_results()`
- **Invocation:** `python unified_benchmark.py --mode qpu-hierarchical-aggregated --sampler qpu --output-json qpu_hier_repaired.json`
- The filename was passed as a CLI argument, not hardcoded.

---

### 1.2 Gurobi Baseline (60s timeout)

| Data File | Path | Description |
|-----------|------|-------------|
| `gurobi_baseline_60s.json` | `./gurobi_baseline_60s.json` | Gurobi ground-truth results with 60-second timeout, 13 rotation scenarios. |

**Used by figures in:**
- `content_report.tex`: Fig 1 (study1_hybrid_performance — via generate_study_plots_v3.py)
- `generate_single_plot_pdfs.py`: Section 1 (prepare_scaling_data_60s)
- `generate_all_report_plots.py`: `load_gurobi_60s()` → comprehensive scaling plot (60s variant)
- `paper_plots/generate_study_plots_v3.py`: Gurobi baseline for study plots

**Generator script:** `unified_benchmark.py` via `unified_benchmark/core.py::save_benchmark_results()`
- **Invocation:** `python unified_benchmark.py --mode gurobi-true-ground-truth --timeout 3600 --output-json gurobi_baseline_60s.json`
- The filename was passed as a CLI argument, not hardcoded.
- **Note:** Output filename kept as `gurobi_baseline_60s.json` for backward compatibility (all loaders reference this name), but actual Gurobi timeout is now **3600 s** so Gurobi can find optimal/near-optimal solutions where previously it timed out at 60 s.

---

### 1.3 Gurobi Timeout Tests (300s)

| Data File | Path | Description |
|-----------|------|-------------|
| `gurobi_timeout_test_20251224_103144.json` | `@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json` | Gurobi results with 300-second timeout (specific file used as default). |
| `gurobi_timeout_test_*.json` | `@todo/gurobi_timeout_verification/gurobi_timeout_test_*.json` | All Gurobi 300s timeout test files (glob pattern; latest by timestamp is used). |

**Used by figures in:**
- `content_report.tex`: Figs 5–8 (via `generate_all_report_plots.py` 300s data path)
- `generate_single_plot_pdfs.py`: Sections 1, 4, 5, 6 (prepare_scaling_data_300s, violation_impact, gap_deep_dive, qpu_advantage)
- `generate_all_report_plots.py`: `load_gurobi_300s()`, `load_violation_impact_data()`, `load_gap_deep_dive_data()`

**Generator script:** `@todo/test_gurobi_timeout.py` (line 494) and `@todo/test_gurobi_timeout_WORKING.py` (line 338)
- Writes to `OUTPUT_DIR / f'gurobi_timeout_test_{timestamp}.json'`
- `OUTPUT_DIR = Path(__file__).parent / 'gurobi_timeout_verification'`
- **Note:** `test_gurobi_timeout.py` updated 2026-03-27: `GUROBI_CONFIG['timeout']` changed **100 s → 3600 s**; `stopped_reason` updated to `'timeout_3600s'`. The specific file `gurobi_timeout_test_20251224_103144.json` (300 s run) is still the hardcoded default in `DEFAULT_GUROBI_300S_FILE` and remains on disk. New 3600 s results will be written to a new timestamped file and automatically picked up by `load_gurobi_300s()` (which selects the **latest** file matching `gurobi_timeout_test_*.json`).

---

### 1.4 QPU Benchmark Results (Study 2 — Decomposition)

| Data File | Path | Description |
|-----------|------|-------------|
| `qpu_benchmark_20251201_160444.json` | `@todo/qpu_benchmark_results/qpu_benchmark_20251201_160444.json` | Small-scale QPU benchmark (10–100 farms, 8 decomposition methods) |
| `qpu_benchmark_20251201_200012.json` | `@todo/qpu_benchmark_results/qpu_benchmark_20251201_200012.json` | Large-scale QPU benchmark (200–1000 farms, selected methods) |
| `qpu_benchmark_20251203_121526.json` | `@todo/qpu_benchmark_results/qpu_benchmark_20251203_121526.json` | HybridGrid(5,9) benchmark (10–1000 farms) |
| `qpu_benchmark_20251203_133144.json` | `@todo/qpu_benchmark_results/qpu_benchmark_20251203_133144.json` | HybridGrid(10,9) benchmark (500–1000 farms) |

**Used by figures in:**
- `content_report.tex`: Figs 2–4 (qpu_benchmark_small_scale, qpu_benchmark_large_scale, qpu_benchmark_comprehensive)
- `generate_single_plot_pdfs.py`: Section 7 (QPU benchmark plots via `_load_qpu_benchmark_data()`)
- `generate_all_report_plots.py`: `_load_qpu_benchmark_data()` → QPU benchmark small/large/comprehensive plots

**Generator script:** `@todo/qpu_benchmark.py` (lines 4897–4904, `save_results()`)
- Writes to `OUTPUT_DIR / f"qpu_benchmark_{ts}.json"`
- `OUTPUT_DIR = Path(__file__).parent / "qpu_benchmark_results"`

---

### 1.5 Comprehensive Hybrid Benchmark (Study 1)

| Data File | Path | Description |
|-----------|------|-------------|
| `comprehensive_benchmark_configs_dwave_20251130_212742.json` | `Benchmarks/COMPREHENSIVE/comprehensive_benchmark_configs_dwave_20251130_212742.json` | Comprehensive hybrid solver benchmark (Gurobi CQM, D-Wave CQM, D-Wave BQM across 10–1000 patches) |

**Used by figures in:**
- `content_report.tex`: Fig 1 (study1_hybrid_performance)
- `paper_plots/generate_study_plots_v3.py`: `load_comprehensive_benchmark()` → hybrid solver panel

**Generator script:** `Benchmark Scripts/comprehensive_benchmark.py` (lines 1204–1227)
- Writes to `f"comprehensive_benchmark_{config_str}{dwave_suffix}_{timestamp}.json"`
- Output directory: `Benchmarks/COMPREHENSIVE/`

---

### 1.6 Decomposition Scaling Results (Appendix)

| Data File | Path | Description |
|-----------|------|-------------|
| `decomposition_scaling_results.json` | `Benchmarks/decomposition_scaling/decomposition_scaling_results.json` | Decomposition overhead timing (Variant A + B, all methods, 5–10000 farms) |
| `solver_comparison_results.json` | `Benchmarks/decomposition_scaling/solver_comparison_results.json` | Gurobi-full vs Gurobi-decomposed vs PT-ICM comparison (time + quality) |

**Used by figures in:**
- `decomposition_benchmark_report.tex`: All 10 figure panels (fig_decomp_time_A/B, fig_partition_count_A/B, fig_max_part_size_A, fig_solver_time_A/B_all, fig_solver_quality_A/B_all, fig_combined_panel)

**Generator scripts:**
- `decomposition_scaling_results.json` ← `Benchmarks/decomposition_scaling/benchmark_decomposition_scaling.py` (line 341)
  - Writes to `out_dir / "decomposition_scaling_results.json"` where `out_dir = Path(__file__).parent`
- `solver_comparison_results.json` ← `Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py` (line 1067)
  - Writes to `out_dir / "solver_comparison_results.json"` where `out_dir = Path(__file__).parent`
  - **Note:** `benchmark_solvers_comparison.py` updated 2026-03-27: `run_all()` now accepts `--timeout` CLI arg (default 3600 s). All Gurobi sub-problem calls (`solve_gurobi_full_A/B`, `solve_gurobi_decomposed_A/B`) now receive the timeout parameter.

---

### 1.7 Raw Input Data (Not Generated by Script)

| Data File | Path | Description |
|-----------|------|-------------|
| `Combined_Food_Data.xlsx` | `Inputs/Combined_Food_Data.xlsx` | 27 crops × 5 food groups with nutritional, environmental, affordability, sustainability scores. Source dataset from GAIN. |

**Used by figures in:**
- `content_report.tex`: Appendix Figs 9–14 (crop sensitivity analysis: 01–06 plots)
- `crop_benefit_weight_analysis.py`: `load_food_data()` reads this file

**Generator:** None — this is raw input data.

---

## 2. Plot Scripts (Plot Figure → Script → Data Files)

### 2.1 Figures in `content_report.tex`

| Fig # | Plot File (in `images/Plots/`) | Plotting Script | Data Files Used |
|-------|-------------------------------|-----------------|-----------------|
| 1 | `study1_hybrid_performance.pdf` | `paper_plots/generate_study_plots_v3.py` | `Benchmarks/COMPREHENSIVE/comprehensive_benchmark_configs_dwave_20251130_212742.json`, `qpu_hier_repaired.json`, `gurobi_baseline_60s.json` |
| 2 | `qpu_benchmark_small_scale.pdf` | `generate_all_report_plots.py` (`_load_qpu_benchmark_data()`) | `@todo/qpu_benchmark_results/qpu_benchmark_20251201_160444.json` |
| 3 | `qpu_benchmark_large_scale.pdf` | `generate_all_report_plots.py` (`_load_qpu_benchmark_data()`) | `@todo/qpu_benchmark_results/qpu_benchmark_20251201_200012.json`, `@todo/qpu_benchmark_results/qpu_benchmark_20251203_121526.json`, `@todo/qpu_benchmark_results/qpu_benchmark_20251203_133144.json` |
| 4 | `qpu_benchmark_comprehensive.pdf` | `generate_all_report_plots.py` (`_load_qpu_benchmark_data()`) | All 4 QPU benchmark JSON files (§1.4) |
| 5 | `quantum_advantage_comprehensive_scaling.pdf` | `generate_all_report_plots.py` (`plot_comprehensive_scaling()`) | `qpu_hier_repaired.json`, `gurobi_baseline_60s.json` |
| 6 | `qpu_advantage_corrected.pdf` | `generate_all_report_plots.py` | `qpu_hier_repaired.json`, `@todo/gurobi_timeout_verification/gurobi_timeout_test_*.json` (300s) |
| 7 | `quantum_advantage_split_analysis.pdf` | `generate_all_report_plots.py` (`plot_split_formulation_analysis()`) | `qpu_hier_repaired.json`, `@todo/gurobi_timeout_verification/gurobi_timeout_test_*.json` (300s) |
| — | `quantum_advantage_objective_gap_analysis.pdf` (commented out) | `generate_all_report_plots.py` (`plot_objective_gap_analysis()`) | `qpu_hier_repaired.json`, `@todo/gurobi_timeout_verification/gurobi_timeout_test_*.json` (300s) |
| 8 | `violation_impact_assessment.pdf` | `generate_all_report_plots.py` (`plot_violation_impact()`) | `qpu_hier_repaired.json`, `@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json` |
| — | `gap_deep_dive.pdf` (commented out) | `generate_all_report_plots.py` (`plot_gap_deep_dive()`) | `qpu_hier_repaired.json`, `@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json` |
| 9 | `01_top_crop_distribution.png` | `crop_benefit_weight_analysis.py` | `Inputs/Combined_Food_Data.xlsx` |
| 10 | `02_benefit_heatmap.png` | `crop_benefit_weight_analysis.py` | `Inputs/Combined_Food_Data.xlsx` |
| 11 | `03_ranking_variability.png` | `crop_benefit_weight_analysis.py` | `Inputs/Combined_Food_Data.xlsx` |
| 12 | `04_sensitivity_w_nutr_val.png` | `crop_benefit_weight_analysis.py` | `Inputs/Combined_Food_Data.xlsx` |
| 13 | `04_sensitivity_w_nutr_den.png` | `crop_benefit_weight_analysis.py` | `Inputs/Combined_Food_Data.xlsx` |
| 14 | `04_sensitivity_w_env_imp.png` | `crop_benefit_weight_analysis.py` | `Inputs/Combined_Food_Data.xlsx` |
| 15 | `04_sensitivity_w_afford.png` | `crop_benefit_weight_analysis.py` | `Inputs/Combined_Food_Data.xlsx` |
| 16 | `04_sensitivity_w_sustain.png` | `crop_benefit_weight_analysis.py` | `Inputs/Combined_Food_Data.xlsx` |
| 17 | `05_spinach_analysis.png` | `crop_benefit_weight_analysis.py` | `Inputs/Combined_Food_Data.xlsx` |
| 18 | `06_parallel_coordinates.png` | `crop_benefit_weight_analysis.py` | `Inputs/Combined_Food_Data.xlsx` |

### 2.2 Figures in `decomposition_benchmark_report.tex`

| Fig # | Plot File | Plotting Script | Data Files Used |
|-------|-----------|-----------------|-----------------|
| 1a | `fig_decomp_time_A.pdf` | `Benchmarks/decomposition_scaling/generate_plots.py` | `Benchmarks/decomposition_scaling/decomposition_scaling_results.json` |
| 1b | `fig_partition_count_A.pdf` | `Benchmarks/decomposition_scaling/generate_plots.py` | `Benchmarks/decomposition_scaling/decomposition_scaling_results.json` |
| 1c | `fig_max_part_size_A.pdf` | `Benchmarks/decomposition_scaling/generate_plots.py` | `Benchmarks/decomposition_scaling/decomposition_scaling_results.json` |
| 2a | `fig_decomp_time_B.pdf` | `Benchmarks/decomposition_scaling/generate_plots.py` | `Benchmarks/decomposition_scaling/decomposition_scaling_results.json` |
| 2b | `fig_partition_count_B.pdf` | `Benchmarks/decomposition_scaling/generate_plots.py` | `Benchmarks/decomposition_scaling/decomposition_scaling_results.json` |
| 3 | `fig_solver_time_A_all.pdf` | `Benchmarks/decomposition_scaling/generate_plots.py` | `Benchmarks/decomposition_scaling/solver_comparison_results.json` |
| 4 | `fig_solver_quality_A_all.pdf` | `Benchmarks/decomposition_scaling/generate_plots.py` | `Benchmarks/decomposition_scaling/solver_comparison_results.json` |
| 5 | `fig_solver_time_B_all.pdf` | `Benchmarks/decomposition_scaling/generate_plots.py` | `Benchmarks/decomposition_scaling/solver_comparison_results.json` |
| 6 | `fig_solver_quality_B_all.pdf` | `Benchmarks/decomposition_scaling/generate_plots.py` | `Benchmarks/decomposition_scaling/solver_comparison_results.json` |
| 7 | `fig_combined_panel.pdf` | `Benchmarks/decomposition_scaling/generate_plots.py` | `Benchmarks/decomposition_scaling/solver_comparison_results.json` |

### 2.3 Plots in `generate_single_plot_pdfs.py` (via `generate_all_report_plots.py`)

`generate_single_plot_pdfs.py` imports all data loaders from `generate_all_report_plots.py`
and produces individual PDF files in `phase3_single_plots/`. It covers the same data
as sections 2.1 Figs 2–8 plus additional per-panel breakdowns. Summary of data flow:

| Section | Plot Functions | Data Loader | Data Files |
|---------|---------------|-------------|------------|
| 1: Comprehensive Scaling | `plot_scaling_objectives_by_vars/farms`, `plot_scaling_time_comparison`, `plot_scaling_qpu_breakdown_by_vars/farms` | `prepare_scaling_data_60s()` → `load_qpu_hierarchical()` + `load_gurobi_60s()` | `qpu_hier_repaired.json`, `gurobi_baseline_60s.json` |
| 2: Split Formulation | `plot_split_objectives_*`, `plot_split_optimality_gap_*`, `plot_split_solve_time_*`, `plot_split_speedup_*`, `plot_split_pure_qpu_time_*`, `plot_split_gurobi_mip_gap_*` | `prepare_scaling_data_300s()` → `load_qpu_hierarchical()` + `load_gurobi_300s()` | `qpu_hier_repaired.json`, `@todo/gurobi_timeout_verification/gurobi_timeout_test_*.json` |
| 3: Objective Gap | 6 plots | `prepare_scaling_data_300s()` | Same as Section 2 |
| 4: Violation Impact | `plot_violation_rate_bars`, `plot_violation_gap_comparison`, `plot_violation_triple_objectives` | `load_violation_impact_data()` | `qpu_hier_repaired.json`, `@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json` |
| 5: Gap Deep Dive | `plot_gap_attribution_bars`, `plot_gap_corrected_comparison`, `plot_gap_correction_effectiveness`, `plot_gap_violations_vs_gap`, `plot_gap_mip_gap_vs_qpu_gap`, `plot_gap_formulation_summary_table` | `load_gap_deep_dive_data()` | `qpu_hier_repaired.json`, `@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json` |
| 6: QPU Advantage | `plot_advantage_*` (6 plots) | `_prepare_qpu_advantage_data()` → `load_qpu_hierarchical()` + `load_gurobi_300s()` | `qpu_hier_repaired.json`, `@todo/gurobi_timeout_verification/gurobi_timeout_test_*.json` |
| 7: QPU Benchmark | Multiple benchmark plots | `_load_qpu_benchmark_data()` | All 4 files in `@todo/qpu_benchmark_results/` (§1.4) |

---

## 3. Data File → Generator Script Summary

| # | Data File | Generator Script | Key Function / Line |
|---|-----------|-----------------|---------------------|
| 1 | `qpu_hier_repaired.json` | `unified_benchmark.py` | `unified_benchmark/core.py::save_benchmark_results()` — CLI arg `--output-json` |
| 2 | `gurobi_baseline_60s.json` | `unified_benchmark.py` | `unified_benchmark/core.py::save_benchmark_results()` — CLI arg `--output-json` |
| 3 | `@todo/gurobi_timeout_verification/gurobi_timeout_test_*.json` | `@todo/test_gurobi_timeout.py` (L494) | `OUTPUT_DIR / f'gurobi_timeout_test_{timestamp}.json'` |
| 3alt | (same) | `@todo/test_gurobi_timeout_WORKING.py` (L338) | Same pattern, alternative version |
| 4 | `@todo/qpu_benchmark_results/qpu_benchmark_20251201_160444.json` | `@todo/qpu_benchmark.py` (L4897–4904) | `OUTPUT_DIR / f"qpu_benchmark_{ts}.json"` |
| 5 | `@todo/qpu_benchmark_results/qpu_benchmark_20251201_200012.json` | `@todo/qpu_benchmark.py` | Same as above (different run) |
| 6 | `@todo/qpu_benchmark_results/qpu_benchmark_20251203_121526.json` | `@todo/qpu_benchmark.py` | Same (HybridGrid 5,9 run) |
| 7 | `@todo/qpu_benchmark_results/qpu_benchmark_20251203_133144.json` | `@todo/qpu_benchmark.py` | Same (HybridGrid 10,9 run) |
| 8 | `Benchmarks/COMPREHENSIVE/comprehensive_benchmark_configs_dwave_20251130_212742.json` | `Benchmark Scripts/comprehensive_benchmark.py` (L1204–1227) | `f"comprehensive_benchmark_{config_str}{dwave_suffix}_{timestamp}.json"` |
| 9 | `Benchmarks/decomposition_scaling/decomposition_scaling_results.json` | `Benchmarks/decomposition_scaling/benchmark_decomposition_scaling.py` (L341) | `out_dir / "decomposition_scaling_results.json"` |
| 10 | `Benchmarks/decomposition_scaling/solver_comparison_results.json` | `Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py` (L1067) | `out_dir / "solver_comparison_results.json"` |
| 11 | `Inputs/Combined_Food_Data.xlsx` | **Raw input data** (not generated by any script) | Source dataset from GAIN |

---

## 4. Hardcoded Data in Plot Scripts

`paper_plots/generate_study_plots_v3.py` also contains **hardcoded data** extracted from
the QPU benchmark JSON files (lines 89–137). This includes decomposition timing data
(`DECOMP_DATA`) and Gurobi timing data (`GUROBI_DATA`) originally sourced from:
- `qpu_benchmark_20251201_160444.json`
- `qpu_benchmark_20251201_200012.json`

These are noted as "REAL DATA from QPU Benchmarks" in comments at line 89.

---

## 5. Complete File Inventory

### Data files (11 unique)
```
./qpu_hier_repaired.json
./gurobi_baseline_60s.json
./@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json
./@todo/qpu_benchmark_results/qpu_benchmark_20251201_160444.json
./@todo/qpu_benchmark_results/qpu_benchmark_20251201_200012.json
./@todo/qpu_benchmark_results/qpu_benchmark_20251203_121526.json
./@todo/qpu_benchmark_results/qpu_benchmark_20251203_133144.json
./Benchmarks/COMPREHENSIVE/comprehensive_benchmark_configs_dwave_20251130_212742.json
./Benchmarks/decomposition_scaling/decomposition_scaling_results.json
./Benchmarks/decomposition_scaling/solver_comparison_results.json
./Inputs/Combined_Food_Data.xlsx
```

### Data generator scripts (7 unique)
```
./unified_benchmark.py                                           → files 1, 2
./@todo/test_gurobi_timeout.py                                   → file 3
./@todo/test_gurobi_timeout_WORKING.py                           → file 3 (alt)
./@todo/qpu_benchmark.py                                         → files 4, 5, 6, 7
./Benchmark Scripts/comprehensive_benchmark.py                   → file 8
./Benchmarks/decomposition_scaling/benchmark_decomposition_scaling.py → file 9
./Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py    → file 10
```

### Plotting scripts (5 unique)
```
./generate_all_report_plots.py          → content_report.tex Figs 2–8 + generate_single_plot_pdfs.py data source
./generate_single_plot_pdfs.py          → Individual PDF breakdowns of Figs 2–8
./paper_plots/generate_study_plots_v3.py → content_report.tex Fig 1
./crop_benefit_weight_analysis.py       → content_report.tex Appendix Figs 9–18
./Benchmarks/decomposition_scaling/generate_plots.py → decomposition_benchmark_report.tex all figures
```

---

## 6. Last Regeneration Status (2026-03-27)

### Completed plot regenerations
| Script | Status | Output |
|--------|--------|--------|
| `generate_all_report_plots.py` | ✅ 9/9 plots | `phase3_results_plots/*.png/pdf` |
| `paper_plots/generate_study_plots_v3.py` | ✅ 3/3 plots | `paper_plots/study1_*.png/pdf`, `study2_*.png/pdf`, `study3_*.png/pdf` |
| `crop_benefit_weight_analysis.py` | ✅ 6 plots + CSV | `crop_weight_analysis/01–06*.png/pdf` |
| `Benchmarks/decomposition_scaling/generate_plots.py` | ✅ 14 PDFs | `Benchmarks/decomposition_scaling/fig_*.pdf` |
| `generate_single_plot_pdfs.py` | ✅ 54/54 plots | `phase3_single_plots/*.pdf` |

### Data regeneration runs in progress (background, started ~19:25–19:27 local)
| Script | Invocation | Output File | Status |
|--------|-----------|-------------|--------|
| `unified_benchmark.py` | `--mode gurobi-true-ground-truth --timeout 3600` | `gurobi_baseline_60s.json` | ⏳ Running |
| `@todo/test_gurobi_timeout.py` | (no args) | `@todo/gurobi_timeout_verification/gurobi_timeout_test_<ts>.json` | ⏳ Running |
| `Benchmarks/decomposition_scaling/benchmark_decomposition_scaling.py` | (no args) | `Benchmarks/decomposition_scaling/decomposition_scaling_results.json` | ✅ Done (19:29) |
| `Benchmarks/decomposition_scaling/benchmark_solvers_comparison.py` | `--timeout 3600` | `Benchmarks/decomposition_scaling/solver_comparison_results.json` | ⏳ Running |

After the in-progress runs complete, re-run the affected plot scripts:
1. `python generate_all_report_plots.py` — picks up new `gurobi_baseline_60s.json` and `gurobi_timeout_test_*.json`
2. `python generate_single_plot_pdfs.py` — same data
3. `python Benchmarks/decomposition_scaling/generate_plots.py` — picks up new `solver_comparison_results.json`
