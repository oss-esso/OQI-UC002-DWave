# Phase 3 Report: Quantum Advantage Analysis

**Project**: OQI-UC002-DWave  
**Date**: December 2025  
**Purpose**: Comprehensive statistical comparison of quantum vs classical optimization for multi-period crop rotation planning.

---

## 📁 Directory Structure

```
Phase3Report/
├── Data/                    # Benchmark results and processed data
│   └── significant/         # Key scenarios demonstrating quantum advantage
├── Docs/                    # LaTeX documentation
├── Plots/                   # Generated visualization plots
├── Scripts/                 # Python scripts for benchmarks and analysis
└── README.md                # This file
```

---

## 🔗 Script-Data-Plot Pairings Index

### 1. Statistical Comparison Test (5-25 Farms, 6 Families)

| Component | File | Description |
|-----------|------|-------------|
| **Script** | [Scripts/statistical_comparison_test.py](Scripts/statistical_comparison_test.py) | Main benchmark: Gurobi vs Clique Decomp vs Spatial-Temporal |
| **Data** | [Data/statistical_comparison_20251214_192625.json](Data/statistical_comparison_20251214_192625.json) | Full results with per-run details |
| **Plots** | [Plots/plot_solution_quality.png](Plots/plot_solution_quality.png) | Solution quality comparison |
| | [Plots/plot_time_comparison.png](Plots/plot_time_comparison.png) | Time comparison (log scale) |
| | [Plots/plot_gap_speedup.png](Plots/plot_gap_speedup.png) | Gap and speedup analysis |
| | [Plots/plot_scaling.png](Plots/plot_scaling.png) | Scaling behavior |
| | [Plots/plot_scaling_loglog.png](Plots/plot_scaling_loglog.png) | Log-log scaling |

**Formulation**: 6 crop families × 3 periods × F farms (F = 5, 10, 15, 20, 25)  
**Methods**: Ground Truth (Gurobi 300s timeout), Clique Decomposition (QPU), Spatial-Temporal (QPU)  
**Variables**: 90-450 binary decision variables

---

### 2. Hierarchical Statistical Test (25-100 Farms, 27→6 Aggregation)

| Component | File | Description |
|-----------|------|-------------|
| **Script** | [Scripts/hierarchical_statistical_test.py](Scripts/hierarchical_statistical_test.py) | Large-scale benchmark with hierarchical decomposition |
| **Data** | [Data/hierarchical_results_20251212_124349.json](Data/hierarchical_results_20251212_124349.json) | Results for 25, 50, 100 farms |
| | [Data/hierarchical_5_farms.json](Data/hierarchical_5_farms.json) ... [hierarchical_30_farms.json](Data/hierarchical_30_farms.json) | Individual farm-size results |
| **Plots** | *Embedded in combined_analysis* | See Combined Analysis below |

**Formulation**: 27 foods aggregated to 6 families × 3 periods × F farms (F = 25, 50, 100)  
**Methods**: Gurobi (300s timeout), Hierarchical QPU (spatial clustering + boundary coordination)  
**Variables**: 2025-8100 (original), 450-1800 (aggregated)

---

### 3. Combined Analysis (Unified View: 5-100 Farms)

| Component | File | Description |
|-----------|------|-------------|
| **Script** | [Scripts/combined_analysis.py](Scripts/combined_analysis.py) | Merges statistical + hierarchical results |
| **Data** | Uses outputs from #1 and #2 above | Combined analysis |
| **Plots** | [Plots/combined_analysis_plots.png](Plots/combined_analysis_plots.png) | Unified visualization |

**Purpose**: Shows quantum advantage progression from small (5 farms) to large (100 farms) problems.

---

### 4. Variable Scaling Analysis

| Component | File | Description |
|-----------|------|-------------|
| **Script** | [Scripts/variable_scaling_analysis.py](Scripts/variable_scaling_analysis.py) | Analysis by number of variables |
| **Data** | [Data/variable_scaling_data.csv](Data/variable_scaling_data.csv) | Processed scaling data |
| **Plots** | [Plots/variable_count_scaling_analysis.png](Plots/variable_count_scaling_analysis.png) | Gap/speedup vs variables |
| | [Plots/plot_gap_speedup_vs_vars.png](Plots/plot_gap_speedup_vs_vars.png) | Gap and speedup vs variables |
| | [Plots/plot_time_vs_vars.png](Plots/plot_time_vs_vars.png) | Time vs variables |
| | [Plots/plot_solution_quality_vs_vars.png](Plots/plot_solution_quality_vs_vars.png) | Quality vs variables |

**Purpose**: Identifies the formulation change boundary and missing data points.

---

### 5. Significant Scenarios Benchmark

| Component | File | Description |
|-----------|------|-------------|
| **Script** | [Scripts/significant_scenarios_benchmark.py](Scripts/significant_scenarios_benchmark.py) | Comprehensive test of key scenarios |
| **Data** | [Data/significant/benchmark_results_20251214_205508.json](Data/significant/benchmark_results_20251214_205508.json) | Primary results |
| | [Data/significant/benchmark_results_20251214_204213.json](Data/significant/benchmark_results_20251214_204213.json) | Additional run |
| | [Data/significant/scenario_definitions.json](Data/significant/scenario_definitions.json) | Scenario specifications |
| | [Data/significant/all_extracted_results.csv](Data/significant/all_extracted_results.csv) | Consolidated CSV |
| | [Data/significant/significant_scenarios_comparison.csv](Data/significant/significant_scenarios_comparison.csv) | Comparison summary |
| **Plots** | ⚠️ **MISSING** | *Plots not yet generated for this analysis* |

**Purpose**: Unified benchmark spanning 5-100 farms with consistent methodology.  
**Note**: This is the most comprehensive benchmark but lacks visualization.

---

### 6. QPU Benchmark (Advanced Methods)

| Component | File | Description |
|-----------|------|-------------|
| **Script** | [Scripts/qpu_benchmark.py](Scripts/qpu_benchmark.py) | Comprehensive QPU testing (5639 lines) |
| **Data** | Multiple `scaling_test_*.json` files | Various scaling experiments |
| | [Data/scaling_study_summary.json](Data/scaling_study_summary.json) | Summary of scaling tests |
| **Plots** | [Plots/comprehensive_scaling.png](Plots/comprehensive_scaling.png) | Comprehensive scaling plot |
| | [Plots/comprehensive_scaling.pdf](Data/comprehensive_scaling.pdf) | PDF version |

**Methods**: Direct QPU, Manual Decomposition (PlotBased, Multilevel, Louvain, Cutset, Spectral)  
**Purpose**: Explores multiple decomposition strategies for large-scale problems.

---

## 📊 Summary Data Files

| File | Description |
|------|-------------|
| [Data/combined_results.csv](Data/combined_results.csv) | Combined results from all benchmarks |
| [Data/summary_20251212_102334.csv](Data/summary_20251212_102334.csv) | Summary statistics |
| [Data/summary_20251212_124349.csv](Data/summary_20251212_124349.csv) | Updated summary |

---

## 📄 Documentation

| File | Description |
|------|-------------|
| [Docs/quantum_classical_comparison_report.tex](Docs/quantum_classical_comparison_report.tex) | Main technical report with results |
| [Docs/statistical_comparison_methodology.tex](Docs/statistical_comparison_methodology.tex) | Detailed problem formulation and methodology |
| [Docs/problem_formulations.tex](Docs/problem_formulations.tex) | Complete formulation specifications (NEW) |

---

## ⚠️ Missing Components

| Category | Missing Item | Status |
|----------|--------------|--------|
| **Plots** | Significant scenarios visualization | 🔴 Not generated |
| **Plots** | Unified comparison across all formulations | 🔴 Not generated |
| **Data** | 30-40 farms data points (gap in scaling) | 🟡 Data gap identified |

---

## 🎯 Problem Formulations Overview

### Formulation A: Native 6 Families
- **Used by**: `statistical_comparison_test.py`
- **Sizes**: 5-25 farms
- **Variables**: 90-450 (6 families × 3 periods × F farms)
- **QPU Methods**: Clique Decomposition, Spatial-Temporal

### Formulation B: 27 Foods → 6 Families (Aggregated)
- **Used by**: `hierarchical_statistical_test.py`, `significant_scenarios_benchmark.py` (for large sizes)
- **Sizes**: 25-100 farms
- **Variables**: 450-1800 (after aggregation)
- **QPU Methods**: Hierarchical decomposition with spatial clustering

### Key Observation
Both formulations share the **same objective function structure**:
1. Base agricultural benefit (linear)
2. Temporal rotation synergies (quadratic)
3. Spatial neighbor interactions (quadratic)
4. Diversity bonus (linear)
5. One-hot penalty (quadratic)

The only difference is whether crop families are native (6) or aggregated from foods (27→6).

---

## 📈 Key Results Summary

| Problem Size | Variables | Gurobi Time | QPU Time | Speedup | Gap |
|--------------|-----------|-------------|----------|---------|-----|
| 5 farms | 90 | 300s (timeout) | ~20s | ~15× | ~15% |
| 10 farms | 180 | 300s (timeout) | ~35s | ~8× | ~14% |
| 20 farms | 360 | 300s (timeout) | ~58s | ~5× | ~13% |
| 25 farms | 450/2025 | 300s (timeout) | ~40s | ~7× | ~8% |
| 50 farms | 900/4050 | 300s (timeout) | ~60s | ~5× | ~7% |
| 100 farms | 1800/8100 | 300s (timeout) | ~100s | ~3× | ~10% |

**Note**: Gurobi hits timeout for all sizes ≥5 farms with the full MIQP formulation. Gaps calculated against timeout solution, not proven optimal.

---

## 📚 Technical Documentation

### LaTeX Reports

1. **[problem_formulations.tex](Docs/problem_formulations.tex)**  
   Complete mathematical specification of optimization formulations with equations for all objective components

2. **[quantum_classical_comparison_report.tex](Docs/quantum_classical_comparison_report.tex)**  
   Statistical comparison methodology and experimental design

3. **[statistical_comparison_methodology.tex](Docs/statistical_comparison_methodology.tex)**  
   Statistical analysis methods and metrics

4. **[benchmark_scenario_analysis.tex](Docs/benchmark_scenario_analysis.tex)** ⭐ **NEW**  
   Comprehensive 25-page analysis of all 4 benchmark configurations:
   - **Configuration A**: Statistical (5-25 farms, 6 families, 300s timeout)
   - **Configuration B**: Hierarchical (25-100 farms, 27→6, 300s timeout)  
   - **Configuration C**: QPU Benchmark (various sizes, 200s timeout)
   - **Configuration D**: Significant Scenarios (5-100 farms, 100s timeout)
   
   **Key Findings**:
   - Results are valid within configurations
   - Cross-configuration comparisons require careful interpretation (different timeouts, formulations, methods)
   - Provides guidelines for reading the all-runtime-traces plot
   - Recommendations for future benchmark standardization

**Compile**: `pdflatex benchmark_scenario_analysis.tex`

---

## 🚀 Quick Start

```bash
# Run statistical comparison test (5-25 farms)
python Scripts/statistical_comparison_test.py

# Run hierarchical test (25-100 farms)
python Scripts/hierarchical_statistical_test.py

# Run significant scenarios benchmark
python Scripts/significant_scenarios_benchmark.py

# Generate combined analysis
python Scripts/combined_analysis.py

# Variable scaling analysis
python Scripts/variable_scaling_analysis.py
```

---

## 📞 Contact

Project: OQI-UC002-DWave  
Date: December 2025
