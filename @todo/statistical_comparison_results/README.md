# Statistical Comparison Test: Quantum vs Classical

This directory contains an isolated statistical comparison test for evaluating quantum advantage in multi-period crop rotation optimization.

## Overview

The test compares:
- **Classical**: Gurobi optimizer (ground truth with timeout)
- **Quantum (Clique Decomp)**: Farm-by-farm decomposition on D-Wave QPU
- **Quantum (Spatial-Temporal)**: Clustered farms + temporal slicing on D-Wave QPU

## Quick Start

```powershell
# Activate environment
conda activate oqi

# Run full test (default: 5, 10, 15, 20 farms × 2 runs each)
python statistical_comparison_test.py

# Quick test (smaller sizes, single run)
python statistical_comparison_test.py --sizes 5 10 --runs 1 --timeout 60

# Custom configuration
python statistical_comparison_test.py --sizes 5 10 15 20 25 --runs 3 --reads 500 --timeout 600
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sizes` | 5 10 15 20 | Farm sizes to test |
| `--runs` | 2 | Runs per method for statistical analysis |
| `--reads` | 100 | QPU reads per subproblem (Phase 2 setting) |
| `--timeout` | 300 | Classical solver timeout (seconds) |
| `--iterations` | 3 | Decomposition refinement iterations |
| `--token` | (default) | D-Wave API token |
| `--no-plots` | False | Skip plot generation |
| `--no-latex` | False | Skip LaTeX report generation |

## Outputs

All outputs are saved to `statistical_comparison_results/`:

### Data Files
- `statistical_comparison_YYYYMMDD_HHMMSS.json` - Complete results with statistics

### Plots (PNG + PDF)
- `plot_solution_quality.pdf` - Objective value comparison (bar chart with error bars)
- `plot_time_comparison.pdf` - Wall time comparison (log scale)
- `plot_gap_speedup.pdf` - Optimality gap and speedup factor
- `plot_scaling.pdf` - Scaling behavior analysis

### Report
- `quantum_classical_comparison_report.tex` - Complete LaTeX technical report

## Test Design

### Problem Formulation
- **Variables**: F farms × 6 crop families × 3 periods
- **Objective**: Maximize agricultural benefit + rotation synergies + diversity
- **Constraints**: At most 2 crops per farm per period

### Quantum Strategy
- Spatial decomposition: 2 farms per cluster
- Temporal decomposition: Solve periods sequentially
- Subproblem size: 12 variables (fits in clique embedding)
- Iterative refinement: 3 iterations with boundary coordination

### Metrics Collected
- Objective value (mean, std, min, max)
- Wall time (mean, std)
- QPU access time
- Constraint violations
- Optimality gap vs ground truth
- Speedup factor

## Building the LaTeX Report

```powershell
cd statistical_comparison_results
pdflatex quantum_classical_comparison_report.tex
```

## Interpreting Results

### Key Metrics

1. **Gap (%)**: `(GT_obj - QT_obj) / GT_obj × 100`
   - Target: <10% is acceptable, <5% is excellent
   - Decreasing gap with scale indicates quantum advantage

2. **Speedup**: `GT_time / QT_time`
   - Values >1 indicate quantum is faster
   - Target: >5× for practical advantage

3. **Violations**: Should be 0 for valid solutions

### Expected Results (Based on Phase 2)

| Farms | Variables | Expected Gap | Expected Speedup |
|-------|-----------|--------------|------------------|
| 5     | 90        | ~7-8%        | ~10-15×          |
| 10    | 180       | ~4-5%        | ~8-10×           |
| 15    | 270       | ~3-4%        | ~8-9×            |
| 20    | 360       | ~2-4%        | ~8-10×           |

## Notes

- Classical timeout affects gap calculation (not proven optimal)
- QPU access time is small fraction of total quantum time
- Most overhead is classical preprocessing and iteration
- Results are deterministic (seeded random number generation)
