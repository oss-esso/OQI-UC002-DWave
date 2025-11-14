# DWave Solve Time Analysis - Summary

## Overview

This analysis examines all DWave quantum annealer runs across both the Legacy and Benchmarks directories to:
1. Calculate total computational time required to reproduce all runs
2. Analyze solve time scaling with problem size
3. Provide estimation formulas for future runs

## Generated Files

### Script
- **`analyze_dwave_solve_times.py`** - Main analysis script that:
  - Scans all DWave result files in Legacy/COMPREHENSIVE and Benchmarks directories
  - Extracts solve times and problem sizes (number of variables)
  - Computes comprehensive statistics
  - Generates visualizations
  - Creates LaTeX document with results

### LaTeX Document & Visualizations
Located in `Latex/`:
- **`dwave_solve_time_analysis.tex`** - Comprehensive LaTeX report
- **`dwave_solve_time_analysis.pdf`** - Compiled PDF report
- **`solve_time_vs_variables.pdf`** - Scatter plot showing solve time vs problem size with polynomial fit
- **`time_per_variable_dist.pdf`** - Distribution of solve time per variable
- **`solve_time_dist.pdf`** - Overall distribution of solve times
- **`solve_time_by_size.pdf`** - Average solve time and efficiency by problem size
- **`scenario_comparison.pdf`** - Comparison across different scenarios

## Key Findings

### Total Runs: 45
- **Total Solve Time**: ~1,870 seconds (0.52 hours, 0.02 days)
- **Average Solve Time**: 41.6 seconds
- **Median Solve Time**: 5.3 seconds

### Problem Size Range
- **Min**: 135 variables
- **Max**: 29,592 variables
- **Average**: 3,088 variables

### Efficiency Metrics
- **Mean Time per Variable**: 0.025349 seconds/variable
- **Median Time per Variable**: 0.018717 seconds/variable

### Scenario Breakdown

#### Benchmarks (New Format)
1. **BQUBO** (Binary Quadratic Unconstrained Binary Optimization)
   - 22 runs
   - Total time: 1,725 seconds (0.48 hours)
   - Mean: 78.4 seconds
   - Mean variables: 4,991

2. **LQ** (Linear Quadratic)
   - 4 runs
   - Total time: 20 seconds
   - Mean: 5.0 seconds
   - Mean variables: 2,531

3. **NLN** (Nonlinear Normalized)
   - 4 runs
   - Total time: 56 seconds
   - Mean: 14.1 seconds
   - Mean variables: 2,531

4. **PATCH**
   - 4 runs
   - Total time: 18.7 seconds
   - Mean: 4.7 seconds
   - Mean variables: 371

#### Legacy (Comprehensive Benchmarks)
5. **Farm_DWave**
   - 4 runs
   - Total time: 21.2 seconds
   - Mean: 5.3 seconds
   - Mean variables: 506

6. **Patch_DWave**
   - 3 runs
   - Total time: 15.9 seconds
   - Mean: 5.3 seconds
   - Mean variables: 477

7. **Patch_DWaveBQM**
   - 4 runs
   - Total time: 13.5 seconds
   - Mean: 3.4 seconds
   - Mean variables: 996

## Time Estimation Formulas

For a problem with `n` variables:

### Linear Approximation
```
t_est = n × 0.025349 seconds
```

### Conservative Estimate (Mean + 1 Std Dev)
```
t_conservative = n × (0.025349 + std_dev) seconds
```

### Example Estimates

| Variables | Expected Time | Conservative Time |
|-----------|---------------|-------------------|
| 100       | 2.5 seconds   | Higher with variance |
| 500       | 12.7 seconds  | |
| 1,000     | 25.3 seconds  | |
| 5,000     | 126.7 seconds | |

## Usage

### Running the Analysis
```bash
python analyze_dwave_solve_times.py
```

### Compiling the LaTeX Document
```bash
cd Latex
pdflatex dwave_solve_time_analysis.tex
pdflatex dwave_solve_time_analysis.tex  # Run twice for references
```

### Viewing Results
- Open `Latex/dwave_solve_time_analysis.pdf` for the full report
- Individual plots are available as separate PDFs in the `Latex/` directory

## Conclusions

1. **Total computational cost to reproduce all DWave runs**: ~0.52 hours
2. **Efficiency**: The BQUBO formulation takes the most time (78.4s average) due to larger problem sizes
3. **Scalability**: Time per variable metric provides a useful estimator for planning future runs
4. **Variability**: Significant variance in solve times suggests problem-specific characteristics matter
5. **Recommendation**: Use conservative estimates for resource planning

## File Locations

```
/Users/edoardospigarolo/Documents/OQI-UC002-DWave/
├── analyze_dwave_solve_times.py          # Analysis script
├── DWAVE_SOLVE_TIME_ANALYSIS_SUMMARY.md  # This file
└── Latex/
    ├── dwave_solve_time_analysis.tex     # LaTeX source
    ├── dwave_solve_time_analysis.pdf     # Final report
    └── *.pdf                             # Individual plots
```

## Notes

- The script automatically handles both Legacy and Benchmarks directory structures
- Problem size (number of variables) is extracted from result files or estimated from n_farms
- All plots use high-resolution (300 DPI) for publication quality
- The analysis includes cross-references and proper citations in the LaTeX document
