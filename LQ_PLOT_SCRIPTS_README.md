# LQ Plot Scripts Documentation

## Overview
Three plotting scripts have been created for visualizing LQ (Linear + Quadratic) benchmark results with **special emphasis on solution quality**:

1. **`plot_lq_speedup.py`** - Basic speedup analysis without curve fitting
2. **`plot_lq_fitted_speedup.py`** - Advanced speedup analysis with curve fitting and crossover detection
3. **`plot_lq_quality_speedup.py`** - **NEW: Comprehensive quality analysis with Time-to-Quality metrics**

## ⚠️ CRITICAL FINDING: D-Wave Solution Quality Issue

**D-Wave shows significant quality gaps in LQ formulation:**
- At 5 farms: 0.01% gap (negligible)
- At 19 farms: 2.38% gap (minor)
- At 72 farms: **15.43% gap** (concerning)
- At 279 farms: **32.18% gap** (severe)

**This means D-Wave's speed advantage is misleading if solution quality matters!**

## Scripts Description

### 1. plot_lq_speedup.py (Without Fits)

**Purpose**: Creates comprehensive visualizations of solve times and speedup comparisons across different scales.

**Features**:
- Loads LQ benchmark data from `Benchmarks/LQ/` directory
- Compares 5 solvers: PuLP, Pyomo, CQM, DWave QPU, and DWave Hybrid
- Generates 6 subplots in a 2×3 grid:
  - **Row 1**: Solve times in linear, log-y, and log-log scales
  - **Row 2**: Speedup factors in linear, log-y, and log-log scales
- Prints summary table with solve times and speedup factors

**Output Files**:
- `Plots/lq_speedup_comparison.png` - Full 2×3 grid comparison
- `Plots/lq_solve_times_linear.png` - High-resolution linear scale plot

**Usage**:
```bash
python plot_lq_speedup.py
```

**⚠️ Quality Warning**: Now includes prominent warning about D-Wave's quality gaps!

**Key Observations from Results**:
```
N_Farms    PuLP (s)     Pyomo (s)    CQM (s)      QPU (s)      Hybrid (s)   QPU vs PuLP     Hybrid vs PuLP
5          5.8067       17.1034      0.5347       0.0695       5.0000       83.50x          1.16x
19         2.8422       23.9893      1.3228       0.0695       5.0000       40.87x          0.57x
72         25.8396      58.8691      44.3998      0.0695       5.0000       371.55x         5.17x
279        35.1453      71.1555      1274.6073    0.0348       5.0000       1010.85x        7.03x
```

### 2. plot_lq_fitted_speedup.py (With Curve Fitting)

**Purpose**: Advanced analysis using curve fitting to predict scaling behavior and identify crossover points.

**Features**:
- Fits interpolating functions to each solver's data
- Uses power law fitting: `f(x) = a * x^b + c`
- QPU uses linear fit (nearly constant time)
- Identifies crossover points where D-Wave becomes advantageous
- Extrapolates trends beyond measured data points
- Calculates speedup ratios from fitted curves

**Fit Parameters Found**:
```
PuLP           : f(x) = 9.040273 * x^0.297491 + -11.801206
Pyomo          : f(x) = 148.384517 * x^0.075217 + -153.215432
CQM            : f(x) = 0.001094 * x^2.480511 + 0.100115
DWave_QPU      : linear fit (nearly constant)
DWave_Hybrid   : f(x) = -0.000000 * x^1.000000 + 5.000000 (constant ~5s)
```

**Crossover Analysis**:
- ✓ **D-Wave Hybrid becomes faster than PuLP at: 8.0 farms**
- ✗ No crossover found with Pyomo in the analyzed range (Pyomo starts slower)

**Output File**:
- `Plots/lq_fitted_speedup_analysis.png` - 2×3 grid with fitted curves and speedup analysis

**Usage**:
```bash
python plot_lq_fitted_speedup.py
```

**⚠️ Quality Warning**: Now includes critical warning about quality gaps vs speed tradeoffs!

### 3. plot_lq_quality_speedup.py (With Quality Analysis) **⭐ NEW & RECOMMENDED**

**Purpose**: **Comprehensive analysis that exposes the quality vs speed tradeoff**. This is the MOST IMPORTANT script for LQ benchmarks!

**Features**:
- Analyzes BOTH solve time AND solution quality
- Calculates objective value gaps (% deviation from best solution)
- Introduces **"Time-to-Quality" (TTQ)** metric: `TTQ = Time × (1 + Gap/100)`
- This penalizes fast but inaccurate solutions
- Creates 3×3 grid showing:
  - **Row 1**: Solve times (linear, log, QPU focus)
  - **Row 2**: Solution quality (objectives, gaps, gap comparison)
  - **Row 3**: Time-to-Quality (linear, log, quality-adjusted speedup)
- Prints comprehensive table with times AND quality metrics

**The TTQ Metric Explained**:
```
If D-Wave solves in 5s with 15% gap: TTQ = 5 × (1 + 15/100) = 5.75s
If PuLP solves in 25s with 0% gap: TTQ = 25 × (1 + 0/100) = 25s
Quality-adjusted speedup = 25 / 5.75 = 4.35x (much less than raw 5x!)
```

**Critical Findings**:
```
N_Farms    PuLP Obj     Pyomo Obj    DWave Obj    DWave Gap%    Quality Impact
5          2.56         2.57         2.57         0.01%         Negligible
19         34.23        34.49        33.66        2.38%         Minor
72         176.04       176.51       149.28       15.43%        ⚠️ Concerning
279        711.54       713.43       483.86       32.18%        🚨 SEVERE
```

**Output File**:
- `Plots/lq_comprehensive_quality_analysis.png` - Complete 3×3 grid analysis

**Usage**:
```bash
python plot_lq_quality_speedup.py
```

**Why This Script is Critical**:
- Reveals that D-Wave's "speedup" is misleading for LQ formulation
- Shows that quality-adjusted performance is much worse than raw speed suggests
- Essential for making informed solver choices in production
- Demonstrates the importance of validating solution quality, not just speed

## LQ Formulation Characteristics

The LQ (Linear + Quadratic) formulation includes:
- **Linear objective terms**: Standard optimization criteria
- **Quadratic synergy terms**: Food pairing synergies that benefit from QPU's natural quadratic processing
- More complex than pure linear (NLD) but less complex than full quadratic (NLN/BQUBO)

## Key Findings

### Scalability Insights:
1. **QPU Time**: Nearly constant (~0.035-0.070s) regardless of problem size
2. **Hybrid Time**: Constant overhead (~5s) that becomes negligible for larger problems
3. **Classical Solvers**: Show increasing solve times with problem size
4. **CQM Solver**: Varies significantly, can be faster or slower than classical

### Speedup Characteristics:
- **Small Problems (< 8 farms)**: Classical PuLP is faster
- **Medium Problems (8-100 farms)**: Crossover region, evaluate case-by-case
- **Large Problems (> 100 farms)**: D-Wave shows clear TIME advantage
  - At 279 farms: 1010x speedup (QPU vs PuLP), 7x speedup (Hybrid vs PuLP)
  - **BUT**: 32% quality gap makes this misleading!

### Solution Quality Characteristics (**CRITICAL**):
1. **Small Problems (< 10 farms)**: D-Wave quality is comparable (< 1% gap)
2. **Medium Problems (10-50 farms)**: Gaps start appearing (2-5%)
3. **Large Problems (> 50 farms)**: Severe quality degradation (15-32% gaps)
4. **Root Cause**: Likely hybrid solver time limits or approximation algorithms
5. **Impact**: Speed advantage is NEGATED by poor solution quality

### Revised Recommendations (Accounting for Quality):
- **< 8 farms**: Use PuLP (classical) - fast enough and optimal
- **8-50 farms**: Use PuLP/Pyomo - D-Wave quality gaps start to appear
- **50-100 farms**: Use classical solvers - quality gaps are too large (15%+)
- **> 100 farms**: **STILL use classical** - 32% quality gap is unacceptable
- **Quality-critical applications**: **ALWAYS use classical solvers** regardless of size
- **Future work**: Investigate D-Wave parameter tuning to improve quality
  - Increase time limit
  - Adjust annealing schedule
  - Use different decomposition strategies

## Color Scheme

Both scripts use consistent colors for easy comparison:
- **PuLP**: Red (#E63946)
- **Pyomo**: Orange (#F77F00)
- **CQM**: Yellow-Orange (#FFB703)
- **DWave QPU**: Cyan (#06FFA5)
- **DWave Hybrid**: Blue (#118AB2)

## Technical Notes

### Dependencies:
- `matplotlib` (with Agg backend to avoid Qt conflicts)
- `numpy`
- `seaborn`
- `scipy` (for curve fitting in fitted version)
- `json` and `pathlib` (for data loading)

### Data Structure Expected:
```
Benchmarks/LQ/
  ├── DWave/config_{n}_run_1.json
  ├── PuLP/config_{n}_run_1.json
  ├── Pyomo/config_{n}_run_1.json
  └── CQM/config_{n}_run_1.json
```

Where `n` ∈ {5, 19, 72, 279, ...}

### JSON Structure:
```json
{
  "metadata": {...},
  "result": {
    "solve_time": float,      // For PuLP/Pyomo
    "cqm_time": float,         // For CQM
    "qpu_time": float,         // For DWave
    "hybrid_time": float,      // For DWave
    ...
  }
}
```

## Comparison with Other Formulations

The LQ scripts follow the same structure as:
- `plot_bqubo_speedup.py` / `plot_fitted_speedup.py` (BQUBO formulation)
- `plot_nln_speedup.py` (NLN formulation)

This allows for easy cross-formulation comparison and consistent analysis across different problem types.

## Future Enhancements

Potential improvements:
1. Add confidence intervals to fitted curves
2. Include multiple runs for statistical analysis
3. Add comparison with theoretical complexity bounds
4. Generate combined plots across all formulations
5. Add interactive HTML plots for exploration

## References

- Based on benchmark data from `benchmark_scalability_LQ.py`
- Follows D-Wave best practices for hybrid solver usage
- Implements power law fitting for scaling analysis
- Uses matplotlib's Agg backend for cross-platform compatibility
