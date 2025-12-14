# Hardness Analysis - Complete Metrics Table

**Date**: December 14, 2025
**Config**: Constant 100 ha, 6 families, 3 periods, Gurobi timeout 300s

## Performance Metrics

| Farms | Vars | Ratio | Area | Solve(s) | Build(s) | Quads | Gap% | Status | Category |
|------:|-----:|------:|-----:|---------:|---------:|------:|-----:|--------|----------|
|   3 |   54 |  0.50 | 100.0 |     0.18 |     0.01 |   540 | 0.00 | OPTIMAL | FAST    |
|   5 |   90 |  0.83 | 100.0 |     0.36 |     0.03 |  1440 | 0.73 | OPTIMAL | FAST    |
|   7 |  126 |  1.17 | 100.0 |     0.73 |     0.04 |  2340 | 0.91 | OPTIMAL | FAST    |
|  10 |  180 |  1.67 | 100.0 |     0.78 |     0.05 |  3420 | 0.96 | OPTIMAL | FAST    |
|  12 |  216 |  2.00 | 100.0 |     2.02 |     0.06 |  4104 | 0.86 | OPTIMAL | FAST    |
|  15 |  270 |  2.50 | 100.0 |     5.30 |     0.08 |  5076 | 0.90 | OPTIMAL | FAST    |
|  18 |  324 |  3.00 | 100.0 |     3.44 |     0.09 |  5724 | 0.96 | OPTIMAL | FAST    |
|  20 |  360 |  3.33 | 100.0 |     6.88 |     0.10 |  6624 | 0.95 | OPTIMAL | FAST    |
|  22 |  396 |  3.67 | 100.0 |     8.53 |     0.11 |  7200 | 0.81 | OPTIMAL | FAST    |
|  25 |  450 |  4.17 | 100.0 |    10.59 |     0.12 |  8172 | 0.85 | OPTIMAL | MEDIUM  |
|  30 |  540 |  5.00 | 100.0 |    14.19 |     0.15 |  9828 | 0.93 | OPTIMAL | MEDIUM  |
|  35 |  630 |  5.83 | 100.0 |    48.22 |     0.17 | 11268 | 0.97 | OPTIMAL | MEDIUM  |
|  40 |  720 |  6.67 | 100.0 |    35.18 |     0.19 | 12816 | 0.91 | OPTIMAL | MEDIUM  |
|  50 |  900 |  8.33 | 100.0 |    99.87 |     0.25 | 15912 | 0.97 | OPTIMAL | MEDIUM  |
|  60 |  900 | 10.00 | 100.0 |   270.68 |     0.24 | 15804 | 0.98 | OPTIMAL | SLOW    |
|  70 |  900 | 11.67 | 100.0 |   154.61 |     0.24 | 15912 | 0.97 | OPTIMAL | SLOW    |
|  80 |  900 | 13.33 | 100.0 |   206.24 |     0.24 | 15912 | 0.90 | OPTIMAL | SLOW    |
|  90 |  900 | 15.00 | 100.0 |   220.11 |     0.24 | 16020 | 0.80 | OPTIMAL | SLOW    |
| 100 |  900 | 16.67 | 100.0 |   198.00 |     0.24 | 15912 | 0.96 | OPTIMAL | SLOW    |

## Summary by Category

### FAST (9 instances)

| Metric | Min | Max | Mean | Std |
|--------|----:|----:|-----:|----:|
| Farms           |   3.00 |  22.00 |  12.44 |   6.77 |
| Variables       |  54.00 | 396.00 | 224.00 | 121.79 |
| Farms/Food      |   0.50 |   3.67 |   2.07 |   1.13 |
| Solve Time (s)  |   0.18 |   8.53 |   3.14 |   3.10 |
| Quadratics      | 540.00 | 7200.00 | 4052.00 | 2314.60 |
| MIP Gap (%)     |   0.00 |   0.96 |   0.79 |   0.31 |

### MEDIUM (5 instances)

| Metric | Min | Max | Mean | Std |
|--------|----:|----:|-----:|----:|
| Farms           |  25.00 |  50.00 |  36.00 |   9.62 |
| Variables       | 450.00 | 900.00 | 648.00 | 173.12 |
| Farms/Food      |   4.17 |   8.33 |   6.00 |   1.60 |
| Solve Time (s)  |  10.59 |  99.87 |  41.61 |  36.03 |
| Quadratics      | 8172.00 | 15912.00 | 11599.20 | 2961.14 |
| MIP Gap (%)     |   0.85 |   0.97 |   0.92 |   0.05 |

### SLOW (5 instances)

| Metric | Min | Max | Mean | Std |
|--------|----:|----:|-----:|----:|
| Farms           |  60.00 | 100.00 |  80.00 |  15.81 |
| Variables       | 900.00 | 900.00 | 900.00 |   0.00 |
| Farms/Food      |  10.00 |  16.67 |  13.33 |   2.64 |
| Solve Time (s)  | 154.61 | 270.68 | 209.93 |  41.86 |
| Quadratics      | 15804.00 | 16020.00 | 15912.00 |  76.37 |
| MIP Gap (%)     |   0.80 |   0.98 |   0.92 |   0.07 |

## Correlations with Solve Time

| Metric | Correlation (r) | Strength |
|--------|----------------:|----------|
| Farms/Food Ratio          |   0.907 | Strong     |
| Number of Farms           |   0.907 | Strong     |
| Constraints               |   0.850 | Strong     |
| Number of Variables       |   0.850 | Strong     |
| Quadratic Terms           |   0.841 | Strong     |
| Build Time                |   0.840 | Strong     |

## QPU Target Recommendations

**Optimal range**: MEDIUM category (25-50 farms)

| Farms | Vars | Solve(s) | Quads | Gap% | Reason |
|------:|-----:|---------:|------:|-----:|--------|
|  25 |  450 |    10.59 |  8172 | 0.85 | Entry point - still solvable |
|  30 |  540 |    14.19 |  9828 | 0.93 | Sweet spot - classical struggles |
|  35 |  630 |    48.22 | 11268 | 0.97 | Moderate difficulty |
|  40 |  720 |    35.18 | 12816 | 0.91 | Classical solver stressed |
|  50 |  900 |    99.87 | 15912 | 0.97 | Upper limit - near timeout |

## Key Findings

1. **Hardness increases with farm count**: Strong correlation (r=0.907)
2. **Quadratic terms drive complexity**: 540 (3 farms) → 15,912 (100 farms)
3. **Sweet spot identified**: 25-50 farms (10-100s solve time)
4. **Area normalization validated**: All within ±0.02% of 100 ha target
5. **MIP gaps consistent**: 0.7-1.0% across all sizes (Gurobi setting: 1%)

## Visualizations

- `comprehensive_hardness_scaling.png` - 6-panel overview
- `plot_solve_time_vs_ratio.png` - Hardness vs farms/food ratio
- `plot_solve_time_vs_farms.png` - Scaling with problem size
- `plot_gap_vs_ratio.png` - Solution quality analysis
- `plot_heatmap_hardness.png` - Distribution matrix
- `plot_combined_analysis.png` - 4-panel combined view