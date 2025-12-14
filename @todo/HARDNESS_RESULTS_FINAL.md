# Hardness Analysis Results - FINAL

**Date**: December 14, 2025  
**Status**: ✅ COMPLETE with proper quadratic terms

## Key Findings

### Hardness Distribution (19 test points, constant 100 ha)

| Category | Farms | Farms/Food | Solve Time | Instances |
|----------|-------|------------|------------|-----------|
| **FAST** | 3-22 | 0.50-3.67 | 0.18-8.53s | 9 |
| **MEDIUM** | 25-50 | 4.17-8.33 | 10.59-99.87s | 5 |
| **SLOW** | 60-100 | 10.00-16.67 | 154-270s | 5 |

### Critical Insight: OPPOSITE Pattern!

**Previous hypothesis was INVERTED:**
- LOW farms/food ratio (< 4.2) = EASY (< 10s)
- HIGH farms/food ratio (> 10) = HARD (> 150s)

**Reason**: More farms = more quadratic terms = harder problem
- 5 farms: 1,440 quadratic non-zeros → 0.36s
- 60 farms: 15,912 quadratic non-zeros → 270.68s
- 100 farms: 15,912 quadratic non-zeros → 198.00s

### QPU Target Zone

**For Quantum Advantage, target MEDIUM range:**
- 25-50 farms (450-900 variables)
- 4.17-8.33 farms/food ratio
- 10-100 second solve time on Gurobi
- 7,000-15,000 quadratic terms
- Embeddable with decomposition

**Specific sweet spots:**
- 25 farms: 450 vars, 10.6s, 7000 quads ✅
- 30 farms: 540 vars, 14.2s, 8640 quads ✅
- 40 farms: 720 vars, 35.2s, 11520 quads ✅
- 50 farms: 900 vars, 99.9s, 14400 quads ✅

## Model Validation

✅ Quadratic terms present (540-15,912 non-zeros)  
✅ Constant area normalization (100 ha ± 5%)  
✅ Binary variables (MIP formulation)  
✅ Rotation frustration matrix (88% negative synergies)  
✅ Spatial k-nearest neighbors (k=4)

## Files Generated

- Results: `hardness_analysis_results/hardness_analysis_results.csv`
- Plots: `hardness_analysis_results/plot_*.png` (5 visualizations)
- Raw data: `hardness_analysis_results/hardness_analysis_results.json`

## Recommendation

**Use 25-50 farm instances for QPU benchmarking:**
- Hard enough to show classical struggle (10-100s)
- Small enough for QPU embedding (< 900 variables)
- In the quantum advantage window
