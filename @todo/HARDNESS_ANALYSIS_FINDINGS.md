# Hardness Analysis Findings

**Date**: December 14, 2025  
**Analysis**: Comprehensive test of 19 farm counts with constant area sampling

## Executive Summary

Ran comprehensive hardness analysis with 19 test points (3-100 farms) using constant total area (~100 ha). **ALL problems solved optimally in < 0.01s**, which contradicts previous hardness investigation findings.

## Root Cause: Missing Quadratic Terms

**CRITICAL ISSUE IDENTIFIED**: The Gurobi model shows `0 quadratic terms` for all instances.

```
Model built: 54 vars, 24 constrs, 0 quadratic terms    ← WRONG!
Model built: 90 vars, 32 constrs, 0 quadratic terms    ← WRONG!
Model built: 450 vars, 112 constrs, 0 quadratic terms  ← WRONG!
```

### Expected vs Actual

Based on hardness investigation (`instant_hardness_analysis.py`), we should have:

| Farms | Expected Quadratics | Actual | Status |
|-------|-------------------|--------|---------|
| 5     | ~1400             | 0      | ❌ MISSING |
| 15    | ~4200             | 0      | ❌ MISSING |
| 25    | ~7000             | 0      | ❌ MISSING |

**Without quadratic terms, the problem is a simple linear integer program**, which Gurobi solves trivially.

## Problem in hardness_comprehensive_analysis.py

The `build_gurobi_model()` function adds rotation and spatial quadratic terms:

```python
# Rotation benefits (temporal quadratics)
for f in farm_names:
    area = land_availability[f]
    for t in range(n_periods - 1):
        for c1 in food_names:
            for c2 in food_names:
                rotation_benefit = rotation_matrix.get(c1, {}).get(c2, 0)
                if rotation_benefit != 0:
                    obj += rotation_benefit * area * x[f, c1, t] * x[f, c2, t+1]

# Spatial synergy (spatial quadratics)
for f1 in farm_names:
    if f1 not in spatial_neighbors:
        continue
    neighbors = spatial_neighbors[f1]
    for f2 in neighbors:
        ...
        obj += synergy * np.sqrt(area1 * area2) * x[f1, c1, t] * x[f2, c2, t]
```

But `rotation_matrix` and `spatial_neighbors` appear to be **empty or not loaded properly**.

##Diagnosis

Looking at the CSV results, the `rotation_benefit_matrix` and `spatial_neighbors` are in the config but may not have meaningful values. From scenarios.py:

```python
'rotation_gamma': 0.35,
'spatial_k_neighbors': 4,
'frustration_ratio': 0.88,
'negative_synergy_strength': -1.5,
```

These parameters exist, but the actual matrices might not be constructed.

## Comparison with Previous Findings

From `HARDNESS_INVESTIGATION_RESULTS.md` and `investigate_hardness.py`:

| Instance | Farms | Status | solve_time |
|----------|-------|--------|------------|
| Previous | 5     | TIMEOUT | > 100s |
| Previous | 15    | TIMEOUT | > 100s |
| Previous | 20    | TIMEOUT | > 100s |
| Previous | 25    | TIMEOUT | > 100s |
| **Current** | **5**     | **OPTIMAL** | **0.00s** ❌ |
| **Current** | **15**    | **OPTIMAL** | **0.00s** ❌ |
| **Current** | **20**    | **OPTIMAL** | **0.00s** ❌ |
| **Current** | **25**    | **OPTIMAL** | **0.00s** ❌ |

This confirms: **The current implementation is NOT testing the same problem**.

## Root Causes

### 1. Rotation Matrix Not Loaded

In `scenarios.py`, `_load_rotation_large_200_food_data()` defines:

```python
'rotation_gamma': 0.35,
'frustration_ratio': 0.88,
```

But doesn't explicitly construct `rotation_benefit_matrix`. This needs to be built from food compatibility data.

### 2. Spatial Neighbors Not Built

`spatial_k_neighbors`: 4` is set, but `spatial_neighbors` dictionary may not be constructed. Need to:
- Calculate farm coordinates (or use random placement)
- Find k-nearest neighbors for each farm
- Build neighbor dict

### 3. Missing Matrix Construction

Previous scripts (like `investigate_hardness.py`) likely built these matrices explicitly before calling Gurobi. Current script assumes they're in the config.

## Action Required

To fix `hardness_comprehensive_analysis.py`:

1. **Build Rotation Matrix**: Either load from config or construct compatibility matrix between food families
2. **Build Spatial Neighbors**: Generate farm coordinates and find k-nearest neighbors
3. **Verify Quadratics**: After model building, check `model.NumQNZs` (quadratic non-zeros)
4. **Validate Problem**: Ensure 5-farm problem takes > 1s (not < 0.01s)

## Validated Results (Current Run)

All results show constant-area normalization worked perfectly:

| Farms | Total Area | CV    | Farms/Food | Variables | Solve Time | Status |
|-------|-----------|-------|------------|-----------|------------|---------|
| 3     | 100.00    | 0.770 | 0.50       | 54        | 0.00s      | OPTIMAL |
| 5     | 100.00    | 0.644 | 0.83       | 90        | 0.00s      | OPTIMAL |
| 7     | 100.00    | 0.908 | 1.17       | 126       | 0.00s      | OPTIMAL |
| 10    | 100.00    | 1.358 | 1.67       | 180       | 0.00s      | OPTIMAL |
| 15    | 100.00    | 1.457 | 2.50       | 270       | 0.00s      | OPTIMAL |
| 20    | 100.00    | 1.536 | 3.33       | 360       | 0.00s      | OPTIMAL |
| 25    | 100.00    | 1.499 | 4.17       | 450       | 0.00s      | OPTIMAL |
| 50    | 99.98     | 1.202 | 8.33       | 900       | 0.01s      | OPTIMAL |
| 100   | 99.98     | 1.202 | 16.67      | 900       | 0.01s      | OPTIMAL |

✅ **Constant area sampling works perfectly**  
❌ **Quadratic terms missing → wrong problem being solved**

## Visualizations Generated

Despite wrong problem formulation, visualizations were created:

1. [plot_solve_time_vs_ratio.png](hardness_analysis_results/plot_solve_time_vs_ratio.png)
2. [plot_solve_time_vs_farms.png](hardness_analysis_results/plot_solve_time_vs_farms.png)
3. [plot_gap_vs_ratio.png](hardness_analysis_results/plot_gap_vs_ratio.png)
4. [plot_heatmap_hardness.png](hardness_analysis_results/plot_heatmap_hardness.png)
5. [plot_combined_analysis.png](hardness_analysis_results/plot_combined_analysis.png)

All show flat solve times (all FAST) because problem is too simple without quadratics.

## Recommendation

1. **Reference Previous Working Code**: Check `investigate_hardness.py` or `test_statistical_gurobi_only.py` for proper matrix construction
2. **Add Matrix Building Functions**: Create helper functions to build rotation and spatial matrices
3. **Validate Against Known Hard Instance**: Test 5-farm case and confirm it takes > 10s
4. **Re-run Analysis**: Once quadratics are added, re-run all 19 test points
5. **Compare with Previous Results**: Validate against hardness investigation findings

## Files for Reference

- Previous hardness analysis: `@todo/HARDNESS_INVESTIGATION_RESULTS.md`
- Previous script (with quadratics): `@todo/investigate_hardness.py`
- Current script (missing quadratics): `@todo/hardness_comprehensive_analysis.py`
- Results (wrong problem): `@todo/hardness_analysis_results/hardness_analysis_results.csv`

---

**STATUS**: Analysis infrastructure ready, but problem formulation needs quadratic terms added before results are valid.
