# LQ Objective Normalization - Implementation Summary

## Overview
The LQ (Linear-Quadratic) solver objective has been updated to include normalized objectives (objective per unit area). This makes objectives comparable across different problem sizes.

## Changes Made

### 1. solver_runner_LQ.py

#### PuLP Solver (Line ~330):
- Added calculation of `total_area` from all allocated areas
- Added `normalized_objective = objective_value / total_area`
- Updated results dict to include:
  - `'normalized_objective'`: Objective per unit area
  - `'total_area'`: Total area allocated across all farms
- Added print statements showing total area and normalized objective

#### Pyomo Solver (Line ~545):
- Added calculation of `total_area` from all allocated areas  
- Added `normalized_objective = objective_value / total_area`
- Updated results dict to include:
  - `'normalized_objective'`: Objective per unit area
  - `'total_area'`: Total area allocated across all farms
- Added print statements showing total area and normalized objective

### 2. benchmark_scalability_LQ.py

#### DWave Result Extraction (Line ~332):
- Added extraction of `dwave_total_area` from solution sample
  - Sums all `Area_*` variables from the best feasible solution
- Added calculation of `dwave_normalized_objective`
- Updated DWave cache result to include:
  - `'normalized_objective'`
  - `'total_area'`
- Added print statements for area and normalized objective

#### PuLP Cache Saving (Line ~277):
- Updated `pulp_cache_result` dict to include:
  - `'normalized_objective'`: from `pulp_results`
  - `'total_area'`: from `pulp_results`

#### Pyomo Cache Saving (Line ~313):
- Updated `pyomo_cache_result` dict to include:
  - `'normalized_objective'`: from `pyomo_results`
  - `'total_area'`: from `pyomo_results`

#### Result Collection (Line ~410):
- Updated the main result dict to include for each solver:
  - `'pulp_normalized_objective'`
  - `'pulp_total_area'`
  - `'pyomo_normalized_objective'`
  - `'pyomo_total_area'`
  - `'dwave_normalized_objective'`
  - `'dwave_total_area'`

## Benefits

1. **Comparability**: Normalized objectives allow fair comparison across problem sizes
2. **Interpretability**: Value per unit area is more meaningful for decision makers
3. **Consistency**: All three solvers (PuLP, Pyomo, DWave) now report normalized objectives
4. **Backwards Compatibility**: Raw objectives are still saved alongside normalized ones

## Usage

After running benchmarks with the updated code:

```python
# Access raw objective
objective = result['result']['objective_value']

# Access normalized objective  
normalized = result['result']['normalized_objective']

# Access total area
total_area = result['result']['total_area']

# Verify: normalized should equal objective / total_area
```

## Verification

The `test_lq_normalization.py` script can be used to verify:
1. All three solvers calculate normalized objectives
2. The calculation is correct (normalized = objective / total_area)
3. All values are saved properly to cache

## Next Steps

To use normalized objectives in plots and analysis:
1. Update `plot_lq_speedup.py` to use `normalized_objective` instead of `objective_value`
2. Update `plot_all_scenarios_comparison.py` to extract and use normalized objectives
3. Update plot titles/labels to indicate "Normalized Objective (per unit area)"
4. Consider adding a comparison plot showing raw vs normalized objectives

## Files Modified

1. `solver_runner_LQ.py` - Added normalization calculation in solvers
2. `benchmark_scalability_LQ.py` - Added normalization to benchmark pipeline
3. `test_lq_normalization.py` - NEW: Test script to verify normalization
4. `analyze_lq_normalization.py` - NEW: Analysis script for existing results

## Formula

```
Normalized Objective = Raw Objective Value / Total Allocated Area

Where:
- Raw Objective = Linear area term + Quadratic synergy term
- Total Allocated Area = Sum of all A[farm, crop] decision variables
```

## Important Notes

1. The raw objective is still preserved for backwards compatibility
2. Normalized objective = 0 if total_area = 0 (degenerate case)
3. Both values are printed during solving for transparency
4. All three solvers (PuLP, Pyomo, DWave) now consistently report both metrics
