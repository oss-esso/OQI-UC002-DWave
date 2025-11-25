# Fix: Unique Food Counting in Food Group Constraints

## Problem

The food group constraint `min_foods` and `max_foods` was incorrectly counting **total (farm, food) selections** instead of **unique foods** selected across all farms.

### Example
- With constraint "Animal-source foods: min_foods=2, max_foods=5"
- **OLD (WRONG)**: If Beef planted on Farm1 and Farm2, count = 2 (same food counted twice!)
- **NEW (CORRECT)**: If Beef planted on Farm1 and Farm2, count = 1 (unique food)

## Solution

Added auxiliary binary variables `U[food]` that equal 1 if a food is selected on ANY farm:
- Linking constraint: `Y[farm, food] <= U[food]` for all farms
- Bound constraint: `U[food] <= sum(Y[farm, food] for all farms)`

Then food group constraints use `U` instead of `Y`:
```python
# CORRECT: Count unique foods
sum(U[food] for food in foods_in_group) >= min_foods

# WRONG: Counts total selections
sum(Y[(farm, food)] for farm in farms for food in foods_in_group) >= min_foods
```

## Files Fixed

### Benchmark Scripts/
1. `solver_runner.py` - Base solver (CQM and PuLP)
2. `solver_runner_LQ.py` - CQM, PuLP, and validation
3. `solver_runner_BINARY.py` - All CQM functions, solve_with_pulp_farm, solve_with_pulp_plots
4. `solver_runner_BQUBO.py` - PuLP function
5. `solver_runner_ROTATION.py` - All CQM functions (including 3-period rotation), all PuLP functions

### @todo/
6. `solver_runner_DECOMPOSED.py` - create_cqm_farm, create_cqm_plots, solve_with_pulp_farm
7. `solver_runner_CUSTOM_HYBRID.py` - create_cqm_farm, create_cqm_plots, solve_with_pulp_farm, solve_with_pulp_plots
8. `result_formatter.py` - Validation logic (already correct - counts unique foods)

## Validation Logic

The validation in these files was **already correct**:
- `comprehensive_benchmark.py` - Uses set() to count unique crops
- `solver_runner_BINARY.py` - Uses `any()` to detect if crop selected anywhere
- `solver_runner_DECOMPOSED.py` - Uses `any()` pattern
- `solver_runner_CUSTOM_HYBRID.py` - Uses `any()` pattern

Fixed validation in:
- `solver_runner_LQ.py` - Changed to use `any()` pattern for unique counting

## Remaining Files with Old Pattern

The following files in `@todo/` still have the old pattern (decomposition strategies):
- `decomposition_benders.py`
- `decomposition_benders_qpu.py`
- `decomposition_admm.py`
- `decomposition_admm_qpu.py`
- `decomposition_dantzig_wolfe.py`
- `decomposition_dantzig_wolfe_qpu.py`

These are used by `benchmark_all_strategies.py` which now validates solutions correctly using the fixed `result_formatter.py`.

## Testing

```bash
# Run benchmark to verify
cd @todo && python benchmark_all_strategies.py --config 10 --strategies benders,admm
```

Expected output:
```
✅ All strategies produced valid solutions
   - No constraint violations
   - No area overflow
```

## Date

Fixed: 2025-11-25
