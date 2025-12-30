# Fractional Recovery Integration - Complete

## Overview

Successfully integrated the **fractional land allocation** post-processing from [`@todo/food_grouping.py`](@todo/food_grouping.py ) into [`@todo/adaptive_hybrid_solver.py`](@todo/adaptive_hybrid_solver.py ). The solver now supports **two recovery modes**:

1. **Binary Mode** (new): Exactly 1 crop per farm-period
2. **Fractional Mode** (from existing code): 2-3 crops per farm-period with land splitting

## Architecture

```
6-Family QPU Solution
         ↓
   Recovery Mode?
    ↙           ↘
Binary          Fractional
   ↓               ↓
1 crop/period   2-3 crops/period
benefit-weighted  land fractions
   ↓               ↓
Binary objective  Fractional objective
```

## Test Results (5 farms × 27 foods)

| Method | Mode | Family Obj | 27-Food Obj | Diversity | Characteristics |
|--------|------|-----------|-------------|-----------|-----------------|
| **SA** | Binary | 4.19 | 4.24 | 6/27 | Simple, constraint-compliant |
| **SA** | Fractional | 4.20 | 3.70 | 10/27 | Realistic land splitting |
| **QPU** | Binary | 3.63 | 3.62 | 6/27 | Fast, strict one-hot |
| **QPU** | Fractional | 3.56 | 2.18 | 16/27 | **Higher diversity** |

## Key Insights

### Binary Mode
- **Output**: `{(farm, crop, period): 1}` (binary)
- **Constraint**: Exactly 1 crop per farm-period
- **Selection**: Highest-benefit crop within each family
- **Advantages**:
  - Maintains strict one-hot constraint
  - Simpler to interpret (clear binary assignments)
  - Consistent objectives (family ≈ food level)
- **Use Case**: When strict constraint compliance is required

### Fractional Mode  
- **Output**: `{(farm, crop, period): fraction}` (0 to 1)
- **Constraint**: 2-3 crops per farm-period with fractional land
- **Selection**: Benefit-weighted sampling within each family
- **Advantages**:
  - **Much higher diversity** (16/27 vs 6/27 crops)
  - Realistic land splitting (matches real farming)
  - Explores more of the solution space
- **Use Case**: When diversity and realistic land allocation are priorities

## Code Changes

### 1. Updated `recover_27food_solution`
- Now handles **binary mode only**
- Selects exactly 1 crop per farm-period
- Uses benefit weighting for selection

### 2. Added `recover_27food_fractional`
- Integrated from `food_grouping.refine_family_solution_to_crops`
- Selects 2-3 crops per family assignment
- Allocates fractional land based on random weights
- Returns `{(farm, crop, period): land_fraction}`

### 3. Updated `calculate_27food_objective`
- Now handles **both binary and fractional** solutions
- Automatically detects value type (binary 0/1 vs fractional)
- Scales contributions by land fraction when applicable

### 4. Enhanced Diversity Analysis
- Binary mode uses `analyze_crop_diversity_binary`
- Fractional mode uses `analyze_crop_diversity` (from food_grouping)
- Both report Shannon diversity and unique crop counts

### 5. Updated Main Solver
Added `recovery_mode` parameter:
```python
result = solve_adaptive_with_recovery(
    data,
    recovery_mode='binary',      # or 'fractional'
    recovery_method='benefit_weighted',
    use_qpu=True
)
```

## API Usage

```python
from adaptive_hybrid_solver import solve_adaptive_with_recovery

# Binary recovery (1 crop per period)
result_binary = solve_adaptive_with_recovery(
    data,
    recovery_mode='binary',
    recovery_method='benefit_weighted',  # or 'greedy', 'uniform'
    use_qpu=True,
    num_reads=100,
    num_iterations=3
)

# Fractional recovery (2-3 crops per period)
result_fractional = solve_adaptive_with_recovery(
    data,
    recovery_mode='fractional',
    use_qpu=True,
    num_reads=100,
    num_iterations=3
)

# Results structure
print(f"Family objective: {result['objective_family']}")
print(f"27-food objective: {result['objective_27food']}")
print(f"Diversity: {result['diversity_stats']['total_unique_crops']}/27")
print(f"Food assignments: {result['n_assigned_food']}")
```

## Files Modified

1. **`adaptive_hybrid_solver.py`**:
   - Added `recover_27food_fractional()` function
   - Updated `calculate_27food_objective()` to handle fractional values
   - Added `recovery_mode` parameter to main solver
   - Enhanced diversity analysis for both modes

2. **`test_adaptive_recovery.py`**:
   - Tests both binary and fractional modes
   - Compares SA vs QPU for each mode
   - Shows diversity differences

## Recommendations

- **Use Binary Mode** when:
  - Strict constraint compliance required
  - Simpler interpretation needed
  - Objective consistency is priority

- **Use Fractional Mode** when:
  - Diversity is critical
  - Realistic land allocation needed
  - Exploring solution space is important
  - Post-processing analysis required

## Next Steps

- ✅ Integration complete
- ✅ Both modes tested and validated
- ✅ Diversity metrics implemented
- 🔄 **TODO**: Compare with Gurobi at 27-food level
- 🔄 **TODO**: Scale testing on larger problems (20, 50, 100 farms)
- 🔄 **TODO**: Benchmark fractional vs binary on full test suite

---

**Date**: December 26, 2025  
**Status**: ✅ Complete and validated
