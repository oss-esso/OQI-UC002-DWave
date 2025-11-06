# Crop Rotation CQM Functions Summary

## Overview
The `solver_runner_ROTATION.py` file now contains **4 CQM creation functions**:
- 2 without rotation (baseline)
- 2 with 3-period rotation

## Functions

### 1. `create_cqm_farm(farms, foods, food_groups, config)`
**Type:** Continuous/Farm formulation (NO rotation)

**Variables:**
- `A_{f,c}` - Continuous area variables [0, farm_area]
- `Y_{f,c}` - Binary indicator variables

**Objective:**
- Linear: Weighted sum of crop values × area

**Constraints:**
- Land availability per farm
- Linking: A >= min_area × Y, A <= farm_area × Y
- Food group min/max (global)

**Use Case:** Traditional optimization with flexible area allocation per farm

---

### 2. `create_cqm_farm_rotation_3period(farms, foods, food_groups, config, gamma=None)`
**Type:** Continuous/Farm formulation WITH 3-period rotation

**Variables:**
- `A_{f,c,t}` - Time-indexed continuous area [0, farm_area] for periods t ∈ {1,2,3}
- `Y_{f,c,t}` - Time-indexed binary indicators

**Objective:**
- Linear: Weighted sum of crop values × area (all periods)
- Quadratic: gamma × R_{c,c'} × A_{f,c,t-1} × A_{f,c',t} (rotation synergy)

**Constraints:**
- Land availability per farm per period
- Linking constraints per period
- Food group min/max per period (global)

**Use Case:** Farm-level rotation planning with flexible area allocation

---

### 3. `create_cqm_plots(farms, foods, food_groups, config)`
**Type:** Binary/Plots formulation (NO rotation)

**Variables:**
- `Y_{p,c}` - Binary assignment (plot p to crop c)

**Objective:**
- Linear: Weighted sum of (plot_area × crop_value × Y)

**Constraints:**
- Each plot assigned to at most one crop
- Min/max plots per crop (converted from area constraints)
- Food group min/max (global)

**Use Case:** Discrete land units (even grid) without rotation

---

### 4. `create_cqm_plots_rotation_3period(farms, foods, food_groups, config, gamma=None)`
**Type:** Binary/Plots formulation WITH 3-period rotation

**Variables:**
- `Y_{p,c,t}` - Time-indexed binary assignment for periods t ∈ {1,2,3}

**Objective:**
- Linear: Weighted sum of (plot_area × crop_value × Y) (all periods)
- Quadratic: gamma × plot_area × R_{c,c'} × Y_{p,c,t-1} × Y_{p,c',t} (rotation synergy)

**Constraints:**
- Each plot assigned to at most one crop per period
- Min/max plots per crop per period
- Food group min/max per period (global)

**Use Case:** Plot-level rotation planning with discrete assignments

---

## Rotation Matrix

All rotation functions load the synergy matrix from:
```
rotation_data/rotation_crop_matrix.csv
```

The matrix `R_{c,c'}` provides rotation benefits/penalties for planting crop c' after crop c.

## Parameters

### Standard Parameters (all functions)
- `farms`: List of farm/plot names
- `foods`: Dictionary of food data (nutritional values, impacts, etc.)
- `food_groups`: Dictionary mapping groups to crops
- `config`: Configuration dict with:
  - `land_availability`: Area per farm/plot
  - `weights`: Objective weights
  - `minimum_planting_area`: Min area per crop
  - `maximum_planting_area`: Max area per crop (plots only)
  - `food_group_constraints`: Min/max foods per group

### Rotation-Specific Parameters
- `gamma`: Rotation synergy weight coefficient (default: 0.1)
  - Higher values prioritize rotation benefits
  - Lower values prioritize immediate crop values

## Return Values

### Non-rotation functions
- `cqm`: ConstrainedQuadraticModel
- `A, Y` (farm) or `Y` (plots): Variable dictionaries
- `constraint_metadata`: Constraint information

### Rotation functions
- `cqm`: ConstrainedQuadraticModel
- `(A, Y)` (farm) or `Y` (plots): Time-indexed variable dictionaries
- `constraint_metadata`: Per-period constraint information

## Usage Example

```python
# Load rotation matrix first
rotation_matrix = load_rotation_matrix()

# Farm rotation with gamma=0.2
cqm, (A, Y), metadata = create_cqm_farm_rotation_3period(
    farms, foods, food_groups, config, gamma=0.2
)

# Plots rotation with default gamma
cqm, Y, metadata = create_cqm_plots_rotation_3period(
    plots, foods, food_groups, config
)
```

## File Updates

**Updated files:**
- `solver_runner_ROTATION.py` - Added `create_cqm_farm_rotation_3period()`
- `solver_runner_ROTATION.py` - Renamed to `create_cqm_plots_rotation_3period()`
- `rotation_benchmark.py` - Updated function call

**Documentation:**
- This file: `ROTATION_FUNCTIONS_SUMMARY.md`
