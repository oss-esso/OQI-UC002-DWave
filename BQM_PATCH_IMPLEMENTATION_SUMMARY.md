# BQM_PATCH Implementation Summary

## Overview
The `solver_runner_PATCH.py` has been successfully adapted to implement the mathematical formulation described in `BQM_PATCH.txt`.

## Key Changes Made

### 1. Variable Structure
**Before:**
- Single variable type: `Y[(farm, food)]` representing binary plantations

**After (BQM_PATCH):**
- **X_{p,c}**: Binary variables for plot-crop assignments (|P| × |C| variables)
  - `X[(plot, crop)] = 1` if plot p is assigned to crop c
  - `X[(plot, crop)] = 0` otherwise
- **Y_c**: Binary variables for crop activation (|C| variables)
  - `Y[crop] = 1` if crop c is grown on at least one plot
  - `Y[crop] = 0` otherwise

### 2. Objective Function
**Mathematical formulation:**
```
max Z = Σ_{p,c} (B_c + λ) * s_p * X_{p,c}
```

Where:
- `B_c` = weighted benefit per unit area for crop c
- `s_p` = area of plot p (from `land_availability`)
- `λ` = idle penalty parameter (default 0.1)

**Implementation:**
```python
B_c = (
    w1 * nutritional_value +
    w2 * nutrient_density -
    w3 * environmental_impact +
    w4 * affordability +
    w5 * sustainability
)
objective = Σ (B_c + λ) * s_p * X_{p,c}
```

### 3. Constraints Implemented

#### Constraint 1: At Most One Crop Per Plot
```
∀p ∈ P: Σ_c X_{p,c} ≤ 1
```
- Allows plots to remain idle (all X_{p,c} = 0)
- Or assigns exactly one crop per plot

#### Constraint 2: X-Y Linking
```
∀p ∈ P, ∀c ∈ C: X_{p,c} ≤ Y_c
```
- If crop c is not selected (Y_c = 0), no plot can be assigned to it

#### Constraint 3: Y Activation
```
∀c ∈ C: Y_c ≤ Σ_p X_{p,c}
```
- If crop c is selected (Y_c = 1), at least one plot must be assigned to it

#### Constraint 4: Area Bounds Per Crop
```
∀c ∈ C: A_c^min ≤ Σ_p (s_p * X_{p,c}) ≤ A_c^max
```
- Minimum area: from `minimum_planting_area` parameter
- Maximum area: from `max_percentage_per_crop * total_land`

#### Constraint 5: Food Group Diversity
```
∀g ∈ G: FG_g^min ≤ Σ_{c in G_g} Y_c ≤ FG_g^max
```
- Ensures minimum and maximum number of crops from each food group

### 4. Functions Updated

#### `create_cqm(farms, foods, food_groups, config)`
- Creates X and Y variables
- Implements all 5 constraint types
- Returns: `(cqm, (X, Y), constraint_metadata)`

#### `solve_with_pulp(farms, foods, food_groups, config)`
- Mirrors CQM formulation exactly in PuLP
- Uses same variable structure and constraints
- Returns results with both X and Y variable values

#### `main(scenario)`
- Updated to handle new variable structure
- Properly extracts and reports X and Y variables
- Updates constraint metadata serialization

### 5. Constraint Metadata Structure
```python
constraint_metadata = {
    'at_most_one_per_plot': {plot: {...}},
    'x_y_linking': {(plot, crop): {...}},
    'y_activation': {crop: {...}},
    'area_bounds_min': {crop: {...}},
    'area_bounds_max': {crop: {...}},
    'food_group_min': {group: {...}},
    'food_group_max': {group: {...}}
}
```

### 6. Result Structure
```python
results = {
    'status': 'Optimal',
    'objective_value': <value>,
    'solve_time': <time>,
    'X_variables': {  # Plot-crop assignments
        'X_plot1_crop1': 1.0,
        'X_plot1_crop2': 0.0,
        ...
    },
    'Y_variables': {  # Crop selections
        'Y_crop1': 1.0,
        'Y_crop2': 0.0,
        ...
    }
}
```

## Advantages of BQM_PATCH Formulation

1. **Implicit Idle Representation**: Reduces binary variables by |P| compared to explicit idle variables
2. **Linear Objective**: All terms are linear, perfect for CQM formulation
3. **Flexible Plot Assignment**: Plots can remain idle without explicit variables
4. **Crop Selection Tracking**: Y variables enable food group diversity constraints
5. **Area-Weighted Benefits**: Accounts for different plot sizes naturally

## Parameter Configuration

The formulation expects the following in config:
```python
config = {
    'parameters': {
        'land_availability': {plot: area, ...},  # s_p values
        'minimum_planting_area': {crop: min_area, ...},  # A_c^min
        'max_percentage_per_crop': {crop: percentage, ...},  # for A_c^max
        'food_group_constraints': {
            group: {'min_foods': n, 'max_foods': m}, ...
        },
        'weights': {
            'nutritional_value': w1,
            'nutrient_density': w2,
            'environmental_impact': w3,
            'affordability': w4,
            'sustainability': w5
        },
        'idle_penalty_lambda': λ  # Default: 0.1
    }
}
```

## Verification

To verify the implementation:
1. Check variable count: |X| = |P| × |C|, |Y| = |C|
2. Check constraint count matches expected
3. Run solver: `python solver_runner_PATCH.py --scenario simple`
4. Verify results with: `python verifier.py <manifest_path>`

## Mathematical Equivalence

The implementation is mathematically equivalent to the BQM_PATCH.txt LaTeX document:
- All constraint types match exactly
- Objective function matches the simplified form
- Variable definitions are identical
- All derived quantities (A_c, A_used, A_idle) can be computed from solution
