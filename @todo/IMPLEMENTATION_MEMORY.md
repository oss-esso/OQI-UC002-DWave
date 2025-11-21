# IMPLEMENTATION MEMORY FILE

## Constraint Reference from Binary Solver

### Farm Constraints (from solver_runner_BINARY.py lines 290-412)

```python
# 1. Land availability (line 341-350)
for farm in farms:
    constraint = sum(A[(farm, food)] for food in foods) <= land_availability[farm]
    
# 2. Min area if selected (line 354-365)
for farm in farms:
    for food in foods:
        if food in min_planting_area:
            constraint = A[(farm, food)] >= min_planting_area[food] * Y[(farm, food)]
            
# 3. Max area if selected (line 367-377)
for farm in farms:
    for food in foods:
        if food in max_planting_area:
            constraint = A[(farm, food)] <= max_planting_area[food] * Y[(farm, food)]
        else:
            constraint = A[(farm, food)] <= land_availability[farm] * Y[(farm, food)]

# 4. Food group MIN (line 381-396)
if food_group_constraints:
    for group_name, group_data in food_groups.items():
        if 'min_foods' in group_data:
            constraint = sum(A[(farm, food)] for farm in farms 
                           for food in group_data['foods']) >= min_foods * total_area
                           
# 5. Food group MAX (line 397-412)
if food_group_constraints:
    for group_name, group_data in food_groups.items():
        if 'max_foods' in group_data:
            constraint = sum(A[(farm, food)] for farm in farms 
                           for food in group_data['foods']) <= max_foods * total_area
```

### Patch Constraints (from solver_runner_BINARY.py lines 517-608)

```python
# 1. One-hot per patch (line 517-527)
for farm in farms:
    constraint = sum(Y[(farm, food)] for food in foods) <= 1
    
# 2. Min plots per crop (line 534-548)
for food in foods:
    if food in min_planting_area and min_planting_area[food] > 0:
        min_plots = ceil(min_planting_area[food] / plot_area)
        constraint = sum(Y[(farm, food)] for farm in farms) >= min_plots
        
# 3. Max plots per crop (line 554-567)
for food in foods:
    if food in max_planting_area:
        max_plots = floor(max_planting_area[food] / plot_area)
        constraint = sum(Y[(farm, food)] for farm in farms) <= max_plots
        
# 4. Food group MIN (line 574-594)
if food_group_constraints:
    for group_name, group_data in food_groups.items():
        if 'min_foods' in group_data:
            constraint = sum(land_availability[farm] * Y[(farm, food)] 
                           for farm in farms 
                           for food in group_data['foods']) >= min_foods * total_land
                           
# 5. Food group MAX (line 595-608)
if food_group_constraints:
    for group_name, group_data in food_groups.items():
        if 'max_foods' in group_data:
            constraint = sum(land_availability[farm] * Y[(farm, food)] 
                           for farm in farms 
                           for food in group_data['foods']) <= max_foods * total_land
```

## Current Implementation Gaps

### Alternative 1 (benchmark_utils_custom_hybrid.py)
- ❌ Missing: max_planting_area constraints (farm)
- ❌ Missing: food_group MAX constraints (farm)
- ❌ Missing: min_plots constraints (patch)
- ❌ Missing: max_plots constraints (patch)
- ❌ Missing: food_group MAX constraints (patch)
- ⚠️ Incorrect: Has one-hot for farm (should only be for patch)

### Alternative 2 (benchmark_utils_decomposed.py)
- ❌ Missing: max_planting_area constraints (farm)
- ❌ Missing: food_group MAX constraints (farm)
- ❌ Missing: min_plots constraints (patch)
- ❌ Missing: max_plots constraints (patch)
- ❌ Missing: food_group MAX constraints (patch)
- ⚠️ Incorrect: Has one-hot for farm (should only be for patch)

## Alternative 2 Architecture Fix

### Current (WRONG)
```
Farm → Gurobi (classical only)
Patch → DWaveSampler (quantum only)
```

### Correct (NEW)
```
Farm → HYBRID DECOMPOSITION:
  Step 1: Solve continuous relaxation (relax Y to [0,1], keep A continuous)
  Step 2: Use Gurobi to get optimal A* and relaxed Y*
  Step 3: Fix A to A*, create binary subproblem for Y only
  Step 4: Convert Y subproblem to BQM
  Step 5: Solve Y subproblem on QPU to get Y**
  Step 6: Combine: final solution uses A* (from step 2) and Y** (from step 5)
  
Patch → Pure Quantum (unchanged):
  Step 1: Create CQM for binary problem (only Y variables)
  Step 2: Convert to BQM
  Step 3: Solve directly on QPU
```

### Implementation Plan for Farm Hybrid Decomposition

```python
def solve_farm_with_hybrid_decomposition(farms, foods, food_groups, config, token):
    """
    Hybrid approach for farm scenario:
    1. Solve continuous relaxation (A continuous, Y relaxed to [0,1]) with Gurobi
    2. Extract A* from solution
    3. Fix A*, create binary subproblem for Y variables only
    4. Convert Y subproblem to BQM
    5. Solve on QPU to get Y**
    6. Combine: return A* and Y**
    """
    # Step 1: Continuous relaxation
    # Replace Binary Y variables with Real Y variables in [0, 1]
    # Solve with Gurobi
    
    # Step 2: Extract A* values
    # These are the continuous areas assigned to each farm-crop pair
    
    # Step 3: Create binary subproblem
    # Only Y variables, with constraints adjusted based on fixed A*
    
    # Step 4: Convert to BQM
    # Use cqm_to_bqm
    
    # Step 5: Solve on QPU
    # Use DWaveSampler or LeapHybridBQMSampler
    
    # Step 6: Combine results
    # Return solution with both A* and Y**
```

## File Modification Checklist

### Code Files
1. `benchmark_utils_custom_hybrid.py`
2. `benchmark_utils_decomposed.py`
3. `solver_runner_DECOMPOSED.py` (major refactor)
4. `test_custom_hybrid.py` (update expectations)
5. `test_decomposed.py` (update expectations)

### Documentation Files
1. `README_CUSTOM_HYBRID.md`
2. `README_DECOMPOSED.md`
3. `technical_report_chapter2.tex`
4. `technical_report_chapter4.tex`
5. `technical_report_chapter5.tex`

## Test Validation Criteria

After changes, verify:
1. ✓ All tests pass (100%)
2. ✓ Constraint counts match binary solver
3. ✓ Farm scenario: NO one-hot constraint
4. ✓ Patch scenario: HAS one-hot constraint
5. ✓ All scenarios have min/max area (or plots)
6. ✓ All scenarios have min/max food groups
7. ✓ Alternative 2 farm uses hybrid decomposition
8. ✓ Alternative 2 patch uses pure quantum

---

Last Updated: Nov 21, 2025
