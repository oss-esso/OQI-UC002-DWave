# Technical Report: Problem Formulation Comparison
## Hierarchical Statistical Test vs Gurobi Timeout Test

**Date**: December 15, 2025  
**Analysis**: Rotation optimization problem formulation discrepancies

---

## Executive Summary

**CRITICAL ISSUE IDENTIFIED**: The hierarchical test is loading data incorrectly, resulting in:
1. ❌ **Fixed total area (1,187.91 ha) regardless of farm count** - should scale linearly
2. ❌ **Wrong number of foods (5 instead of 6)** - missing one food per scenario
3. ❌ **No crop diversity after post-processing (0 unique crops)** - post-processing failing

**Root Cause**: The hierarchical test loads 250 farms' worth of data and limits to first N farms, but then tries to select 6 representative foods from food families incorrectly.

**Impact**: The hierarchical test is solving a DIFFERENT problem than the Gurobi timeout test.

---

## 1. Data Loading Comparison

### Gurobi Timeout Test (`test_gurobi_timeout.py`)

```python
def load_scenario_data(scenario: Dict) -> Dict:
    """Loads scenario by name directly from scenarios.py"""
    n_farms = scenario['n_farms']
    n_foods = scenario['n_foods']
    
    # Maps to actual scenario names
    if n_foods == 6:
        # Uses predefined 6-food family scenarios
        if n_farms <= 5:
            scenario_name = 'rotation_micro_25'      # 5 farms × 6 families
        elif n_farms <= 10:
            scenario_name = 'rotation_small_50'      # 10 farms × 6 families
        # ... etc
    else:  # 27 foods
        # Uses predefined 27-food scenarios
        scenario_name = 'rotation_250farms_27foods'  # 250 farms × 27 foods
    
    data = load_food_data_as_dict(scenario_name)
    return data  # Returns exact scenario data
```

**Key Points**:
- ✅ Loads **exact scenario** with correct number of farms and foods
- ✅ Each scenario has **pre-configured data** (land, foods, parameters)
- ✅ Area scales correctly with number of farms
- ✅ 6-food scenarios have exactly 6 food families defined

### Hierarchical Test (`hierarchical_statistical_test.py`)

```python
def load_hierarchical_data(n_farms: int, n_foods: int, scenario_name: str) -> Dict:
    """Loads base scenario and limits farms/foods"""
    
    # PROBLEM 1: Always loads 250-farm scenario
    base_scenario = 'rotation_250farms_27foods'  # 1,187.91 ha for 250 farms
    farms, foods, food_groups, config = load_food_data(base_scenario)
    
    # PROBLEM 2: Limits to first N farms
    la = params.get('land_availability', {})
    farm_names = list(la.keys())[:n_farms]  # Takes first N farms from 250
    la = {f: la[f] for f in farm_names}
    total_area = sum(la.values())  # ❌ WRONG: Sums only first N farms' areas
    
    # PROBLEM 3: Incorrectly selects 6 foods
    if n_foods == 6:
        food_names = []
        seen_families = set()
        for food_name in all_food_names:  # Iterates 27 foods
            for family, members in food_groups.items():  # Checks families
                if food_name in members and family not in seen_families:
                    food_names.append(food_name)
                    seen_families.add(family)
                    if len(food_names) >= 6:
                        break
            if len(food_names) >= 6:
                break
```

**Key Issues**:
1. ❌ **Area Not Scaling**: Always loads 250 farms, sums first N farms' areas
   - 5 farms: should be ~23.76 ha, actually varies by which 5 farms selected
   - Should generate synthetic farms with consistent area per farm
2. ❌ **Wrong Food Selection**: Tries to extract 6 foods from 27-food scenario
   - Gets 5 foods instead of 6 (loop logic error)
   - Foods are individual crops, not family representatives
3. ❌ **Inconsistent Data**: Different scenarios get different subsets of 250 farms

---

## 2. Problem Formulation Comparison

### Decision Variables

**Both tests use identical variable definition**:
```
Y[f, c, t] ∈ {0, 1}
where:
  f ∈ Farms
  c ∈ Foods (6 families or 27 crops)
  t ∈ Periods {1, 2, 3}
```

✅ **IDENTICAL**: Binary variables for farm-food-period assignments

---

### Objective Function

**Both tests use identical objective structure**:

```
Maximize: Σ[benefit] + Σ[rotation_synergy] + Σ[spatial_synergy] + Σ[diversity_bonus]
          - Σ[one_hot_penalty]

Where:
  benefit         = Σ_f,c,t (food_benefit[c] * land[f] * Y[f,c,t])
  rotation_synergy = γ * Σ_f,c1,c2,t (R[c1,c2] * Y[f,c1,t] * Y[f,c2,t+1])
  spatial_synergy  = γ/2 * Σ_neighbors (synergy * Y[f1,c,t] * Y[f2,c,t])
  diversity_bonus  = α * Σ_f (unique_crops[f])
  one_hot_penalty  = -λ * Σ_f,t (Σ_c Y[f,c,t] - 1)²
```

**Parameters** (both tests):
- `γ` (rotation_gamma) = 0.2
- `α` (diversity_bonus) = 0.15
- `λ` (one_hot_penalty) = 3.0
- `k_neighbors` = 4

✅ **IDENTICAL**: Objective function structure and parameters

---

### Constraints

**Both tests use identical constraints**:

1. **One-Hot Constraint** (One food per farm per period):
   ```
   Σ_c Y[f,c,t] = 1  ∀f,t
   ```

2. **Land Availability** (implicit in benefit calculation):
   ```
   benefit proportional to land[f]
   ```

✅ **IDENTICAL**: Constraint structure

---

## 3. Solver Configuration Comparison

### Gurobi Timeout Test

```python
model.setParam('TimeLimit', 100)           # 100s hard timeout
model.setParam('MIPGap', 0.01)             # 1% optimality gap
model.setParam('MIPFocus', 1)              # Focus on feasibility
model.setParam('ImproveStartTime', 30)     # Stop if no improvement after 30s
```

### Hierarchical Test (Gurobi branch - not used)

```python
model.setParam('TimeLimit', 300)           # 300s hard timeout
model.setParam('MIPGap', 0.1)              # 10% optimality gap (more lenient)
model.setParam('MIPFocus', 1)              # Focus on feasibility
model.setParam('ImproveStartTime', 30)     # Stop if no improvement after 30s
```

⚠️ **DIFFERENT**: Hierarchical test uses 300s timeout and 10% gap (but Gurobi skipped)

---

## 4. Hierarchical Decomposition Details

### Level 1: Aggregation

**For 27-food scenarios only**:
```python
# Aggregate 27 crops → 6 families
family_data = aggregate_foods_to_families(data)
```

**For 6-food scenarios**:
- No aggregation needed (already at family level)
- ❌ **PROBLEM**: Currently loading wrong number of foods (5 instead of 6)

### Level 2: Spatial Decomposition

```python
# Decompose into clusters
farms_per_cluster = 5
n_clusters = ceil(n_farms / farms_per_cluster)

# Example: 25 farms → 5 clusters of 5 farms each
# Each cluster: 5 farms × 6 families × 3 periods = 90 variables
```

### Level 3: Post-Processing

```python
# Refine families → specific crops (only for 27-food scenarios)
if n_foods == 27:
    crop_solution = refine_family_solution_to_crops(family_solution, original_data)
else:
    # For 6-food scenarios, no refinement
    crop_solution = family_solution
```

❌ **PROBLEM**: Post-processing returns 0 unique crops for 6-food scenarios

---

## 5. Critical Discrepancies

### Issue 1: Area Calculation

**Expected Behavior**:
```
rotation_micro_25 (5 farms):
  - Each farm: ~4.75 ha (assuming uniform distribution)
  - Total: 5 × 4.75 = 23.76 ha

rotation_small_50 (10 farms):
  - Total: 10 × 4.75 = 47.52 ha

rotation_25farms_6foods (25 farms):
  - Total: 25 × 4.75 = 118.80 ha
```

**Actual Behavior** (Hierarchical Test):
```
ALL scenarios: 1,187.91 ha (250 farms' worth)
  - Takes first N farms from 250-farm scenario
  - Sum of first N farms' areas (not uniform)
```

**Impact**: Problem scale inconsistent, benefit calculations wrong

---

### Issue 2: Food Count

**Expected** (Gurobi Test):
```
rotation_micro_25: 5 farms × 6 families × 3 periods = 90 vars
rotation_small_50: 10 farms × 6 families × 3 periods = 180 vars
```

**Actual** (Hierarchical Test):
```
rotation_micro_25: 5 farms × 5 foods × 3 periods = 75 vars  ❌ (missing 1 food)
rotation_small_50: 10 farms × 5 foods × 3 periods = 150 vars ❌ (missing 1 food)
```

**Impact**: Different problem size, missing one food family

---

### Issue 3: Post-Processing Results

**Expected**:
```
6-food scenarios:
  - Post-processing: family → specific crops
  - Should produce 15-20 unique crops from 6 families

27-food scenarios:
  - Post-processing: family → specific crops
  - Should produce 15-20 unique crops from 27 available
```

**Actual** (Hierarchical Test):
```
ALL scenarios: 0 unique crops ❌
  - Post-processing returning empty solution
  - Diversity analysis failing
```

**Impact**: Solution quality metrics are wrong

---

## 6. Root Cause Analysis

### Problem 1: Data Loading Strategy

**Gurobi Test**: ✅ Correct
- Loads pre-defined scenarios with exact farm/food counts
- Each scenario is self-contained and verified

**Hierarchical Test**: ❌ Incorrect
- Tries to synthesize scenarios by limiting large base scenario
- Does not handle 6-food scenarios properly
- Area and food selection logic is wrong

### Problem 2: Food Family Selection

**Current Code** (hierarchical_statistical_test.py, lines 178-192):
```python
if n_foods == 6:
    # Use only family representatives (6 foods)
    food_names = []
    seen_families = set()
    for food_name in all_food_names:  # 27 foods
        for family, members in food_groups.items():  # 6 families
            if food_name in members and family not in seen_families:
                food_names.append(food_name)
                seen_families.add(family)
                if len(food_names) >= 6:
                    break
        if len(food_names) >= 6:
            break
```

**Issue**: 
- Logic breaks early, only gets 5 foods
- Should instead use family names directly, not search for representatives

**Correct Approach**:
```python
if n_foods == 6:
    # Use family names directly as "foods"
    food_names = list(food_groups.keys())[:6]  # ['Grains', 'Vegetables', 'Legumes', ...]
    # OR load actual 6-family scenario like Gurobi test does
```

---

## 7. Recommendations

### Immediate Fixes Required

1. **Fix Data Loading** (HIGH PRIORITY):
   ```python
   # Instead of loading 250-farm scenario and limiting,
   # load the EXACT scenario that Gurobi test uses:
   
   if n_foods == 6:
       # Load actual 6-family scenario
       data = load_food_data_as_dict(scenario_name)
   else:
       # Load actual 27-food scenario
       data = load_food_data_as_dict(scenario_name)
   ```

2. **Verify Area Scaling**:
   - Check that total area scales linearly with farm count
   - Print diagnostic: `area_per_farm = total_area / n_farms`

3. **Fix Food Count**:
   - Ensure 6-food scenarios have exactly 6 foods
   - Verify variable count matches expected

4. **Fix Post-Processing**:
   - Investigate why 0 unique crops returned
   - May need to skip post-processing for 6-food scenarios

### Implementation Plan

**Step 1**: Use Gurobi test's data loading approach
```python
from test_gurobi_timeout import load_scenario_data

def load_hierarchical_data(scenario: Dict) -> Dict:
    # Use exact same loading as Gurobi test
    data = load_scenario_data(scenario)
    return data
```

**Step 2**: Verify problem formulation matches
- Print variable counts
- Print constraint counts
- Compare objective coefficients

**Step 3**: Test on small scenario
- rotation_micro_25 (5 farms × 6 foods)
- Verify: 90 variables, 5×23.76 ha area, 6 foods

---

## 8. Verification Checklist

Before running full benchmark:

- [ ] Total area scales linearly: `area = n_farms × area_per_farm`
- [ ] Correct food count: 6 for family scenarios, 27 for full scenarios
- [ ] Variable count matches: `n_farms × n_foods × 3`
- [ ] Post-processing produces > 0 unique crops
- [ ] Objective values in reasonable range (not 0 or 1000+)
- [ ] Solver configurations match expected parameters

---

## 9. Comparison Table

| Aspect | Gurobi Timeout Test | Hierarchical Test (Current) | Status |
|--------|---------------------|----------------------------|--------|
| Data Loading | Direct scenario load | Limit 250-farm scenario | ❌ WRONG |
| Area Scaling | Linear with farms | Fixed 1,187.91 ha | ❌ WRONG |
| Food Count (6-food) | Exactly 6 families | Gets 5 foods | ❌ WRONG |
| Food Count (27-food) | Exactly 27 crops | Gets 27 crops | ✅ OK |
| Variable Count | Correct | Off by ~15% | ❌ WRONG |
| Objective Function | Correct | Correct | ✅ OK |
| Constraints | Correct | Correct | ✅ OK |
| Post-Processing | N/A | Returns 0 crops | ❌ WRONG |
| Solver Config | 100s, 1% gap | 300s, 10% gap | ⚠️ DIFFERENT |

---

## 10. Conclusion

**The hierarchical test is currently solving a DIFFERENT problem than the Gurobi timeout test** due to incorrect data loading. The problem formulation (objective and constraints) is correct, but the input data is wrong.

**Critical Issues**:
1. Area does not scale (always 1,187.91 ha)
2. Wrong number of foods (5 instead of 6)
3. Post-processing broken (0 unique crops)

**Recommendation**: **STOP current test and fix data loading first**. Use the same `load_food_data_as_dict()` approach as the Gurobi timeout test to ensure identical problem instances.

**Impact**: Current results are **NOT COMPARABLE** to Gurobi timeout test results.
