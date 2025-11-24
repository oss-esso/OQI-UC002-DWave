# Decomposition Strategy Bugs - Area Overflow Issue

## Problem Summary

Benders and Dantzig-Wolfe decomposition strategies are producing **INVALID** solutions that violate the fundamental land availability constraint:

$$\sum_{c \in \mathcal{C}} A_{f,c} \leq L_f \quad \forall f \in \mathcal{F}$$

## Evidence

**Config 25 (25 farms, 100 ha total):**

| Strategy | Total Area Used | Overflow | Objective | Status |
|----------|----------------|----------|-----------|---------|
| PuLP (baseline) | 100.0 ha | 0% | 0.388 | ✅ Valid |
| Benders | 210.0 ha | **+110%** | 0.480 | ❌ INVALID |
| Dantzig-Wolfe | 288.5 ha | **+188%** | 0.864 | ❌ INVALID |

## Root Causes

### 1. Dantzig-Wolfe: Column Accumulation Without Bounds

**File:** `decomposition_dantzig_wolfe.py`, lines 199-202

```python
for col_idx, weight in final_solution.items():
    if weight > 1e-6:
        col = columns[col_idx]
        for key, value in col['allocation'].items():
            A_solution[key] += weight * value  # ← UNBOUNDED ACCUMULATION
```

**Issue:** When multiple columns contribute to the same `(farm, food)` pair, their weighted allocations are summed without checking if the total exceeds `L_f` for that farm.

**Example:**
- Column 1: Farm1_Spinach = 50 ha (weight=0.6)
- Column 2: Farm1_Spinach = 80 ha (weight=0.5)
- **Result:** Farm1_Spinach = 50×0.6 + 80×0.5 = **70 ha**
- Even if Farm1 capacity is only 4 ha!

### 2. Benders: Subproblem Doesn't Enforce Per-Farm Limits

**File:** `decomposition_benders.py`, lines 220-240

The subproblem LP optimizes `A` variables but the land availability constraints may not be properly aggregated or the dual variables aren't correctly passed back to generate valid cuts.

### 3. Missing Global Validation

Neither strategy verifies that:
```python
total_allocated = sum(A_dict.values())
assert total_allocated <= sum(farms.values()), "Global land overflow!"
```

## Required Fixes

### Fix 1: Add Per-Farm Capacity Constraints in RMP (Dantzig-Wolfe)

```python
# In solve_restricted_master()
for farm in farms:
    rmp.addConstr(
        gp.quicksum(
            x[col_idx] * col['allocation'].get((farm, food), 0.0)
            for col_idx, col in enumerate(columns)
            for food in foods
        ) <= farms[farm],
        name=f"Farm_Capacity_{farm}"
    )
```

### Fix 2: Verify Benders Cuts Include Land Constraints

Ensure subproblem dual variables for land constraints are correctly used in optimality cuts.

### Fix 3: Add Post-Solution Projection

After reconstruction, project solution back to feasible space:

```python
# For each farm, if overflow detected:
for farm in farms:
    farm_total = sum(A_solution.get((farm, c), 0) for c in foods)
    if farm_total > farms[farm]:
        scale = farms[farm] / farm_total
        for c in foods:
            A_solution[(farm, c)] *= scale
```

### Fix 4: Enhanced Validation in benchmark_all_strategies.py

✅ **Already implemented:** Global area overflow detection in comparison table.

## Theoretical Maximum Check

With corrected benefit calculation (environmental_impact subtracted):

**Top Foods by Benefit:**
1. Spinach: 0.4300
2. Pork: 0.3091
3. Long bean: 0.2886

**Theoretical Maximum Objective:** 0.4300 (all land to Spinach)

Any objective > 0.43 is **PHYSICALLY IMPOSSIBLE** and indicates constraint violation.

## Action Items

- [ ] Fix Dantzig-Wolfe column aggregation with per-farm capacity constraints
- [ ] Fix Benders cut generation to properly include land availability duals
- [ ] Add projection/scaling to ensure all solutions respect global land limit
- [ ] Add assertion checks in both strategies before returning results
- [ ] Update validation to fail strategies that produce infeasible solutions

## References

- LaTeX specification: See constraint definitions in technical documentation
- PuLP baseline: `Benchmarks/COMPREHENSIVE/Farm_PuLP/config_25_run_1.json`
- Test results: `Benchmarks/ALL_STRATEGIES/all_strategies_config_25_*.json`
