# D-Wave CQM Infeasibility Issue Analysis

## Problem Summary

D-Wave's Hybrid CQM solver is returning solutions that violate the "at most one crop per plot" constraint, which should be a simple hard constraint.

### Observed Behavior

From `Benchmarks/COMPREHENSIVE/Patch_DWave/config_10_run_1.json`:
- **Status**: "Infeasible"
- **Violations**: 9 out of 10 plots have multiple crops assigned
- **Example violations**:
  - Patch1: 3 crops (Durian, Egg, Peanuts)
  - Patch2: 3 crops (Chicken, Cucumber, Tofu)
  - Patch9: 3 crops (Long bean, Pork, Pumpkin)

## Root Cause Analysis

### 1. Constraint Formulation is Correct

The CQM constraint in `solver_runner_BINARY.py` (line 518-521):
```python
cqm.add_constraint(
    sum(Y[(farm, food)] for food in foods) - 1 <= 0,
    label=f"Max_Assignment_{farm}"
)
```

Where `Y[(farm, food)]` are **Binary** variables (line 474).

This correctly encodes: **sum(Y) ≤ 1** for binary variables, meaning **at most one Y can be 1**.

### 2. Why D-Wave CQM Fails

D-Wave's Hybrid CQM Solver has known limitations:

#### A. **Soft Constraint Treatment**
- The hybrid solver uses a **classical preprocessor + quantum annealer**
- During CQM→BQM conversion, constraints become **penalty terms** with Lagrange multipliers
- Auto-tuned penalty weights may be **insufficient** to enforce hard constraints

#### B. **Conflicting Constraints**
- Minimum planting area constraints may require **more plots than available**
- Example: If 27 foods each need minimum 1 plot, but only 10 plots exist → **mathematically infeasible**
- Solver then returns "best effort" solution violating constraints

#### C. **Time Limit**
- Hybrid solver has internal time limits (default ~5 seconds)
- May terminate before finding feasible solution

#### D. **Objective Dominance**
- Solver may prioritize **maximizing objective** over satisfying constraints
- Penalty weights not scaled high enough relative to objective coefficients

## Verification Steps

### 1. Check if Problem is Actually Infeasible

Run PuLP/Gurobi on same instance:
```python
# In comprehensive_benchmark.py, check Patch_PuLP results
# If Gurobi also reports "Infeasible", problem formulation has conflicts
```

### 2. Identify Constraint Conflicts

Use diagnostic tool:
```python
from Utils.diagnose_cqm_infeasibility import analyze_cqm_constraints

cqm, Y, metadata = solver_runner.create_cqm_plots(plots, foods, food_groups, config)
analyze_cqm_constraints(cqm)
```

This will identify if:
- Sum of minimum plots required > available plots
- Food group constraints create impossibilities

## Recommended Solutions

### Solution 1: Relax Minimum Planting Area Constraints ⭐ **RECOMMENDED**

**Why**: Minimum area constraints likely cause infeasibility

**Implementation**:
```python
# In config creation (comprehensive_benchmark.py, line 166)
'minimum_planting_area': {
    food: 0.0  # Changed from 0.0001 - allow any food to NOT be planted
    for food in foods
}
```

### Solution 2: Increase CQM Solver Time Limit

**Why**: Give solver more time to find feasible solution

**Implementation**:
```python
# In solve_with_dwave_cqm (solver_runner_BINARY.py, line 1007)
sampleset = sampler.sample_cqm(
    cqm, 
    label="Food Optimization - Professional Run",
    time_limit=30  # Increase from default ~5s to 30s
)
```

### Solution 3: Add Constraint Weights (Advanced)

**Why**: Explicitly prioritize constraint satisfaction

**Note**: CQM doesn't directly support constraint weights, but you can:

```python
# Option A: Make "at most one" an equality constraint
cqm.add_constraint(
    sum(Y[(farm, food)] for food in foods) == 1,  # EXACTLY one (or use 0 for idle)
    label=f"Exactly_One_Assignment_{farm}"
)

# Option B: Add slack variables with heavy penalties
```

### Solution 4: Reduce Food Group Diversity Requirements

**Why**: May be forcing too many different foods to be planted

**Implementation**:
```python
# In config creation
'food_group_constraints': {
    group: {
        'min_foods': 0,  # Changed from 1 - allow groups to be empty
        'max_foods': len(food_list)
    }
    for group, food_list in food_groups.items()
}
```

### Solution 5: Use BQM Directly with Manual Penalty Scaling

**Why**: Better control over penalty weights

**Implementation**: See `Tests/README_constraint_investigation.md` for manual BQM construction with custom penalty multipliers (100x-1000x objective scale).

## Comparison with PuLP/Gurobi

**PuLP/Gurobi Results** (from `Patch_PuLP/config_10_run_1.json`):
- Check if Gurobi finds problem **feasible or infeasible**
- If **feasible**: D-Wave solver issue
- If **infeasible**: Formulation issue (likely minimum area constraints)

## Action Plan

1. **Immediate** - Check PuLP results to determine if problem is actually feasible
2. **Quick Fix** - Set `minimum_planting_area` to 0.0 for all foods
3. **Diagnostic** - Run `analyze_cqm_constraints()` to identify conflicts
4. **Testing** - Re-run benchmark with relaxed constraints
5. **Advanced** - If still infeasible, increase solver `time_limit` or use manual BQM

## Expected Outcome

With `minimum_planting_area = 0.0`:
- Each plot should have **0 or 1 crop** assigned (not 2-3)
- D-Wave should return **feasible solutions**
- Objective value should be comparable to Gurobi

## References

- `Tests/README_constraint_investigation.md` - BQM penalty analysis
- `CQM_DIAGNOSTIC_GUIDE.md` - CQM debugging strategies
- `Utils/diagnose_cqm_infeasibility.py` - Diagnostic tool
