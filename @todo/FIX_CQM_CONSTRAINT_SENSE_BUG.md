# Fix: CQM Constraint Sense Comparison Bug

**Date:** 2025-11-28  
**Issue:** CQM formulations were returning `INFEASIBLE` status when solved with Gurobi  
**Root Cause:** String comparison of `constraint.sense` failed because dimod uses `Sense` enum objects

---

## Problem Description

When running the comprehensive embedding and solving benchmark, CQM formulations were failing with:
- Status 3 = INFEASIBLE (Gurobi)
- All constraints appeared valid (mathematically feasible)
- Irreducible Inconsistent Subsystem (IIS) showed subset of constraints as contradictory

Example output:
```
[1/90] n_farms=5, form=CQM, decomp=None
  Building CQM...
  Solving CQM (135 vars, 6 constraints)...
  [DONE] in 0.01s (status=3)  ← INFEASIBLE!
```

## Root Cause

The constraint sense comparison in `comprehensive_embedding_and_solving_benchmark.py` and other files used string comparison:

```python
if constraint.sense == '<=':
    model.addConstr(constr_expr <= constraint.rhs, name=label)
elif constraint.sense == '>=':
    model.addConstr(constr_expr >= constraint.rhs, name=label)
else:  # ==
    model.addConstr(constr_expr == constraint.rhs, name=label)
```

However, dimod's `ConstrainedQuadraticModel` uses **Sense enum objects**:
- `Sense.Le` (not the string `'<='`)
- `Sense.Ge` (not the string `'>='`)
- `Sense.Eq` (not the string `'=='`)

This caused **ALL constraint comparisons to fail** and fall through to the `else` clause, converting all constraints to **equality constraints** (`==`). This made the problem infeasible.

For example, a constraint like:
```
sum(Y[patch_0, :]) <= 5  # Each patch can select at most 5 foods
```

Was incorrectly converted to:
```
sum(Y[patch_0, :]) == 5  # Each patch MUST select exactly 5 foods (WRONG!)
```

With 5 patches requiring exactly 5 foods each (25 total) and a minimum of 13 foods required globally, Gurobi presolve eliminated 130 out of 135 variables and found the problem infeasible.

## Solution

Convert sense to string first, then compare:

```python
sense_str = str(constraint.sense)
if sense_str == 'Sense.Le' or constraint.sense == '<=':
    model.addConstr(constr_expr <= constraint.rhs, name=label)
elif sense_str == 'Sense.Ge' or constraint.sense == '>=':
    model.addConstr(constr_expr >= constraint.rhs, name=label)
elif sense_str == 'Sense.Eq' or constraint.sense == '==':
    model.addConstr(constr_expr == constraint.rhs, name=label)
else:
    raise ValueError(f"Unknown constraint sense: {constraint.sense}")
```

This handles both:
- **dimod Sense enum objects** (`Sense.Le`, `Sense.Ge`, `Sense.Eq`)
- **String comparisons** (`'<='`, `'>='`, `'=='`) for compatibility

## Files Fixed

1. **`@todo/comprehensive_embedding_and_solving_benchmark.py`** (line 633-638)
   - Main benchmark script CQM solving function
   
2. **`Utils/diagnose_bqm_constraint_violations.py`** (lines 492, 665)
   - CQM constraint checking in diagnostics
   
3. **`@todo/debug_cqm_infeasibility.py`** (new debug script)
   - Created to identify and test the fix

## Verification

After the fix:
```bash
$ conda run -n oqi python @todo/debug_cqm_infeasibility.py

Building Patch CQM (5 patches, 27 foods)...
  Variables: 135
  Constraints: 6

Solving with Gurobi...
Optimize a model with 6 rows, 135 columns and 270 nonzeros
...
Solution status: 2  ← OPTIMAL!
Status name: OPTIMAL

Optimal objective: -25.000000
Active variables: 25

Selections per patch:
  Patch 0: 5 foods
  Patch 1: 5 foods
  Patch 2: 5 foods
  Patch 3: 5 foods
  Patch 4: 5 foods
  Total: 25 foods
```

## Additional Issue: BQM Time Limits

The benchmark also showed:
```
[2/90] n_farms=5, form=BQM, decomp=None
  Solving BQM (157 vars, 10431 quad)...
  [DONE] in 300.03s (status=9)  ← TIME_LIMIT
```

Status 9 = `TIME_LIMIT` means Gurobi hit the 300-second timeout without finding an optimal solution. This is **expected** for large BQM problems with 10,431 quadratic terms - these are NP-hard problems that may require:

1. **Longer time limits** (600s, 1200s, or more)
2. **Better decomposition strategies** (to reduce problem size)
3. **Quantum/hybrid solvers** (D-Wave QPU for actual quantum annealing)
4. **Heuristic methods** (simulated annealing, local search)

The benchmark should report partial solutions if available (`status=9` may have incumbent solutions) and allow configuration of time limits per problem size.

## Impact

This fix resolves the core issue preventing CQM formulations from solving. The decomposition strategies should now work correctly as they rely on the same constraint handling logic.

## Next Steps

1. ✅ Fix constraint sense comparisons (DONE)
2. ⚠️ Address BQM time limit issues (configure longer timeouts or better decomposition)
3. ⚠️ Test decomposition strategies (Benders, ADMM, Dantzig-Wolfe) with fixed CQM solving
4. ⚠️ Implement better handling of TIME_LIMIT status (report partial solutions)
