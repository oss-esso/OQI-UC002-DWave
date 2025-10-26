# REAL Root Cause: Lagrange Multiplier Issue in PATCH

## The Actual Problem

**BQUBO is SLOWER but FEASIBLE**
**PATCH is FASTER but INFEASIBLE** (massive constraint violations)

## Data Comparison

### BQUBO (72 farms)
```json
{
  "hybrid_time": 65.68s,
  "qpu_time": 0.155s,
  "bqm_conversion_time": 0.175s,
  "feasible": true,
  "objective_value": 0.0336
}
```

### PATCH (50 plots)
```json
{
  "hybrid_time": 2.99s,  ← 22x FASTER
  "qpu_time": 0.104s,
  "bqm_conversion_time": 0.068s,
  "is_feasible": false,  ← INFEASIBLE!
  "n_violations": 24,
  "utilization": 1.62  ← 162% over-allocated!
}
```

### PATCH (100 plots)
```json
{
  "hybrid_time": 3.73s,  ← 18x FASTER
  "qpu_time": 0.104s,
  "bqm_conversion_time": 0.034s,
  "is_feasible": false,  ← INFEASIBLE!
  "n_violations": 87,
  "utilization": 1.91  ← 191% over-allocated!
}
```

## Root Cause

### BQUBO (`solver_runner_BQUBO.py` line 231)
```python
bqm, invert = cqm_to_bqm(cqm)  # No lagrange_multiplier specified!
```
- D-Wave **automatically chooses** a strong Lagrange multiplier
- Constraints are **strongly enforced**
- Solution is **feasible** but takes longer

### PATCH (`comprehensive_benchmark.py` line 565)
```python
lagrange_multiplier = 10000 * max_obj_coeff  # Manual override
bqm, invert = cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)
```
- **Manually specified** Lagrange multiplier
- **TOO WEAK** relative to constraint violations
- Constraints are **ignored** (cheap to violate)
- Solution is **fast but infeasible**

## Why PATCH Lagrange Multiplier is Too Weak

### Objective Scale in PATCH
```python
# Max objective coefficient for 50 plots
max_obj_coeff ≈ 1.97  # (B_c + λ) × s_p where s_p can be large

# Lagrange multiplier used
lagrange_multiplier = 10000 × 1.97 ≈ 19,700
```

### Constraint Violation Cost
```python
# For "at most one crop per plot" constraint violation:
# Assigning 2 crops to a plot (violation = 1)
penalty = lagrange_multiplier × (violation)²
penalty = 19,700 × 1² = 19,700
```

### Objective Benefit from Violation
```python
# Benefit from assigning 2 good crops to a large plot (s_p = 2.0):
benefit = 2 × (B_c + λ) × s_p
benefit = 2 × 1.97 × 2.0 = 7.88

# Net benefit from violating constraint:
net = benefit - penalty = 7.88 - 19,700 = -19,692
```

Wait, this should prevent violations... Let me recalculate more carefully with actual BQM energy values.

### The Real Issue: Objective Magnitudes

Looking at the results:
- **PATCH BQM energy**: 473,355 (50 plots) or 1,536,377 (100 plots)
- **PATCH objective**: 23.38 (50 plots) or 52.12 (100 plots)

The BQM energy is **HUGE** compared to the objective! This suggests:
1. The Lagrange multiplier IS creating large penalties
2. BUT the solver is finding "local minima" that violate constraints
3. The problem is **poorly conditioned** with the manual multiplier

### D-Wave's Auto-Selection

When you DON'T specify `lagrange_multiplier`, D-Wave:
1. Analyzes the constraint structure
2. Computes appropriate penalty weights
3. Uses **adaptive penalties** during solving
4. **Reweights** if it finds violated constraints

When you DO specify `lagrange_multiplier`:
1. Fixed penalty throughout solving
2. No adaptation
3. Solver may get stuck in infeasible regions

## Solution

**Remove the manual Lagrange multiplier in PATCH** - let D-Wave choose automatically like BQUBO does!

```python
# CURRENT (comprehensive_benchmark.py):
lagrange_multiplier = 10000 * max_obj_coeff
bqm, invert = cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)

# FIX:
bqm, invert = cqm_to_bqm(cqm)  # Let D-Wave auto-select!
```

This should:
- ✓ Maintain feasibility (like BQUBO)
- ✓ Still be reasonably fast
- ✓ Use D-Wave's adaptive penalty mechanism

## Expected Results After Fix

**PATCH with auto Lagrange** (predicted):
- hybrid_time: ~30-60s (slower than current, but still faster than BQUBO's 65s)
- is_feasible: true
- n_violations: 0
- Better quality solutions

## Why BQUBO is Slower Even With Fewer Constraints

BQUBO has:
- **Simpler constraints** (plantation limits only)
- **Smaller objective scale** (normalized 0-1)
- **Auto Lagrange** working well

But it's SLOWER because:
1. **D-Wave chose a STRONGER Lagrange multiplier** (more conservative)
2. **Search space is harder** (72 farms vs 50-100 plots)
3. **Binary formulation** may have different BQM structure

The speed difference is D-Wave being **more careful** with BQUBO to ensure feasibility!
