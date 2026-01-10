# β Parameter Analysis: Monoculture Penalty Issue

## Problem Summary

The user correctly identified that the current β parameter values lead to **excessive monoculture penalties** that don't match the paper's stated ~17-20% yield loss.

## Current Implementation

### Found in `unified_benchmark/scenarios.py` (lines 314-356):

```python
def build_rotation_matrix(
    n_foods: int,
    frustration_ratio: float = 0.7,
    negative_strength: float = -0.8,  # ← This is β
    seed: int = 42
) -> np.ndarray:
    """Build a rotation synergy matrix."""
    # ...
    for i in range(n_foods):
        for j in range(n_foods):
            if i == j:
                # Same crop: strong negative (avoid monoculture)
                R[i, j] = negative_strength * 1.5  # ← R_c,c = -0.8 × 1.5 = -1.2
```

### Current Values:
- **β (negative_strength)**: `-0.8`
- **Monoculture penalty**: `R_c,c = -0.8 × 1.5 = -1.2`
- **Scaling factor (rotation_gamma)**: `0.2` (found in all solvers)

### Effective Penalty Calculation:

The penalty enters the objective as:
```
Penalty = -rotation_gamma × R_c,c × area_frac
        = -0.2 × (-1.2) × area_frac
        = 0.24 × area_frac
```

This is added as a **quadratic penalty term** for Y[f,c,t] × Y[f,c,t+1] pairs.

## The Mathematical Issue

**CONFIRMED FROM crop_rotation.tex (line 36):**

```latex
R_{c,c'}: rotation synergy matrix: numerical effect (benefit if positive, 
penalty if negative) from planting c' in period t after crop c in period t-1
```

**Formulation from crop_rotation.tex (line 56):**

```latex
Temporal = γ_rot × Σ Σ Σ Σ R_{c,c'} × A_{f,c,t-1} × A_{f,c',t}
```

This is a **LINEAR contribution to objective**, NOT exponential!

### Current Implementation Analysis:

**The rotation term adds to objective value:**
```
contribution = rotation_gamma × R_c,c' × area_frac
             = 0.2 × R_c,c' × area_frac
```

**For monoculture (c = c'):**
```
R_c,c = -0.8 × 1.5 = -1.2
contribution = 0.2 × (-1.2) × area_frac = -0.24 × area_frac
```

This is a **penalty added to objective**, NOT a yield multiplier!

## Paper/User Specification Issue

**The user's concern about 17-20% yield loss assumes exponential formulation:**
- exp(-1.2) ≈ 0.30 → 70% yield loss
- eResolution: Two Different Interpretation Paths

### Path 1: User expects exponential formulation (like agronomic models)

If the paper/specification truly means **yield multiplier**:
```
yield = base_yield × exp(R_c,c')
```

Then for 17-20% loss:
```
exp(-β × 1.5) = 0.80-0.83  (keep 80-83% of yield)
-β × 1.5 = ln(0.80) ≈ -0.223
β = -0.223 / 1.5 ≈ -0.15
```

**This would require changing the formulation entirely** - not just parameters.

### Path 2: Current linear formulation is correct

If the formulation is correct (linear contribution to objective):

**Monoculture penalty effect:**
```
For base_benefit = 1.0 per unit area:
Without monoculture: objective += 1.0 × area_frac
With monoculture: objective += 1.0 × area_frac + 0.2 × (-1.2) × area_frac
                            = (1.0 - 0.24) × area_frac
                            = 0.76 × area_frac
```

**This gives 24% reduction in objective contribution** for monoculture, which is CLOSE to the 17-20% target!

CurAnalysis Summary

### Confirmed Facts:

1. **Formulation is LINEAR (from crop_rotation.tex)**
   ```
   Objective += γ_rot × R_{c,c'} × area_product
   ```
   NOT exponential yield multiplier

2. **Current Parameters:**
   - β (negative_strength) = -0.8
   - Monoculture penalty: R_c,c = -0.8 × 1.5 = -1.2
   - Scaling factor: rotation_gamma = 0.2
   - **Effective monoculture penalty: 24% reduction in objective**

3. **User's 17-20% target:**
   - Current 24% is close but slightly higher
   - To get 17-20%, need one of:
     - Reduce β: -0.57 to -0.67 (instead of -0.8)
     - Reduce rotation_gamma: 0.14 to 0.17 (instead of 0.2)
     - Reduce monoculture multiplier: 1.06 to 1.25 (instead of 1.5)

### Key Insight:

**The user's exponential interpretation (exp(-1.2) ≈ 70% loss) does NOT apply to our formulation!**

Our formulation uses **additive penalties** in the objective function, not **multiplicative yield effects**.

### Questions for User:

1. **Is the 17-20% figure from a paper that uses exponential yield models?**
   - If so, those parameters don't directly transfer to linear objective formulations

2. **Does "17-20% yield loss" mean:**
   - (a) Actual crop yield (kg/hectare) is 17-20% lower? (exponential model)
   - (b) Objective function value is 17-20% lower? (linear model - what we have)

3. **Should we adjust parameters to match the 17-20% target in our linear formulation?**
   - Current 24% is reasonable and not far off
   - Easy adjustment: rotation_gamma = 0.15 would give ~18% penalty

## Critical Questions to Resolve:

1. **Does the paper use exponential or linear formulation?**
   - Check formulations.tex and the paper PDF
   - Look for exp() terms in rotation synergy

2. **What is the actual rotation penalty formulation?**
   - Is it added to objective: `obj += rotation_gamma × R_c,c'`?
   - Or applied to yield: `yield × exp(R_c,c')`?

3. **What β values were used in paper results?**
   - Check paper's parameter tables
   - Look for sensitivity analysis on rotation parameters

4. **Is rotation_gamma = 0.2 correct?**
   - This scales all rotation effects by 20%
   - Should it be 1.0 for full effect?

## Files to Check:

1. **formulations.tex** - Mathematical formulation of rotation term
2. **Paper PDF** - Parameter values and yield loss specifications
3. **unified_benchmark/gurobi_solver.py** - How Gurobi applies rotation penalties
4. **unified_benchmark/miqp_scorer.py** - How MIQP objective computes rotation synergy

## Recommendation:

Need to:
1. Read formulations.tex to confirm the exact mathematical formulation
2. Verify if the 17-20% is a yield multiplier or an objective difference
3. Adjust β and/or rotation_gamma to match paper specifications
4. Re-run benchmarks with corrected parameters
