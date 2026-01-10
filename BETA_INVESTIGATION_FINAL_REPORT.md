# β Parameter Investigation - Final Report

## Your Question

You correctly identified that with β = -0.8 and the monoculture penalty R_c,c = -β × 1.5 = -1.2, if this enters **exponentially** as exp(-1.2) ≈ 0.30, it would cause 70% yield loss (far more than the 17-20% target).

## Answer: Your Exponential Interpretation Doesn't Apply

### Our Formulation is LINEAR, Not Exponential

**From `Latex/crop_rotation.tex` (authoritative formulation document):**

```latex
Objective = Base_Benefit + γ_rot × Σ R_{c,c'} × Y_{f,c,t-1} × Y_{f,c',t}
```

This is a **linear addition** to the objective, not a yield multiplier!

### What the Parameters Actually Do

**Current Parameters (from `unified_benchmark/scenarios.py`):**
```python
beta = -0.8  # negative_strength
R_c,c = -0.8 × 1.5 = -1.2  # monoculture penalty
rotation_gamma = 0.2  # scaling factor
```

**Effect on Objective:**
```
Base benefit: 1.0 × area_frac
Monoculture contribution: 1.0 × area_frac + 0.2 × (-1.2) × area_frac
                        = 0.76 × area_frac
```

**Result: 24% reduction in objective value for monoculture**

### Comparison to Your Exponential Concern

**Your calculation (DOES NOT APPLY):**
```
exp(-1.2) ≈ 0.30  → 70% yield loss ❌ Wrong model
```

**Actual effect (LINEAR MODEL):**
```
penalty = 0.2 × (-1.2) = -0.24  → 24% objective reduction ✓ Correct
```

## Current Status

✅ **Current 24% penalty is close to your 17-20% target**

✅ **Parameters are reasonable for the linear formulation**

✅ **No exponential yield loss occurs**

## If You Want Exactly 17-20% Penalty

Three options (choose one):

### Option 1: Adjust rotation_gamma (simplest)
```python
# In unified_benchmark/quantum_solvers.py, gurobi_solver.py, miqp_scorer.py
rotation_gamma = 0.15  # Instead of 0.2 → gives 18% penalty
```

### Option 2: Adjust β
```python
# In unified_benchmark/scenarios.py:build_rotation_matrix
negative_strength = -0.6  # Instead of -0.8 → gives 18% penalty
```

### Option 3: Adjust monoculture multiplier
```python
# In unified_benchmark/scenarios.py:build_rotation_matrix
R[i, j] = negative_strength * 1.125  # Instead of 1.5 → gives 18% penalty
```

## Verification

Run the provided script to see all calculations:
```bash
python verify_beta_parameters.py
```

## Key Takeaway

**The 70% loss you calculated assumes an exponential yield model:**
```
yield = base_yield × exp(R_c,c)  ← NOT our formulation
```

**Our model uses linear objective penalties:**
```
objective = base_benefit + rotation_gamma × R_c,c  ← Our actual formulation
```

**These are fundamentally different models!** The exponential interpretation does not apply to our linear quadratic optimization formulation. Current parameters give 24% objective penalty, which is very close to your 17-20% target and reasonable for this model type.

## References

- **Formulation**: `Latex/crop_rotation.tex` (lines 56-73)
- **Implementation**: `unified_benchmark/scenarios.py` (lines 314-356)
- **Solver usage**: `unified_benchmark/quantum_solvers.py`, `gurobi_solver.py` (rotation_gamma = 0.2)
