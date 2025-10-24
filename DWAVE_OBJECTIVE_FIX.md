# DWave Objective Calculation Fix

## Problem Identified

The DWave objectives were vastly different from PuLP objectives due to incorrect objective calculation from BQM solutions.

### Root Cause

When using `cqm_to_bqm()` to convert a CQM to a BQM:
1. The BQM discretizes continuous variables and returns an `invert` function
2. The BQM energy corresponds to the **discretized problem**, not the original CQM
3. Simply using `-best.energy` gives the BQM objective, not the **actual CQM objective**

### Example of the Problem

From your benchmark results:
```
N_Patches    PuLP Obj     DWave Obj    Difference
5            0.093866     -0.024386    Wrong!
10           2.133776     -133.035522  WAY off!
15           3.532290     -5.444623    Wrong!
25           6.355462     -77.234295   Terrible!
```

The DWave objectives are negative when they should be positive, and the magnitudes are completely wrong.

## Solution

### The Correct Approach

To get the actual objective value from a BQM solution:

1. **Use the `invert` function** to convert BQM variables back to CQM variables
2. **Recalculate the objective** using the original CQM formulation with the converted variables

### Implementation

Created a helper function in `benchmark_scalability_PATCH.py`:

```python
def calculate_objective_from_bqm_sample(sample, invert, patches, foods, config):
    """
    Calculate the actual objective value from a BQM solution.
    
    Args:
        sample: BQM sample dictionary
        invert: Invert function from cqm_to_bqm
        patches: List of patch names
        foods: Dictionary of food data
        config: Configuration dictionary
    
    Returns:
        Objective value (maximization)
    """
    # Convert BQM sample back to CQM variables
    cqm_sample = invert(sample)
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    idle_penalty = params.get('idle_penalty_lambda', 0.1)
    
    # Calculate objective: sum_{p,c} (B_c + λ) * s_p * X_{p,c}
    objective = 0.0
    
    for plot in patches:
        s_p = land_availability[plot]
        
        for crop in foods:
            # Get X_{p,c} value from sample
            x_var_name = f"X_{plot}_{crop}"
            x_val = cqm_sample.get(x_var_name, 0)
            
            # Calculate B_c (weighted benefit for crop c)
            B_c = (
                weights['nutritional_value'] * foods[crop]['nutritional_value'] +
                weights['nutrient_density'] * foods[crop]['nutrient_density'] +
                weights['environmental_impact'] * (1 - foods[crop]['environmental_impact']) +
                weights['affordability'] * foods[crop]['affordability'] +
                weights['sustainability'] * foods[crop]['sustainability']
            )
            
            # Add contribution
            objective += (B_c + idle_penalty) * s_p * x_val
    
    return objective
```

### Changes Made

1. **`benchmark_scalability_PATCH.py`**:
   - Added `calculate_objective_from_bqm_sample()` function
   - Fixed DWave objective calculation (line ~309):
     ```python
     # OLD (WRONG):
     dwave_objective = -best.energy
     
     # NEW (CORRECT):
     dwave_objective = calculate_objective_from_bqm_sample(
         best.sample, invert, patches, foods, config
     )
     ```
   - Fixed Simulated Annealing objective calculation similarly

2. **TODO: Fix `solver_runner_PATCH.py`**:
   - Lines 608, 632: Need to use the same approach
   - Will need to pass `patches`, `foods`, `config` to calculate correct objectives

## Why This Matters

### For Benchmarking
- **Solution Quality**: Can now correctly compare DWave and PuLP solution quality
- **Time-to-Quality**: Accurate objectives enable proper TTQ metrics
- **Trust**: Results are now scientifically valid

### For Production Use
- **Verification**: Can verify that DWave solutions are actually good
- **Debugging**: Can identify if poor performance is due to the solver or the formulation
- **Optimization**: Can tune parameters based on actual objective values

## Next Steps

1. ✅ Fixed `benchmark_scalability_PATCH.py`
2. ⏳ Need to fix `solver_runner_PATCH.py` 
3. ⏳ Re-run benchmarks with corrected objective calculation
4. ⏳ Verify that DWave objectives now match PuLP (or are close)
5. ⏳ Update any cached results or regenerate benchmark cache

## Technical Notes

### Why Not Just Use `-best.energy`?

The BQM energy includes:
- Penalty terms from constraint violations in the discretized problem
- Discretization artifacts from converting continuous variables
- The objective value in the BQM's native representation

These don't directly correspond to the original CQM objective. You must:
1. Invert back to CQM space
2. Evaluate the original objective function

### Variable Name Convention

In the BQM, variables are named as strings like `"X_patch_1_Wheat"`.
The `invert()` function maps these back to the original CQM variable names.

## References

- DWave Ocean SDK: `dimod.cqm_to_bqm()`
- CQM to BQM conversion: https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/cqm.html
