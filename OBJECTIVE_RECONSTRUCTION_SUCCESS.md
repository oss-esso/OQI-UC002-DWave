# Solution Comparison - Objective Reconstruction

## Date: October 26, 2025

## What We Implemented

Added **objective reconstruction** from BQM solutions to make them comparable with CQM solutions.

### New Function: `calculate_original_objective()`

```python
def calculate_original_objective(solution, farms, foods, land_availability, weights, idle_penalty):
    """
    Calculate the original CQM objective from a solution.
    Reconstructs: sum_{p,c} (B_c + λ) * s_p * X_{p,c}
    """
    objective = 0.0
    for plot in farms:
        s_p = land_availability[plot]
        for crop in foods:
            var_name = f"X_{plot}_{crop}"
            x_pc = solution.get(var_name, 0)
            if x_pc > 0:
                B_c = calculate_weighted_benefit(crop, foods, weights)
                objective += (B_c + idle_penalty) * s_p * x_pc
    return objective
```

## Test Results (Config=5, 5 patches, 0.226 ha)

### Objectives Now Comparable! ✅

| Solver | Objective | Method |
|--------|-----------|---------|
| **Patch_PuLP** | **0.0706** | Direct CQM optimization |
| **Patch_GurobiQUBO** | **0.1355** | BQM → Reconstructed CQM objective |

### Key Observations

1. **Both objectives are now in same range** ✅
   - Before: QUBO was 3.794 (50x too high!)
   - After: QUBO is 0.1355 (comparable scale)

2. **Different solutions** ⚠️
   - PuLP objective: 0.0706
   - Gurobi QUBO objective: 0.1355 (1.9x higher)
   - This means the BQM solver found a **different** (possibly better!) solution

3. **Why different?**
   - BQM formulation may have different local optima
   - Penalty coefficients affect the solution space
   - 30s timeout means QUBO might not be fully optimal

## What This Means

### ✅ Can Now Compare
- **Objective values** across all solvers (same metric!)
- **Solution quality** - which solver finds better allocations
- **Timing vs Quality** trade-offs

### 📊 Next Steps: Visualize Solutions

We should extract and compare:
1. **Y_c values**: Which crops are selected?
2. **X_{p,c} values**: Plot-to-crop assignments
3. **Area allocated**: How much land per crop?
4. **Benefit breakdown**: Which crops contribute most?

## Files Modified

1. **solver_runner_PATCH.py**:
   - Added `calculate_original_objective()` function
   - Updated `solve_with_gurobi_qubo()` to accept config parameters
   - Returns reconstructed objective in results

2. **comprehensive_benchmark.py**:
   - Imports `calculate_original_objective()`
   - Passes config to Gurobi QUBO solver
   - Calculates objective for D-Wave BQM solver
   - Updates result files with reconstructed objectives

## Example Output

```
================================================================================
SOLVING QUBO WITH GUROBI (BQM → GUROBI MODEL)
================================================================================
  BQM Variables: 89
  BQM Linear terms: 89
  BQM Quadratic terms: 381
  
  ✅ Optimal solution found
  BQM Energy: -0.114326          ← BQM space (with penalties)
  Original CQM Objective: 0.135450 ← Reconstructed CQM space! ✅
  Active variables: 20
  Solve time: 3.402 seconds
```

## Result File Structure

```json
{
  "status": "Optimal",
  "objective_value": 0.13545,  // ← Reconstructed CQM objective! ✅
  "bqm_energy": -0.11433,       // ← Original BQM energy
  "solve_time": 3.402,
  "note": "objective_value is reconstructed from BQM solution; comparable to CQM"
}
```

## Validation Questions

### Why is QUBO objective (0.1355) > PuLP objective (0.0706)?

**Possible reasons:**
1. **Different solutions**: BQM found a different local optimum
2. **Better solution**: QUBO might actually be better!
3. **Constraint violations**: Need to check if QUBO satisfies all constraints
4. **Timeout**: 30s might not be enough for convergence

### How to Verify?

Need to check:
1. ✅ **Objective calculation** - Same formula applied
2. ⚠️ **Constraint satisfaction** - Does QUBO solution violate constraints?
3. ⚠️ **Variable assignments** - Compare actual Y_c and X_{p,c} values

## Next Implementation Steps

### 1. Add Constraint Validation
```python
def validate_solution(solution, farms, foods, constraints):
    """Check if BQM solution satisfies original CQM constraints"""
    violations = []
    
    # Check: At most one crop per plot
    for plot in farms:
        assigned = sum(solution.get(f"X_{plot}_{crop}", 0) for crop in foods)
        if assigned > 1:
            violations.append(f"Plot {plot}: {assigned} crops assigned")
    
    # Check: Area bounds
    # Check: Food group constraints
    # etc...
    
    return violations
```

### 2. Extract and Compare Solutions
```python
def extract_solution_summary(solution, farms, foods, land_availability):
    """Extract Y_c and X_{p,c} assignments for visualization"""
    crops_selected = []
    plot_assignments = []
    
    for crop in foods:
        if solution.get(f"Y_{crop}", 0) > 0:
            crops_selected.append(crop)
            total_area = sum(
                solution.get(f"X_{plot}_{crop}", 0) * land_availability[plot]
                for plot in farms
            )
            plot_assignments.append({
                'crop': crop,
                'area': total_area,
                'plots': [plot for plot in farms 
                         if solution.get(f"X_{plot}_{crop}", 0) > 0]
            })
    
    return {'crops': crops_selected, 'assignments': plot_assignments}
```

### 3. Visualization
- Bar chart: Crops selected (Y_c) by each solver
- Stacked bar: Area allocated per crop
- Heat map: Plot-crop assignments (X_{p,c})
- Comparison: PuLP vs QUBO solutions side-by-side

## Summary

**MAJOR ACHIEVEMENT** ✅: We can now compare objectives across all solvers!

**What changed:**
- ❌ Before: BQM energy (3.794) was meaningless
- ✅ After: Reconstructed CQM objective (0.1355) is comparable

**What we learned:**
- Gurobi QUBO found a different solution (0.1355 vs 0.0706)
- Need to validate constraints to see if it's actually better
- Need to visualize solutions to understand differences

**Next steps:**
1. Add constraint validation
2. Extract and compare Y_c, X_{p,c} values
3. Create visualization comparing solutions
4. Run full benchmark with all configs
