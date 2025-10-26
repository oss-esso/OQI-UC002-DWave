# BQM Energy vs CQM Objective - Critical Distinction

## Date: October 26, 2025

## The Problem Discovered

When comparing solver results for `config_5`:

| Solver | Objective Value | Total Area | Per Hectare |
|--------|----------------|------------|-------------|
| Farm_PuLP | 0.778 | 2.49 ha | 0.31 /ha |
| Patch_PuLP | 0.071 | 0.226 ha | 0.31 /ha |
| **Patch_GurobiQUBO** | **3.794** | **0.226 ha** | **16.8 /ha** ❌ |

The QUBO objective was **~50x higher** than the CQM objective! This is completely wrong and makes comparison impossible.

## Root Cause

**BQM (Binary Quadratic Model) energy is NOT the same as CQM (Constrained Quadratic Model) objective!**

### What Happens During CQM → BQM Conversion

When `cqm_to_bqm(cqm)` is called:

1. **Constraints are converted to penalty terms**:
   ```
   Original: x + y <= 10  (hard constraint)
   BQM:      penalty * (x + y - 10)²  (soft penalty, large coefficient)
   ```

2. **Objective is mixed with penalties**:
   ```
   BQM Energy = Original Objective + Σ(Penalty Terms)
   ```

3. **The BQM tries to minimize**:
   - Original objective (small coefficients)
   - Constraint violations (LARGE penalty coefficients)

### Why This Matters

The BQM energy includes:
- ✅ The original objective function
- ❌ **Large penalty terms** for any constraint violations
- ❌ **Slack variables** introduced during conversion
- ❌ **Auxiliary variables** for inequality constraints

**Result**: BQM energy is in a completely different numerical range and cannot be compared to the original CQM objective.

## The Incorrect Code

### Before (WRONG):
```python
# In solve_with_gurobi_qubo()
objective_value = model.ObjVal  # This is BQM energy
bqm_energy = objective_value
solution_objective = -bqm_energy + bqm.offset  # ❌ WRONG! Can't recover CQM objective this way

return {
    'objective_value': solution_objective,  # ❌ Not comparable!
    'bqm_energy': objective_value
}
```

### After (CORRECT):
```python
# In solve_with_gurobi_qubo()
bqm_energy = model.ObjVal  # This is BQM energy (includes penalties)

return {
    'objective_value': None,  # ✅ Explicitly None - not comparable!
    'bqm_energy': bqm_energy,  # ✅ Report BQM energy separately
    'note': 'BQM energy includes penalty terms and is not directly comparable to CQM objective'
}
```

## What We Should Compare

### ✅ Valid Comparisons

**1. Solve Times:**
- How long does each solver take?
- Gurobi on CQM vs Gurobi on BQM vs D-Wave BQM vs D-Wave CQM

**2. Solution Quality (same formulation only):**
- Compare Gurobi CQM with D-Wave CQM (both using CQM)
- Compare Gurobi QUBO with D-Wave BQM (both using BQM)

**3. Solution Feasibility:**
- Does the solution satisfy all original constraints?
- Need to validate BQM solutions against original CQM constraints

### ❌ Invalid Comparisons

**1. Objective Values Across Formulations:**
- ❌ CQM objective vs BQM energy
- ❌ Gurobi CQM objective vs Gurobi QUBO objective

**2. Direct Energy Comparison:**
- BQM energy depends on penalty coefficients chosen during conversion
- Different penalty weights = different energy scales

## How to Properly Compare BQM Solutions

To compare a BQM solution with a CQM solution, you must:

### Option 1: Reconstruct CQM Objective
```python
# Get variable assignments from BQM solution
solution = extract_solution_from_bqm(bqm_result)

# Calculate original CQM objective using those assignments
cqm_objective = evaluate_cqm_objective(solution, original_objective_function)

# Check constraint satisfaction
constraints_satisfied = validate_constraints(solution, original_constraints)
```

### Option 2: Convert CQM Solution to BQM Energy
```python
# Get variable assignments from CQM solution
cqm_solution = pulp_result['solution']

# Evaluate BQM energy for this solution
bqm_energy = bqm.energy(cqm_solution)
```

## Updated Result Files

### New Structure
```json
{
  "status": "Optimal",
  "objective_value": null,  // ← Now explicitly null
  "bqm_energy": -0.11432635268400437,  // ← BQM energy (includes penalties)
  "solve_time": 3.567,
  "note": "BQM energy includes penalty terms and is not directly comparable to CQM objective"
}
```

### What to Report

**For Gurobi QUBO and D-Wave BQM:**
- ✅ `bqm_energy` - The energy in BQM space
- ✅ `solve_time` - How long it took
- ✅ `success` - Whether a solution was found
- ❌ `objective_value` - Set to `null` (not comparable)

**For Gurobi CQM and D-Wave CQM:**
- ✅ `objective_value` - The original CQM objective
- ✅ `solve_time` - How long it took
- ✅ `success` - Whether optimal solution was found

## Implications for Plotting

### What We CAN Plot

1. **Solve Time Comparison:**
   ```python
   # All solvers - compare timing
   plt.bar(['Gurobi CQM', 'D-Wave CQM', 'Gurobi QUBO', 'D-Wave BQM'], 
           [t_gcqm, t_dcqm, t_gqubo, t_dbqm])
   ```

2. **CQM Objectives Only:**
   ```python
   # Only Gurobi CQM and D-Wave CQM
   plt.plot(configs, gurobi_cqm_objectives, label='Gurobi CQM')
   plt.plot(configs, dwave_cqm_objectives, label='D-Wave CQM')
   ```

3. **BQM Energies Only:**
   ```python
   # Only Gurobi QUBO and D-Wave BQM
   plt.plot(configs, gurobi_qubo_energies, label='Gurobi QUBO')
   plt.plot(configs, dwave_bqm_energies, label='D-Wave BQM')
   ```

### What We CANNOT Plot

1. ❌ CQM objectives and BQM energies on same axis
2. ❌ "Solution quality" comparing CQM to BQM
3. ❌ Any "objective value" comparison across formulations

## Quantum Advantage Claims

### ✅ Valid Claims

**Speed Comparison:**
- "D-Wave BQM solves QUBO 10x faster than Gurobi QUBO"
- "Gurobi on CQM is 50x faster than Gurobi on BQM"
- "D-Wave CQM handles constraints better than pure BQM approach"

**Scalability:**
- "Classical QUBO solver timeout increases exponentially"
- "Quantum solver maintains consistent runtime"

### ❌ Invalid Claims

**Solution Quality:**
- ❌ "D-Wave BQM finds better solutions" (can't compare energies directly)
- ❌ "Gurobi QUBO objective is worse" (different formulation)

**Without Validation:**
- Any claim about solution quality requires:
  1. Reconstructing CQM objective from BQM solution, OR
  2. Validating constraint satisfaction

## Recommended Fixes for Future

### Short-term (Implemented)
- ✅ Set `objective_value` to `null` for BQM solvers
- ✅ Add explanatory note to results
- ✅ Report `bqm_energy` separately
- ✅ Update documentation

### Long-term (TODO)
1. **Add objective reconstruction**:
   - Extract variable assignments from BQM solution
   - Calculate original CQM objective
   - Report both BQM energy and CQM objective

2. **Add constraint validation**:
   - Check if BQM solution satisfies original constraints
   - Report violation count and magnitude

3. **Unified comparison**:
   - Convert all results to same space (either CQM or BQM)
   - Enable direct objective comparison

## Summary

**Key Takeaway**: BQM energy and CQM objective are fundamentally different metrics living in different spaces. They cannot and should not be directly compared without proper transformation.

**What We Fixed**:
- Removed misleading "objective_value" from BQM results
- Clearly documented that BQM energy includes penalties
- Set expectations for valid vs invalid comparisons

**What We Can Compare**:
- ✅ Solve times across all solvers
- ✅ CQM objectives (Gurobi CQM vs D-Wave CQM)
- ✅ BQM energies (Gurobi QUBO vs D-Wave BQM)

**What We Cannot Compare (yet)**:
- ❌ CQM objectives vs BQM energies
- ❌ Solution quality across formulations
