# Fixes for LQ Solver Validation and Pyomo Small Values

## Date: 2025-11-17

## Problems Identified

### 1. Pyomo Using Wrong Solver (IPOPT)
**Problem**: Pyomo was using IPOPT, which is a **continuous NLP solver** that treats binary variables as continuous values in [0, 1]. This resulted in all variable values being tiny decimals (1e-8 to 1e-10) instead of proper binary 0/1 values.

**Root Cause**: The solver priority list had IPOPT first, which is inappropriate for Mixed-Integer Quadratic Programming (MIQP) problems.

**Fix**: Changed solver priority to prefer MIQP-capable solvers:
1. Gurobi (commercial MIQP - best)
2. CPLEX (commercial MIQP - excellent)
3. SCIP (open-source MIQCP - good)
4. CBC/GLPK (MIP only, will linearize quadratic terms)
5. **Excluded IPOPT** with clear error message explaining why

### 2. Model Formulation Bug - Y Can Be 1 When A Is 0
**Problem**: The linking constraints allowed binary selection variables (Y) to be set to 1 even when area (A) was 0:
- Constraint: `A >= min_area * Y` 
- When `min_area = 0` (which was the case for all crops), this becomes `A >= 0 * Y = 0`
- This allows Y=1 when A=0, which is incorrect

**Impact**: 
- The quadratic synergy bonus term (`synergy_bonus_weight * Y[crop1] * Y[crop2]`) is POSITIVE
- Gurobi maximized synergy bonuses by setting ALL Y=1, even when A=0
- This resulted in incorrect solutions claiming all crops were selected

**Fix**: Added epsilon constraint to minimum area:
```python
A_min = min_planting_area.get(c, 0)
if A_min == 0:
    A_min = 0.001  # 0.001 hectares = 10 square meters minimum
```

This ensures `A >= 0.001 * Y`, which forces Y=0 when A=0.

### 3. Solution Dictionary Key Mismatch
**Problem**: The `extract_solution_summary()` function expected dictionary keys with prefixes:
- `A_Farm1_Wheat` for area variables
- `Y_Farm1_Wheat` for selection variables

But the code was creating a single dictionary with keys like `Farm1_Wheat` that appeared twice (overwriting each other when merging `areas` and `selections`).

**Fix**: Changed solution dictionary creation to use proper prefixes:
```python
solution = {}
for key, val in areas.items():
    solution[f"A_{key}"] = val
for key, val in selections.items():
    solution[f"Y_{key}"] = val
```

### 4. Validation Logic Enhancement
**Problem**: Validation was checking `y_val > 0.5` to determine if a crop was selected, but:
- Needed to handle near-binary values (0.1-0.9) as violations
- Food group constraints were summing fractional Y values instead of counting selections

**Fix**: 
- Enhanced linking constraint validation to detect non-binary Y values (0.1 <= Y <= 0.9) as violations
- Changed food group validation to count selections properly: `sum(1 if Y > 0.9 else 0)`

## Results

### Before Fixes
- **Pyomo with IPOPT**: All variables had tiny values (1e-8 to 1e-10)
- **Solution summary**: Reported 0 crops selected despite positive area allocation
- **Validation**: Failed with food group constraint violations
- **Binary variables**: Not truly binary (fractional values like 0.000000052)

### After Fixes
- **Pyomo with Gurobi**: Proper binary values (exactly 0.0 or 1.0)
- **Solution summary**: Correctly reports actual crops selected
- **Validation**: Passes all constraints
- **Objective values**: Identical between PuLP and Pyomo (73.125000 for simple scenario)

## Testing
Tested with 'simple' scenario:
- **PuLP**: 73.125000 objective, 1 crop (Soybeans) on 3 farms, validation PASSED
- **Pyomo**: 73.125000 objective, 1 crop (Soybeans) on 3 farms, validation PASSED

## Files Modified
- `Benchmark Scripts/solver_runner_LQ.py`
  - Fixed Pyomo solver selection (lines ~730-760)
  - Added epsilon to minimum area constraints (lines ~700 and ~368)
  - Fixed solution dictionary creation (lines ~848 and ~448)
  - Enhanced validation logic (lines ~520-560)
  - Improved output formatting (removed problematic Unicode characters)

## Recommendations
1. **Always use Gurobi or CPLEX** for MIQP problems when available
2. **Set minimum planting areas** in scenario configurations to avoid the epsilon workaround
3. **Validate binary variable values** after solving to ensure solver enforced constraints properly
4. **Test with multiple solvers** to ensure model formulation is robust
