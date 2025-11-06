# Rotation Synergy Linearization - McCormick Relaxation

## Overview

The PuLP rotation solvers now use **McCormick relaxation** to linearize quadratic rotation synergy terms, enabling classical MIP solvers (like Gurobi) to solve the full rotation problem including synergy benefits.

---

## Problem: Quadratic Terms

### Original Quadratic Objective

For rotation, we want to maximize:
```
Objective = Linear_Value + Rotation_Synergy

where:
  Linear_Value = Σ_{f,c,t} B_c * A_{f,c,t}  (or Y_{p,c,t} for plots)
  
  Rotation_Synergy = Σ_{f,c,c',t} gamma * R_{c,c'} * Y_{f,c,t-1} * Y_{f,c',t}
```

**Problem:** PuLP cannot handle the quadratic term `Y_{t-1} * Y_t` directly.

---

## Solution: McCormick Relaxation

### Technique

For each product of two binary variables `Y1 * Y2`, introduce an auxiliary binary variable `Z`:

```
Z = Y1 * Y2
```

This is enforced by three linear constraints:
```
Z <= Y1          (if Y1=0, then Z=0)
Z <= Y2          (if Y2=0, then Z=0)
Z >= Y1 + Y2 - 1 (if both Y1=1 and Y2=1, then Z=1)
```

### Application to Rotation

For each rotation synergy term `Y_{f,c,t-1} * Y_{f,c',t}`:

1. **Create auxiliary variable:**
   ```python
   Z_{f,c,c',t} = Binary variable
   ```

2. **Add McCormick constraints:**
   ```python
   Z_{f,c,c',t} <= Y_{f,c,t-1}
   Z_{f,c,c',t} <= Y_{f,c',t}
   Z_{f,c,c',t} >= Y_{f,c,t-1} + Y_{f,c',t} - 1
   ```

3. **Replace quadratic term in objective:**
   ```python
   # Original (quadratic):
   gamma * R_{c,c'} * Y_{f,c,t-1} * Y_{f,c',t}
   
   # Linearized:
   gamma * R_{c,c'} * Z_{f,c,c',t}
   ```

---

## Implementation Details

### Farm Formulation

**Variables Created:**
- Original: `A_{f,c,t}` (continuous), `Y_{f,c,t}` (binary)
- Auxiliary: `Z_{f,c,c',t}` (binary) for each rotation pair

**Objective:**
```python
goal = Σ B_c * A_{f,c,t}  +  Σ gamma * R_{c,c'} * Z_{f,c,c',t}
       ^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       Linear crop values      Linearized rotation synergy
```

**Constraints:**
- All original constraints (land, linking, food groups)
- Plus 3 × n_rotation_pairs McCormick constraints

### Plots Formulation

**Variables Created:**
- Original: `Y_{p,c,t}` (binary)
- Auxiliary: `Z_{p,c,c',t}` (binary) for each rotation pair

**Objective:**
```python
goal = Σ area_p * B_c * Y_{p,c,t}  +  Σ gamma * area_p * R_{c,c'} * Z_{p,c,c',t}
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       Linear crop values              Linearized rotation synergy
```

**Constraints:**
- All original constraints (plot assignments, food groups)
- Plus 3 × n_rotation_pairs McCormick constraints

---

## Complexity Analysis

### Variable Count

**Without Linearization (Linear Only):**
- Farm: 2 × F × C × 3 = 6FC variables
- Plots: 1 × P × C × 3 = 3PC variables

**With Linearization (Full Rotation):**
- Farm: 6FC + (F × C² × 2) variables
  - Auxiliary Z variables: F × C² for periods 2 and 3
- Plots: 3PC + (P × C² × 2) variables
  - Auxiliary Z variables: P × C² for periods 2 and 3

### Constraint Count

**Additional McCormick Constraints:**
- 3 constraints per Z variable
- Total: 3 × F × C² × 2 (farm) or 3 × P × C² × 2 (plots)

### Example: 10 farms, 5 crops

**Farm formulation:**
- Original: 300 variables (150 continuous + 150 binary)
- With linearization: 300 + 500 = 800 variables (150 continuous + 650 binary)
- Additional constraints: 1,500

**Plots formulation:**
- Original: 150 binary variables
- With linearization: 150 + 500 = 650 binary variables
- Additional constraints: 1,500

---

## Accuracy

### McCormick Relaxation Properties

1. **Exact for Binary Variables:** When Y1, Y2 ∈ {0,1}, the McCormick constraints force Z = Y1 × Y2 exactly.

2. **No Approximation:** This is an **exact** linearization, not an approximation.

3. **Optimality Preserved:** The optimal solution of the linearized problem equals the optimal solution of the original quadratic problem.

---

## Performance Comparison

| Solver | Quadratic Support | Rotation Synergy | Speed | Scalability |
|--------|-------------------|------------------|-------|-------------|
| **PuLP (Original)** | ❌ No | ❌ Not included | Fast | Good |
| **PuLP (Linearized)** | ✅ Yes (via McCormick) | ✅ Included | Moderate | Good |
| **D-Wave CQM** | ✅ Yes (native) | ✅ Included | Moderate | Excellent |
| **D-Wave BQM** | ✅ Yes (native) | ✅ Included | Moderate | Excellent |
| **Gurobi QUBO** | ✅ Yes (native) | ✅ Included | Fast | Very Good |

---

## Usage

### Farm Rotation (Linearized)

```python
from solver_runner_ROTATION import solve_with_pulp_farm_rotation

# Solve with linearized rotation synergy
model, results = solve_with_pulp_farm_rotation(
    farms, foods, food_groups, config, gamma=0.2
)

# Access results
areas = results['areas']  # A_{f,c,t} values
selections = results['selections']  # Y_{f,c,t} values
rotation_pairs = results['rotation_pairs']  # Z_{f,c,c',t} values
objective = results['objective_value']  # Includes rotation synergy!
```

### Plots Rotation (Linearized)

```python
from solver_runner_ROTATION import solve_with_pulp_plots_rotation

# Solve with linearized rotation synergy
model, results = solve_with_pulp_plots_rotation(
    plots, foods, food_groups, config, gamma=0.15
)

# Access results
solution = results['solution']  # Y_{p,c,t} values
rotation_pairs = results['rotation_pairs']  # Z_{p,c,c',t} values
objective = results['objective_value']  # Includes rotation synergy!
```

---

## Key Differences from Linear-Only Version

### Before (Linear Only)

```python
print("NOTE: PuLP solving LINEAR approximation only (rotation synergy not included)")

# Objective only had linear terms
goal = sum(B_c * A_{f,c,t})
# Missing: rotation synergy!
```

### After (Linearized with McCormick)

```python
print("NOTE: PuLP using McCormick relaxation to linearize rotation synergy")

# Objective has linear + linearized quadratic terms
goal = sum(B_c * A_{f,c,t}) + sum(gamma * R_{c,c'} * Z_{f,c,c',t})
#      ^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#      Linear                  Linearized rotation synergy
```

---

## Benefits

✅ **Full Rotation Synergy:** Now included in PuLP solvers
✅ **Exact Solution:** No approximation, optimal solution guaranteed
✅ **Classical Solver:** Can use Gurobi MIP (faster than quantum for small problems)
✅ **Deterministic:** Unlike quantum solvers, always returns same result
✅ **GPU Acceleration:** Gurobi can use GPU for barrier method

---

## When to Use

### Use Linearized PuLP When:
- Problem size: Small to medium (< 1000 variables)
- Need deterministic results
- Have Gurobi license
- Want fast solve times for smaller problems
- Need exact optimal solution

### Use D-Wave/QUBO When:
- Problem size: Large to very large (> 1000 variables)
- Can accept near-optimal solutions
- Want quantum-classical hybrid approach
- Problem is already in QUBO form
- McCormick linearization creates too many variables

---

## Technical References

**McCormick Relaxation:**
- McCormick, G. P. (1976). "Computability of global solutions to factorable nonconvex programs: Part I - Convex underestimating problems"

**Application:**
- Linearization converts MIQP (Mixed-Integer Quadratic Program) to MILP (Mixed-Integer Linear Program)
- MILP is efficiently solved by modern MIP solvers like Gurobi, CPLEX

**Complexity:**
- Adds O(C²) auxiliary variables per farm/plot per period transition
- Adds O(C²) constraints per farm/plot per period transition
- Still polynomial size, scalable for practical problems

---

## Summary

The PuLP rotation solvers now provide **full rotation synergy modeling** through McCormick relaxation:

1. ✅ **Complete Objective:** Linear crop values + quadratic rotation synergy
2. ✅ **Exact Solution:** No approximation, optimal solution guaranteed
3. ✅ **Classical Solver:** Uses proven MIP technology (Gurobi)
4. ✅ **Easy to Use:** Same interface as other solvers
5. ✅ **Production Ready:** Tested and validated

This gives users a choice:
- **Linearized PuLP:** Classical MIP, exact, fast for small-medium problems
- **D-Wave Quantum:** Quantum-hybrid, handles larger problems, stochastic
- **Gurobi QUBO:** Classical QUBO, GPU-accelerated, deterministic

