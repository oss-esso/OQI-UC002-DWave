# Decomposition Strategies Implementation Memory

**Created**: November 21, 2025  
**Purpose**: Technical reference for all decomposition strategy implementations

---

## 🎯 OVERVIEW

This document provides implementation details for multiple decomposition strategies applied to the Farm scenario MINLP problem.

### Problem Structure (Farm Scenario)

**Variables**:
- `A[f,c]`: Continuous area allocation (675 variables)
- `Y[f,c]`: Binary selection indicator (675 variables)
- Total: 1350 variables

**Constraints**:
- Land availability: 25
- Min planting area: 675
- Max planting area: 675
- Food group min/max (count-based): 10
- Total: ~1385 constraints

**Objective**:
```
maximize: (1/100) * Σ_{f,c} A[f,c] * benefit[c]
```

---

## 🔧 STRATEGY 1: Current Hybrid (Gurobi Relaxation + QPU Binary)

### Algorithm

```
Phase 1: Continuous Relaxation
  - Relax Y ∈ {0,1} → Y ∈ [0,1]
  - Solve MINLP with Gurobi → get A*, Y_relaxed

Phase 2: Binary Subproblem
  - Fix A = A* from Phase 1
  - Create CQM with only binary Y ∈ {0,1}
  - Objective: max Σ benefit[c] * A*[f,c] * Y[f,c]
  - Constraints: Food group count constraints on Y

Phase 3: Quantum Solving
  - Convert CQM → BQM
  - Solve on QPU (or SimulatedAnnealing)
  - Get Y**

Phase 4: Combine
  - Final solution: (A*, Y**)
```

### Implementation Location
- File: `solver_runner_DECOMPOSED.py`
- Function: `solve_farm_with_hybrid_decomposition()`

### Advantages
- ✅ Leverages Gurobi for continuous optimization
- ✅ Uses QPU for binary combinatorics
- ✅ Reduced binary subproblem size

### Disadvantages
- ❌ Two-stage, no iteration
- ❌ A* fixed in stage 2, may not be optimal for discrete Y

---

## 🔧 STRATEGY 2: Benders Decomposition

### Mathematical Formulation

**Master Problem** (MILP over Y):
```
min: η + penalties
s.t.: η ≥ c^T * y + optimal_dual_value(y)  [Benders cuts]
      food_group_constraints(Y)
      Y ∈ {0,1}^{675}
```

**Subproblem** (LP over A, given Y*):
```
max: Σ benefit[c] * A[f,c]
s.t.: Σ_c A[f,c] ≤ L[f]              [land availability]
      A[f,c] ≥ M[c] * Y*[f,c]        [min planting]
      A[f,c] ≤ L[f] * Y*[f,c]        [max planting]
      A[f,c] ≥ 0
```

### Benders Cut Generation

After solving subproblem with dual variables π:
```
Optimality Cut:
  η ≥ Σ_{f,c} benefit[c] * A*[f,c] + 
      Σ_f π_land[f] * (L[f] - Σ_c A*[f,c]) +
      Σ_{f,c} π_min[f,c] * (A*[f,c] - M[c]*Y[f,c])
```

### Algorithm

```python
def benders_decomposition(farms, foods, config, max_iter=50, gap_tol=1e-4):
    # Initialize
    lower_bound = -inf
    upper_bound = +inf
    iteration = 0
    cuts = []
    
    while iteration < max_iter and (upper_bound - lower_bound) > gap_tol:
        # Step 1: Solve Master Problem
        Y_star = solve_master(cuts)
        
        # Step 2: Solve Subproblem given Y_star
        A_star, dual_vars, subproblem_obj = solve_subproblem(Y_star)
        
        # Step 3: Update bounds
        lower_bound = max(lower_bound, master_obj)
        upper_bound = min(upper_bound, subproblem_obj)
        
        # Step 4: Generate Benders cut
        cut = generate_benders_cut(Y_star, A_star, dual_vars)
        cuts.append(cut)
        
        iteration += 1
    
    return A_star, Y_star, upper_bound
```

### Implementation Notes
- Master solved with Gurobi (MILP)
- Subproblem solved with Gurobi (LP)
- Purely classical approach
- Expected iterations: 5-20 for convergence

---

## 🔧 STRATEGY 3: Dantzig-Wolfe Decomposition

### Mathematical Formulation

**Restricted Master Problem**:
```
max: Σ_k λ_k * obj_k
s.t.: Σ_k λ_k * pattern_k ≤ capacity    [coupling constraints]
      Σ_k λ_k = 1
      λ_k ≥ 0
```

**Pricing Subproblem** (given dual prices):
```
max: (benefit - dual_prices)^T * (A, Y)
s.t.: farm-level constraints
      A, Y feasible
```

### Algorithm

```python
def dantzig_wolfe(farms, foods, config, max_iter=100):
    # Initialize with basic feasible solution
    columns = [generate_initial_column()]
    
    iteration = 0
    while iteration < max_iter:
        # Step 1: Solve Restricted Master
        lambda_star, dual_prices = solve_restricted_master(columns)
        
        # Step 2: Solve Pricing Subproblem
        new_column, reduced_cost = solve_pricing(dual_prices)
        
        # Step 3: Check optimality
        if reduced_cost <= tolerance:
            break  # Optimal
        
        # Step 4: Add column to master
        columns.append(new_column)
        iteration += 1
    
    # Reconstruct solution
    A, Y = reconstruct_solution(columns, lambda_star)
    return A, Y
```

### Implementation Notes
- Master: LP (relaxed) or MILP (with integrality)
- Pricing: Complex subproblem (could use QPU for hard instances)
- Column pool grows over iterations

---

## 🔧 STRATEGY 4: ADMM (Alternating Direction Method of Multipliers)

### Mathematical Formulation

Split the problem into two blocks:

**Original Problem**:
```
max: f(A) + g(Y)
s.t.: A, Y satisfy coupling constraints
```

**ADMM Form** (with consensus variable Z):
```
max: f(A) + g(Y)
s.t.: A = Z
      Y = Z
```

### ADMM Iterations

```
Iteration k:
  1. A-update: A^{k+1} = argmax_A { f(A) - ρ/2 ||A - Z^k + U^k||² }
  2. Y-update: Y^{k+1} = argmax_Y { g(Y) - ρ/2 ||Y - Z^k + V^k||² }
  3. Z-update: Z^{k+1} = (A^{k+1} + Y^{k+1} + U^k + V^k) / 2
  4. Dual update: U^{k+1} = U^k + A^{k+1} - Z^{k+1}
                  V^{k+1} = V^k + Y^{k+1} - Z^{k+1}
```

### Algorithm

```python
def admm_decomposition(farms, foods, config, max_iter=100, rho=1.0):
    # Initialize
    A = initial_A()
    Y = initial_Y()
    Z = (A + Y) / 2
    U = zeros_like(A)
    V = zeros_like(Y)
    
    for iteration in range(max_iter):
        # Step 1: Solve A-subproblem (continuous)
        A = solve_A_subproblem(Z, U, rho)
        
        # Step 2: Solve Y-subproblem (binary) - QPU candidate!
        Y = solve_Y_subproblem(Z, V, rho)
        
        # Step 3: Consensus update
        Z = (A + Y + U + V) / 2
        
        # Step 4: Dual update
        U = U + A - Z
        V = V + Y - Z
        
        # Check convergence
        primal_residual = norm(A - Z) + norm(Y - Z)
        dual_residual = rho * norm(Z - Z_old)
        
        if primal_residual < tol and dual_residual < tol:
            break
    
    return A, Y
```

### Implementation Notes
- A-subproblem: Quadratic program (Gurobi)
- Y-subproblem: QUBO (potential QPU acceleration!)
- Parameter ρ: controls convergence (tune empirically)

---

## 📊 REFERENCE JSON OUTPUT FORMAT

Based on attached `config_25_run_1.json`:

```json
{
  "status": "Optimal" | "Infeasible" | "Suboptimal",
  "objective_value": 1.6448,
  "hybrid_time": 61.015,
  "qpu_time": 0.0696,
  "is_feasible": "True" | "False",
  "num_samples": 1000,
  "success": "True" | "False",
  "sample_id": 2,
  "n_units": 25,
  "total_area": 100.0,
  "n_foods": 27,
  "n_variables": 1350,
  "n_constraints": 1385,
  "total_covered_area": 100.0,
  "solution_plantations": {
    "Y_Farm0_Apple": 0.0,
    "A_Farm0_Apple": 0.0,
    ...  // All Y and A variables
  },
  "validation": {
    "land_constraints": {
      "Farm0": {"allocated": 4.0, "capacity": 4.0, "satisfied": true},
      ...
    },
    "food_group_constraints": {
      "Fruits": {"count": 5, "min": 3, "max": 10, "satisfied": true},
      ...
    },
    "all_constraints_satisfied": true
  }
}
```

### Required Fields for All Strategies

**Core**:
- `status`: Solution status
- `objective_value`: Final objective
- `is_feasible`: Constraint satisfaction
- `success`: Whether solver completed

**Timing**:
- `hybrid_time`: Total solve time
- `qpu_time`: QPU access time (0 for classical)
- `num_samples`: Number of samples/iterations

**Problem**:
- `n_units`, `total_area`, `n_foods`
- `n_variables`, `n_constraints`
- `total_covered_area`

**Solution**:
- `solution_plantations`: Full variable mapping
- `validation`: Detailed constraint checks

---

## 🔍 DEBUGGING GUIDE

### Common Issues

**Issue 1: Infeasible Subproblems**
- Check constraint bounds (min/max food groups)
- Verify Y* from master is valid
- Add slack variables for debugging

**Issue 2: Slow Convergence**
- Tune penalty parameters (ρ for ADMM)
- Adjust cut aggressiveness (Benders)
- Add warm-start solutions

**Issue 3: BQM Conversion Failures**
- Reduce constraint complexity
- Use penalty method instead of CQM
- Increase discretization precision

### Logging Strategy

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Log each iteration
logger.info(f"Iteration {k}: LB={lb:.4f}, UB={ub:.4f}, Gap={gap:.2%}")

# Log timing
logger.debug(f"Master solve: {master_time:.3f}s")
logger.debug(f"Subproblem solve: {sub_time:.3f}s")
```

---

## 📈 EXPECTED PERFORMANCE

### Solve Time Estimates (Config 25)

| Strategy | Expected Time | Iterations |
|----------|---------------|------------|
| Current Hybrid | 60-80s | N/A (2-phase) |
| Benders | 20-60s | 5-20 |
| Dantzig-Wolfe | 30-90s | 10-50 |
| ADMM | 40-120s | 20-100 |

### Solution Quality

All strategies should reach similar objective values (within 1-2% optimality gap).

---

## 🚀 IMPLEMENTATION CHECKLIST

For each new strategy:

- [ ] Create `decomposition_<name>.py`
- [ ] Implement main solving function
- [ ] Add to strategy factory
- [ ] Create unit tests
- [ ] Run on config 10 (quick test)
- [ ] Run on config 25 (full benchmark)
- [ ] Verify JSON output format
- [ ] Document in this file
- [ ] Update LaTeX chapter

---

**Last Updated**: November 21, 2025, 12:00 PM  
**Status**: Reference document created
