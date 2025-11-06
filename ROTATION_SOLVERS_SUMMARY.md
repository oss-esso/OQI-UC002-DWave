# Rotation Solvers Summary

## Overview
The `solver_runner_ROTATION.py` file now contains **7 solver functions**:
- 4 PuLP/Gurobi solvers (2 without rotation, 2 with rotation)
- 2 D-Wave solvers
- 1 Gurobi QUBO solver

## PuLP/Gurobi Solvers

### 1. `solve_with_pulp_farm(farms, foods, food_groups, config)`
**Formulation:** Farm (continuous) WITHOUT rotation

**Variables:**
- `A_{f,c}` - Continuous area [0, land_f]
- `Y_{f,c}` - Binary indicator

**Objective:** Linear crop values only

**Constraints:**
- Land availability per farm
- Linking: A >= min_area × Y, A <= land_f × Y
- Food group min/max (global)

**Use Case:** Single-period farm optimization with flexible areas

---

### 2. `solve_with_pulp_farm_rotation(farms, foods, food_groups, config, gamma=None)`
**Formulation:** Farm (continuous) WITH 3-period rotation

**Variables:**
- `A_{f,c,t}` - Time-indexed continuous area [0, land_f]
- `Y_{f,c,t}` - Time-indexed binary indicator

**Objective:** LINEAR ONLY (rotation synergy NOT included)
- ⚠️ **NOTE:** PuLP cannot handle quadratic terms
- For full quadratic rotation, use D-Wave or Gurobi QUBO

**Constraints:**
- Land availability per farm per period
- Linking per period: A >= min_area × Y, A <= land_f × Y
- Food group min/max per period

**Use Case:** Multi-period farm planning (linear approximation only)

---

### 3. `solve_with_pulp_plots(farms, foods, food_groups, config)`
**Formulation:** Plots (binary) WITHOUT rotation

**Variables:**
- `Y_{p,c}` - Binary assignment

**Objective:** Linear crop values only

**Constraints:**
- Each plot assigned to at most one crop
- Food group min/max (global)

**Use Case:** Single-period plot assignment

---

### 4. `solve_with_pulp_plots_rotation(farms, foods, food_groups, config, gamma=None)`
**Formulation:** Plots (binary) WITH 3-period rotation

**Variables:**
- `Y_{p,c,t}` - Time-indexed binary assignment

**Objective:** LINEAR ONLY (rotation synergy NOT included)
- ⚠️ **NOTE:** PuLP cannot handle quadratic terms
- For full quadratic rotation, use D-Wave or Gurobi QUBO

**Constraints:**
- Each plot assigned to at most one crop per period
- Food group min/max per period

**Use Case:** Multi-period plot assignment (linear approximation only)

---

## D-Wave Solvers

### 5. `solve_with_dwave_cqm(cqm, token)`
**Purpose:** Solve CQM using D-Wave Leap Hybrid CQM Sampler

**Supports:**
- Continuous and binary variables
- Linear AND quadratic objectives ✅
- Full rotation synergy modeling

**Returns:**
- sampleset, hybrid_time, qpu_time

**Use Case:** Full rotation problems with quadratic terms

---

### 6. `solve_with_dwave_bqm(cqm, token)`
**Purpose:** Convert CQM to BQM and solve with D-Wave Leap Hybrid BQM Sampler

**Supports:**
- Binary variables (discretizes continuous)
- Linear AND quadratic objectives ✅
- Better QPU utilization

**Returns:**
- sampleset, hybrid_time, qpu_time, bqm_conversion_time, invert

**Use Case:** Large rotation problems needing more QPU time

---

## Gurobi QUBO Solver

### 7. `solve_with_gurobi_qubo(bqm, ...)`
**Purpose:** Solve BQM using Gurobi's native QUBO solver

**Supports:**
- Binary variables only
- Linear AND quadratic objectives ✅
- GPU acceleration (if available)

**Parameters:**
- `bqm`: Binary Quadratic Model
- `farms`, `foods`, `land_availability`, etc. (optional for validation)
- `time_limit`: Gurobi time limit (default: 100s)

**Returns:**
- Dict with status, solution, objective_value, solve_time, validation

**Use Case:** High-performance QUBO solving with Gurobi

---

## Key Differences: Linear vs Quadratic

| Aspect | PuLP Solvers | D-Wave/Gurobi QUBO |
|--------|--------------|---------------------|
| **Quadratic Terms** | ❌ Not supported | ✅ Fully supported |
| **Rotation Synergy** | ❌ Not included | ✅ Included |
| **Objective Accuracy** | Linear approximation only | Full objective |
| **Solver Type** | Classical LP/MIP | Quantum-hybrid/QUBO |
| **Best For** | Baseline comparison | Production rotation |

---

## Usage Patterns

### Single-Period Optimization

```python
# Farm formulation
model, results = solve_with_pulp_farm(farms, foods, food_groups, config)

# Plots formulation
model, results = solve_with_pulp_plots(farms, foods, food_groups, config)
```

### 3-Period Rotation (Linear Approximation)

```python
# Farm rotation (linear only)
model, results = solve_with_pulp_farm_rotation(
    farms, foods, food_groups, config, gamma=0.1
)

# Plots rotation (linear only)
model, results = solve_with_pulp_plots_rotation(
    farms, foods, food_groups, config, gamma=0.1
)
```

### 3-Period Rotation (Full Quadratic)

```python
# Create rotation CQM
cqm, vars, metadata = create_cqm_farm_rotation_3period(
    farms, foods, food_groups, config, gamma=0.1
)

# Solve with D-Wave CQM
sampleset, hybrid_time, qpu_time = solve_with_dwave_cqm(cqm, token)

# OR solve with D-Wave BQM
sampleset, hybrid_time, qpu_time, conv_time, invert = solve_with_dwave_bqm(cqm, token)

# OR convert to BQM and solve with Gurobi QUBO
bqm, invert = cqm_to_bqm(cqm)
result = solve_with_gurobi_qubo(bqm, farms, foods, food_groups, 
                                land_availability, weights, idle_penalty, config)
```

---

## Solver Selection Guide

### Use PuLP Solvers When:
- Need baseline comparison
- Linear approximation is acceptable
- Want fast classical solving
- Don't need rotation synergy

### Use D-Wave CQM When:
- Need full rotation synergy
- Have continuous variables (farm formulation)
- Want quantum-classical hybrid approach
- Problem size: medium to large

### Use D-Wave BQM When:
- Need full rotation synergy
- All variables are binary (plots formulation)
- Want maximum QPU utilization
- Problem size: large to very large

### Use Gurobi QUBO When:
- Need full rotation synergy
- Have Gurobi license with QUBO support
- Want GPU acceleration
- Need deterministic results

---

## Performance Considerations

### PuLP/Gurobi MIP
- **Speed:** Fast (seconds to minutes)
- **Scalability:** Good (hundreds of variables)
- **Limitation:** No quadratic objectives

### D-Wave CQM
- **Speed:** Moderate (10-60 seconds)
- **Scalability:** Excellent (thousands of variables)
- **Advantage:** Handles continuous + binary + quadratic

### D-Wave BQM
- **Speed:** Moderate (10-60 seconds)
- **Scalability:** Excellent (thousands of variables)
- **Advantage:** More QPU time, better for quadratic

### Gurobi QUBO
- **Speed:** Fast with GPU (seconds)
- **Scalability:** Very good (thousands of variables)
- **Advantage:** GPU acceleration, deterministic

---

## Important Notes

1. **PuLP Limitation:** PuLP solvers do NOT include rotation synergy quadratic terms. They solve only the linear part of the objective.

2. **Gamma Parameter:** Only used by D-Wave and Gurobi QUBO solvers. PuLP solvers accept it for API consistency but don't use it.

3. **Validation:** All solvers return solutions in compatible formats for validation.

4. **Time Limits:** 
   - PuLP: 300s default
   - D-Wave: No explicit limit (Leap handles it)
   - Gurobi QUBO: 100s default (configurable)

5. **Rotation Matrix:** Must exist at `rotation_data/rotation_crop_matrix.csv` for rotation functions.

---

## Complete Solver Mapping

| Problem Type | Formulation | Rotation | Solver Function |
|--------------|-------------|----------|-----------------|
| Single-period | Farm | No | `solve_with_pulp_farm()` |
| Single-period | Plots | No | `solve_with_pulp_plots()` |
| 3-period (linear) | Farm | Yes* | `solve_with_pulp_farm_rotation()` |
| 3-period (linear) | Plots | Yes* | `solve_with_pulp_plots_rotation()` |
| 3-period (full) | Farm | Yes | D-Wave CQM + `create_cqm_farm_rotation_3period()` |
| 3-period (full) | Plots | Yes | D-Wave BQM + `create_cqm_plots_rotation_3period()` |
| 3-period (full) | Any | Yes | Gurobi QUBO (via BQM conversion) |

\* Linear approximation only - rotation synergy not included

