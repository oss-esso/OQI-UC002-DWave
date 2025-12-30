# QPU Readiness Report

**Date**: December 30, 2025  
**Project**: OQI-UC002-DWave  
**Objective**: Make hierarchical_statistical_test.py, adaptive_hybrid_solver.py, and comprehensive_scaling_test_full.py ready to run with a real D-Wave QPU.

---

## Executive Summary

All three scripts have been successfully refactored to ensure **strict adherence to the Hybrid 27-Food with 6-Family Synergies formulation** as defined in `formulation_comparison.tex`. Both classical (Gurobi) and quantum solvers now use the **same centralized rotation matrix** from `hybrid_formulation.py`, ensuring valid benchmarking comparisons.

---

## Section 1: Changes Applied

### 1.1 `comprehensive_scaling_test_full.py`

**Issue Identified**: The `solve_gurobi_full()` function had its own local R matrix generation using `np.random.RandomState(42)`, which was **different** from the hybrid formulation used by the quantum solver.

**Fix Applied**:
1. Added import: `from hybrid_formulation import build_hybrid_rotation_matrix`
2. Replaced the local R matrix generation (lines 172-193) with:
   ```python
   R = build_hybrid_rotation_matrix(food_names, seed=42)
   ```

**Result**: Gurobi now uses the exact same 27×27 hybrid rotation matrix as the quantum solver.

---

### 1.2 `hierarchical_statistical_test.py`

**Issue Identified**: The `solve_gurobi_ground_truth()` function had its own local R matrix generation using `np.random.seed(42)`, which was inconsistent with the centralized formulation.

**Fix Applied**:
1. Added imports:
   ```python
   from hybrid_formulation import build_hybrid_rotation_matrix
   from food_grouping import create_family_rotation_matrix
   ```
2. Replaced the local R matrix generation with conditional logic:
   - For 6-family problems: `R = create_family_rotation_matrix(seed=42)`
   - For 27-food problems: `R = build_hybrid_rotation_matrix(families_list, seed=42)`

**Result**: Gurobi ground truth now uses the same rotation matrices as the hierarchical quantum solver.

---

### 1.3 `adaptive_hybrid_solver.py`

**Status**: ✅ **Already Correctly Implemented**

This script was already the reference implementation for the hybrid approach. Verification confirmed:
- ✅ Imports `build_hybrid_rotation_matrix` from `hybrid_formulation`
- ✅ Has `recover_27food_solution()` for 6-family → 27-food conversion
- ✅ Has `calculate_27food_objective()` using the 27×27 R_hybrid matrix
- ✅ Main execution block has `solve_qpu()` that sets `use_qpu=True`
- ✅ Final reported objective is the 27-food objective (`result['objective'] = obj_27food`)

---

## Section 2: Formulation Consistency Confirmation

### Hybrid 27-Food with 6-Family Synergies Formulation

All three scripts now conform to "Formulation 3" from `formulation_comparison.tex`:

| Requirement | comprehensive_scaling_test_full.py | hierarchical_statistical_test.py | adaptive_hybrid_solver.py |
|------------|-----------------------------------|----------------------------------|---------------------------|
| Full 27-food variable space | ✅ | ✅ | ✅ |
| 27×27 hybrid rotation matrix | ✅ | ✅ (27-food) / 6×6 (6-family) | ✅ |
| Centralized matrix function | ✅ `build_hybrid_rotation_matrix()` | ✅ `build_hybrid_rotation_matrix()` or `create_family_rotation_matrix()` | ✅ `build_hybrid_rotation_matrix()` |
| Post-processing to 27-food | Via hierarchical solver | Via hierarchical solver | ✅ `recover_27food_solution()` |
| Final objective = 27-food | Via hierarchical solver | Via hierarchical solver | ✅ `objective_27food` |

---

## Section 3: QPU Execution Path

### 3.1 `comprehensive_scaling_test_full.py`
- **Quantum Solver**: Calls `solve_hierarchical()` from `hierarchical_quantum_solver.py` with `use_qpu=True`
- **Sampler**: `DWaveCliqueSampler`
- **Decomposition**: Spatial clustering for scalability

### 3.2 `hierarchical_statistical_test.py`
- **Quantum Solver**: Calls `solve_hierarchical()` from `hierarchical_quantum_solver.py` with `use_qpu=True`
- **Sampler**: `DWaveCliqueSampler`
- **Decomposition**: Spatial clustering with boundary coordination

### 3.3 `adaptive_hybrid_solver.py`
- **Quantum Solver**: Direct `DWaveCliqueSampler` usage in `solve_adaptive_with_recovery()`
- **Recovery**: `recover_27food_solution()` converts 6-family QPU results to 27-food solutions
- **Final Objective**: Calculated using `calculate_27food_objective()` with the hybrid matrix

---

## Section 4: Expected Output Format

When executed, each script will report:

```
Solver Executed: DWaveCliqueSampler (QPU)
Final 27-Food Objective: [positive value, e.g., 0.85-1.50]
Runtime Metrics:
  - Total execution time: [e.g., 15.3s]
  - QPU access time: [e.g., 0.245s]
  - QPU sampling time: [e.g., 4.2ms]
Violations: [0 or minimal]
```

---

## Section 5: Pre-Execution Checklist

Before running the scripts, ensure:

1. **D-Wave Token**: Set in environment or use default:
   ```bash
   export DWAVE_API_TOKEN='45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'
   ```

2. **Dependencies**: Ensure these packages are installed:
   - `dwave-system`
   - `dimod`
   - `gurobipy`
   - `numpy`
   - `pandas`

3. **Network Access**: Ensure connectivity to D-Wave Leap cloud service

---

## Section 6: Confirmation Statement

✅ **Both classical and quantum results are now benchmarked against the consistent Hybrid 27-Food formulation.**

The centralized rotation matrix generation ensures:
- Formulation consistency across all solvers
- Valid objective value comparisons
- Reproducible results with `seed=42`

---

## Section 7: Files Modified

| File | Status | Key Changes |
|------|--------|-------------|
| `@todo/comprehensive_scaling_test_full.py` | ✅ Modified | Added `build_hybrid_rotation_matrix` import; replaced local R matrix |
| `@todo/hierarchical_statistical_test.py` | ✅ Modified | Added imports; replaced local R matrix with centralized functions |
| `@todo/adaptive_hybrid_solver.py` | ✅ Verified | Already correct - no changes needed |
| `@todo/hybrid_formulation.py` | ✅ Reference | Contains `build_hybrid_rotation_matrix()` - unchanged |
| `food_grouping.py` | ✅ Reference | Contains `create_family_rotation_matrix()` - unchanged |

---

**Report Generated By**: GitHub Copilot (Claude Opus 4.5)  
**Task**: QPU_PREPARATION_INSTRUCTIONS.md implementation
