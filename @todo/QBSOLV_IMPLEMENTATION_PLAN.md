# QBSolv Integration Plan

## Status: ✅ COMPLETED (2024-12-04)

## 1. Objective

This document outlines the strategy for integrating the `qbsolv` algorithm into this project. The goal is to leverage the official, albeit discontinued, `qbsolv` implementation to solve large Quadratic Unconstrained Binary Optimization (QUBO) problems efficiently, rather than re-implementing the algorithm from scratch.

## 2. Key References

*   **Technical Paper:** [Partitioning Optimization Problems for Hybrid Classical/Quantum Execution](https://www.dwavesys.com/sites/default/files/partitioning_QUBOs_for_hybrid_exec-web.pdf) (Booth, Reinhardt, Roy, 2017)
*   **Official GitHub Repository:** [https://github.com/dwavesystems/qbsolv](https://github.com/dwavesystems/qbsolv)

## 3. Analysis of the `dwavesystems/qbsolv` Repository

A review of the official GitHub repository reveals the following:

*   **Hybrid Implementation:** The core solver is written in C/C++ for high performance, exactly as implied in the technical paper.
*   **Python Bindings:** The repository includes Python bindings, but they are **deprecated and incompatible with Python 3.10+** due to old dependencies (dimod<0.11.0, numpy incompatibility).
*   **Build System:** The C/C++ core is built using `cmake` and `make`. 
*   **Solution:** We use the compiled C/C++ executable via subprocess calls instead of the deprecated Python bindings.

## 4. Implementation (Completed)

### Step 1: Clone the `qbsolv` Repository ✅

```bash
mkdir -p external
git clone https://github.com/dwavesystems/qbsolv.git external/qbsolv
```

### Step 2: Build the C/C++ Core Library ✅

```bash
cd external/qbsolv
mkdir -p build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make
```

The compiled executable is located at: `external/qbsolv/build/qbsolv`

### Step 3: Python Wrapper (Alternative Approach) ✅

Instead of using the deprecated Python bindings (which have dependency conflicts with Python 3.10+), we created a subprocess-based wrapper that communicates with the compiled executable via temporary files.

**File Created: `src/qbsolv_runner.py`**

Key features:
- `solve_qubo_with_qbsolv()`: Main function to solve QUBO problems
- `solve_qubo_dict_with_qbsolv()`: Convenience function for dict-format QUBOs
- `QBSolvSampler`: dimod-compatible sampler class for easy integration
- Supports scipy sparse matrices, numpy arrays, and dictionary formats
- Configurable parameters: `num_repeats`, `subproblem_size`, `timeout`, `target`, `seed`, etc.

### Step 4: Testing ✅

All tests pass:
- Simple 2-variable QUBO: ✅
- Dictionary format QUBO: ✅
- QBSolvSampler class: ✅
- 100-variable random QUBO: ✅

## 5. Usage Examples

### Basic Usage

```python
from scipy.sparse import dok_matrix
from src.qbsolv_runner import solve_qubo_with_qbsolv, QBSolvSampler

# Create a QUBO
Q = dok_matrix((3, 3))
Q[0, 0] = -1
Q[1, 1] = -1
Q[2, 2] = -1
Q[0, 1] = 2
Q[1, 2] = 2

# Solve it
energy, solution = solve_qubo_with_qbsolv(Q, num_repeats=50, seed=42)
print(f"Energy: {energy}, Solution: {solution}")
```

### Using the Sampler Class

```python
from src.qbsolv_runner import QBSolvSampler

sampler = QBSolvSampler(num_repeats=50)
result = sampler.sample_qubo({(0,0): -1, (1,1): -1, (0,1): 2})
print(f"Best energy: {result['energy']}")
print(f"Best solution: {result['sample']}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_repeats` | 50 | Times to repeat main loop after finding new optimal |
| `subproblem_size` | 47 | Size of subproblems for partitioning |
| `timeout` | None | Timeout in seconds |
| `target` | None | Target energy to stop when found |
| `seed` | None | Random seed for reproducibility |
| `maximize` | False | Find maximum instead of minimum |
| `algorithm` | 'o' | 'o' for original, 'd' for diversity-based |

## 6. Files Created/Modified

- `external/qbsolv/` - Cloned qbsolv repository
- `external/qbsolv/build/qbsolv` - Compiled executable
- `src/qbsolv_runner.py` - Python wrapper module

## 7. Next Steps (Optional Improvements)

1. Integrate `solve_qubo_with_qbsolv` into existing optimization workflows
2. Add support for D-Wave QPU as subproblem solver (requires D-Wave credentials)
3. Benchmark against existing solvers in the project
4. Add more comprehensive tests for edge cases
