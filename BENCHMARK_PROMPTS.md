# LLM Coder Production Loop

This document outlines a series of prompts for an LLM coder to follow. The development process should follow this production loop:

1.  **Analyze**: Understand the user's request and the existing codebase. Use file reading and search tools to analyze the relevant files.
2.  **Code**: Implement the required changes based on the analysis.
3.  **Test**: Run existing or new tests to verify that the changes work as expected and do not introduce regressions.
4.  **Repeat**: Continue this loop for each task until all requirements are met.

## Task List

1.  **Verify GPU Usage for Gurobi**: Ensure that the Gurobi solver is correctly configured to use the GPU in the `benchmark_scalability_PATCH.py` script.
2.  **Enable Gurobi to Solve QUBO Models**: Modify `solver_runner_PATCH.py` to allow Gurobi to solve QUBO models derived from the `cqm_to_bqm` function. Update `benchmark_scalability_PATCH.py` to use this new functionality.
3.  **Create a Comprehensive Benchmark**: Develop a new benchmark script that runs a variety of solvers on two different problem scenarios (farms and patches) and collects performance data.
4.  **Plot Comprehensive Results**: Create a plotting script to visualize the results from the comprehensive benchmark, comparing the performance of all solver configurations.

---

### **Prompt 1: Verify GPU Usage for Gurobi**

**Analyze:**

Read the contents of `benchmark_scalability_PATCH.py` and `solver_runner_PATCH.py`. Look for the section of the code where the Gurobi solver is invoked. Check the Gurobi parameters to see if GPU usage is enabled. The key parameters to look for are `Method=2` (for the barrier method) and `Crossover=0`.

**Code:**

If GPU usage is not enabled, modify the `gurobi_options` in `solver_runner_PATCH.py` to include the following:

```python
gurobi_options = [
    ('Method', 2),           # Barrier method (GPU-accelerated)
    ('Crossover', 0),        # Disable crossover to keep computation on GPU
    ('BarHomogeneous', 1),   # Homogeneous barrier (more GPU-friendly)
    ('Threads', 0),          # Use all available CPU threads for parallelization
    ('MIPFocus', 1),         # Focus on finding good solutions quickly
    ('Presolve', 2),         # Aggressive presolve
]
```

**Test:**

Run the `benchmark_scalability_PATCH.py` script and monitor the system's GPU usage to confirm that Gurobi is utilizing the GPU during the solve process.

---

### **Prompt 2: Enable Gurobi to Solve QUBO Models**

**Analyze:**

Examine the `solver_runner_PATCH.py` file. You will need to add a new function that can take a BQM object, convert it into a Gurobi model, and solve it.

**Code:**

1.  In `solver_runner_PATCH.py`, add a new function `solve_with_gurobi_qubo(bqm)`. This function should:
    *   Accept a `dimod.BinaryQuadraticModel` as input.
    *   Convert the BQM to a Gurobi model. You can do this by iterating over the linear and quadratic terms of the BQM and adding them to a new Gurobi model.
    *   Solve the Gurobi model.
    *   Return the solution and the solve time.

2.  In `benchmark_scalability_PATCH.py`, modify the `run_benchmark` function to:
    *   Call `cqm_to_bqm` to get the BQM.
    *   Call the new `solve_with_gurobi_qubo` function with the BQM.
    *   Record the solve time and results for the Gurobi QUBO solver.

**Test:**

Update the tests in `benchmark_scalability_PATCH.py` to include checks for the Gurobi QUBO solver, ensuring it produces valid results and that its performance is recorded correctly.

---

### **Prompt 3: Create a Comprehensive Benchmark**

**Analyze:**

Review the existing benchmark scripts, `benchmark_scalability_PATCH.py` and `benchmarks.py`, to understand how they are structured. The new benchmark will need to handle two different scenarios and multiple solver configurations.

**Code:**

Create a new file named `comprehensive_benchmark.py`. This script should:

1.  Accept an integer `n_samples` as a command-line argument.
2.  In parallel, call `farm_sampler.py` and `patch_sampler.py` to generate `n_samples` of farms and patches. Store the total area for each set of samples.
3.  Create two scenarios:
    *   **Farm Scenario**: Based on the generated farms.
    *   **Patch Scenario**: Based on the generated patches.
4.  For the **Farm Scenario**:
    *   Run the DWaveCQM solver.
    *   Run PuLP with the Gurobi solver.
    *   Record the classical solve time for PuLP/Gurobi and the hybrid and QPU times for D-Wave.
5.  For the **Patch Scenario**:
    *   Run PuLP and DWaveCQM in CQM form.
    *   Convert the CQM to a BQM.
    *   Run the DWaveBQM solver.
    *   Run the Gurobi QUBO solver.
    *   Record the solve times for all four of these configurations (PuLP, DWaveCQM, DWaveBQM, Gurobi QUBO), including hybrid and QPU times for the D-Wave solvers.
6.  Save all results to a JSON file.

**Test:**

Create a test suite for `comprehensive_benchmark.py` that runs a small number of samples and verifies that the output JSON file contains the correct structure and data for all solver configurations.

---

### **Prompt 4: Plot Comprehensive Results**

**Analyze:**

Examine the plotting scripts `plot_patch_quality_speedup.py` and `plot_patch_speedup.py`. The new plotting script will need to be able to handle the eight different solver configurations from the comprehensive benchmark.

**Code:**

Create a new file named `plot_comprehensive_results.py`. This script should:

1.  Load the JSON data produced by `comprehensive_benchmark.py`.
2.  Use the plotting functions from the existing plotting scripts as a baseline.
3.  Generate plots that show the performance of all eight solver configurations on the same axes. This should include:
    *   A plot of solve times vs. problem size.
    *   A plot of solution quality (objective value) vs. problem size.
    *   A speedup plot comparing the quantum solvers to the classical solvers.
4.  Ensure the plots are clearly labeled, with a legend that distinguishes between all eight lines.

**Test:**

Run the `plot_comprehensive_results.py` script with a sample JSON file and visually inspect the generated plots to ensure they are accurate and easy to understand.
