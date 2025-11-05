# Instruction-Prompt: Crop Rotation Benchmark Implementation

This document provides instructions for an LLM agent to create a new benchmark for a 3-period crop rotation optimization problem.

## 1. LLM Agent Workflow

Follow this iterative loop for development:
1.  **Read:** Carefully read these instructions and all referenced files to understand the task.
2.  **Analyze:** Analyze the existing code structure, formulations, and the new requirements from the LaTeX document.
3.  **Code:** Write new code and modify existing files as required.
4.  **Run:** Execute the code, run tests, and verify that the implementation is correct and produces the expected output.
5.  **Consider:** Evaluate the results, identify issues or areas for improvement, and plan the next iteration.

## 2. Research and Task Planning

1.  **Task Manager:** Use a task manager to create a detailed, step-by-step task list for implementing the new benchmark. This will help track progress and ensure all requirements are met.
2.  **Context Research:** Use your context research tools to thoroughly analyze the provided files. Create a memory file summarizing the key concepts, variables, objective function, and constraints from `@Latex/crop_rotation.tex`. This is critical for a correct implementation.

## 3. Benchmark and Solver Development Instructions

### Objective

The primary goal is to create a new benchmark and solver runner that implements a 3-period crop rotation model.

-   **New Benchmark File:** `rotation_benchmark.py`
-   **New Solver Runner:** `solver_runner_ROTATION.py`

### Core Task: Implement Crop Rotation Model

The new implementation must be based on the binary (plot-level) formulation described in detail in `@Latex/crop_rotation.tex`.

-   **Time Periods:** The model spans 3 time periods ($t \in \{1, 2, 3\}$).
-   **Decision Variables:** Use plot-level binary assignment variables indexed by time: $Y_{p,c,t} \in \{0,1\}$.
-   **Objective Function:** The objective is to maximize the total area-normalized value, including a quadratic synergy term for crop rotations between consecutive periods.

    $$
    \max\; Z \;=\; \frac{1}{A_{\text{tot}}}\Bigg[
    \sum_{t=1}^3 \sum_{p\in\mathcal F}\sum_{c\in\mathcal C} a_p\,B_c\,Y_{p,c,t}
    \;+\; \gamma
    \sum_{t=2}^3 \sum_{p\in\mathcal F}\sum_{c\in\mathcal C}\sum_{c'\in\mathcal C}
    a_p\,R_{c,c'}\,Y_{p,c,t-1}\,Y_{p,c',t}
    \Bigg]
    $$

-   **Constraints:** All constraints must be enforced on a per-period basis. The most important one is the single-plot assignment:
    $$
    \sum_{c\in\mathcal C} Y_{p,c,t} \;<\; 1 \quad \forall p,\;t.
    $$

### Key Files

-   **Base Benchmark:** `@comprehensive_benchmark.py`
-   **Base Solver Runner:** `@solver_runner_BINARY.py`
-   **Rotation Formulation:** `@Latex/crop_rotation.tex`
-   **Rotation Matrix Generator:** `@rotation_matrix.py`

### Implementation Steps

#### Step 1: Generate the Rotation Matrix

1.  **Inspect `rotation_matrix.py`:** This script generates the crop-to-crop rotation synergy matrix $R$. It saves its output CSV files to `/mnt/data`.
2.  **Modify Path (IMPORTANT):** The path `/mnt/data` may not be suitable. **Edit `rotation_matrix.py`** and change the output directory to a local path, for example: `outdir = Path("./")`.
3.  **Run the script:**
    ```bash
    python rotation_matrix.py
    ```
4.  **Verify Output:** Ensure that `rotation_crop_matrix.csv` is created in the specified directory. This file will be loaded by the new solver runner.

#### Step 2: Create the New Solver Runner

1.  **Copy File:** Duplicate `@solver_runner_BINARY.py` and name it `solver_runner_ROTATION.py`.
2.  **Modify CQM Creation:**
    -   Update the `create_cqm_plots` function (or create a new one like `create_cqm_rotation`) to build the 3-period model.
    -   **Load the Rotation Matrix:** Load the `rotation_crop_matrix.csv` file into a pandas DataFrame or NumPy array.
    -   **Time-Indexed Variables:** The variables `Y` must be indexed by plot, crop, and time period `t`.
    -   **Update Objective:** Rebuild the objective function to match the formula from the `.tex` file, including both the linear value term and the quadratic rotation synergy term. Remember the normalization by $A_{\text{tot}}$
    -   **Update Constraints:** Ensure all constraints, especially the single-plot assignment, are indexed by time period `t` and applied correctly for each period.

#### Step 3: Create the New Benchmark Script

1.  **Copy File:** Duplicate `@comprehensive_benchmark.py` and name it `rotation_benchmark.py`.
2.  **Update Imports:** Modify `rotation_benchmark.py` to import and use the new `solver_runner_ROTATION.py`.
3.  **Update Benchmark Logic:**
    -   The main benchmark loop runs scenarios for different numbers of plots. This structure can be kept.
    -   The functions `run_binary_scenario` should be adapted to call the new CQM creation and solving functions from `solver_runner_ROTATION.py`.
    -   The results processing and saving logic should be updated to handle the output of the 3-period model.

#### Step 4: Testing

1.  Create a new test file `test_rotation_benchmark.py`.
2.  In this file, write a simple test that runs the `rotation_benchmark.py` for a very small instance (e.g., 3 plots, 3 crops).
3.  The test should verify that the benchmark runs without errors and that the output results have the expected structure.
4.  Run the test to validate your implementation.
