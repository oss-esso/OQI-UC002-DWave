# Comprehensive QPU Readiness Plan

**Objective**: Make `hierarchical_statistical_test.py`, `adaptive_hybrid_solver.py`, and `comprehensive_scaling_test_full.py` ready to run with a real D-Wave QPU. The final implementation must return realistic objective values and ensure both quantum and classical solvers adhere strictly to the **Hybrid 27-Food with 6-Family Synergies** formulation defined in `formulation_comparison.tex`.

---

## Section 1: Core Principle - Strict Adherence to the Hybrid Formulation

All modifications must conform to "Formulation 3: Hybrid 27-Food with 6-Family Synergies" from `formulation_comparison.tex`. This is the most critical constraint.

### Key Requirements of the Hybrid Formulation:
1.  **Full Variable Space**: The optimization problem must be defined over the full 27 foods. The variable `Y_f,c,t` must represent a specific food `c`, not a family. There should be **no pre-aggregation of the 27-food data** into 6 families before building the model.
2.  **Structured Synergy Matrix**: The rotation synergy matrix `R` must be a **27x27 matrix**. This matrix (`R_hybrid`) is constructed by:
    a. First, creating a `6x6` family template (`R_template`) based on defined rules (monoculture penalty, frustration, positive synergy).
    b. Then, expanding this template to `27x27` by mapping each of the 27 foods to its corresponding family and adding small, food-level noise. This ensures foods in the same family have similar, but not identical, synergy values.
3.  **Solver-Agnostic Formulation**: Both the classical solver (Gurobi) and the quantum solver (whether direct or decomposed) must be benchmarked against this **exact same objective function and `R_hybrid` matrix**.

---

## Section 2: Task 1 - Refactor `comprehensive_scaling_test_full.py` for Formulation Consistency

This script currently contains a major formulation mismatch between the Gurobi and quantum solvers, making their comparison invalid.

### Sub-Task 2.1: Centralize the Hybrid Rotation Matrix Generation
1.  Locate the file `hybrid_formulation.py`, which is already used by `adaptive_hybrid_solver.py`.
2.  Ensure it contains a function `build_hybrid_rotation_matrix(food_names, seed)` that perfectly implements **Algorithm 2** from `formulation_comparison.tex`. The existing function in `adaptive_hybrid_solver.py` is a good reference. This function must return a `27x27` numpy array.

### Sub-Task 2.2: Fix the Gurobi Formulation (`solve_gurobi_full`)
1.  Open `comprehensive_scaling_test_full.py`.
2.  In the `solve_gurobi_full` function, **delete the entire block of code that generates a local `R` matrix** using `np.random.RandomState(42)`. This is the source of the formulation error.
3.  Import `build_hybrid_rotation_matrix` from `hybrid_formulation.py`.
4.  Call `R = build_hybrid_rotation_matrix(food_names, seed=42)` to generate the correct `27x27` hybrid matrix.
5.  Verify that the Gurobi objective function correctly uses this new `R` matrix for the quadratic rotation synergy term, iterating through all 27 foods (`n_foods`).

### Sub-Task 2.3: Align the Quantum Solver (`solve_quantum_qpu`)
1.  The `solve_quantum_qpu` function calls `solve_hierarchical` from `hierarchical_quantum_solver.py`.
2.  You must investigate `hierarchical_quantum_solver.py` and ensure that the BQM it constructs for the QPU also derives its quadratic terms from the **same hybrid formulation**.
3.  **Modification Guideline**:
    *   The hierarchical solver is expected to simplify the problem (e.g., to 6 families) *for the QPU hardware*. However, the objective function it is a proxy for must be the 27-food hybrid one.
    *   Ensure the `solve_hierarchical` function uses `create_family_rotation_matrix` (the `6x6` template) to build its BQM, as this is the basis of the hierarchical approach.
    *   The key is the **post-processing**. After the QPU returns a 6-family solution, it must be expanded back to a 27-food solution, and the final, reported objective value must be calculated using the full `27x27` `R_hybrid` matrix, exactly as specified in `adaptive_hybrid_solver.py`.
4.  Verify that `solve_hierarchical` has been fixed as per `HIERARCHICAL_SOLVER_DEBUG_SUMMARY.md` to correctly handle solution formats and return a final `solution` key.

---

## Section 3: Task 2 - Verify `adaptive_hybrid_solver.py` for QPU Execution

This script appears to be the reference implementation for the hybrid approach. Your task is to validate its QPU execution path and ensure the results are sound.

### Sub-Task 3.1: Enable and Run QPU Mode
1.  In `adaptive_hybrid_solver.py`, locate the main execution block (`if __name__ == '__main__':`).
2.  Ensure the `solve_qpu(data, ...)` function is called. This function sets `use_qpu=True`.
3.  Run the script. The primary goal is to execute the `DWaveCliqueSampler` path successfully.

### Sub-Task 3.2: Confirm Correct Objective Calculation
1.  The script already calculates two objectives: `objective_family` and `objective_27food`.
2.  Confirm that the final, reported objective is `objective_27food`. This value is the "realistic value" the user wants, as it's calculated on the full, un-aggregated 27-food solution using the `27x27` `R_hybrid` matrix.
3.  Log the final `objective_27food` and `violations`. The objective should be positive and violations should be zero or very low.

---

## Section 4: Task 3 - Prepare `hierarchical_statistical_test.py`

The contents of this file are unknown, but it must be updated using the same principles.

### Sub-Task 4.1: Analyze and Refactor
1.  Read the file to identify which solver functions it calls (e.g., a local Gurobi implementation, `solve_hierarchical`, or `solve_adaptive_with_recovery`).
2.  **If it has a Gurobi implementation**: Apply the exact same fix as in **Sub-Task 2.2**. Replace its local `R` matrix generation with a call to the centralized `build_hybrid_rotation_matrix`.
3.  **If it calls a quantum solver**:
    *   Locate the solver call and ensure the `use_qpu=True` flag (or equivalent) is set.
    *   Trace the data flow to ensure that the final reported objective value is the one calculated on the 27-food level, post-recovery, using the `R_hybrid` matrix. If it only reports the 6-family objective, you must add the recovery and final objective calculation steps, mirroring the logic in `adaptive_hybrid_solver.py`.

---

## Section 5: Final Verification and Reporting

After implementing the changes in all three files, perform the following verification steps.

### Sub-Task 5.1: Execute All Scripts
1.  Run `comprehensive_scaling_test_full.py`.
2.  Run `adaptive_hybrid_solver.py`.
3.  Run `hierarchical_statistical_test.py`.

### Sub-Task 5.2: Create a Final Report
For each script, document the following in a `QPU_Readiness_Report.md`:
1.  **Script Name**.
2.  **Solver Executed**: Confirm that it ran on the QPU (e.g., `DWaveCliqueSampler`).
3.  **Final Objective Value**: Report the final **27-food objective**. It must be a positive, realistic number.
4.  **Runtime Metrics**:
    *   Total script execution time.
    *   `qpu_access_time` and `qpu_sampling_time` for quantum runs.
5.  **Violations**: Report the final violation count.
6.  **Confirmation of Formulation**: Add a concluding sentence confirming that "Both classical and quantum results are now benchmarked against the consistent Hybrid 27-Food formulation."
This report will serve as the final deliverable confirming the task is complete.
