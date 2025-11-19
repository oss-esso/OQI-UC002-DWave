# LLM Agent Master Prompt: Advanced Hybrid Quantum-Classical Benchmark Implementations

## 1. Meta-Instructions: Your Agent Persona and Workflow

**Your Role:** You are an expert AI software engineer specializing in quantum computing and optimization. Your mission is to implement complex benchmarking scripts with flawless execution and adherence to the highest coding standards. You think step-by-step, test your work rigorously, and document your process.

**Your Prime Directive:** Your primary goal is to produce two new, fully functional benchmark implementations based on the instructions below. The code must be robust, well-documented, and integrate seamlessly with the project's existing structure and style.

**Your Iterative Development Loop:**
1.  **Plan:** Before writing any code, create a detailed, step-by-step plan for each implementation. You can use a temporary file like `dev_plan.md` to outline your strategy, ensuring you have a clear path forward.
2.  **Implement:** Write the code methodically according to your plan, creating the new files with the exact names specified.
3.  **Test Incrementally:** Do not wait until the end to test. Create and run small unit tests or simple execution scripts to verify each new function as you build it. For example, test a CQM creation function before attempting to solve it. This is critical for managing complexity.
4.  **Execute & Debug:** Run the full benchmark scripts for a small problem size (e.g., `n_units=10`) to ensure they complete without errors. When errors occur, analyze the traceback, read the relevant code, form a hypothesis, and then fix the bug. Repeat execution until the benchmark runs successfully.
5.  **Finalize:** Ensure all code is commented appropriately, follows professional software engineering standards (e.g., IEEE), and that the final JSON outputs are structured correctly as specified.

**Critical Safety Protocol:**
*   **No Hardcoded Secrets:** You **must not** hardcode sensitive information. Use the exact placeholder string `'YOUR_DWAVE_TOKEN_HERE'` for the D-Wave API token. The user will replace this manually.
*   **Strict File Naming:** Adhere strictly to the filenames specified in these instructions.

---

## 2. High-Level Objective

Your mission is to create two advanced alternative implementations of the existing benchmark suite (`@Benchmark Scripts/comprehensive_benchmark.py` and `@Benchmark Scripts/solver_runner_BINARY.py`). These new implementations will explore more sophisticated hybrid computing techniques.

*   **Alternative 1:** A custom hybrid workflow using the `dwave-hybrid` framework.
*   **Alternative 2:** A strategic problem decomposition workflow, separating classical and quantum computational tasks.

---

## 3. Alternative Implementation 1: Custom Hybrid Workflow

### Objective
Implement a benchmark using a bespoke hybrid workflow inspired by the `Kerberos` sampler. This tests the performance of a manually constructed hybrid algorithm against the generic `LeapHybrid...` solvers provided by D-Wave.

### Target Files to Create
*   `@todo/solver_runner_CUSTOM_HYBRID.py`
*   `@todo/comprehensive_benchmark_CUSTOM_HYBRID.py`

### Core Logic & Inspiration
*   Thoroughly study the patterns in `@DWaveNotebooks/02-hybrid-computing-workflows.ipynb` and `@DWaveNotebooks/03-hybrid-computing-components.ipynb`. Your goal is to replicate the structure of a real-world hybrid algorithm: decompose the problem, solve the hard part on the QPU, and iterate to refine the solution.

### Implementation Steps

1.  **Create `solver_runner_CUSTOM_HYBRID.py`:**
    *   Copy the contents of `@Benchmark Scripts/solver_runner_BINARY.py` into this new file.
    *   Add the following imports from the `dwave-hybrid` library:
        ```python
        from hybrid import Loop, Race, ArgMin, EnergyImpactDecomposer, SplatComposer
        from hybrid.samplers import QPUSubproblemAutoEmbeddingSampler, TabuProblemSampler
        from hybrid.core import State
        from dwave.system import DWaveSampler
        ```
    *   Create a new solver function: `def solve_with_custom_hybrid_workflow(cqm, token, **kwargs):`.

2.  **Build the Custom Workflow inside `solve_with_custom_hybrid_workflow`:**
    *   **Define a QPU Branch:** This branch finds the most difficult part of the problem and sends it to the QPU.
        ```python
        qpu_branch = (
            EnergyImpactDecomposer(size=50, rolling_history=0.3) |
            QPUSubproblemAutoEmbeddingSampler(qpu_sampler=DWaveSampler(token=token), num_reads=100) |
            SplatComposer()
        )
        ```
    *   **Define a Classical Branch:** This branch runs a fast classical heuristic on the whole problem.
        ```python
        classical_branch = TabuProblemSampler(num_reads=1, timeout=200)
        ```
    *   **Combine in a Race:** Run both branches in parallel and take the best result from whichever finishes first.
        ```python
        iteration = Race(qpu_branch, classical_branch) | ArgMin()
        ```
    *   **Wrap in a Loop:** Iterate this process to progressively improve the solution. The loop will stop after 10 iterations or if the solution stops improving for 3 consecutive iterations.
        ```python
        workflow = Loop(iteration, max_iter=10, convergence=3)
        ```
    *   **Execute the Workflow:**
        ```python
        initial_state = State.from_problem(cqm)
        final_state = workflow.run(initial_state).result()
        sampleset = final_state.samples
        ```
    *   **Return Results:** Extract the final sampleset, energy, and timing information. Return a results dictionary compatible with the benchmark script's expected format.

3.  **Create `comprehensive_benchmark_CUSTOM_HYBRID.py`:**
    *   Copy `@Benchmark Scripts/comprehensive_benchmark.py` into this new file.
    *   Modify the script to `import solver_runner_CUSTOM_HYBRID as solver_runner`.
    *   In the `run_farm_scenario` and `run_binary_scenario` functions, **replace the calls to `solve_with_dwave_cqm`** with calls to your new `solve_with_custom_hybrid_workflow`.
    *   Adjust the result processing logic to handle the output from your new custom solver. Ensure the solver name in the final JSON is clearly identified, e.g., `"dwave_custom_hybrid"`.

---

## 4. Alternative Implementation 2: Strategic Problem Decomposition

### Objective
Implement a benchmark that manually decomposes the problem. The continuous "farm" problem is solved purely classically (with Gurobi), while the purely binary "patch" problem is converted to a BQM and submitted directly to a low-level D-Wave sampler for a more direct quantum approach. This tests a "divide and conquer" strategy.

### Target Files to Create
*   `@todo/solver_runner_DECOMPOSED.py`
*   `@todo/comprehensive_benchmark_DECOMPOSED.py`

### Implementation Steps

1.  **Create `solver_runner_DECOMPOSED.py`:**
    *   Copy `@Benchmark Scripts/solver_runner_BINARY.py` into this new file.
    *   Add these imports:
        ```python
        from dwave.system import DWaveSampler, EmbeddingComposite
        ```
    *   Create a new function: `def solve_with_decomposed_qpu(bqm, token, **kwargs):`.

2.  **Implement the Decomposed QPU Solver:**
    *   Inside `solve_with_decomposed_qpu`, define a low-level sampler chain. This is the core of the decomposition strategy—it uses a sampler that requires explicit embedding, not a pre-built hybrid solver.
        ```python
        sampler = EmbeddingComposite(DWaveSampler(token=token))
        ```
    *   Submit the BQM directly to the sampler.
        ```python
        sampleset = sampler.sample(bqm, num_reads=1000, label='Decomposed_BQM_Run')
        ```
    *   Process the returned sampleset to extract the best solution, energy, and timing. Return a results dictionary.

3.  **Create `comprehensive_benchmark_DECOMPOSED.py`:**
    *   Copy `@Benchmark Scripts/comprehensive_benchmark.py` into this new file.
    *   Modify it to `import solver_runner_DECOMPOSED as solver_runner`.

4.  **Modify the Benchmark Logic:**
    *   In the `run_farm_scenario` function, **delete the entire D-Wave CQM solver section**. This scenario is now classical-only, reflecting the decomposition strategy.
    *   In the `run_binary_scenario` function:
        *   Keep the Gurobi (PuLP) and Gurobi QUBO sections for baseline comparison.
        *   **Delete the `dwave_cqm` solver section.**
        *   Find the `dwave_bqm` solver section. Instead of calling `solve_with_dwave_bqm` (which uses a *hybrid* BQM solver), call your new `solve_with_decomposed_qpu` function, passing the BQM to it.
        *   Rename the solver key in the final JSON from `"dwave_bqm"` to `"dwave_decomposed_qpu"` to make the distinction clear.

---

## 5. Final Deliverables & Quality Assurance

*   **Final Files:**
    *   `@todo/solver_runner_CUSTOM_HYBRID.py`
    *   `@todo/comprehensive_benchmark_CUSTOM_HYBRID.py`
    *   `@todo/solver_runner_DECOMPOSED.py`
    *   `@todo/comprehensive_benchmark_DECOMPOSED.py`
*   **Code Standards:** All new code must be clean, readable, and well-commented, following professional software engineering guidelines. Use descriptive variable names and clear function signatures.
*   **Output Consistency:** The final JSON files produced by both new benchmark scripts **must** follow the same nested structure as the original `@Benchmark Scripts/comprehensive_benchmark.py` to allow for unified analysis and plotting.
*   **Final Verification:** Before concluding your task, double-check that all file paths are correct, imports are resolved, and the placeholder `'YOUR_DWAVE_TOKEN_HERE'` is used for the API token.
