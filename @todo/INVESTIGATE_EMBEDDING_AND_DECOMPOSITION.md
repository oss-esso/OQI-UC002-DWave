# LLM Agent Task: Resolve Benders QPU Embedding Failure for Large Problems

**Objective:** The current Benders decomposition strategy (`decomposition_benders_qpu.py`) fails to find a minor-embedding on the QPU for problems with more than 10 farms. This is a critical issue preventing the solution of large-scale problems. Your task is to investigate and implement solutions to overcome this embedding limitation.

**Reference Documents:**
- **Problem Evidence:** `@@todo/study_embedding_scaling.py` and its output `@@todo/embedding_scaling_study_20251126_183012.json`. The JSON file shows a 100% embedding failure rate for problems with 10 or more farms.
- **Decomposition Idea:** The paper `main.tex` ("QAOA-in-QAOA: solving large-scale MaxCut problems on small quantum machines") outlines a hierarchical "divide-and-conquer" approach. You will adapt this methodology.

---

## 1. Investigation & Implementation Plan

You are to pursue two primary investigation paths.

### Path 1: Advanced Embedding Techniques

Your goal is to find an embedding for the existing BQM without changing the problem structure.

**Tasks:**
1.  **Analyze the BQM Structure:** Investigate the BQM generated from the Benders master problem in `decomposition_benders_qpu.py`. Understand its connectivity (graph density, node degrees). Is it structured or random?
2.  **Tune `minorminer`:** The `find_embedding` function in `minorminer` has numerous parameters. Systematically experiment with them. Create a script to test various combinations of:
    - `tries`: Increase the number of attempts.
    - `chainlength_patience`: Adjust the patience for finding shorter chains.
    - Different finders (e.g., `trellis`).
3.  **Research D-Wave Tools:** Consult D-Wave's documentation for the latest embedding tools or structured samplers that might be a better fit than `EmbeddingComposite`. For example, `DWaveCliqueSampler` could be effective if the problem has clique structures.
4.  **Report Findings:** Document which, if any, `minorminer` parameters or alternative D-Wave tools lead to a successful embedding for a 15-farm problem.

### Path 2: Hierarchical Graph Decomposition (QAOA-in-QAOA Adaptation)

If Path 1 fails or proves inefficient, implement a graph decomposition strategy for the Benders master problem, inspired by the `main.tex` paper.

**Tasks:**
1.  **Create a New Strategy File:** Create `@@todo/decomposition_benders_hierarchical.py`. This file will contain the new solver logic.
2.  **Implement Graph Partitioning:**
    - In the new file, write a function that takes the BQM from the Benders master problem and partitions its graph structure into smaller subgraphs.
    - Use a standard graph partitioning library like `networkx.algorithms.community` to implement community detection (e.g., Louvain method), as suggested in the paper to keep dense subgraphs intact.
3.  **Implement Subproblem Solver:**
    - For each subgraph, create a smaller BQM.
    - Solve each sub-BQM on the QPU (or simulator) using the existing `solve_with_decomposed_qpu` logic. Store the resulting solutions.
4.  **Implement Merging-Problem Formulation:**
    - This is the most critical step. As per Theorem 1 in `main.tex`, the process of merging solutions can be formulated as a new, smaller optimization problem.
    - Your task is to construct a new, smaller BQM (the "merging BQM") where each variable represents a choice for a subgraph's solution (e.g., whether to flip its spin configuration, if applicable). The weights of this merging BQM should be determined by the strength of the connections (external edges) *between* the original subgraphs.
5.  **Implement Hierarchical Solving:**
    - The merging BQM itself might be too large to embed. Implement a recursive or hierarchical loop: if the merging BQM is too large, partition it and solve its subproblems, then solve a final meta-merging problem.
    - The final solution is reconstructed by applying the decisions from the highest-level merge down to the individual subgraph solutions.
6.  **Integrate with Factory:** Add your new `BendersHierarchicalStrategy` to `decomposition_strategies.py` so it can be called by the benchmark scripts.

---

## 2. Deliverables

1.  A **new Markdown file** `@@todo/EMBEDDING_INVESTIGATION_REPORT.md` summarizing your findings from both investigation paths. For Path 1, report the `minorminer` parameters that worked, if any. For Path 2, explain the design of your hierarchical solver.
2.  The new solver file `@@todo/decomposition_benders_hierarchical.py`.
3.  Modifications to `@@todo/decomposition_strategies.py` to include the new strategy.
4.  A new benchmark script `@@todo/benchmark_hierarchical.py` to test your new solver on problem sizes from 5 to 30 farms.
5.  JSON output files from your benchmark runs, demonstrating that your new strategy successfully solves problems with >= 10 farms.
