# Verification Prompt: Project Codebase Integrity Check

**Objective:** Verify that the code and data artifacts described in the LaTeX report `@@todo/report/content_report.tex` exist within the project workspace. Your primary focus should be the `@todo/` directory, but you should also inspect other relevant directories like `src/`, `Utils/`, `Inputs/`, and the root directory for scripts.

**Methodology:** For each numbered item below, search the filesystem for corresponding files or code snippets. Report your findings in a structured format, noting the file path where the artifact was found and your confidence (High/Medium/Low) that it matches the description.

---

## Verification Checklist:

### **1. Core Problem Formulations**
The report details two main problem formulations. Verify their existence.

*   **1.1. Formulation A (Binary Crop Allocation):**
    *   **Description:** A CQM for binary crop allocation with 27 crops and linear objectives.
    *   **Action:** Search for files and code implementing "Formulation A" or a "binary crop allocation" model. Look for CQM definitions related to this.

*   **1.2. Formulation B (Hierarchical Multi-Period Rotation):**
    *   **Description:** An enhanced CQM with 6 aggregated crop families, a 3-period rotation, and quadratic synergy terms (temporal and spatial). It is described as being designed to be "frustrated" and challenging for classical solvers.
    *   **Action:** Search for files and code implementing "Formulation B", "multi-period rotation", or the specific objective function components: `Benefit`, `Temporal`, `Spatial`, `Penalty`, `Diversity`. Pay close attention to the definition of the `Rotation Matrix R` with its `monoculture penalty` and `frustration ratio`.

### **2. Decomposition Strategies**
The report describes eight distinct decomposition methods for solving the problem on a QPU. These are the core technical contribution. Find the implementation for each. They are likely located in the `@todo/` directory.

*   **Actions (for each method 2.1 through 2.8):**
    *   Search for Python files or modules named after the method (e.g., `plot_based_decomposition.py`).
    *   Search within files for class or function names matching the method (e.g., `class PlotBasedDecomposition:`).
    *   Look for code that matches the pseudocode and complexity analysis provided in the report.

*   **2.1. Direct QPU Embedding:** A baseline method that tries to embed the whole problem.
*   **2.2. Plot-Based Decomposition:** Partitions the problem by farm.
*   **2.3. Multilevel Partitioning:** Groups farms into clusters of size `k`.
*   **2.4. Louvain Community Detection:** Uses modularity maximization to partition the variable interaction graph.
*   **2.5. CQM-First Decomposition:** Partitions at the CQM level *before* BQM conversion.
*   **2.6. Coordinated Master-Subproblem:** A two-level method where a master problem selects crops (`U` variables) and subproblems assign them.
*   **2.7. Spectral Clustering:** Uses eigenvectors of the graph Laplacian for partitioning.
*   **2.8. HybridGrid Decomposition:** Partitions both farms and crops into a 2D grid.

### **3. Data and Preprocessing**
Verify that the data sources and preprocessing steps mentioned are present.

*   **3.1. Crop Database:**
    *   **Description:** Data for 27 crops across 5 food groups, with scores for nutrition, environment, etc. Sourced from GAIN for Bangladesh and Indonesia.
    *   **Action:** Look in the `Inputs/` directory for data files (`.csv`, `.json`) containing this crop data. Check for scripts that load and process this data.

*   **3.2. Crop Family Aggregation:**
    *   **Description:** A process to aggregate the 27 crops into 6 families for Formulation B.
    *   **Action:** Find a script or function that performs this aggregation (e.g., a function `aggregate_crops` or similar).

*   **3.3. Farm Data and Spatial Layout:**
    *   **Description:** Farm areas sampled from a skewed distribution and assigned coordinates on a 2D grid with a 4-nearest-neighbor graph.
    *   **Action:** Search for code that generates farm data, samples sizes, and constructs a k-nearest neighbor graph.

### **4. Solver and Hardware Configuration**
Check for code that configures and calls the classical and quantum solvers.

*   **4.1. D-Wave QPU Configuration:**
    *   **Description:** Mentions `Advantage` quantum annealer, `Pegasus` topology, `annealing_time`, `chain_strength`, and `num_reads`.
    *   **Action:** Find the code that calls the D-Wave API (e.g., `dwave-ocean-sdk`). Check if the parameters mentioned are being set. Look for use of `DWaveCliqueSampler` and `MinorMiner`.

*   **4.2. Gurobi Classical Baseline:**
    *   **Description:** Gurobi 12.0.1, 300-second timeout, 1% MIP gap, `MIPFocus=1`.
    *   **Action:** Find the code that calls the Gurobi solver (`gurobipy`). Check for parameter-setting lines that configure the timeout, `MIPGap`, and other mentioned settings.

### **5. BQM Conversion**
The report details the process of converting the CQMs to BQMs, including the penalty method and the calculation of linear biases (`h`) and quadratic couplings (`J`).

*   **Action:** Search for a function or module responsible for `cqm_to_bqm` conversion. Verify that it constructs the `h` and `J` terms by combining the objective function with penalties for constraints (one-hot, rotation, etc.), as described in the report. Look for the logic that determines penalty weights (`lambda`).

---

**Final Output Format:**
Please provide a summary of your findings in a markdown file. Create a section for each of the 5 main verification categories above. For each sub-point, state:
*   **Status:** `FOUND` or `NOT FOUND`.
*   **Location:** The file path(s) where the artifact was located.
*   **Confidence:** `High`, `Medium`, or `Low`. High confidence means the implementation closely matches the report's description and pseudocode. Low confidence means only a keyword or a loosely related function was found.
*   **Notes:** Any brief comments on discrepancies or partial matches.