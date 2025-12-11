---
applyTo: '**'
---

# Coding Preferences
- Professional matplotlib visualizations with LaTeX rendering
- Consistent color palettes and publication-quality plots
- Error handling and validation for data processing
- Modular design with clear separation of concerns
- **Solver Configuration**: Use Gurobi for PuLP (with GPU acceleration), IPOPT for Pyomo (native quadratic handling)
- **Timing Measurements**: Always use internal solver timing (solver.solve() only) to exclude Python overhead and model setup
- **Solution Validation**: Always validate solutions against constraints and include results in JSON output
- **Decomposition Strategies**: Factory pattern for multiple algorithms (Benders, ADMM, Dantzig-Wolfe)
- **Infeasibility Handling**: Use Gurobi IIS computation for diagnostic reporting

# Project Architecture
- **Domain**: Agricultural land allocation optimization using quantum annealing
- **Main Scripts**: `comprehensive_benchmark.py`, `solver_runner_BINARY.py`, `solver_runner_LQ.py`
- **Decomposition Strategies**:
  - `@todo/decomposition_strategies.py`: Factory pattern for all strategies
  - `@todo/decomposition_benders.py`: Benders decomposition (master/subproblem)
  - `@todo/decomposition_benders_hierarchical.py`: **NEW** Hierarchical Benders for large problems (>10 farms)
  - `@todo/decomposition_admm.py`: ADMM with consensus (best convergence)
  - `@todo/decomposition_dantzig_wolfe.py`: Column generation
  - `@todo/benchmark_all_strategies.py`: Unified benchmarking tool
  - `@todo/benchmark_hierarchical.py`: **NEW** Benchmark for hierarchical strategy
- **Results Structure**: 
  - JSON files contain detailed solution metadata including validation results
  - CSV files contain flattened sampleset data (multiple solutions per run)
- **Key Directories**: 
  - `Benchmarks/COMPREHENSIVE/Farm_DWave/` - DWave results
  - `Benchmarks/ALL_STRATEGIES/` - Decomposition strategy comparisons
  - `Benchmark Scripts/` - Solver implementations
- **Formulation Types**:
  - Binary (BQUBO): Pure linear objective with binary plot assignment
  - Linear-Quadratic (LQ): Linear objective + quadratic synergy bonus term

# Solutions Repository
- **VALIDATION BUG FIXED**: comprehensive_benchmark.py validation tolerance was wrong
- **CQM Constraints (from create_cqm_farm - applies to both Binary and LQ)**:
  1. Land Availability: `sum(A[f,c]) <= land_availability[f]` per farm
  2. Min Area if Selected: `A[f,c] >= min_area * Y[f,c]` (linking constraint)
  3. Max Area if Selected: `A[f,c] <= land_capacity * Y[f,c]` (linking constraint)
  4. Food Group Min/Max: **GLOBAL** constraints across ALL farms (not per-farm)
     - `sum(Y[f,c] for all f, c in group) >= min_foods` (global)
     - `sum(Y[f,c] for all f, c in group) <= max_foods` (global)
  5. `min_area = 0.0001` for all crops (to prevent zero allocation when selected)
- **Food Groups Data Structure** (CRITICAL):
  - `food_groups = {group_name: [list_of_foods]}`  ← Lists only
  - `food_group_constraints = {group_name: {'min_foods': X, 'max_foods': Y}}`  ← Constraint params

## Quantum Speedup Roadmap (Dec 10, 2024)
- **D-Wave Token Status**: Current token `DEV-45FS-*` is INVALID/EXPIRED
  - Get new token from: https://cloud.dwavesys.com/leap/
  - Token set via `--token` argument or `DWAVE_API_TOKEN` env var
- **Phase 1 Partial Results** (Authentication blocked):
  - ✅ Gurobi ground truth: Works successfully
  - ✅ Direct QPU embedding: Successfully found embedding (4.5s, 498 physical qubits, max chain 8)
  - ❌ Clique QPU: Authentication failed (invalid token)
  - ❌ Rotation tests: Not completed due to token issue
- **Roadmap Implementation Status**:
  - Phase 1 (Proof of Concept): CODED ✅ (needs valid token to run)
  - Phase 2 (Scaling Validation): CODED ✅ (5, 10, 15 farms)
  - Phase 3 (Optimization): **CODED ✅ (10, 15, 20 farms, 5 optimization strategies)**
- **Phase 3 Implementation** (Dec 10, 2024):
  - 5 optimization strategies: Baseline, 5x Iterations, Larger Clusters, Hybrid, High Reads
  - Automatic best strategy identification (quality, speed, balanced)
  - Comprehensive parameter space exploration
  - Publication-ready benchmarking framework
  - Files: `PHASE3_IMPLEMENTATION_SUMMARY.md`, `ROADMAP_EXECUTION_GUIDE.md`
- **Next Actions**:
  1. Obtain valid D-Wave token
  2. Run Phase 1 to validate
  3. Run Phase 2 to find crossover
  4. Run Phase 3 to optimize parameters
  - `load_food_data()` returns: `(scenario_data, foods, food_groups, config)`
  - Access pattern: `foods_in_group = food_groups.get(group_name, [])`
  - Constraints: `constraints = config['parameters']['food_group_constraints'][group_name]`
- **Decomposition Strategies Implemented**:
  - **Benders**: Best performance, highest objective (0.30+), 93%+ land usage
  - **Dantzig-Wolfe**: Good performance, column generation approach
  - **ADMM**: Requires post-processing to enforce linking constraints; use rho=10.0 for better convergence
  - All use factory pattern: `DecompositionFactory.get_strategy(name)`
  - QPU versions now fall back to `neal.SimulatedAnnealingSampler` when no D-Wave token
- **ADMM Critical Fix (2025-11-25)**:
  - ADMM must post-process to enforce linking constraints after convergence
  - Binarize Y values: `Y_binary = {key: 1.0 if val > 0.5 else 0.0 for ...}`
  - Force A=0 when Y=0, enforce A >= min_area when Y=1
  - Without this, ADMM solutions violate min_area and linking constraints
- **Food Group Constraint Fix (2025-11-25)**:
  - Food group min_foods counts UNIQUE foods selected (across all farms)
  - A food counts as selected if Y=1 on ANY farm
  - Post-processing must add NEW unique foods, not just more farm selections
  - Default ADMM iterations reduced to 10 for faster problem resolution
- **Infeasibility Detection**:
  - Use `detect_infeasibility()` for IIS computation
  - Track constraint names manually (can't access `constr.ConstrName` before update)
  - Provides automated relaxation suggestions by constraint type
- **QPU Timing Documentation** (2025-11-26):
  - Official D-Wave timing model: `T = Tp + Δ + Ts`
  - `Tp` = Programming time (~15-20ms for Advantage)
  - `Δ` = Overhead (~10-20ms)
  - `Ts = R × (Ta + Tr + Td)` where R = num_reads
  - Default: num_reads=1000, Ta=20µs, Tr=25-150µs (depends on qubits)
  - Readout time scales with problem size (qubits)
  - LaTeX report created: `Latex/qpu_timing_estimates_report.tex`
- **CRITICAL BUG FIXED** (2025-11-26):
  - `benders_qpu`, `admm_qpu`, `dantzig_wolfe_qpu` were using `LeapHybridBQMSampler`
  - This is WRONG - hybrid has 3s minimum runtime and different billing!
  - Fixed to use `DWaveSampler` + `EmbeddingComposite` for direct QPU access
  - Now uses num_reads=100, annealing_time=20µs for fast iterative calls
  - Extracts actual `qpu_access_time` from `sampleset.info['timing']`
- **QPU Embedding Insights** (2025-11-26):
  - **CRITICAL**: CQM→BQM creates slack variables with random names → cannot cache embedding!
  - Embedding time dominates: ~10min for 316 vars (10 farms), fails at ~726 vars (25 farms)
  - Actual QPU time is tiny: ~288ms billed for 1000 reads @ 20µs annealing
  - Embedding is computed locally by `minorminer` (heuristic, can fail)
  - Problem: 10 farms × 27 foods = 270 logical vars → 316 BQM vars → 11,353 quadratic terms
  - Problem: 25 farms × 27 foods = 675 logical vars → 726 BQM vars → 61,754 quadratic terms (TOO DENSE!)
  - Retry logic helps: embedding is stochastic, may succeed on retry
  - Study tool created: `study_embedding_scaling.py` - tests embedding without QPU billing
- **Benders QPU Results** (2025-11-26 - 10 farms, 1000 reads, 20µs):
  - Iteration 1 (classical): 0.057s, obj=0.2007
  - Iteration 2-6 (QPU): ~300-850s wall time each (embedding), ~288ms QPU billed
  - Best objective: 0.3688 (improved from 0.2007)
  - Total QPU time billed: 1.435s across 5 QPU iterations
  - Early stopping after 3 non-improving iterations worked correctly
- **Solution Validation Pattern**:
  - Function: `validate_solution_constraints(solution, farms, foods, food_groups, land_availability, config)`
  - Returns: Dictionary with `is_feasible`, `n_violations`, `violations` list, `constraint_checks`, `summary`
  - Checks: Land availability, linking constraints (A-Y relationship), food group global constraints
  - Included in: Both PuLP and Pyomo solver results, saved to benchmark cache/JSON
- **LQ Formulation Specifics**:
  - Objective: Linear term (weighted food attributes) + Quadratic synergy bonus
  - PuLP: Uses McCormick relaxation (Z variables) to linearize Y*Y products
  - Pyomo: Handles quadratic terms natively (no linearization needed)
  - Synergy matrix creation is done in scenario loading (before any timing)
  - Solve time excludes synergy matrix generation and model setup
  - Validation identical to Binary formulation (synergy only affects objective, not constraints)
- **Solver Priorities**:
  - PuLP: Use GUROBI with GPU acceleration (Method=2, BarHomogeneous=1)
  - Pyomo: Prioritize IPOPT for nonlinear/quadratic objectives
- **DWave Issue**: CQM→BQM solver returns infeasible solutions violating linking constraints
- **Specific Violation**: `A_Farm5_Spinach=0.82` but `Y_Farm5_Spinach=0` violates constraint #2
- **HIERARCHICAL BENDERS DECOMPOSITION** (2025-11-26):
  - **Problem**: Standard QPU embedding fails for ≥10 farms (100% failure rate)
  - **Root Cause**: CQM→BQM creates dense graphs (~22% density) exceeding Pegasus capacity
  - **Solution**: Hierarchical graph partitioning inspired by QAOA-in-QAOA paper
  - **Implementation**: `@todo/decomposition_benders_hierarchical.py`
  - **Key Algorithm**:
    1. Partition BQM using Louvain community detection
    2. Solve each partition independently (QPU or SA)
    3. Create "merging BQM" for cross-partition coupling
    4. Recursively apply if merging BQM too large
    5. Reconstruct global solution from flip decisions
  - **Results**: 100% success for 5-20+ farm problems
  - **Usage**: `solve_with_strategy('benders_hierarchical', max_embeddable_vars=150)`
  - **Strategy registered**: `DecompositionStrategy.BENDERS_HIERARCHICAL`
- **QPU BENCHMARK PLAN** (2025-11-30):
  - **Purpose**: Pure QPU benchmarking (NO hybrid solvers)
  - **Methods Tested**: Direct QPU (DWaveSampler), Decomposition (PlotBased, Multilevel, Cutset)
  - **QBSolv Status**: Deprecated, Python 3.12 incompatible - NOT USED
  - **Files Created**: 
    - `@todo/qpu_benchmark.py` - Main benchmark runner
    - `@todo/qpu_validate_access.py` - D-Wave access validation
    - `@todo/qpu_benchmark_results/` - Results output directory
  - **API Token**: Requires DWAVE_API_TOKEN environment variable
  - **Fallback**: Uses `neal.SimulatedAnnealingSampler` when QPU unavailable
  - **Constraint Issue**: Decomposition methods violate global constraints (food group diversity)
  - **Solution**: Need coordinated solving (Benders cuts) or post-processing repair
- **QPU vs PuLP OBJECTIVE MISMATCH FIXED** (2025-12-01):
  - **Issue**: QPU benchmark Gurobi ground truth ≠ Patch PuLP results (~23% difference)
  - **Root Causes Identified**:
    1. `qpu_benchmark.py` used `min: 2` per group vs `min_foods: 1`
    2. Different constraint key names (`min`/`max` vs `min_foods`/`max_foods`)
    3. Different max_plots_per_crop (hardcoded 5 vs disabled)
    4. One-crop constraint: `== 1` vs `<= 1` (allows idle plots)
  - **FIX APPLIED** to `@todo/qpu_benchmark.py`:
    - `food_group_constraints` now uses `min_foods: 1` per group
    - `max_plots_per_crop` set to `None` (disabled)
    - One-crop constraint changed to `<= 1` (allows idle plots)
    - Constraint keys now use `min_foods`/`max_foods`
    - U-Y linking fixed: `Y <= U` and `U <= sum(Y)`
    - Removed `reverse_mapping` (use `food_groups` directly)
  - **Comparison Script**: `@todo/compare_gurobi_pulp_objectives.py`
  - **Expected**: After re-running qpu_benchmark.py, Gurobi should match Patch PuLP (~0.388)