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
- **Decomposition Strategies** (NEW):
  - `@todo/decomposition_strategies.py`: Factory pattern for all strategies
  - `@todo/decomposition_benders.py`: Benders decomposition (master/subproblem)
  - `@todo/decomposition_admm.py`: ADMM with consensus (best convergence)
  - `@todo/decomposition_dantzig_wolfe.py`: Column generation
  - `@todo/benchmark_all_strategies.py`: Unified benchmarking tool
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
  - `load_food_data()` returns: `(scenario_data, foods, food_groups, config)`
  - Access pattern: `foods_in_group = food_groups.get(group_name, [])`
  - Constraints: `constraints = config['parameters']['food_group_constraints'][group_name]`
- **Decomposition Strategies Implemented**:
  - **ADMM**: Best performance, converges in 6 iterations (config 10)
  - **Benders**: Functional, needs cut refinement for better convergence
  - **Dantzig-Wolfe**: Implemented, column generation approach
  - All use factory pattern: `DecompositionFactory.get_strategy(name)`
- **Infeasibility Detection**:
  - Use `detect_infeasibility()` for IIS computation
  - Track constraint names manually (can't access `constr.ConstrName` before update)
  - Provides automated relaxation suggestions by constraint type
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