# Comprehensive Decomposition Benchmark Plan

## Objective
Create a complete benchmark comparing ALL formulations, decomposition strategies, and solvers for both Farm and Patch scenarios at scale (25 units).

## Key Insight from Binary Plot Study
- **Ultra-Sparse BQM** successfully embeds up to 25 plots (675 vars, 8775 quadratic, 3.9% density)
- **Louvain decomposition** creates ~27 tiny partitions (~25 vars each) - all embed in <20s
- **Plot-based decomposition** creates 5 partitions (135 vars each) - all embed in ~120s
- Direct embedding of CQM→BQM and Direct BQM **fail** at 10+ plots

## Architecture Overview

```
SCENARIO
├── Farm (continuous + binary)
│   ├── CQM (native)
│   │   └── Solver: Gurobi (via pyomo/model conversion)
│   │
│   └── BQM (converted from CQM)
│       ├── Direct solve attempts (expect: FAIL for 25 farms)
│       ├── Decomposition strategies:
│       │   ├── 1. Louvain graph partitioning
│       │   ├── 2. Plot-based partitioning
│       │   ├── 3. Energy-impact (dwave-hybrid)
│       │   └── 4. QBSolv (placeholder - no token)
│       └── Solver for each partition: Gurobi QUBO
│
└── Patch (binary only)
    ├── CQM (native)
    │   └── Solver: Gurobi
    │
    ├── BQM (converted from CQM)
    │   ├── Direct solve attempts (expect: FAIL for 25 plots)
    │   └── Same 4 decomposition strategies → Gurobi QUBO
    │
    ├── Direct BQM (no slack variables)
    │   ├── Direct solve attempts (expect: FAIL for 25 plots)
    │   └── Same 4 decomposition strategies → Gurobi QUBO
    │
    └── Ultra-Sparse BQM (minimal quadratic terms)
        ├── Direct solve attempts (expect: SUCCESS for 25 plots!)
        └── Same 4 decomposition strategies → Gurobi QUBO
```

## Formulations to Test

### Farm Scenario (25 farms × 27 foods)
1. **CQM (native)** - continuous areas + binary selections
2. **BQM (from CQM)** - fully binary via slack variables

### Patch Scenario (25 patches × 27 foods)
1. **CQM (native)** - binary only, but with constraints
2. **BQM (from CQM)** - binary with slack variables
3. **Direct BQM** - binary, minimal slack
4. **Ultra-Sparse BQM** - binary, ultra-minimal quadratic terms

## Decomposition Strategies (Applied to BQM variants)

For each BQM formulation that's too large to embed directly:

1. **Louvain Graph Partitioning**
   - Use `networkx.algorithms.community.louvain_communities`
   - Partition into subgraphs of ~25-150 variables
   - Solve each partition independently
   - Merge solutions (QAOA-in-QAOA style)

2. **Plot-based Partitioning**
   - Group by plots (5 plots per partition for 25-plot problems)
   - Each partition: 135 variables (5 plots × 27 foods)
   - Natural problem structure preservation

3. **Energy-Impact Decomposition**
   - Use `hybrid.decomposers.EnergyImpactDecomposer`
   - BFS traversal from high-energy variables
   - Partition size: 50-150 variables

4. **QBSolv** (placeholder)
   - Would use D-Wave's QBSolv for automatic decomposition
   - Currently: empty placeholder (no token/setup)
   - Document as "future work"

5. **Multilevel Decomposition (ML-QLS type)**
   - Based on the approach in `graph_decomp_QLS.tex`.
   - **Implementation**:
     - **Coarsening**: Create a hierarchy of smaller graphs. Use a library like `KaHIP` to perform multilevel graph coarsening (e.g., via maximum weight matching).
     - **Initial Solve**: Solve the problem on the coarsest graph using Gurobi or test its embedding.
     - **Uncoarsening & Refinement**: Project the solution back to finer graphs. At each level, use a local search heuristic (like QLS, but implemented with classical subproblem solvers) to refine the solution. The subproblems can be solved with Gurobi and their embedding characteristics studied.

6. **Sequential Cut-Set Reduction**
   - Based on the approach in `graph_decomp_sequential.tex`.
   - **Implementation**:
     - **Iterative Reduction**: Repeatedly find a minimum vertex cut set `K` that partitions the graph.
     - **Subproblem Solving**: For the smaller partition `V2`, solve the subproblem for all `2^|K|` binary assignments of variables in `K`. Use Gurobi for this.
     - **Reweighting**: Use the subproblem solutions to build and solve a system of linear equations to determine new weights for edges within `K`, effectively absorbing the smaller partition's contribution.
     - **Termination**: Stop when the graph is small enough or the minimum cut set size exceeds a threshold. The final reduced graph is then solved or embedded.

## Solvers to Apply

### For CQM formulations:
1. **Gurobi** (classical MINLP/MILP)
2. **LeapHybridCQMSampler** (D-Wave quantum-classical hybrid)

### For BQM formulations (whole or partitions):
1. **Gurobi QUBO** (classical quadratic solver)
2. **Embedding time study** (no QPU solve - just embedding feasibility)

## Metrics to Collect

For each combination of (Scenario, Formulation, Decomposition, Solver):

1. **Model Construction**
   - Variables count
   - Constraints count
   - Quadratic terms count
   - Graph density
   - Construction time

2. **Decomposition** (if applied)
   - Number of partitions
   - Partition sizes
   - Decomposition time
   - Merging strategy

3. **Embedding** (for BQM only)
   - Embedding time
   - Success/failure
   - Physical qubits used
   - Max chain length
   - Embedding attempts

4. **Solving**
   - Solver time
   - Total time (construction + decomposition + solving)
   - Objective value
   - Feasibility
   - QPU access time (if applicable)

5. **Solution Quality**
   - Objective value
   - Constraint violations
   - Gap from best known solution

## Implementation Steps

### Phase 1: Core Infrastructure (Files to create/modify)
- [ ] `comprehensive_decomposition_benchmark.py` - Main benchmark script
- [ ] `decomposition_framework.py` - Unified decomposition interface
- [ ] `solver_framework.py` - Unified solver interface
- [ ] `result_collector.py` - Standardized result collection

### Phase 2: Formulation Builders
- [ ] `build_farm_cqm()` - Farm scenario CQM
- [ ] `build_patch_cqm()` - Patch scenario CQM
- [ ] `build_patch_direct_bqm()` - Direct binary formulation
- [ ] `build_patch_ultra_sparse_bqm()` - Ultra-sparse formulation
- [ ] `cqm_to_bqm_converter()` - Wrapper with metadata

### Phase 3: Decomposition Implementations
- [ ] `LouvainDecomposer` class
- [ ] `PlotBasedDecomposer` class
- [ ] `EnergyImpactDecomposer` class (wrapper)
- [ ] `QBSolvDecomposer` class (placeholder)
- [ ] `MultilevelDecomposer` class
- [ ] `SequentialCutSetDecomposer` class
- [ ] `merge_partition_solutions()` - QAOA-in-QAOA style merging

### Phase 4: Solver Wrappers
- [ ] `GurobiCQMSolver` - Solve CQM with Gurobi
- [ ] `GurobiQUBOSolver` - Solve BQM/partition with Gurobi
- [ ] `DWaveCQMSolver` - Solve with LeapHybridCQMSampler
- [ ] `EmbeddingStudy` - Test embedding without QPU solve

### Phase 5: Benchmark Execution
- [ ] Run all combinations for Farm scenario (25 farms)
- [ ] Run all combinations for Patch scenario (25 patches)
- [ ] Collect comprehensive results JSON
- [ ] Generate comparison plots

### Phase 6: Analysis & Reporting
- [ ] `COMPREHENSIVE_DECOMPOSITION_RESULTS.md` - Main results document
- [ ] Embedding feasibility table
- [ ] Solve time comparison charts
- [ ] Solution quality comparison
- [ ] Recommendations for each problem size/type

## Expected Results Summary

### Farm Scenario (25 farms × 27 foods)
| Formulation | Decomposition | Embedding | Gurobi | DWave CQM |
|-------------|---------------|-----------|--------|-----------|
| CQM | N/A | N/A | ✅ | ✅ |
| BQM (from CQM) | None | ❌ FAIL | ❌ | N/A |
| BQM (from CQM) | Louvain | ✅ (~27 parts) | ✅ | N/A |
| BQM (from CQM) | Plot-based | ✅ (~5 parts) | ✅ | N/A |
| BQM (from CQM) | Energy-impact | ✅ | ✅ | N/A |
| BQM (from CQM) | QBSolv | 🔄 Placeholder | 🔄 | N/A |
| BQM (from CQM) | ML-QLS | ✅ | ✅ | N/A |
| BQM (from CQM) | Seq. Cut-Set | ✅ | ✅ | N/A |

### Patch Scenario (25 patches × 27 foods)
| Formulation | Decomposition | Embedding | Gurobi | DWave CQM |
|-------------|---------------|-----------|--------|-----------|
| CQM | N/A | N/A | ✅ | ✅ |
| BQM (from CQM) | None | ❌ FAIL | ❌ | N/A |
| BQM (from CQM) | Louvain | ✅ | ✅ | N/A |
| BQM (from CQM) | Plot-based | ✅ | ✅ | N/A |
| BQM (from CQM) | Energy-impact | ✅ | ✅ | N/A |
| BQM (from CQM) | ML-QLS | ✅ | ✅ | N/A |
| BQM (from CQM) | Seq. Cut-Set | ✅ | ✅ | N/A |
| Direct BQM | None | ❌ FAIL | ❌ | N/A |
| Direct BQM | Louvain | ✅ | ✅ | N/A |
| Direct BQM | Plot-based | ✅ | ✅ | N/A |
| Direct BQM | Energy-impact | ✅ | ✅ | N/A |
| Direct BQM | ML-QLS | ✅ | ✅ | N/A |
| Direct BQM | Seq. Cut-Set | ✅ | ✅ | N/A |
| **Ultra-Sparse BQM** | **None** | **✅ SUCCESS** | **✅** | N/A |
| Ultra-Sparse BQM | Louvain | ✅ | ✅ | N/A |
| Ultra-Sparse BQM | Plot-based | ✅ | ✅ | N/A |
| Ultra-Sparse BQM | Energy-impact | ✅ | ✅ | N/A |
| Ultra-Sparse BQM | ML-QLS | ✅ | ✅ | N/A |
| Ultra-Sparse BQM | Seq. Cut-Set | ✅ | ✅ | N/A |

## File Structure

```
@todo/
├── COMPREHENSIVE_DECOMPOSITION_PLAN.md (this file)
├── comprehensive_decomposition_benchmark.py (main script)
├── decomposition_framework.py (decomposition strategies)
├── solver_framework.py (solver wrappers)
├── result_collector.py (result standardization)
├── formulation_builders.py (CQM/BQM construction)
└── results/
    └── comprehensive_decomposition_results_YYYYMMDD_HHMMSS.json
```

## Estimated Execution Time

- Farm CQM (Gurobi): ~30s
- Farm CQM (DWave): ~60s
- Farm BQM decomposed (6 strategies × Gurobi): ~6 × 120s = 12min
- Patch CQM (Gurobi): ~20s
- Patch CQM (DWave): ~60s
- Patch BQMs decomposed (3 formulations × 6 strategies × Gurobi): ~18 × 120s = 36min
- Embedding studies (no solve): ~5min

**Total estimated time: ~55 minutes**

## Next Steps

1. Create modular framework files
2. Implement formulation builders
3. Implement decomposition strategies
4. Implement solver wrappers
5. Create main benchmark script
6. Execute and collect results
7. Generate analysis report

---

**Status**: Plan complete, ready for implementation
**Created**: 2025-11-27
**Author**: Claudette (GitHub Copilot)
