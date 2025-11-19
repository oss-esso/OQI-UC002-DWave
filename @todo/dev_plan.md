# Development Plan: Advanced Hybrid Quantum-Classical Benchmark Implementations

## Overview
This document outlines the implementation strategy for creating two advanced alternative implementations of the benchmark suite exploring sophisticated hybrid computing techniques.

## Task Type Classification
**Identified Task Type**: Feature Implementation - Creating advanced quantum-classical hybrid workflows  
**Expert Role**: Quantum Computing & Optimization Engineer specializing in hybrid algorithms

## Architecture Analysis

### Existing Codebase Structure
1. **solver_runner_BINARY.py**: Handles binary formulation with CQM/BQM conversion
   - Functions: create_cqm_plots, solve_with_pulp_plots, solve_with_dwave_bqm
   - Uses LeapHybridBQMSampler after CQM-to-BQM conversion
   
2. **comprehensive_benchmark.py**: Orchestrates multi-solver benchmarks
   - Generates farm (continuous) and patch (binary) scenarios
   - Runs parallel solver comparisons (Gurobi, D-Wave CQM, D-Wave BQM)
   - Saves results to JSON

3. **DWave Notebooks**: Provide hybrid workflow patterns
   - Notebook 02: Demonstrates custom workflows using dwave-hybrid
   - Key patterns: EnergyImpactDecomposer, QPUSubproblemAutoEmbeddingSampler, Loop, Race, Parallel

## Alternative 1: Custom Hybrid Workflow (CUSTOM_HYBRID)

### Objective
Create a bespoke hybrid workflow using dwave-hybrid framework, inspired by Kerberos sampler, to manually construct hybrid algorithms.

### Implementation Steps

#### 1. solver_runner_CUSTOM_HYBRID.py
Based on: solver_runner_BINARY.py

**New Function**: `solve_with_custom_hybrid_workflow(cqm, token, **kwargs)`

**Workflow Structure** (inspired by Kerberos):
```python
from hybrid import (
    Loop, Race, Parallel, ArgMin,
    EnergyImpactDecomposer, QPUSubproblemAutoEmbeddingSampler, SplatComposer,
    InterruptableTabuSampler, SimulatedAnnealingProblemSampler,
    State
)

# Define QPU branch: decompose → sample on QPU → compose
qpu_branch = (
    EnergyImpactDecomposer(size=<subproblem_size>, rolling=True, rolling_history=0.85) |
    QPUSubproblemAutoEmbeddingSampler(num_reads=100) |
    SplatComposer()
)

# Define classical branches
tabu_branch = InterruptableTabuSampler(timeout=<timeout_ms>)
sa_branch = InterruptableTabuSampler(timeout=<timeout_ms>)

# Racing branches: run until QPU completes
racing_branches = Race(
    tabu_branch,
    sa_branch,
    qpu_branch
) | ArgMin()

# Loop until convergence or max iterations
workflow = Loop(racing_branches, max_iter=<max_iterations>, convergence=<convergence_threshold>)
```

**Key Parameters**:
- subproblem_size: 40-50 variables (based on notebook examples)
- timeout_ms: 200-500ms for tabu/SA
- max_iterations: 10-20
- convergence_threshold: 3-5 iterations without improvement

**Return Format**:
```python
{
    'status': 'Optimal' | 'Converged' | 'Max Iterations',
    'objective_value': <best_energy>,
    'solve_time': <total_time_seconds>,
    'qpu_access_time': <qpu_time_seconds>,
    'iterations': <num_iterations>,
    'solver_name': 'dwave_custom_hybrid'
}
```

#### 2. comprehensive_benchmark_CUSTOM_HYBRID.py
Based on: comprehensive_benchmark.py

**Modifications**:
1. Import from `@todo/solver_runner_CUSTOM_HYBRID.py`
2. Add `solve_with_custom_hybrid_workflow` to binary scenario solvers
3. Update result processing to handle custom hybrid output
4. Ensure solver name is `"dwave_custom_hybrid"` in JSON

**Integration Point**:
```python
# In run_binary_scenario function
if dwave_token:
    try:
        print("\n  [4/6] D-Wave Custom Hybrid Solver...")
        result_custom = solve_with_custom_hybrid_workflow(cqm, token=dwave_token)
        # Process and cache results
    except Exception as e:
        print(f"    ❌ Custom hybrid failed: {e}")
```

## Alternative 2: Strategic Problem Decomposition (DECOMPOSED)

### Objective
Manually decompose the problem: continuous "farm" solved classically (Gurobi), binary "patch" converted to BQM and submitted to low-level D-Wave sampler.

### Implementation Steps

#### 1. solver_runner_DECOMPOSED.py
Based on: solver_runner_BINARY.py

**New Function**: `solve_with_decomposed_qpu(bqm, token, **kwargs)`

**Workflow Structure**:
```python
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.embedding import embed_bqm, unembed_sampleset

# Low-level sampler chain with explicit embedding
sampler_qpu = DWaveSampler()
sampler = EmbeddingComposite(sampler_qpu)

# Sample on QPU
sampleset = sampler.sample(bqm, num_reads=<num_reads>, annealing_time=<annealing_time>)

# Extract timing
qpu_access_time = sampleset.info.get('timing', {}).get('qpu_access_time', 0) / 1e6
```

**Key Parameters**:
- num_reads: 100-1000 (more reads for direct QPU access)
- annealing_time: 20-100 microseconds
- chain_strength: Auto-calculated or specified

**Return Format**:
```python
{
    'status': 'Optimal',
    'objective_value': <best_energy>,
    'solve_time': <total_time_seconds>,
    'qpu_access_time': <qpu_time_seconds>,
    'qpu_programming_time': <programming_time>,
    'num_reads': <num_reads>,
    'solver_name': 'dwave_decomposed_qpu'
}
```

#### 2. comprehensive_benchmark_DECOMPOSED.py
Based on: comprehensive_benchmark.py

**Strategic Decomposition Logic**:
```python
def run_decomposed_benchmark(sample_data, dwave_token):
    results = {}
    
    # 1. Farm scenario: Classical only (Gurobi)
    print("Farm Scenario: Classical Gurobi (continuous variables)")
    farm_result = solve_with_pulp_farm(farms, foods, food_groups, config)
    results['farm_classical'] = farm_result
    
    # 2. Patch scenario: Quantum BQM (binary variables)
    print("Patch Scenario: Quantum BQM (low-level QPU)")
    cqm_patch, Y, metadata = create_cqm_plots(farms, foods, food_groups, config)
    bqm, invert = cqm_to_bqm(cqm_patch)
    patch_result = solve_with_decomposed_qpu(bqm, token=dwave_token)
    results['patch_quantum'] = patch_result
    
    return results
```

**Modifications**:
1. Add decomposition logic to separate farm vs patch
2. Route farm to Gurobi, patch to QPU
3. Collect and merge results
4. Ensure solver names are distinct: `"farm_gurobi_classical"`, `"patch_dwave_qpu"`

## Testing Strategy

### Incremental Testing

#### CUSTOM_HYBRID Tests
1. **Unit Test**: Test workflow construction
```python
# Test 1: Verify workflow builds without errors
workflow = build_custom_hybrid_workflow(cqm, token)
assert workflow is not None
```

2. **Integration Test**: Run on small problem (n_units=10)
```python
# Test 2: Execute workflow and verify output
initial_state = State.from_problem(bqm)
result_state = workflow.run(initial_state).result()
assert 'samples' in result_state
assert result_state.samples.first.energy < initial_state.samples.first.energy
```

#### DECOMPOSED Tests
1. **Unit Test**: Test BQM conversion
```python
# Test 1: Verify CQM to BQM conversion
cqm, Y, metadata = create_cqm_plots(farms, foods, food_groups, config)
bqm, invert = cqm_to_bqm(cqm)
assert len(bqm.variables) > 0
```

2. **Integration Test**: Run low-level sampler
```python
# Test 2: Execute decomposed solver
result = solve_with_decomposed_qpu(bqm, token)
assert result['status'] == 'Optimal'
assert result['qpu_access_time'] > 0
```

### Full Execution Tests
- Run both comprehensive benchmarks with `BENCHMARK_CONFIGS = [10]`
- Verify JSON outputs are correct
- Check solver names are properly labeled
- Validate timing metrics are captured

## Code Quality Standards

### IEEE Professional Standards
1. **Documentation**: All functions have comprehensive docstrings
2. **Comments**: Inline comments explain non-obvious logic
3. **Naming**: Clear, descriptive variable and function names
4. **Error Handling**: Try-except blocks with meaningful messages
5. **Modularity**: Functions have single, well-defined purposes

### Security
- Use placeholder `'YOUR_DWAVE_TOKEN_HERE'` for D-Wave API token
- No hardcoded credentials

### Consistency
- Follow existing codebase style
- Match parameter naming conventions
- Use consistent formatting (PEP 8)

## Expected Outputs

### CUSTOM_HYBRID Outputs
```
@todo/solver_runner_CUSTOM_HYBRID.py
@todo/comprehensive_benchmark_CUSTOM_HYBRID.py
Benchmarks/CUSTOM_HYBRID/
  results_config_10_run_1.json
  results_config_15_run_1.json
  ...
```

### DECOMPOSED Outputs
```
@todo/solver_runner_DECOMPOSED.py
@todo/comprehensive_benchmark_DECOMPOSED.py
Benchmarks/DECOMPOSED/
  results_config_10_run_1.json
  results_config_15_run_1.json
  ...
```

## JSON Output Format
```json
{
  "scenario_type": "farm" | "patch",
  "solver_name": "dwave_custom_hybrid" | "dwave_decomposed_qpu",
  "config": {...},
  "results": {
    "status": "Optimal",
    "objective_value": 0.XXX,
    "solve_time": X.XX,
    "qpu_access_time": X.XX,
    "iterations": X,
    "...": "..."
  }
}
```

## Execution Order
1. ✅ Create dev_plan.md (this file)
2. ⏳ Implement solver_runner_CUSTOM_HYBRID.py
3. ⏳ Implement comprehensive_benchmark_CUSTOM_HYBRID.py
4. ⏳ Test CUSTOM_HYBRID with n_units=10
5. ⏳ Debug and fix CUSTOM_HYBRID
6. ⏳ Implement solver_runner_DECOMPOSED.py
7. ⏳ Implement comprehensive_benchmark_DECOMPOSED.py
8. ⏳ Test DECOMPOSED with n_units=10
9. ⏳ Debug and fix DECOMPOSED
10. ⏳ Final validation and quality check

## Risk Mitigation
- **Missing Libraries**: Install required packages (dwave-hybrid, dwave-system)
- **API Token**: Ensure user sets token before running
- **Timeouts**: Configure reasonable timeouts to prevent long hangs
- **Memory**: Monitor memory usage for large problems
- **Compatibility**: Ensure Ocean SDK version compatibility

## Success Criteria
- [ ] Both implementations execute without errors
- [ ] JSON outputs are correctly formatted
- [ ] Solver names are properly identified
- [ ] Timing metrics are accurate
- [ ] Code follows IEEE standards
- [ ] No hardcoded credentials
- [ ] Integration with existing codebase is seamless
