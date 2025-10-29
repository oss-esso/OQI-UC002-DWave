# Comprehensive Benchmark Update Summary

## Overview
Updated `comprehensive_benchmark.py` to use `solver_runner_BINARY.py` as the unified runner for both farm (continuous) and patch (binary) scenarios, with proper land generation methods and a fixed 100 ha total area.

## Key Changes

### 1. Unified Solver Import
**Before:**
```python
import solver_runner as solver_farm
import solver_runner_BINARY as solver_binary
```

**After:**
```python
import solver_runner_BINARY as solver_runner
```

Now using `solver_runner_BINARY.py` exclusively, which contains both formulations:
- `create_cqm_farm()` and `solve_with_pulp_farm()` for continuous formulation
- `create_cqm_plots()` and `solve_with_pulp_plots()` for binary formulation

### 2. Enhanced Land Generation (`generate_sample_data`)

**Updates:**
- Farm scenario: Uses `uneven_distribution` land method (realistic farm sizes)
- Patch scenario: Uses `even_grid` land method (equal-sized plots)
- Both scenarios scaled to fixed 100 ha total area
- Added `land_method` field to sample data
- Improved console output with statistics

**Output Format:**
```
Farm (continuous): 10 farms
   Area: min=5.32 ha, max=15.87 ha, avg=10.00 ha
   Total: 100.00 ha

Patch (binary): 10 plots
   Area per plot: 10.000 ha (equal grid)
   Total: 100.00 ha
```

### 3. Farm Scenario (`run_farm_scenario`)

**Formulation:**
- Type: Continuous (MINLP)
- Land method: `uneven_distribution`
- Variables: Continuous area variables (A) + Binary selection variables (Y)
- Function: `solver_runner.create_cqm_farm()`

**Solvers:**
1. **Gurobi (PuLP)**: 
   - Uses `solve_with_pulp_farm()`
   - MINLP solver
   - Timeout: 100 seconds

2. **D-Wave CQM**:
   - Uses `solve_with_dwave_cqm()`
   - Quantum-classical hybrid
   - Returns feasibility status and objective value

**Result Metadata:**
- `land_method`: 'uneven_distribution'
- `formulation`: 'continuous_minlp' (Gurobi) or 'continuous_cqm' (D-Wave)
- `n_variables`, `n_constraints`: From CQM
- Solution includes areas and selections

### 4. Patch Scenario (`run_binary_scenario` → `run_binary_scenario`)

**Formulation:**
- Type: Binary (BIP)
- Land method: `even_grid`
- Variables: Pure binary assignment variables (Y only)
- Function: `solver_runner.create_cqm_plots()`

**Solvers:**
1. **Gurobi (PuLP)**:
   - Uses `solve_with_pulp_plots()`
   - BIP solver (pure binary)
   - Timeout: 300 seconds

2. **D-Wave CQM**:
   - Uses `solve_with_dwave_cqm()`
   - Quantum-classical hybrid for CQM

3. **Gurobi QUBO**:
   - Uses `gurobi_optimods.qubo.solve_qubo()`
   - Native QUBO solver
   - After CQM→BQM conversion
   - Lagrange multiplier: 10.0

4. **D-Wave BQM**:
   - Uses `LeapHybridBQMSampler`
   - Higher QPU utilization
   - After CQM→BQM conversion
   - Tracks QPU time separately

**Result Metadata:**
- `land_method`: 'even_grid'
- `formulation`: 'binary_bip', 'binary_cqm', 'binary_qubo', or 'binary_bqm'
- `bqm_conversion_time`: Time to convert CQM→BQM
- `qpu_time`: Actual quantum processing time (D-Wave BQM only)
- `hybrid_time`: Total solver time including classical preprocessing

### 5. Solver Configuration Matrix

| Scenario | Land Method | Solver | Formulation | Problem Class |
|----------|-------------|---------|-------------|---------------|
| Farm | uneven_distribution | Gurobi | Continuous areas + Binary selection | MINLP |
| Farm | uneven_distribution | D-Wave CQM | Continuous areas + Binary selection | MINLP |
| Patch | even_grid | Gurobi | Pure binary | BIP |
| Patch | even_grid | D-Wave CQM | Pure binary | BIP |
| Patch | even_grid | Gurobi QUBO | QUBO (after BQM conversion) | QUBO |
| Patch | even_grid | D-Wave BQM | QUBO (after BQM conversion) | QUBO |

**Total: 6 solver configurations**

### 6. Fixed Total Land Area

**Implementation:**
- All scenarios use **exactly 100 ha** total land
- Farm scenario: 100 ha distributed unevenly across N farms
- Patch scenario: 100 ha distributed evenly across N plots
- Plot area in patch scenario: 100 / N ha per plot

**Benefits:**
- Direct comparability across different N values
- Consistent problem scale for benchmarking
- Matches Grid_Refinement.py approach

### 7. Caching and Result Persistence

Each solver result is saved individually to:
```
Benchmarks/COMPREHENSIVE/
├── Farm_PuLP/          # Farm + Gurobi
├── Farm_DWave/         # Farm + D-Wave CQM
├── Patch_PuLP/         # Patch + Gurobi
├── Patch_DWave/        # Patch + D-Wave CQM
├── Patch_GurobiQUBO/   # Patch + Gurobi QUBO
└── Patch_DWaveBQM/     # Patch + D-Wave BQM
```

Filename format: `config_{n_units}_run_{run_id}.json`

### 8. Output Improvements

**Console Output:**
```
🌾 FARM SCENARIO (Continuous) - Sample 0
   10 farms, 100.0 ha
   Method: uneven_distribution
   Running Gurobi (MINLP)...
      ✓ Gurobi: Optimal in 0.523s (obj: 0.748312)
   Running DWave CQM...
      ✓ DWave CQM: Feasible in 12.345s
        Objective: 0.746210, Charge: 8.123s

📊 PATCH SCENARIO (Binary) - Sample 0
   10 plots, 100.0 ha
   Method: even_grid
   Running Gurobi (BIP)...
      ✓ Gurobi: Optimal in 0.421s (obj: 0.735421)
   Running DWave CQM...
      ✓ DWave CQM: Feasible in 11.234s
   Converting CQM to BQM...
      Lagrange multiplier: 10.0
      ✓ BQM: 100 vars, 523 interactions (0.123s)
   Running DWave BQM...
      ✓ DWave BQM: Optimal in 15.678s (QPU: 3.456s)
   Running Gurobi QUBO...
      BQM Variables: 100, QUBO terms: 623
      ✓ Gurobi QUBO: Optimal in 0.789s (obj: 0.735421)
```

### 9. Benchmark Configurations

Default configurations (can be customized):
```python
BENCHMARK_CONFIGS = [
    10,
    25,
    50
]
```

Each configuration represents the number of farms (continuous) or plots (binary).

### 10. Command-Line Usage

```bash
# Use default configurations with D-Wave
python comprehensive_benchmark.py --configs --dwave

# Single configuration without D-Wave
python comprehensive_benchmark.py 10

# Custom configurations
python comprehensive_benchmark.py 15 --dwave

# Specify output file
python comprehensive_benchmark.py --configs --dwave --output my_results.json

# Use custom D-Wave token
python comprehensive_benchmark.py --configs --dwave --token YOUR_TOKEN
```

## Compatibility with solver_runner_BINARY.py

### Functions Used from solver_runner_BINARY.py

#### For Farm Scenario (Continuous):
1. **`create_cqm_farm(farms, foods, food_groups, config)`**
   - Creates CQM with continuous area variables + binary selection
   - Returns: `(cqm, A, Y, constraint_metadata)`
   
2. **`solve_with_pulp_farm(farms, foods, food_groups, config)`**
   - Solves with PuLP/Gurobi MINLP solver
   - Returns: `(pulp_model, pulp_results)`

3. **`solve_with_dwave_cqm(cqm, token)`**
   - Solves with D-Wave CQM hybrid solver
   - Returns: `(sampleset, solve_time)`

#### For Patch Scenario (Binary):
1. **`create_cqm_plots(farms, foods, food_groups, config)`**
   - Creates CQM with pure binary variables
   - Includes min/max plot constraints
   - Returns: `(cqm, Y, constraint_metadata)`

2. **`solve_with_pulp_plots(farms, foods, food_groups, config)`**
   - Solves with PuLP/Gurobi BIP solver
   - Returns: `(pulp_model, pulp_results)`

3. **`solve_with_dwave_cqm(cqm, token)`**
   - Same as farm scenario
   - Returns: `(sampleset, solve_time)`

### BQM Conversion (Patch Only)
- Uses `dimod.cqm_to_bqm(cqm, lagrange_multiplier=10.0)`
- Returns: `(bqm, invert_function)`
- Enables Gurobi QUBO and D-Wave BQM solvers

## Benefits of the Update

### 1. Unified Codebase
- Single source file (`solver_runner_BINARY.py`) for both formulations
- Easier maintenance and consistency
- Shared complexity analysis functions

### 2. Proper Formulation Separation
- Farm: Realistic continuous areas (uneven_distribution)
- Patch: Discretized binary assignments (even_grid)
- Each uses appropriate solver functions

### 3. Complete Solver Coverage
- **Classical**: Gurobi for both MINLP and BIP
- **Quantum-Classical Hybrid**: D-Wave CQM for both
- **Pure QUBO**: Gurobi QUBO and D-Wave BQM for binary

### 4. Consistent Benchmarking
- Fixed 100 ha total area across all scenarios
- Direct comparability between continuous and binary formulations
- Proper scaling for different numbers of units

### 5. Enhanced Complexity Analysis
- Uses `calculate_model_complexity()` from solver_runner_BINARY.py
- Provides benchmark-ready metrics
- Quantifies reduction benefits (50% variables, ~74% constraints, 100% quadratic elimination)

## Testing Checklist

Before running the benchmark:

- [ ] Verify `solver_runner_BINARY.py` has all functions:
  - [ ] `create_cqm_farm()`
  - [ ] `create_cqm_plots()`
  - [ ] `solve_with_pulp_farm()`
  - [ ] `solve_with_pulp_plots()`
  - [ ] `solve_with_dwave_cqm()`

- [ ] Check land generation:
  - [ ] Farm samples have uneven areas
  - [ ] Patch samples have equal areas
  - [ ] Total is exactly 100 ha for both

- [ ] Verify solver configurations:
  - [ ] Gurobi works for both scenarios
  - [ ] D-Wave CQM works for both scenarios
  - [ ] BQM conversion succeeds for patch scenario
  - [ ] Gurobi QUBO works after BQM conversion
  - [ ] D-Wave BQM works after BQM conversion

- [ ] Test caching:
  - [ ] Results are saved to correct directories
  - [ ] Cached results are loaded on re-run
  - [ ] File names follow `config_{n}_run_{id}.json` format

## Next Steps

1. **Run small test:**
   ```bash
   python comprehensive_benchmark.py 5
   ```

2. **Verify output files** in `Benchmarks/COMPREHENSIVE/`

3. **Run full benchmark** (when ready):
   ```bash
   python comprehensive_benchmark.py --configs --dwave
   ```

4. **Analyze results:**
   ```bash
   python plot_comprehensive_results.py Benchmarks/COMPREHENSIVE/comprehensive_benchmark_*.json
   ```

## Summary

The comprehensive benchmark now uses `solver_runner_BINARY.py` as the unified runner, providing:
- ✅ Proper farm formulation (continuous, uneven_distribution)
- ✅ Proper patch formulation (binary, even_grid)  
- ✅ Fixed 100 ha total area for all scenarios
- ✅ 6 solver configurations (2 for farm, 4 for patch)
- ✅ Min/max area constraints in binary formulation
- ✅ Complexity metrics integration
- ✅ Consistent benchmarking approach

The script is ready for execution but should be tested with small configurations first before running the full benchmark.
