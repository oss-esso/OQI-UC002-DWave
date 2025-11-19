# Custom Hybrid Workflow Implementation - README

## Overview

This implementation creates a custom hybrid quantum-classical workflow using the `dwave-hybrid` framework, inspired by the Kerberos sampler architecture.

## Files Created

### Core Implementation
1. **solver_runner_CUSTOM_HYBRID.py** - Main solver with custom hybrid workflow
   - `solve_with_custom_hybrid_workflow()` - Custom hybrid algorithm implementation
   - Uses Racing Branches: Tabu, SA, and QPU running in parallel
   - Iterates until convergence or max iterations

2. **comprehensive_benchmark_CUSTOM_HYBRID.py** - Simplified benchmark script
   - Clean, focused implementation
   - Easy command-line interface
   - Supports both farm and patch scenarios

3. **benchmark_utils_custom_hybrid.py** - Modular utility functions
   - Data generation functions
   - Solver execution wrappers
   - Result saving and summary printing
   - Split for easy testing and maintenance

4. **test_custom_hybrid.py** - Unit tests
   - Tests data generation
   - Tests CQM creation
   - Tests hybrid framework availability
   - Tests workflow construction

## Architecture

### Custom Hybrid Workflow

```
Input: CQM → Convert to BQM → Create Initial State
                                     ↓
                        ┌────────────┴────────────┐
                        │    Loop (max_iter=15)   │
                        │                         │
                        │  ┌────────────────────┐ │
                        │  │  Racing Branches   │ │
                        │  │                    │ │
                        │  │  • Tabu Search     │ │
                        │  │  • Simulated Ann.  │ │
                        │  │  • QPU Branch:     │ │
                        │  │    - Decompose     │ │
                        │  │    - QPU Sample    │ │
                        │  │    - Compose       │ │
                        │  └────────────────────┘ │
                        │           ↓             │
                        │      ArgMin (select)    │
                        │           ↓             │
                        │    Check Convergence    │
                        └─────────────────────────┘
                                     ↓
                              Final Solution
```

### Workflow Components

1. **Decomposition**: `EnergyImpactDecomposer`
   - Selects subproblem of 40 variables
   - Rolling history to cover different parts
   
2. **QPU Sampling**: `QPUSubproblemAutoEmbeddingSampler`
   - Automatic embedding on QPU
   - 100 reads per iteration
   
3. **Composition**: `SplatComposer`
   - Merges subproblem solutions back
   
4. **Classical Samplers**: `InterruptableTabuSampler`, `SimulatedAnnealingProblemSampler`
   - Run in parallel with QPU
   - Interrupted when QPU completes

5. **Selection**: `ArgMin`
   - Selects best solution from racing branches
   
6. **Iteration**: `Loop`
   - Continues until convergence (3 iterations) or max (15 iterations)

## Usage

### Running Tests

```powershell
conda activate oqi
cd @todo
python test_custom_hybrid.py
```

Expected output:
```
================================================================================
CUSTOM HYBRID WORKFLOW - UNIT TESTS
================================================================================

[TEST 1: Data Generation]
  ✓ Farm data generation
  ✓ Patch data generation

[TEST 2: CQM Creation]
  ✓ Farm CQM: 36 vars, 39 constraints
  ✓ Patch CQM: 18 vars, 3 constraints

[TEST 3: Hybrid Framework Availability]
  ✓ dwave-hybrid imported successfully
  ✓ HYBRID_AVAILABLE flag is True

[TEST 4: Workflow Construction]
  ✓ solve_with_custom_hybrid_workflow function imported
  ✓ Workflow parameters validated

================================================================================
ALL TESTS PASSED ✓
================================================================================
```

### Running Benchmark (Gurobi only, no D-Wave)

```powershell
conda activate oqi
cd @todo
python comprehensive_benchmark_CUSTOM_HYBRID.py --config 10
```

### Running Benchmark with D-Wave

```powershell
conda activate oqi
cd @todo

# Option 1: Set environment variable
$env:DWAVE_API_TOKEN = "YOUR_TOKEN_HERE"
python comprehensive_benchmark_CUSTOM_HYBRID.py --config 10

# Option 2: Pass token directly
python comprehensive_benchmark_CUSTOM_HYBRID.py --config 10 --token YOUR_TOKEN_HERE
```

### Command-Line Options

- `--config N`: Number of units to test (default: 10)
- `--token TOKEN`: D-Wave API token
- `--output-dir DIR`: Output directory (default: Benchmarks/CUSTOM_HYBRID)

## Output

Results are saved to `Benchmarks/CUSTOM_HYBRID/results_config_N_TIMESTAMP.json`:

```json
{
  "n_units": 10,
  "total_land": 100.0,
  "scenarios": {
    "farm": {
      "n_units": 10,
      "n_variables": 120,
      "n_constraints": 130,
      "solvers": {
        "gurobi": {
          "solver": "gurobi",
          "status": "Optimal",
          "objective_value": 0.XXX,
          "solve_time": X.XX,
          "success": true
        },
        "custom_hybrid": {
          "solver": "custom_hybrid",
          "status": "Converged",
          "objective_value": 0.XXX,
          "solve_time": X.XX,
          "qpu_access_time": X.XXXX,
          "iterations": X,
          "success": true
        }
      }
    },
    "patch": {
      ...
    }
  },
  "metadata": {
    "timestamp": "20251119_HHMMSS",
    "config": 10,
    "dwave_enabled": true
  }
}
```

## Design Principles

### Modular Architecture
- **Separation of Concerns**: Each file has a single, clear responsibility
- **Testability**: Utilities split out for easy unit testing
- **Maintainability**: Short, focused files (< 300 lines each)

### Professional Standards (IEEE)
- **Documentation**: Comprehensive docstrings for all functions
- **Error Handling**: Try-except blocks with meaningful messages
- **Security**: No hardcoded credentials (uses placeholder)
- **Logging**: Clear progress messages and summaries

### Best Practices
- **DRY**: Shared utilities prevent code duplication
- **KISS**: Simple, straightforward implementation
- **Testing First**: Unit tests verify components before integration
- **Fail Fast**: Tests check requirements before running expensive benchmarks

## Dependencies

Required packages (install via `conda activate oqi`):
- `dwave-hybrid` - Custom hybrid workflow framework
- `dwave-system` - D-Wave system access
- `dimod` - BQM/CQM models
- `pulp` - Classical solver interface
- Standard libraries: `json`, `time`, `argparse`, etc.

## Next Steps

After custom hybrid workflow is complete:

1. **Alternative 2: Decomposed QPU Implementation**
   - Create `solver_runner_DECOMPOSED.py`
   - Create `comprehensive_benchmark_DECOMPOSED.py`
   - Test with `test_decomposed.py`

2. **Comparison Analysis**
   - Compare custom hybrid vs standard CQM/BQM solvers
   - Analyze QPU utilization
   - Evaluate convergence patterns

3. **Documentation**
   - Update main README with findings
   - Create performance comparison charts
   - Document best practices for hybrid workflows

## Troubleshooting

### Import Error: dwave-hybrid not found
```powershell
pip install dwave-hybrid
```

### No D-Wave Token
Set environment variable or pass via command line:
```powershell
$env:DWAVE_API_TOKEN = "YOUR_TOKEN"
```

### Tests Fail
Check that `oqi` conda environment is activated:
```powershell
conda activate oqi
conda list | Select-String "dwave"
```

## Contact

For questions about this implementation, refer to the dev_plan.md for detailed architecture documentation.
