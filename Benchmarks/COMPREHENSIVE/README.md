# Comprehensive Benchmark - Quantum vs Classical Solver Comparison

This directory contains comprehensive benchmark results comparing quantum and classical optimization solvers across different problem scenarios.

## Overview

The comprehensive benchmark tests two main scenarios:

### 1. Farm Scenario (Large-scale)
- **Problem Type**: Full-scale farm optimization
- **Solvers Tested**:
  - Gurobi (via PuLP) - Classical solver on CQM formulation
  - D-Wave CQM - Quantum hybrid solver (when enabled)
- **Goal**: Test on realistic large-scale problems

### 2. Patch Scenario (Medium-scale)  
- **Problem Type**: Smaller patch optimization
- **Solvers Tested**:
  - Gurobi (via PuLP) - Classical solver on CQM formulation
  - D-Wave CQM - Quantum hybrid solver (when enabled)
  - Gurobi QUBO - Classical solver on BQM (QUBO) formulation
  - D-Wave BQM - Quantum BQM solver (when enabled)
- **Goal**: Demonstrate quantum advantage by comparing solver performance on different formulations

## Key Insight: Quantum Advantage Demonstration

The benchmark is designed to show quantum advantage by testing the **same problem** with **different formulations**:

1. **Classical Gurobi (CQM)**: Solves the original constrained problem efficiently (~0.05s)
2. **Gurobi QUBO (BQM)**: Struggles on the QUBO formulation (~300s+, often hits time limit)
3. **D-Wave BQM**: Efficiently solves the QUBO formulation using quantum annealing

This demonstrates that:
- Classical solvers excel at constrained optimization (CQM)
- Classical solvers struggle when problems are converted to QUBO (BQM)
- Quantum solvers can efficiently solve QUBO formulations

## Running Benchmarks

### Basic Usage

```bash
# Run with a specific number of samples (no D-Wave)
python comprehensive_benchmark.py 5

# Run with predefined configurations (5, 10, 15, 20, 25 samples)
python comprehensive_benchmark.py --configs

# Run with D-Wave enabled
python comprehensive_benchmark.py 5 --dwave
python comprehensive_benchmark.py --configs --dwave

# Custom output filename
python comprehensive_benchmark.py 5 --output my_results.json
```

### Configuration Options

The benchmark can use predefined configurations defined in `BENCHMARK_CONFIGS`:
```python
BENCHMARK_CONFIGS = [5, 10, 15, 20, 25]
```

These represent the number of farms/patches to test for each scenario.

### Environment Variables

For D-Wave support:
```bash
# Set D-Wave API token
export DWAVE_API_TOKEN="your-token-here"  # Linux/Mac
set DWAVE_API_TOKEN=your-token-here       # Windows CMD
$env:DWAVE_API_TOKEN="your-token-here"    # Windows PowerShell

# Or pass token directly
python comprehensive_benchmark.py 5 --dwave --token "your-token-here"
```

## Benchmark Results

Results are automatically saved to `Benchmarks/COMPREHENSIVE/` with timestamped filenames:

```
comprehensive_benchmark_5samples_classical_20251026_120835.json
comprehensive_benchmark_configs_dwave_20251026_123456.json
```

### Result Structure

```json
{
  "metadata": {
    "timestamp": "2025-10-26T12:08:35",
    "n_samples": 5,
    "total_runtime": 300.5,
    "dwave_enabled": false,
    "scenarios": ["farm", "patch"],
    "solvers": {
      "farm": ["gurobi", "dwave_cqm"],
      "patch": ["gurobi", "dwave_cqm", "gurobi_qubo", "dwave_bqm"]
    }
  },
  "farm_results": [...],
  "patch_results": [...],
  "summary": {
    "farm_samples_completed": 5,
    "patch_samples_completed": 5,
    "total_solver_runs": 30
  }
}
```

## Timing Breakdown

Each solver result includes detailed timing:

- **Farm/Patch + Gurobi**: 
  - `solve_time`: Total time including model setup
  - `solver_time`: Pure solver time

- **Farm/Patch + D-Wave CQM**:
  - `solve_time`: Total time including communication
  - `qpu_time`: Actual quantum processing time
  - `hybrid_time`: Classical preprocessing time

- **Patch + Gurobi QUBO**:
  - `solve_time`: QUBO solving time
  - `bqm_conversion_time`: CQM → BQM conversion time
  - `bqm_energy`: Energy of BQM solution

- **Patch + D-Wave BQM**:
  - `solve_time`: Total BQM solving time
  - `qpu_time`: Quantum processing time
  - `hybrid_time`: Classical preprocessing time
  - `bqm_conversion_time`: CQM → BQM conversion time

## Performance Notes

### Classical Solver (Gurobi on CQM)
- Very fast (~0.05-0.5s)
- Handles constraints natively
- Optimal for constrained problems

### Classical Solver (Gurobi on BQM/QUBO)
- **Much slower** (300s+ per problem)
- Penalties replace constraints
- Often hits time limit
- Shows why QUBO is challenging for classical solvers

### Quantum Solver (D-Wave CQM)
- Fast for constrained problems (~5-30s)
- Hybrid classical-quantum approach
- Handles constraints better than pure QUBO

### Quantum Solver (D-Wave BQM)
- Specialized for QUBO problems
- Can outperform classical QUBO solvers
- Benefits from quantum annealing

## Plotting Results

After running benchmarks, visualize with:

```bash
python plot_comprehensive_results.py Benchmarks/COMPREHENSIVE/comprehensive_benchmark_*.json
```

## Cache System

The benchmark supports caching to avoid re-running expensive computations. Cache files are stored per solver type in subdirectories:

```
Benchmarks/COMPREHENSIVE/
├── CQM/
│   ├── config_5_run_1.json
│   └── config_10_run_1.json
├── DWave/
│   ├── config_5_run_1.json
│   └── config_10_run_1.json
├── PuLP/
│   ├── config_5_run_1.json
│   └── config_10_run_1.json
└── README.md
```

## Expected Runtime

Without D-Wave (classical only):
- 1 sample: ~10 minutes (300s per Gurobi QUBO)
- 5 samples: ~50 minutes
- Full configs (5,10,15,20,25): ~5 hours

With D-Wave enabled:
- 1 sample: ~1 minute
- 5 samples: ~5 minutes
- Full configs: ~30 minutes

**Note**: Gurobi QUBO has a 300-second time limit per problem. Without this, it could take hours per problem!

## Troubleshooting

### Missing Dependencies
```bash
pip install dimod dwave-system dwave-ocean-sdk pulp
```

### Gurobi Not Found
Make sure Gurobi is installed and licensed:
```bash
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

### D-Wave Connection Issues
Verify your token:
```bash
dwave ping --client hybrid
```

### Memory Issues
For large problem sizes, increase available memory or reduce `BENCHMARK_CONFIGS`.

## Analysis Tips

1. **Compare solve times**: Look at `solve_time` across solvers
2. **Check solution quality**: Compare `objective_value` across solvers
3. **Quantum advantage**: Compare D-Wave BQM vs Gurobi QUBO times
4. **Scalability**: Plot solve time vs problem size
5. **Timing breakdown**: Analyze QPU time vs total time for D-Wave

## References

- [D-Wave Ocean SDK Documentation](https://docs.ocean.dwavesys.com/)
- [Gurobi Optimizer](https://www.gurobi.com/documentation/)
- [PuLP Documentation](https://coin-or.github.io/pulp/)
