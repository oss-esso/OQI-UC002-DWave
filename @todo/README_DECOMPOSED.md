# Decomposed QPU Implementation - README

## Overview

This implementation demonstrates **Strategic Problem Decomposition** for quantum-classical optimization:
- **Farm scenarios**: Classical-only optimization (Gurobi MINLP)
- **Patch scenarios**: Quantum-only optimization (low-level QPU direct access)

## Philosophy

Different problem types benefit from different solvers:
- **Continuous problems with constraints** → Classical solvers excel
- **Pure binary problems** → Quantum annealers provide unique advantages

This approach maximizes each solver's strengths rather than using hybrid solvers for everything.

## Files Created

### Core Implementation
1. **solver_runner_DECOMPOSED.py** - Solver with decomposed QPU function
   - `solve_with_decomposed_qpu()` - Low-level QPU access via DWaveSampler
   - Direct control over embedding, annealing parameters
   - Maximum QPU utilization without hybrid overhead

2. **comprehensive_benchmark_DECOMPOSED.py** - Benchmark script
   - Strategic decomposition: farm→classical, patch→quantum
   - Command-line options for QPU parameters
   - Clean separation of solver types

3. **benchmark_utils_decomposed.py** - Modular utilities
   - `run_farm_classical()` - Classical-only farm solver
   - `run_patch_quantum()` - Quantum-only patch solver
   - Clean separation of concerns

4. **test_decomposed.py** - Unit tests
   - Tests data generation
   - Tests BQM conversion
   - Tests low-level sampler availability
   - All components verified independently

## Architecture

### Strategic Decomposition Flow

```
Input Problem
      ↓
  ┌───┴───┐
  │       │
Farm    Patch
(Cont)  (Bin)
  │       │
  ↓       ↓
Gurobi  BQM Conv
MINLP      ↓
  │    DWaveSampler
  │    + Embedding
  │       ↓
  └───┬───┘
      ↓
   Results
```

### Low-Level QPU Access

```
BQM → DWaveSampler → EmbeddingComposite
                            ↓
                    Find Minor-Embedding
                            ↓
                      Program QPU
                            ↓
                    Anneal (20-100μs)
                            ↓
                      Read Results
                            ↓
                     Return Samples
```

## Usage

### Running Tests

```powershell
conda activate oqi
cd @todo
python test_decomposed.py
```

Expected output:
```
================================================================================
DECOMPOSED QPU WORKFLOW - UNIT TESTS
================================================================================

[TEST 1: Data Generation]
  ✓ Farm data generation
  ✓ Patch data generation

[TEST 2: CQM Creation]
  ✓ Farm CQM: 36 vars, 39 constraints
  ✓ Patch CQM: 18 vars, 3 constraints

[TEST 3: BQM Conversion]
  ✓ BQM Conversion: 18 vars, X interactions
  ✓ Invert function available

[TEST 4: Low-Level QPU Sampler Availability]
  ✓ DWaveSampler imported successfully
  ✓ LOWLEVEL_QPU_AVAILABLE flag is True

[TEST 5: Decomposed Solver Function]
  ✓ solve_with_decomposed_qpu function imported
  ✓ Function signature validated
  ✓ QPU parameters validated

================================================================================
ALL TESTS PASSED ✓
================================================================================
```

### Running Benchmark (Classical only)

```powershell
conda activate oqi
cd @todo
python comprehensive_benchmark_DECOMPOSED.py --config 10
```

### Running Benchmark with D-Wave QPU

```powershell
conda activate oqi
cd @todo

# Option 1: Set environment variable
$env:DWAVE_API_TOKEN = "YOUR_TOKEN_HERE"
python comprehensive_benchmark_DECOMPOSED.py --config 10

# Option 2: Pass token directly
python comprehensive_benchmark_DECOMPOSED.py --config 10 --token YOUR_TOKEN_HERE

# Option 3: Custom QPU parameters
python comprehensive_benchmark_DECOMPOSED.py --config 10 --token YOUR_TOKEN --num-reads 2000 --annealing-time 50
```

### Command-Line Options

- `--config N`: Number of units to test (default: 10)
- `--token TOKEN`: D-Wave API token
- `--output-dir DIR`: Output directory (default: Benchmarks/DECOMPOSED)
- `--num-reads N`: Number of QPU reads (default: 1000)
- `--annealing-time T`: Annealing time in microseconds (default: 20)

## Output

Results saved to `Benchmarks/DECOMPOSED/results_config_N_TIMESTAMP.json`:

```json
{
  "n_units": 10,
  "total_land": 100.0,
  "scenarios": {
    "farm": {
      "strategy": "classical_only",
      "solvers": {
        "gurobi": {
          "solver": "gurobi",
          "status": "Optimal",
          "objective_value": 0.XXX,
          "solve_time": X.XX,
          "solver_type": "classical_minlp"
        }
      }
    },
    "patch": {
      "strategy": "quantum_only",
      "solvers": {
        "decomposed_qpu": {
          "solver": "decomposed_qpu",
          "status": "Optimal",
          "objective_value": 0.XXX,
          "solve_time": X.XX,
          "qpu_access_time": 0.XXXX,
          "qpu_programming_time": 0.XXXX,
          "qpu_sampling_time": 0.XXXX,
          "num_reads": 1000,
          "solver_type": "quantum_annealing",
          "qpu_config": {
            "chip_id": "Advantage_system6.X",
            "topology": "pegasus",
            "num_reads": 1000,
            "annealing_time": 20
          }
        }
      }
    }
  }
}
```

## Design Principles

### Strategic Decomposition
- **Match solver to problem type**: Classical for continuous, quantum for binary
- **Maximize strengths**: Use each solver where it excels
- **Avoid overhead**: No hybrid solver overhead for problems that don't need it

### Low-Level QPU Control
- **Direct access**: DWaveSampler instead of LeapHybrid solvers
- **Explicit embedding**: EmbeddingComposite for automatic minor-embedding
- **Parameter control**: Direct control over num_reads, annealing_time, chain_strength
- **Timing visibility**: Detailed QPU timing breakdown

### Professional Standards (IEEE)
- **Documentation**: Comprehensive docstrings explaining decomposition strategy
- **Error Handling**: Try-except blocks with meaningful messages
- **Security**: No hardcoded credentials
- **Modularity**: Clean separation of farm/patch solving logic

## Key Differences from Other Approaches

### vs. Hybrid Solvers (CQM/BQM)
- **No preprocessing overhead**: Direct QPU access
- **Maximum QPU time**: All computation on quantum hardware
- **Explicit control**: Full control over annealing parameters
- **Pure quantum**: No classical post-processing

### vs. Custom Hybrid Workflow
- **Simpler**: No complex workflow construction
- **Specialized**: Farm→classical, patch→quantum (not mixed)
- **Deterministic**: Predictable solver routing
- **Focused**: Each problem type gets optimal solver

## Dependencies

Required packages:
- `dwave-system` - DWaveSampler for low-level QPU access
- `dwave-ocean-sdk` - Ocean tools (dimod, etc.)
- `pulp` - Classical solver interface
- Standard libraries: `json`, `time`, `argparse`, etc.

## QPU Parameters Guide

### num_reads
- **Default**: 1000
- **Range**: 1-10000
- **Effect**: More reads = better sampling of energy landscape
- **Trade-off**: More reads = longer QPU time = higher cost

### annealing_time
- **Default**: 20 μs
- **Range**: 1-2000 μs
- **Effect**: Longer annealing = more time for quantum evolution
- **Trade-off**: Longer time = higher cost, diminishing returns

### chain_strength
- **Default**: Auto-calculated
- **Effect**: Strength of logical qubit chains during embedding
- **Guide**: Auto is usually best; manual only for advanced tuning

## Troubleshooting

### Import Error: DWaveSampler not found
```powershell
pip install dwave-system
```

### No D-Wave Token
Set environment variable or pass via command line:
```powershell
$env:DWAVE_API_TOKEN = "YOUR_TOKEN"
```

### QPU Access Errors
Check QPU availability:
```python
from dwave.system import DWaveSampler
sampler = DWaveSampler()
print(sampler.properties)
```

### Tests Fail
Ensure `oqi` environment is activated:
```powershell
conda activate oqi
conda list | Select-String "dwave"
```

## Performance Expectations

### Farm Scenario (Classical)
- **Solver**: Gurobi MINLP
- **Speed**: Fast (< 1s for small problems)
- **Quality**: Optimal solutions guaranteed

### Patch Scenario (Quantum)
- **Solver**: DWaveSampler QPU
- **Speed**: Depends on num_reads and annealing_time
- **Quality**: High-quality solutions, not guaranteed optimal
- **QPU Time**: Typically 0.01-0.1s for 1000 reads

## Next Steps

1. **Run Tests**: Verify all components work
2. **Run Small Benchmark**: Test with --config 10
3. **Analyze Results**: Compare classical vs quantum performance
4. **Scale Up**: Try larger configurations if successful
5. **Compare Approaches**: Evaluate vs CQM/BQM/Custom Hybrid

## References

- **Dev Plan**: `dev_plan.md` - Detailed architecture
- **Solver Implementation**: `solver_runner_DECOMPOSED.py`
- **D-Wave Docs**: https://docs.ocean.dwavesys.com/
- **DWaveSampler**: https://docs.ocean.dwavesys.com/en/stable/docs_system/reference/samplers.html
