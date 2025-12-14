# Significant Scenarios Benchmark - Quick Start Guide

## Overview

This benchmark tests **6 carefully selected scenarios** spanning from small (5 farms) to large (100 farms) problems, comparing:
- **Gurobi** (classical MIP solver)
- **D-Wave QPU** (quantum-classical hybrid methods)

## Scenarios Tested

| # | Scenario | Farms | Foods | Variables | QPU Method | Expected Speedup |
|---|----------|-------|-------|-----------|------------|------------------|
| 1 | rotation_micro_25 | 5 | 6 | 90 | clique_decomp | 11.5× |
| 2 | rotation_small_50 | 10 | 6 | 180 | clique_decomp | 6.2× |
| 3 | rotation_medium_100 | 20 | 6 | 360 | clique_decomp | 5.2× |
| 4 | rotation_large_25farms_27foods | 25 | 27 | 2025 | hierarchical | 5.0× |
| 5 | rotation_xlarge_50farms_27foods | 50 | 27 | 4050 | hierarchical | 4.5× |
| 6 | rotation_xxlarge_100farms_27foods | 100 | 27 | 8100 | hierarchical | 2.5× |

## What's Tracked

### Performance Metrics
- **Objective Value**: Total benefit achieved (higher = better)
- **Runtime**: Wall clock time (seconds)
- **Speedup**: `gurobi_time / qpu_time`

### Quality Metrics
- **Gap**: `(gurobi_obj - qpu_obj) / gurobi_obj × 100%`
  - **Positive gap** = QPU found worse solution than Gurobi
  - **Negative gap** = QPU found better solution than Gurobi (!)
  - Target: < 20% gap acceptable for practical use

### Constraint Validation
- **Rotation violations**: Same crop in consecutive periods
- **Diversity violations**: Farm not growing any crop in a period
- **Area violations**: Area allocation errors
- **Total violations**: Sum of all above

## How to Run

### Prerequisites
```bash
# IMPORTANT: Activate the oqi conda environment first!
conda activate oqi

# Ensure you're in the @todo directory
cd /Users/edoardospigarolo/Documents/OQI-UC002-DWave/@todo

# Verify D-Wave token is set (already configured in script)
echo $DWAVE_API_TOKEN

# Run preflight check to verify all dependencies
python preflight_check.py
```

### Run the Benchmark
```bash
# CRITICAL: Must activate oqi environment first!
conda activate oqi

# Option 1: Direct execution
python significant_scenarios_benchmark.py

# Option 2: As executable
./significant_scenarios_benchmark.py

# Option 3: With output logging
python significant_scenarios_benchmark.py 2>&1 | tee benchmark_run_$(date +%Y%m%d_%H%M%S).log
```

### Expected Runtime
- **Small scenarios (5-20 farms)**: ~5-10 minutes each
- **Large scenarios (25-100 farms)**: ~10-20 minutes each
- **Total benchmark**: ~60-90 minutes

**Note**: This will consume D-Wave QPU credits! Estimated usage:
- ~100 QPU calls per scenario × 6 scenarios = ~600 QPU calls
- Cost depends on your D-Wave plan

## Output Files

Results are saved to `significant_scenarios_results/`:

```
significant_scenarios_results/
├── benchmark_results_YYYYMMDD_HHMMSS.json  # Detailed results
└── benchmark_results_YYYYMMDD_HHMMSS.csv   # Tabular format
```

## Understanding the Results

### Example Output
```
Scenario                       Gurobi Obj      QPU Obj      Gap %    Speedup
--------------------------------------------------------------------------------
rotation_micro_25                    4.08         3.75      +8.2%     11.50×
rotation_small_50                    7.17         6.49      +9.6%      6.15×
rotation_medium_100                 14.89        12.98     +12.9%      5.26×
...
--------------------------------------------------------------------------------
Average Gap: +10.2%
Average Speedup: 6.8×
```

### Interpretation

**Good Results**:
- Gap < 20%: QPU solution quality acceptable
- Speedup > 3×: Significant time savings
- Violations = 0: Feasible solution

**Concerning Results**:
- Gap > 30%: QPU solution much worse
- Speedup < 1×: QPU slower than Gurobi
- Violations > 0: Constraint violations detected

## Configuration

Edit `significant_scenarios_benchmark.py` to adjust:

### Gurobi Settings
```python
GUROBI_CONFIG = {
    'timeout': 300,           # 5 minutes max
    'mip_gap': 0.1,          # 10% optimality gap
    'mip_focus': 1,          # Prioritize feasible solutions
    'improve_start_time': 30, # Stop if no improvement for 30s
}
```

### QPU Settings
```python
QPU_CONFIG = {
    'num_reads': 100,         # QPU samples per call
    'farms_per_cluster': 5,   # For hierarchical decomposition
    'num_iterations': 3,      # Boundary coordination iterations
}
```

## Troubleshooting

### Import Errors
```bash
# If clique_decomposition or hierarchical_quantum_solver not found:
ls -la @todo/*.py | grep -E "(clique|hierarchical)"

# Verify they exist and are importable
python -c "from clique_decomposition import solve_rotation_clique_decomposition"
python -c "from hierarchical_quantum_solver import solve_hierarchical"
```

### D-Wave Connection Issues
```bash
# Test D-Wave connection
python -c "from dwave.system import DWaveCliqueSampler; sampler = DWaveCliqueSampler(); print('Connected!')"
```

### Gurobi License Issues
```bash
# Verify Gurobi license
python -c "import gurobipy; m = gurobipy.Model(); print('Gurobi OK')"
```

## Next Steps

After running the benchmark:

1. **Analyze Results**: Check CSV file for detailed metrics
2. **Identify Patterns**: Which method performs best where?
3. **Adjust Parameters**: Tune QPU settings if gaps are too large
4. **Generate Plots**: Use results to create comparison visualizations
5. **Write Report**: Document findings for technical paper

## Notes

- **First 3 scenarios (5-20 farms)** use clique decomposition - good for small problems
- **Last 3 scenarios (25-100 farms)** use hierarchical decomposition - scales to large problems
- **Gap can be negative** if QPU finds better solution than Gurobi (rare but possible!)
- **Gurobi timeout** means it couldn't find optimal solution in 5 minutes
- **Constraint violations** should ideally be zero - investigate if > 0

## Contact

For issues or questions about this benchmark, check:
- Main project README: `/Users/edoardospigarolo/Documents/OQI-UC002-DWave/README.md`
- Scenario definitions: `@todo/significant_scenarios/complete_scenarios_inventory.json`
- Analysis notes: `@todo/SIGNIFICANT_SCENARIOS_ANALYSIS.md`
