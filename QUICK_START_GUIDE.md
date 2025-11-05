# Quick Start Guide: 3-Period Crop Rotation Benchmark

## Prerequisites
- Conda environment `oqi` activated
- D-Wave API token (optional, for running actual solvers)

## Step 1: Generate Rotation Matrix
First, generate the crop-to-crop rotation synergy matrix:

```bash
conda activate oqi
python rotation_matrix.py
```

**Output:** Creates `rotation_data/` directory with:
- `rotation_crop_matrix.csv` (27×27 crop synergy matrix)
- `rotation_group_matrix.csv` (5×5 food group matrix)
- `group_env_means.csv` (environmental impacts)

## Step 2: Run Test
Verify the implementation works correctly:

```bash
python test_rotation_benchmark.py
```

**Expected Output:**
```
================================================================================
TEST: 3-Period Crop Rotation Benchmark
================================================================================

[Test 1] Generating rotation scenario with 3 plots...
  ✓ Generated scenario: 3 plots, 10.0 ha

[Test 2] Creating rotation configuration...
  ✓ Created config: 27 foods, 5 food groups

[Test 3] Running rotation scenario (CQM creation only)...
  ✓ Created CQM successfully
    Variables: 243
    Constraints: 120
  ✓ Variable count verified: 243 (3 plots × 27 crops × 3 periods)

================================================================================
✅ ALL TESTS PASSED
================================================================================
```

## Step 3: Run Small Benchmark (No D-Wave Token)
Test CQM creation for multiple configurations:

```bash
python rotation_benchmark.py --configs 5 10
```

This will:
- Generate scenarios with 5 and 10 plots
- Create the 3-period rotation CQM
- Report variable/constraint counts and build times
- Skip D-Wave solver (no token provided)

## Step 4: Run Full Benchmark (With D-Wave Token)
To actually solve the problems with D-Wave:

```bash
export DWAVE_API_TOKEN="your-dwave-token-here"
python rotation_benchmark.py --configs 5 10 15 --gamma 0.1
```

**Parameters:**
- `--configs 5 10 15`: Test with 5, 10, and 15 plots
- `--gamma 0.1`: Rotation synergy weight (default: 0.1)
- `--total-land 100.0`: Total land area in hectares (default: 100)

**Output:** Creates `benchmark_rotation_3period_YYYYMMDD_HHMMSS.json` with:
- Problem statistics (variables, constraints, build time)
- D-Wave CQM solution (objective value, solve time, QPU time)
- Solution summary (per-period crop assignments)
- Linear value and rotation synergy components

## Understanding the Output

### CQM Statistics
For n plots, 27 crops, 3 periods:
- **Variables**: n × 27 × 3 (binary variables Y_{p,c,t})
- **Constraints**: 
  - Plot assignments: n × 3 (one per plot per period)
  - Min/max crop constraints: 27 × 2 × 3 = 162
  - Food group constraints: 5 × 2 × 3 = 30

### Objective Components
The solution reports:
- **Total Objective**: Combined value (to maximize)
- **Linear Value**: Crop values across all periods
- **Rotation Synergy**: Bonus from beneficial crop rotations

Example:
```json
{
  "objective_value": 0.523456,
  "linear_value": 0.498123,
  "rotation_synergy": 0.025333
}
```

## Tuning Parameters

### Gamma (Rotation Weight)
- **Low (0.01-0.05)**: Prioritizes crop values, minimal rotation consideration
- **Medium (0.1-0.2)**: Balanced approach (recommended)
- **High (0.5-1.0)**: Strong emphasis on rotation synergy

Test different values:
```bash
python rotation_benchmark.py --configs 10 --gamma 0.05
python rotation_benchmark.py --configs 10 --gamma 0.1
python rotation_benchmark.py --configs 10 --gamma 0.5
```

## Troubleshooting

### "Rotation matrix not found"
Run `python rotation_matrix.py` first to generate the matrix.

### Import errors in IDE
The code is correct - these are just missing type stubs. The actual modules are installed in the conda environment.

### "No D-Wave token"
Either:
1. Set environment variable: `export DWAVE_API_TOKEN="token"`
2. Use command-line: `--dwave-token "token"`
3. Run without solver (creates CQM only)

### Large problem sizes
For plots > 20, CQM creation may take several minutes. Start with smaller configs to verify functionality.

## File Structure
```
rotation_data/
├── rotation_crop_matrix.csv       # Crop-to-crop synergy matrix
├── rotation_group_matrix.csv      # Food group synergy matrix
└── group_env_means.csv            # Environmental impact data

rotation_benchmark.py              # Main benchmark script
solver_runner_ROTATION.py          # Solver implementation
test_rotation_benchmark.py         # Test script
rotation_matrix.py                 # Matrix generator
benchmark_rotation_3period_*.json  # Results (after running)
```

## Next Steps
1. ✅ Basic functionality verified
2. Run with small configs (5, 10 plots)
3. Analyze rotation patterns in solutions
4. Compare different gamma values
5. Scale up to larger problems (15, 20+ plots)
6. Visualize crop rotation sequences
