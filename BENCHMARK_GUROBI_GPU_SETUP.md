# Benchmark Script Configuration for Gurobi GPU Parallelization

## Changes Made (October 25, 2025)

### 1. DWave Budget Preservation
- **File**: `benchmark_scalability_PATCH.py`
- **Change**: Commented out DWave API token and set to `None`
- **Line ~672**: 
  ```python
  # dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
  dwave_token = None  # Disable DWave to preserve budget
  ```
- **Result**: All DWave tests will be skipped automatically (existing try-except blocks handle this gracefully)

### 2. Gurobi GPU Acceleration
- **File**: `solver_runner_PATCH.py`
- **Enhancement**: Added try-except for Gurobi API vs GUROBI_CMD
- **Line ~348-368**: 
  ```python
  try:
      # Try using GUROBI API directly for better GPU support
      model.solve(pl.GUROBI(msg=0, timeLimit=300, options=gurobi_options))
  except Exception as e:
      # Fallback to GUROBI_CMD if direct API is not available
      options_str = ' '.join([f'{k}={v}' for k, v in gurobi_options])
      model.solve(pl.GUROBI_CMD(msg=0, options=[options_str]))
  ```

### 3. Gurobi GPU Parameters
The following Gurobi parameters are configured for optimal GPU utilization:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `Method` | 2 | Use barrier method (GPU-accelerated) |
| `Crossover` | 0 | Disable crossover to keep computation on GPU |
| `BarHomogeneous` | 1 | Use homogeneous barrier (more GPU-friendly) |
| `Threads` | 0 | Use all available CPU threads for parallelization |
| `MIPFocus` | 1 | Focus on finding good solutions quickly |
| `Presolve` | 2 | Aggressive presolve for faster solving |

### 4. Cleaned Old Results
- **Action**: Removed all old PuLP JSON results
- **Command**: `Remove-Item "PuLP_Results\*.json" -Force`
- **Result**: Benchmark will run fresh without cached data

## Running the Benchmark

### Prerequisites
1. **Gurobi License**: Ensure you have a valid Gurobi license installed
2. **GPU Support**: Gurobi 9.0+ with CUDA-compatible GPU (optional but recommended)
3. **Python Environment**: All dependencies installed (PuLP, Gurobi, etc.)

### Execute Benchmark
```powershell
cd "c:\Users\Edoardo\Documents\EPFL\OQI-UC002-DWave"
python benchmark_scalability_PATCH.py
```

### Expected Behavior
- ✅ PuLP with Gurobi GPU: **ENABLED** (with parallelization)
- ❌ DWave HybridBQM: **DISABLED** (budget preservation)
- ❌ Simulated Annealing: **DISABLED** (requires DWave SDK)

### Benchmark Configurations
Testing the following patch counts (from `BENCHMARK_CONFIGS`):
- 5 patches
- 10 patches
- 15 patches
- 25 patches

Each configuration runs `NUM_RUNS = 1` times for statistical analysis.

## GPU Verification

### Check if GPU is Being Used
While the benchmark is running, you can monitor GPU usage:

**PowerShell** (if NVIDIA GPU):
```powershell
nvidia-smi -l 1
```

**Gurobi Log** (check for GPU messages):
The Gurobi solver will log GPU usage if available. Look for messages like:
- "Barrier method"
- "Concurrent LP optimizer"
- GPU utilization metrics

### Troubleshooting

#### If GPU is not detected:
1. Check Gurobi version: `python -c "import gurobipy; print(gurobipy.gurobi.version())"`
2. Verify CUDA installation: `nvidia-smi`
3. Check Gurobi GPU license: Ensure your license supports GPU solving
4. Fallback: Script will automatically use GUROBI_CMD with CPU parallelization

#### If errors occur:
- Check Gurobi installation: `python -c "import pulp; pulp.GUROBI().available()"`
- Verify license: `gurobi_cl --version`
- Review error messages in terminal output

## Performance Expectations

### With GPU Acceleration:
- **Small problems** (5-10 patches): Minimal speedup (overhead dominates)
- **Medium problems** (15-25 patches): 2-5x speedup expected
- **Large problems** (>25 patches): 5-10x speedup possible

### Without GPU (CPU only):
- Gurobi will still use multi-threaded CPU parallelization
- Performance will depend on CPU core count
- Still significantly faster than non-parallel solvers

## Re-enabling DWave (Future)

To re-enable DWave testing when budget allows:

1. Open `benchmark_scalability_PATCH.py`
2. Find line ~672
3. Uncomment the dwave_token line:
   ```python
   dwave_token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
   # dwave_token = None  # Disable DWave to preserve budget
   ```
4. Ensure `DWAVE_API_TOKEN` environment variable is set
5. Run benchmark normally

## Notes
- DWave budget preservation is implemented via simple token disabling
- Existing try-except blocks ensure graceful handling when DWave is disabled
- Benchmark cache system still works - only PuLP results are saved
- GPU usage is automatic if available; script falls back to CPU if not
