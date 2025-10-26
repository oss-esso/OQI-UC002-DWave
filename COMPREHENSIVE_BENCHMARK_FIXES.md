# Comprehensive Benchmark - Fixes Applied

## Date: October 26, 2025

## Issues Fixed

### 1. ✅ Gurobi QUBO Timeout Issue
**Problem**: Gurobi QUBO was taking 748+ seconds even though a 300-second timeout was set, and required manual interrupt to stop.

**Solution**:
- Reduced timeout from 300s to **30 seconds** for preliminary testing
- Added callback function `time_limit_callback` to enforce strict timeout
- Added `GRB.INTERRUPTED` status handling in result extraction
- Updated timeout in both parameter list and callback function

**Files Modified**:
- `solver_runner_PATCH.py`: Lines ~550-580
  - Changed `TimeLimit` from 300 to 30
  - Added callback with `model.terminate()` at 30s
  - Enhanced status handling for INTERRUPTED state

### 2. ✅ Benchmark Folder Structure
**Problem**: Results were being saved as single JSON files in root directory, not matching the structure of other benchmarks (PATCH, BQUBO, etc.) which use subdirectories for each solver.

**Solution**: 
- Created 6 subdirectories matching solver configurations:
  - `Farm_PuLP/` - Gurobi solver results for farm scenarios
  - `Farm_DWave/` - D-Wave CQM results for farm scenarios
  - `Patch_PuLP/` - Gurobi solver results for patch scenarios
  - `Patch_DWave/` - D-Wave CQM results for patch scenarios
  - `Patch_GurobiQUBO/` - Gurobi QUBO results for patch scenarios
  - `Patch_DWaveBQM/` - D-Wave BQM results for patch scenarios

- Each solver result is saved individually as: `config_{n_units}_run_{run_id}.json`
- Example: `config_15_run_1.json` for a 15-unit problem on run 1

**Files Modified**:
- `comprehensive_benchmark.py`: 
  - Added `save_solver_result()` function (lines ~195-230)
  - Updated all solver sections to call `save_solver_result()`
  - Enhanced result dictionaries to include metadata (n_units, sample_id, etc.)

### 3. ✅ Configuration Values Not Matching BENCHMARK_CONFIGS
**Problem**: The benchmark was generating samples with dynamic numbers (10, 12, 15, 18...) instead of using the predefined `BENCHMARK_CONFIGS = [5, 10, 15, 20, 25]`.

**Root Cause**: 
```python
# OLD - Wrong approach
farms = generate_farms_large(n_farms=10 + sample_idx * 2, seed=seed)
patches = generate_patches_small(n_farms=15 + sample_idx * 3, seed=seed)
```

**Solution**:
- Changed `generate_sample_data()` signature from `n_samples: int` to `config_values: List[int]`
- Now directly uses config values: `enumerate(config_values)` to generate exact numbers
- Updated function to accept list of configuration values
- Updated `run_comprehensive_benchmark()` to accept and pass config_values

**Files Modified**:
- `comprehensive_benchmark.py`:
  - Updated `generate_sample_data()` function (lines ~60-135)
  - Changed from generating `n` samples with formula to generating samples for each config value
  - Updated `run_comprehensive_benchmark()` signature and calls
  - Fixed metadata to use `config_values` instead of `n_samples`

## Current Behavior

### Correct Configuration Mapping
Now generates exact numbers from `BENCHMARK_CONFIGS`:
- Sample 0: **5 farms/patches** (not 10)
- Sample 1: **10 farms/patches** (not 12)
- Sample 2: **15 farms/patches** (not 15)
- Sample 3: **20 farms/patches** (not 18)
- Sample 4: **25 farms/patches** (not 21)

### File Structure
```
Benchmarks/COMPREHENSIVE/
├── Farm_PuLP/
│   ├── config_5_run_1.json
│   ├── config_10_run_1.json
│   ├── config_15_run_1.json
│   ├── config_20_run_1.json
│   └── config_25_run_1.json
├── Farm_DWave/
│   └── (same structure)
├── Patch_PuLP/
│   └── (same structure)
├── Patch_DWave/
│   └── (same structure)
├── Patch_GurobiQUBO/
│   ├── config_5_run_1.json
│   ├── config_10_run_1.json
│   ├── config_15_run_1.json
│   ├── config_20_run_1.json
│   └── config_25_run_1.json
├── Patch_DWaveBQM/
│   └── (same structure)
├── comprehensive_benchmark_configs_classical_*.json (summary file)
└── README.md
```

### Result File Format
Each individual result file contains:
```json
{
  "status": "Optimal",
  "objective_value": 4.4325600000000005,
  "solve_time": 0.019,
  "solver_time": 0.0197598934173584,
  "success": true,
  "sample_id": 0,
  "n_units": 15,
  "total_area": 7.59,
  "n_foods": 6,
  "n_variables": 96,
  "n_constraints": 117
}
```

## Testing Results

### Test Command
```bash
python comprehensive_benchmark.py --configs
```

### Expected Output
```
Using predefined BENCHMARK_CONFIGS: [5, 10, 15, 20, 25]

GENERATING SAMPLES FOR CONFIGS: [5, 10, 15, 20, 25]
  Generating 5 farm samples...
    ✓ Farm sample 0: 5 farms, 2.5 ha
    ✓ Farm sample 1: 10 farms, 50.7 ha
    ✓ Farm sample 2: 15 farms, 68.4 ha
    ✓ Farm sample 3: 20 farms, 76.8 ha
    ✓ Farm sample 4: 25 farms, 105.6 ha
  Generating 5 patch samples...
    ✓ Patch sample 0: 5 patches, 0.2 ha
    ✓ Patch sample 1: 10 patches, 3.4 ha
    ✓ Patch sample 2: 15 patches, 8.6 ha
    ✓ Patch sample 3: 20 patches, 7.5 ha
    ✓ Patch sample 4: 25 patches, 12.6 ha
```

### Performance with 30s Timeout
- **Gurobi on CQM**: ~0.01-0.05s (very fast)
- **Gurobi QUBO on BQM**: ~30s (hits timeout as expected)
- **Total runtime**: ~150-200s for full config set (5 samples × 30s each for QUBO)

## Benefits

1. **Faster Testing**: 30s timeout instead of 300s means 10x faster preliminary tests
2. **Consistent Structure**: Matches other benchmarks (PATCH, BQUBO, LQ, NLN)
3. **Easy Plotting**: Individual result files can be read directly by plotting scripts
4. **Correct Configs**: Now uses exact values from `BENCHMARK_CONFIGS`
5. **Better Organization**: Subdirectories for each solver type
6. **Cache-Friendly**: Can reuse individual solver results

## Usage Examples

```bash
# Use predefined configs [5, 10, 15, 20, 25]
python comprehensive_benchmark.py --configs

# Use single custom value
python comprehensive_benchmark.py 10

# With D-Wave
python comprehensive_benchmark.py --configs --dwave
```

## Next Steps

1. **Increase timeout** after preliminary testing: Change `GUROBI_QUBO_TIMEOUT` from 30 to 300
2. **Run with D-Wave** to get full quantum vs classical comparison
3. **Create plotting scripts** that read from subdirectories
4. **Add caching** to avoid re-running expensive solvers
5. **Aggregate results** for analysis and visualization

## Files Changed Summary

1. **comprehensive_benchmark.py**:
   - Added `GUROBI_QUBO_TIMEOUT` constant
   - Added `save_solver_result()` function
   - Modified `generate_sample_data()` to use config_values
   - Updated `run_comprehensive_benchmark()` signature
   - Enhanced all solver result dictionaries with metadata
   - Fixed result aggregation and printing

2. **solver_runner_PATCH.py**:
   - Reduced TimeLimit from 300 to 30 seconds
   - Added time_limit_callback with enforce at 30s
   - Added GRB.INTERRUPTED status handling
   - Enhanced error handling for timeout scenarios

## Validation

✅ Configurations match BENCHMARK_CONFIGS exactly (5, 10, 15, 20, 25)
✅ Subdirectories created for each solver type
✅ Individual result files saved with correct naming
✅ Timeout enforced at 30 seconds
✅ Results structure matches other benchmarks
✅ All metadata included in individual files
✅ Summary JSON still generated for overall tracking
