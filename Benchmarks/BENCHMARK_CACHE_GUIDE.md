# Benchmark Caching System Documentation

## Overview

The benchmark caching system provides intelligent result storage and retrieval to:
- **Save time**: Skip already-completed benchmark runs
- **Incremental saving**: Results saved immediately after each run
- **Organized storage**: Structured folder hierarchy by benchmark type and solver
- **Smart resumption**: Automatically detect and continue incomplete benchmarks

## Folder Structure

```
Benchmarks/
├── BQUBO/
│   ├── CQM/
│   │   ├── config_5_run_1.json
│   │   ├── config_5_run_2.json
│   │   └── ...
│   ├── PuLP/
│   │   └── ...
│   └── DWave/
│       └── ...
├── NLD/
│   ├── CQM/
│   ├── PuLP/
│   ├── Pyomo/
│   └── DWave/
├── NLN/
│   ├── CQM/
│   ├── PuLP/
│   ├── Pyomo/
│   └── DWave/
└── LQ/
    ├── CQM/
    ├── PuLP/
    ├── Pyomo/
    └── DWave/
```

## File Format

Each result file (e.g., `config_5_run_1.json`) contains:

```json
{
  "metadata": {
    "benchmark_type": "NLN",
    "solver": "PuLP",
    "n_farms": 5,
    "run_number": 1,
    "timestamp": "2025-10-23T..."
  },
  "result": {
    "solve_time": 0.123,
    "status": "Optimal",
    "objective_value": 0.548,
    "solution": {...},
    "n_foods": 10,
    "problem_size": 50,
    ...
  },
  "cqm": {  // Only in CQM files
    "num_variables": 100,
    "num_constraints": 50,
    ...
  }
}
```

## Usage

### Check Cache Status

```bash
# Overview of all benchmarks
python check_benchmark_cache.py

# Detailed status for specific benchmark
python check_benchmark_cache.py --detailed NLN

# Detailed status for all benchmarks
python check_benchmark_cache.py --detailed all

# Export summary to JSON
python check_benchmark_cache.py --export --output my_summary.json
```

### Run Benchmarks with Caching

The caching is **automatically integrated** into the benchmark scripts. Just run them normally:

```bash
python benchmark_scalability_NLN.py
python benchmark_scalability_NLD.py
python benchmark_scalability_BQUBO.py
```

The scripts will:
1. Check which runs already exist
2. Skip completed runs
3. Only execute missing runs
4. Save results incrementally after each run

### Adjusting NUM_RUNS

If you've already run benchmarks with `NUM_RUNS = 1` and then change to `NUM_RUNS = 5`, the system will automatically:
1. Detect the 1 existing run
2. Only execute the 4 remaining runs needed
3. Combine all 5 runs for statistical analysis

Example:
```python
# In benchmark_scalability_NLN.py
NUM_RUNS = 5  # Changed from 1

# When you run it:
# - Finds 1 existing run in cache
# - Executes only runs 2, 3, 4, 5
# - Aggregates all 5 runs for statistics
```

## API Reference

### BenchmarkCache Class

```python
from benchmark_cache import BenchmarkCache

# Initialize
cache = BenchmarkCache()

# Get existing runs for a configuration
existing_runs = cache.get_existing_runs('NLN', 'PuLP', n_farms=72)
# Returns: [1, 2, 3] (list of run numbers)

# Get runs still needed
runs_needed = cache.get_runs_needed('NLN', n_farms=72, target_runs=5)
# Returns: {'PuLP': [4, 5], 'Pyomo': [1, 2, 3, 4, 5], ...}

# Save a result
cache.save_result(
    benchmark_type='NLN',
    solver='PuLP',
    n_farms=72,
    run_num=1,
    result_data={
        'solve_time': 0.123,
        'objective_value': 0.548,
        ...
    }
)

# Load a result
result = cache.load_result('NLN', 'PuLP', n_farms=72, run_num=1)

# Get all results for a configuration
all_results = cache.get_all_results('NLN', 'PuLP', n_farms=72)

# Print status report
cache.print_cache_status('NLN', [5, 19, 72, 279, 1096, 1535], target_runs=5)
```

## Integration Pattern for Benchmark Scripts

### Step 1: Add Imports

```python
from benchmark_cache import BenchmarkCache, serialize_cqm
```

### Step 2: Update `run_benchmark` Function

```python
def run_benchmark(n_farms, run_number=1, total_runs=1, cache=None, save_to_cache=True):
    # ... existing code ...
    
    # After each solver completes, save to cache:
    if save_to_cache and cache:
        result_data = {
            'solve_time': solve_time,
            'status': status,
            'objective_value': obj_value,
            'n_foods': n_foods,
            'problem_size': problem_size,
            ...
        }
        cache.save_result(BENCHMARK_TYPE, SOLVER_NAME, n_farms, run_number, result_data)
```

### Step 3: Update `main` Function

```python
def main():
    # Initialize cache
    cache = BenchmarkCache()
    
    # Print cache status
    cache.print_cache_status(BENCHMARK_TYPE, BENCHMARK_CONFIGS, NUM_RUNS)
    
    for n_farms in BENCHMARK_CONFIGS:
        # Load existing results from cache
        existing_results = cache.get_all_results(BENCHMARK_TYPE, PRIMARY_SOLVER, n_farms)
        config_results = convert_cached_to_result_format(existing_results)
        
        # Determine which runs are needed
        runs_needed = cache.get_runs_needed(BENCHMARK_TYPE, n_farms, NUM_RUNS)
        primary_runs_needed = runs_needed.get(PRIMARY_SOLVER, [])
        
        # Only run missing benchmarks
        for run_num in primary_runs_needed:
            result = run_benchmark(n_farms, run_number=run_num, total_runs=NUM_RUNS, 
                                 cache=cache, save_to_cache=True)
            config_results.append(result)
        
        # Continue with aggregation as before...
```

## Benefits

### Time Savings
- **No redundant runs**: Never re-run the same configuration
- **Incremental progress**: Stop and resume anytime
- **Flexible NUM_RUNS**: Change target runs without losing progress

### Data Organization
- **Structured storage**: Easy to find and analyze results
- **Version tracking**: Timestamps on every result
- **Solver comparison**: All solvers in parallel folders

### Robustness
- **Crash recovery**: Results saved immediately, not at end
- **Data integrity**: JSON format, easy to inspect and validate
- **Reproducibility**: Complete metadata for each run

## Troubleshooting

### Clear Cache for Specific Configuration

```python
import os
import glob

# Remove all runs for config 72 in NLN benchmark
for solver in ['CQM', 'PuLP', 'Pyomo', 'DWave']:
    pattern = f'Benchmarks/NLN/{solver}/config_72_run_*.json'
    for file in glob.glob(pattern):
        os.remove(file)
        print(f"Removed: {file}")
```

### Re-run Specific Run Number

```python
# Just delete that specific file
os.remove('Benchmarks/NLN/PuLP/config_72_run_3.json')

# Or in the benchmark script, set save_to_cache=True and run_number=3
result = run_benchmark(72, run_number=3, total_runs=5, cache=cache, save_to_cache=True)
```

### Inspect a Cached Result

```python
import json

with open('Benchmarks/NLN/PuLP/config_72_run_1.json', 'r') as f:
    data = json.load(f)
    print(f"Solve time: {data['result']['solve_time']}")
    print(f"Objective: {data['result']['objective_value']}")
    print(f"Timestamp: {data['metadata']['timestamp']}")
```

## Complete Integration Status

### ✅ Fully Integrated
- **NLN Benchmark**: Complete with all solvers (CQM, PuLP, Pyomo, DWave)

### 🔧 Ready for Integration
- **NLD Benchmark**: Pattern defined, needs manual application
- **BQUBO Benchmark**: Pattern defined, needs manual application

### Integration Checklist for Remaining Benchmarks

For NLD and BQUBO, apply these changes:

1. ✅ Add imports: `from benchmark_cache import BenchmarkCache, serialize_cqm`
2. ⬜ Update `run_benchmark` signature with `cache` and `save_to_cache` parameters
3. ⬜ Add caching after CQM creation
4. ⬜ Add caching after each solver (PuLP, Pyomo, DWave)
5. ⬜ Update `main()` to:
   - Initialize `BenchmarkCache()`
   - Print cache status
   - Load existing results
   - Determine runs needed
   - Only execute missing runs
   - Combine cached and new results for aggregation

## Example: Complete Workflow

```bash
# 1. Check current cache status
python check_benchmark_cache.py --detailed NLN

# 2. Run benchmark (will use cache automatically)
python benchmark_scalability_NLN.py

# 3. Benchmark output shows:
#    - "Loaded 3 existing runs from cache"
#    - "Need to run: [4, 5]"
#    - "✓ Saved PuLP result: config_72_run_4"
#    - "✓ Saved PuLP result: config_72_run_5"

# 4. Check updated cache status
python check_benchmark_cache.py --detailed NLN

# 5. Export summary for analysis
python check_benchmark_cache.py --export

# 6. Results are in organized folders ready for analysis
tree Benchmarks/NLN/
```

## Future Enhancements

Possible future improvements:
- **Result comparison**: Compare current run with cached results
- **Performance tracking**: Track solver performance trends over time
- **Automatic validation**: Verify result consistency
- **Cache compression**: Compress old results to save space
- **Remote caching**: Sync cache with cloud storage
- **Parallel execution**: Run multiple configs in parallel

## Notes

- Cache is stored in `Benchmarks/` folder relative to script location
- JSON format ensures human-readable and version-control friendly
- Folder structure created automatically on first use
- No external database required - simple file-based system
- Compatible with any Python JSON-serializable data

---

**Status**: NLN benchmark fully integrated and tested ✅  
**Next Steps**: Apply same pattern to NLD and BQUBO benchmarks  
**Documentation**: Complete and ready for use
