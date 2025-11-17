# D-Wave SampleSet Storage Utility - Usage Guide

## Overview

The `Utils/save_dwave_sampleset.py` utility provides a standardized way to save D-Wave SampleSet objects to pandas DataFrames for future analysis. This ensures that all solution data from D-Wave solvers is preserved, not just the best solution.

## File Structure

Samplesets are saved in the following directory structure:

```
Benchmarks/
└── {BENCHMARK_TYPE}/
    └── {Scenario}_{Solver}/
        ├── config_{N}_run_{M}.json          # Best solution results (existing)
        └── samplesets/
            └── {benchmark}_{scenario}_{solver}_config{N}_run{M}_{timestamp}.csv
```

### Example:
```
Benchmarks/
└── COMPREHENSIVE/
    └── Patch_DWave/
        ├── config_10_run_1.json
        └── samplesets/
            └── comprehensive_Patch_DWave_config10_run1_20251117_143052.csv
```

## Filename Format

```
{benchmark_lower}_{scenario}_{solver}_[config{N}]_[run{M}]_{timestamp}.csv
```

- `benchmark_lower`: Benchmark type in lowercase (e.g., "comprehensive", "lq", "patch")
- `scenario`: Scenario type (e.g., "Farm", "Patch")
- `solver`: Solver type (e.g., "DWave", "DWaveBQM")
- `config{N}`: Configuration ID (optional, e.g., "config10")
- `run{M}`: Run ID (optional, e.g., "run1")
- `timestamp`: Date and time in format YYYYMMDD_HHMMSS

### Examples:
- `comprehensive_Patch_DWave_config10_run1_20251117_143052.csv`
- `lq_Farm_DWave_config25_run3_20251117_150230.csv`
- `rotation_Patch_DWaveBQM_config15_20251117_162045.csv`

## Usage in Benchmark Scripts

### Basic Usage

```python
from Utils.save_dwave_sampleset import save_sampleset_to_dataframe

# After getting sampleset from D-Wave
sampleset, solve_time, qpu_time = solver.solve_with_dwave_cqm(cqm, token)

# Save the complete sampleset
filepath = save_sampleset_to_dataframe(
    sampleset=sampleset,
    benchmark_type='COMPREHENSIVE',
    scenario_type='Patch',
    solver_type='DWave',
    config_id=10,
    run_id=1
)
print(f"Saved sampleset to: {filepath}")
```

### With Error Handling

```python
# Save complete sampleset to DataFrame for future analysis
try:
    sampleset_path = save_sampleset_to_dataframe(
        sampleset=sampleset,
        benchmark_type='COMPREHENSIVE',
        scenario_type='Farm',
        solver_type='DWave',
        config_id=sample_data['n_units'],
        run_id=1
    )
    print(f"  ✓ Saved sampleset to: {os.path.basename(sampleset_path)}")
except Exception as e:
    print(f"  Warning: Failed to save sampleset: {e}")
```

### Convenience Function (Auto-parsing)

```python
from Utils.save_dwave_sampleset import save_sampleset_from_benchmark

# Automatically parse scenario and solver from directory name
filepath = save_sampleset_from_benchmark(
    sampleset=sampleset,
    benchmark_category='COMPREHENSIVE',
    solver_dir='Patch_DWave',  # Automatically parsed to scenario='Patch', solver='DWave'
    config_id=10,
    run_id=1
)
```

## DataFrame Structure

The saved CSV files include:

### Metadata Columns (added automatically):
- `benchmark_type`: Benchmark category
- `scenario_type`: Scenario name
- `solver_type`: Solver used
- `config_id`: Configuration ID (if provided)
- `run_id`: Run ID (if provided)
- `timestamp`: When the file was created

### SampleSet Data (from D-Wave):
- `energy`: Energy value for each sample
- `num_occurrences`: Number of times this sample was found
- `is_feasible`: Whether the sample satisfies all constraints
- Variable columns: One column per variable in the problem (e.g., `Y_Farm1_Wheat`, `Y_Farm2_Corn`, etc.)

## Loading Saved Data

```python
from Utils.save_dwave_sampleset import load_sampleset_dataframe
import pandas as pd

# Load a specific sampleset
df = load_sampleset_dataframe('path/to/sampleset.csv')

# Or use pandas directly
df = pd.read_csv('path/to/sampleset.csv')

# Filter for feasible solutions
feasible = df[df['is_feasible'] == True]

# Get best energy
best_solution = df.loc[df['energy'].idxmin()]

# Analyze energy distribution
print(df['energy'].describe())
```

## Finding Saved Samplesets

```python
from Utils.save_dwave_sampleset import list_saved_samplesets

# List all samplesets
all_files = list_saved_samplesets()

# Filter by benchmark type
comprehensive_files = list_saved_samplesets(benchmark_type='COMPREHENSIVE')

# Filter by scenario and solver
patch_dwave_files = list_saved_samplesets(
    benchmark_type='COMPREHENSIVE',
    scenario_type='Patch',
    solver_type='DWave'
)

for filepath in patch_dwave_files:
    print(filepath)
```

## Integration Status

### ✅ Updated Benchmark Scripts:
- **comprehensive_benchmark.py**: Saves samplesets for:
  - Farm D-Wave CQM
  - Patch D-Wave CQM
  - Patch D-Wave BQM

### 🔄 To Be Updated:
Other benchmark scripts should follow the same pattern:
- `benchmark_scalability_LQ.py`
- `benchmark_scalability_NLN.py`
- `benchmark_scalability_PATCH.py`
- `rotation_benchmark.py`
- Any other scripts using D-Wave solvers

## Benefits

1. **Complete Data Preservation**: Saves ALL solutions, not just the best one
2. **Easy Analysis**: CSV format can be opened in Excel, Python, R, etc.
3. **Standardized Naming**: Consistent file naming across all benchmarks
4. **Organized Storage**: Clear directory structure
5. **Future-Proof**: All solution data available for re-analysis
6. **Metadata Tracking**: Benchmark type, config, run, and timestamp stored with data

## Example Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load a sampleset
df = pd.read_csv('comprehensive_Patch_DWave_config10_run1_20251117_143052.csv')

# Plot energy distribution
plt.figure(figsize=(10, 6))
plt.hist(df['energy'], bins=50)
plt.xlabel('Energy')
plt.ylabel('Count')
plt.title(f"Energy Distribution - {df['scenario_type'][0]} {df['solver_type'][0]}")
plt.show()

# Compare feasible vs infeasible
print(f"Feasible solutions: {df['is_feasible'].sum()}")
print(f"Infeasible solutions: {(~df['is_feasible']).sum()}")

# Analyze variable patterns in best solutions
best_10 = df.nsmallest(10, 'energy')
variable_cols = [col for col in df.columns if col.startswith('Y_')]
print("Top 10 solutions variable patterns:")
print(best_10[variable_cols])
```

## Notes

- Sampleset files can be large for problems with many variables
- CSV format is human-readable and compatible with many tools
- The original JSON result files still contain the best solution summary
- Samplesets complement the JSON results with complete solution data
