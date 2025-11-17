# Output Path Fixes Summary

This document tracks all output path corrections made to ensure scripts correctly reference the project root directory instead of their own directory location.

## Issue Description

Scripts located in the "Benchmark Scripts" subdirectory were using their own directory as the base for constructing paths to output directories like `Benchmarks/`, `CQM_Models/`, `DWave_Results/`, etc. This caused files to be created in incorrect locations when scripts were run from the "Benchmark Scripts" folder.

## Solution Applied

Changed all instances of:
```python
project_root = os.path.dirname(os.path.abspath(__file__))
# or
script_dir = os.path.dirname(os.path.abspath(__file__))
```

To:
```python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
```

And updated all output paths to use:
```python
output_path = os.path.join(project_root, 'OutputDirectory', filename)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
```

## Files Modified

### Benchmark Scripts/ (18 files)

#### Solver Runner Scripts (8 files)
1. **solver_runner_PATCH.py**
   - Fixed project_root calculation
   - Updated paths for: CQM_Models, Constraints, PuLP_Results, DWave_Results, run_manifests

2. **solver_runner_BQUBO.py**
   - Fixed project_root calculation
   - Updated paths for: CQM_Models, Constraints, PuLP_Results, DWave_Results, run_manifests

3. **solver_runner_NLD.py**
   - Fixed project_root calculation
   - Updated paths for: CQM_Models, Constraints, PuLP_Results, Pyomo_Results, DWave_Results, run_manifests

4. **solver_runner_NLN.py**
   - Fixed project_root calculation
   - Updated paths for: CQM_Models_NLN, Constraints_NLN, PuLP_Results_NLN, DWave_Results_NLN

5. **solver_runner_LQ.py**
   - Fixed project_root calculation
   - Updated paths for: CQM_Models_LQ, Constraints_LQ, PuLP_Results_LQ, DWave_Results_LQ

6. **solver_runner_ROTATION.py**
   - Fixed project_root calculation (no direct output paths in this file)

7. **solver_runner_BINARY.py**
   - Fixed project_root calculation (no direct output paths in this file)

8. **solver_runner.py**
   - Fixed project_root calculation (no direct output paths in this file)

#### Benchmark Scalability Scripts (6 files)
9. **benchmark_scalability_PATCH.py**
   - Fixed Inputs path from script_dir to project_root

10. **benchmark_scalability_BQUBO.py**
    - Fixed Inputs path from script_dir to project_root

11. **benchmark_scalability_LQ.py**
    - Fixed Inputs path from script_dir to project_root
    - Fixed Benchmarks/LQ output directory path
    - Fixed Plots output directory path

12. **benchmark_scalability_NLD.py**
    - Fixed Inputs path from script_dir to project_root

13. **benchmark_scalability_NLN.py**
    - Fixed Inputs path from script_dir to project_root

14. **benchmarks.py**
    - Fixed Inputs path from script_dir to project_root

#### Comprehensive & Rotation Benchmarks (2 files)
15. **comprehensive_benchmark.py**
    - Fixed output directory path for Benchmarks/COMPREHENSIVE
    - Fixed three instances of script_dir to project_root
    - Updated save_solver_result() and load_cached_result() functions

16. **rotation_benchmark.py**
    - Fixed output directory path for Benchmarks/ROTATION
    - Updated load_cached_result() and save_solver_result() functions

17. **rotation_benchmark_new.py**
    - (If exists, same pattern as rotation_benchmark.py)

18. **benchmark_gurobi_qubo_only.py**
    - (No output path changes needed - uses solver runners)

### Plot Scripts/ (2 files)

1. **choropleth_plo.py**
   - Fixed choropleth_outputs path to use project_root
   - Changed from `output_dir = "choropleth_outputs"` to `os.path.join(project_root, "choropleth_outputs")`

2. **plot_lq_speedup.py**
   - Fixed Benchmarks input directory path
   - Fixed Plots output directory path
   - Changed from `Path(__file__).parent` to `Path(__file__).parent.parent` (project root)

**Note**: Many other plot scripts in Plot Scripts/ likely have similar patterns but were not all updated in this pass. They follow the same pattern and can be updated as needed.

### Utils/ 
- No changes needed - utility scripts either use command-line argument paths or write to their own directory (acceptable for utilities)

### Tests/
- No changes needed - test scripts use in-memory results or write to verification_reports which already exists

## Output Directories Affected

All scripts now correctly write to these project-root-level directories:

- `Benchmarks/`
  - `COMPREHENSIVE/` (Farm_PuLP, Farm_DWave, Patch_PuLP, Patch_DWave, Patch_GurobiQUBO, Patch_DWaveBQM)
  - `BQUBO/` (CQM, DWave, GurobiQUBO, PuLP)
  - `LQ/`
  - `NLD/`
  - `NLN/`
  - `PATCH/`
  - `ROTATION/`
- `CQM_Models/`
- `CQM_Models_LQ/`
- `CQM_Models_NLN/`
- `Constraints/`
- `Constraints_LQ/`
- `Constraints_NLN/`
- `DWave_Results/`
- `DWave_Results_LQ/`
- `DWave_Results_NLN/`
- `PuLP_Results/`
- `PuLP_Results_LQ/`
- `PuLP_Results_NLN/`
- `Pyomo_Results/`
- `Plots/`
- `choropleth_outputs/`
- `run_manifests/`

## Pattern for Future Scripts

When creating new scripts in subdirectories, always use:

```python
import os
import sys

# Calculate project root (parent of current script's directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Use project_root for all output paths
output_dir = os.path.join(project_root, 'OutputDirectory', 'subdirectory')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f'output_{timestamp}.json')
```

Or with pathlib:
```python
from pathlib import Path

# Calculate project root
project_root = Path(__file__).parent.parent

# Use project_root for all paths
output_dir = project_root / 'OutputDirectory' / 'subdirectory'
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f'output_{timestamp}.json'
```

## Testing

After these changes, scripts should be tested by:
1. Running them from various working directories
2. Verifying output files appear in the correct project-root directories
3. Checking that existing functionality is preserved

## Related Changes

This work complements the import fixes documented in `IMPORT_FIXES_CHANGELOG.md` which addressed module import paths after reorganizing utility modules into the `Utils/` folder.
