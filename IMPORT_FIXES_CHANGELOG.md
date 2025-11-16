# Import Fixes Changelog

This file tracks all import statement changes made during the repository reorganization.

## Summary

All import statements have been successfully updated to reflect the new repository structure where utility modules were moved to the `Utils/` folder.

## Module Relocations

The following modules were moved from root directory to `Utils/`:
- `patch_sampler.py` → `Utils/patch_sampler.py`
- `farm_sampler.py` → `Utils/farm_sampler.py`
- `benchmark_cache.py` → `Utils/benchmark_cache.py`
- `constraint_validator.py` → `Utils/constraint_validator.py`
- `piecewise_approximation.py` → `Utils/piecewise_approximation.py`
- `enhanced_dwave_solver.py` → `Utils/enhanced_dwave_solver.py`

## Import Patterns Applied

### For files importing from Utils (outside Utils folder):
```python
from Utils.module_name import ...
```

### For files within Utils importing other Utils modules:
```python
from .module_name import ...  # Relative import
```

### For files importing from Benchmark Scripts folder:
Added sys.path modification to handle spaces in folder name:
```python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'Benchmark Scripts'))
```

## Files Modified

### Benchmark Scripts/ (17 files)
1. `benchmark_scalability_PATCH.py` - Updated patch_sampler, benchmark_cache, constraint_validator imports
2. `benchmark_scalability_BQUBO.py` - Updated farm_sampler, benchmark_cache imports
3. `benchmark_scalability_LQ.py` - Updated farm_sampler, benchmark_cache imports
4. `benchmark_scalability_NLD.py` - Updated farm_sampler, benchmark_cache imports
5. `benchmark_scalability_NLN.py` - Updated farm_sampler, benchmark_cache, piecewise_approximation imports
6. `benchmarks.py` - Updated farm_sampler import
7. `rotation_benchmark.py` - Updated patch_sampler import
8. `rotation_benchmark_new.py` - Updated patch_sampler import
9. `comprehensive_benchmark.py` - Updated farm_sampler, patch_sampler imports
10. `solver_runner_BINARY.py` - Updated patch_sampler, farm_sampler imports
11. `solver_runner_PATCH.py` - Updated patch_sampler import
12. `solver_runner_ROTATION.py` - Updated patch_sampler, farm_sampler imports
13. `solver_runner_NLN.py` - Updated piecewise_approximation import
14. `solver_runner_BQUBO.py` - No changes needed
15. `solver_runner_LQ.py` - No changes needed
16. `solver_runner_NLD.py` - No changes needed
17. `benchmark_gurobi_qubo_only.py` - Updated patch_sampler import

### Plot Scripts/ (1 file)
1. `choropleth_plo.py` - Updated farm_sampler import

### Tests/ (17 files)
1. `test_patch_dwave_bqm_constraints.py` - Updated patch_sampler import, added sys.path for Benchmark Scripts
2. `test_patch_cqm_constraints.py` - Updated patch_sampler import, updated sys.path
3. `test_lq_normalization.py` - Updated benchmark_cache import, updated sys.path
4. `test_lq_formulation.py` - Updated farm_sampler import, updated sys.path
5. `test_lagrange_multipliers.py` - Updated patch_sampler import, updated sys.path
6. `test_lagrange_fix.py` - Updated patch_sampler import, updated sys.path
7. `test_fixed_constraints.py` - Updated patch_sampler import, updated sys.path
8. `test_enhanced_solver.py` - Updated patch_sampler, enhanced_dwave_solver imports, added sys.path
9. `test_constraint_validation.py` - Updated constraint_validator import, added sys.path
10. `test_comprehensive_benchmark.py` - Updated farm_sampler, patch_sampler imports, updated sys.path
11. `simulate_constraint_violations.py` - Updated patch_sampler import, added sys.path
12. `run_constraint_investigation.py` - Updated patch_sampler import, added sys.path
13. `test_rotation_benchmark.py` - Updated sys.path
14. `test_objective_fix.py` - Added sys.path
15. `test_dwave_cost_estimation.py` - Added sys.path
16. `advanced_bqm_analysis.py` - Added sys.path
17. `test_comprehensive_quick.py` - No changes needed

### Utils/ (15 files)
1. `normalize_lq_by_land.py` - Updated to relative import
2. `diagnose_gurobi_qubo.py` - Updated to relative import, updated sys.path
3. `diagnose_bqm_constraint_violations.py` - Updated to relative imports, updated sys.path
4. `compare_pulp_dwave_constraints.py` - Updated to relative import, added sys.path
5. `check_benchmark_cache.py` - Updated to relative import
6. `check_all_constraints.py` - Updated to relative import, added sys.path
7. `analyze_constraints.py` - Updated to relative imports, added sys.path
8. `analyze_bqm_formulations.py` - Updated to relative imports, updated sys.path
9. `Grid_Refinement.py` - Updated to relative imports, updated sys.path
10. `gurobi_qubo_comparison.py` - Updated to relative imports, updated sys.path
11. `verify_solvers.py` - Added sys.path
12. `patch_sampler.py` - No changes (referenced by others)
13. `farm_sampler.py` - No changes (referenced by others)
14. `benchmark_cache.py` - No changes (referenced by others)
15. `constraint_validator.py` - No changes (referenced by others)

### New Files Created
- `Utils/__init__.py` - Package initialization file for Utils module

## Verification

All old import patterns have been successfully replaced:
- ✅ No remaining `from patch_sampler import`
- ✅ No remaining `from farm_sampler import`
- ✅ No remaining `from benchmark_cache import`
- ✅ No remaining `from constraint_validator import`
- ✅ No remaining `from enhanced_dwave_solver import`
- ✅ No remaining `from piecewise_approximation import`

All imports now correctly reference the `Utils` package or use relative imports within Utils.

## Date: 2025-11-16

## Notes

- The "Benchmark Scripts" folder name contains a space, which is handled via sys.path manipulation
- Files within Utils use relative imports (`.module_name`) for other Utils modules
- Files outside Utils use absolute imports (`Utils.module_name`)
- All scripts that import from Benchmark Scripts now have proper sys.path setup
