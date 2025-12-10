# QPU Benchmark Enhancement Summary

## Changes Made

This update enables the QPU benchmark to work with small synthetic scenarios designed for direct quantum embedding.

### 1. Modified `qpu_benchmark.py`

#### Added Scenario Support to `run_benchmark()` function:
- **New parameter**: `scenarios: List[str]` - accepts scenario names instead of farm scales
- **Logic**: When scenarios are provided, uses `load_problem_data_from_scenario()` instead of `load_problem_data()`
- **Dual mode operation**: 
  - Traditional: `--scale 25 50 100` (uses farm counts)
  - New: `--scenario micro_6 tiny_24` (uses named scenarios)

#### Updated Direct QPU Embedding Logic:
```python
# Before: Only attempted for n_farms <= 15
if n_farms <= 15:
    # attempt direct QPU

# After: Attempted for small scenarios OR tiny scales
is_small_scenario = use_scenarios and scenario_name in SYNTHETIC_SCENARIOS
is_tiny_scale = n_farms <= 15
attempt_direct_qpu = is_small_scenario or is_tiny_scale
```

This ensures all synthetic scenarios (`micro_6` through `medium_160`) automatically attempt direct QPU embedding.

#### Enhanced Command-Line Arguments:
```python
parser.add_argument('--scenario', '--scenarios', nargs='+', dest='scenarios',
                    help='Test small synthetic scenarios (e.g., micro_6 tiny_24 small_60). '
                         'Available: micro_6, micro_12, tiny_24, tiny_40, small_60, small_80, '
                         'small_100, medium_120, medium_160')

parser.add_argument('--all-small', action='store_true',
                    help='Test all small synthetic scenarios (micro_6 through medium_160)')
```

#### Smart Method Defaults:
```python
# For small scenarios: focus on direct embedding
if scenarios is not None:
    methods = ['ground_truth', 'direct_qpu']
    
# For larger scales: use decomposition methods
else:
    methods = ['ground_truth', 'direct_qpu', 'coordinated', 
               'decomposition_PlotBased_QPU', ...]
```

#### Updated `print_summary()`:
- Changed scale column from 6 chars to 20 chars to accommodate scenario names
- Displays scenario name (e.g., "micro_6") instead of just farm count when available
- Maintains backward compatibility with traditional scale-based runs

### 2. Created `test_qpu_small_scenarios.py`

Validation script that:
- Tests all 9 synthetic scenarios load correctly
- Verifies expected variable counts (6, 12, 25, 36, 56, 80, 99, 120, 154)
- Checks farm and food counts match specifications
- Provides usage examples

### 3. Created `QPU_SMALL_SCENARIOS_GUIDE.md`

Comprehensive documentation covering:
- Available scenarios with specs
- Usage examples
- Expected embedding performance
- Troubleshooting guide
- Comparison with traditional benchmarks

## Key Features

### ✅ Direct QPU Embedding for Small Problems
Small scenarios (6-160 variables) now automatically attempt direct embedding without decomposition.

### ✅ Backward Compatible
Traditional scale-based benchmarking still works:
```bash
python qpu_benchmark.py --scale 25 50 100
```

### ✅ Flexible Scenario Testing
```bash
# Single scenario
python qpu_benchmark.py --scenario micro_6

# Multiple scenarios
python qpu_benchmark.py --scenario micro_6 tiny_24 small_60

# All small scenarios
python qpu_benchmark.py --all-small
```

### ✅ Existing Scenarios in `scenarios.py`
Leverages already-implemented synthetic scenarios:
- `micro_6`, `micro_12`, `tiny_24`, `tiny_40`
- `small_60`, `small_80`, `small_100`
- `medium_120`, `medium_160`

## Usage Examples

### Test Direct QPU Embedding on Smallest Problem
```bash
python qpu_benchmark.py --scenario micro_6
```
Expected: 6 variables (2 plots × 2 foods + 2 U vars), should embed trivially.

### Compare QPU vs Gurobi on Multiple Small Scales
```bash
python qpu_benchmark.py --scenario micro_6 tiny_24 small_60 --methods ground_truth direct_qpu
```

### Full Small Scenario Suite
```bash
python qpu_benchmark.py --all-small
```
Benchmarks all 9 scenarios with default methods (ground_truth + direct_qpu).

## Testing

Verify implementation:
```bash
python test_qpu_small_scenarios.py
```

Expected output: ✅ All scenarios load successfully with correct variable counts.

## What Remains Unchanged

- All existing decomposition methods still work
- Traditional scale-based benchmarking unchanged
- CQM/BQM building logic unchanged
- Ground truth (Gurobi) solving unchanged
- Output format and structure unchanged (except added `scenario` field in results)

## Benefits

1. **Rapid prototyping**: Test formulation changes on 6-variable problems (seconds vs minutes)
2. **Embedding validation**: Verify problem structure embeds efficiently before scaling
3. **QPU testing**: Direct quantum annealing without decomposition overhead
4. **Educational**: Small problems easier to understand and debug
5. **Progressive scaling**: Test micro → tiny → small → medium → large systematically

## Files Modified

- `@todo/qpu_benchmark.py` (~120 lines changed)
  - `run_benchmark()` function signature and logic
  - `main()` argument parsing and workflow
  - `print_summary()` display formatting

## Files Created

- `@todo/test_qpu_small_scenarios.py` (validation script)
- `@todo/QPU_SMALL_SCENARIOS_GUIDE.md` (user documentation)

---

**Status**: ✅ Implementation complete and tested
**Next steps**: Run actual QPU benchmarks on small scenarios to validate embedding performance
