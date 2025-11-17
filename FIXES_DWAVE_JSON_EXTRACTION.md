# DWave JSON Solution Extraction Fix

## Date: 2025-11-17

## Problem
The DWave benchmark results JSON files were being saved with empty dictionaries for:
- `solution_areas`: {}
- `land_data`: {}
- `validation`: {}

This made it impossible to analyze DWave solutions or compare them with PuLP/Pyomo results.

## Root Cause
The solution extraction code was commented out with `# Can add if needed`, leaving only placeholder empty dictionaries.

## Fix Applied

### File: `benchmark_scalability_LQ.py`

**Before:**
```python
dwave_cache_result = {
    'status': 'Optimal' if dwave_feasible else 'Infeasible',
    # ... other fields ...
    'solution_areas': {},  # Can add if needed
    'land_data': {},  # Can add if needed
    'validation': {}  # Can add if validation is performed
}
```

**After:**
```python
# Extract solution variables and run validation if feasible solution found
dwave_solution_areas = {}
dwave_solution_selections = {}
dwave_land_data = {}
dwave_validation = {}

if dwave_feasible and sampleset:
    from solver_runner_LQ import validate_solution_constraints, extract_solution_summary
    
    # Extract best solution
    best = feasible_sampleset.first
    
    # Build solution dictionary with A_ and Y_ prefixes (matching PuLP/Pyomo format)
    solution = {}
    for var_name, var_value in best.sample.items():
        # DWave CQM variables are named like "Area_Farm1_Wheat" and "Y_Farm1_Wheat"
        if var_name.startswith('Area_'):
            # Convert "Area_Farm1_Wheat" to "A_Farm1_Wheat"
            key = var_name.replace('Area_', 'A_')
            solution[key] = var_value
            # Also store in areas dict without prefix for backwards compatibility
            farm_crop_key = var_name.replace('Area_', '')
            dwave_solution_areas[farm_crop_key] = var_value
        elif var_name.startswith('Y_'):
            # Y variables already have correct prefix
            solution[var_name] = var_value
            # Also store in selections dict without prefix
            farm_crop_key = var_name.replace('Y_', '')
            dwave_solution_selections[farm_crop_key] = var_value
    
    # Calculate land usage per farm
    land_availability = config['parameters']['land_availability']
    for farm in farms:
        farm_total = sum(solution.get(f"A_{farm}_{crop}", 0) for crop in foods)
        dwave_land_data[farm] = farm_total
    
    # Run validation
    dwave_validation = validate_solution_constraints(
        solution, farms, foods, food_groups, land_availability, config
    )
    
    # Extract solution summary
    dwave_solution_summary = extract_solution_summary(solution, farms, foods, land_availability)
else:
    dwave_solution_summary = {}

dwave_cache_result = {
    'status': 'Optimal' if dwave_feasible else 'Infeasible',
    # ... other fields ...
    'solution_areas': dwave_solution_areas,
    'land_data': dwave_land_data,
    'validation': dwave_validation
}
```

### Additional Fix: Variable Initialization
Added initialization of `sampleset` and `feasible_sampleset` to `None` at the start to prevent reference errors when DWave token is not provided.

## Features Added

1. **Solution Extraction**: Properly extracts area (A) and selection (Y) variables from DWave sampleset
2. **Variable Name Mapping**: Converts DWave naming convention (`Area_Farm1_Wheat`) to internal format (`A_Farm1_Wheat`)
3. **Land Usage Calculation**: Computes total area used per farm
4. **Constraint Validation**: Runs the same validation logic as PuLP/Pyomo to check:
   - Land availability constraints
   - Linking constraints (A and Y relationship)
   - Food group constraints
5. **Solution Summary**: Generates summary statistics (crops selected, utilization, etc.)

## Result

DWave benchmark JSON files now include:
- **solution_areas**: Complete dictionary of all area allocations per farm-crop combination
- **land_data**: Total area used per farm
- **validation**: Full constraint validation results including:
  - `is_feasible`: Boolean indicating if solution is valid
  - `violations`: List of constraint violations (if any)
  - `constraint_checks`: Breakdown by constraint type
  - `n_violations`: Count of total violations
  - `pass_rate`: Percentage of constraints satisfied

This enables:
- Direct comparison of DWave solutions with PuLP/Pyomo
- Detection of constraint violations in quantum solutions
- Analysis of solution quality and feasibility
- Debugging of model formulation issues

## Testing

To verify the fix works:
```powershell
cd "d:\Projects\OQI-UC002-DWave\Benchmark Scripts"
Remove-Item "D:\Projects\OQI-UC002-DWave\Benchmarks\LQ\DWave\config_10_run_1.json" -Force
python benchmark_scalability_LQ.py
```

Then check the generated JSON has populated `solution_areas`, `land_data`, and `validation` fields.
