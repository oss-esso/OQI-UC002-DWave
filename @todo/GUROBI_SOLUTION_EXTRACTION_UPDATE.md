# Gurobi Timeout Test Solution Extraction Update

**Date:** December 22, 2025  
**File Updated:** `@todo/test_gurobi_timeout.py`

## Problem Identified

The original `gurobi_timeout_test_20251215_152938.json` output only contained summary statistics:
- `objective`, `runtime`, `mip_gap`, `status`, etc.
- **Missing:** Actual solution variables, area allocations, and validation results

This made it impossible to compare solutions directly with the hierarchical solver output (`hierarchical_30_farms.json`), which includes:
- `solution_selections`: Binary decision variables (0/1)
- `solution_areas`: Area allocations (binary × farm_area)
- `validation`: Constraint verification with detailed violations

## Changes Implemented

### 1. **Enhanced Result Structure**

Updated to match hierarchical solver format with nested structure:

```python
result = {
    'metadata': {
        'benchmark_type': 'GUROBI_TIMEOUT_TEST',
        'solver': 'gurobi',
        'scenario': scenario['name'],
        'n_farms': scenario['n_farms'],
        'n_foods': scenario['n_foods'],
        'n_periods': scenario['n_periods'],
        'timestamp': datetime.now().isoformat()
    },
    'result': {
        'scenario': scenario['name'],
        'n_vars': scenario['n_vars'],
        'status': 'unknown',
        'objective_value': None,
        'solve_time': None,
        'mip_gap': None,
        'hit_timeout': False,
        'hit_improve_limit': False,
        'stopped_reason': 'unknown',
        'solution_selections': {},  # Binary decision variables
        'solution_areas': {},       # Area allocations
        'total_covered_area': 0.0,
        'validation': {},           # Constraint verification
        'solver': 'gurobi',
        'success': False,
    }
}
```

### 2. **Solution Extraction**

Added complete solution extraction after Gurobi solve:

```python
if model.SolCount > 0:
    # Extract binary decision variables
    for i, farm in enumerate(farm_names):
        farm_area = land_availability[farm]
        for j, food in enumerate(food_names):
            for t in range(1, n_periods + 1):
                var = Y[(i, j, t)]
                val = var.X  # Binary value (0 or 1)
                
                var_name = f"{farm}_{food}_t{t}"
                result['result']['solution_selections'][var_name] = float(val)
                
                # Calculate area allocation (binary × farm_area)
                area = val * farm_area
                result['result']['solution_areas'][var_name] = float(area)
    
    # Calculate total covered area
    result['result']['total_covered_area'] = sum(result['result']['solution_areas'].values())
```

### 3. **Solution Validation**

Added constraint verification matching hierarchical solver:

```python
# Check 1: One-hot constraint (1-2 crops per farm per period)
for i, farm in enumerate(farm_names):
    for t in range(1, n_periods + 1):
        crops_selected = sum(
            result['result']['solution_selections'][f"{farm}_{food}_t{t}"]
            for food in food_names
        )
        if crops_selected < 1 or crops_selected > 2:
            violations.append({
                'type': 'one_hot_violation',
                'farm': farm,
                'period': t,
                'crops_selected': crops_selected,
                'expected': '1-2 crops'
            })

# Check 2: Rotation constraint (no same crop consecutive periods)
for i, farm in enumerate(farm_names):
    for j, food in enumerate(food_names):
        for t in range(1, n_periods):
            val_t = result['result']['solution_selections'][f"{farm}_{food}_t{t}"]
            val_t1 = result['result']['solution_selections'][f"{farm}_{food}_t{t+1}"]
            if val_t > 0.5 and val_t1 > 0.5:  # Both selected
                violations.append({
                    'type': 'rotation_violation',
                    'farm': farm,
                    'food': food,
                    'periods': f"t{t} and t{t+1}",
                    'message': 'Same crop in consecutive periods'
                })

result['result']['validation'] = {
    'is_valid': len(violations) == 0,
    'n_violations': len(violations),
    'violations': violations[:10],  # Limit to first 10
    'summary': f"{'Valid' if len(violations) == 0 else 'Invalid'}: {len(violations)} violations found"
}
```

## Output Format Comparison

### Before (Old Format)
```json
{
  "scenario": "rotation_micro_25",
  "n_vars": 90,
  "status": "timeout",
  "objective": 6.166826839679835,
  "runtime": 100.30700588226318,
  "mip_gap": 234.4526167876634,
  "hit_timeout": true,
  "stopped_reason": "timeout_300s"
}
```

### After (New Format - Matches Hierarchical Solver)
```json
{
  "metadata": {
    "benchmark_type": "GUROBI_TIMEOUT_TEST",
    "solver": "gurobi",
    "scenario": "rotation_micro_25",
    "n_farms": 5,
    "n_foods": 6,
    "n_periods": 3,
    "timestamp": "2025-12-22T10:30:00.123456"
  },
  "result": {
    "scenario": "rotation_micro_25",
    "n_vars": 90,
    "status": "timeout",
    "objective_value": 6.166826839679835,
    "solve_time": 100.30700588226318,
    "mip_gap": 234.4526167876634,
    "hit_timeout": true,
    "stopped_reason": "timeout_300s",
    "success": true,
    "solver": "gurobi",
    "solution_selections": {
      "Farm_1_Beef_t1": 0.0,
      "Farm_1_Chicken_t1": 1.0,
      "Farm_1_Egg_t1": 0.0,
      ...
    },
    "solution_areas": {
      "Farm_1_Beef_t1": 0.0,
      "Farm_1_Chicken_t1": 3.33,
      "Farm_1_Egg_t1": 0.0,
      ...
    },
    "total_covered_area": 100.0,
    "validation": {
      "is_valid": true,
      "n_violations": 0,
      "violations": [],
      "summary": "Valid: 0 violations found"
    }
  }
}
```

## Benefits

1. **Direct Comparability**: Output now matches `hierarchical_30_farms.json` structure
2. **Binary Variables**: All decision variables (0/1) are explicitly saved
3. **Area Allocations**: Physical area allocations computed from binary decisions
4. **Validation**: Constraint violations detected and reported
5. **Metadata**: Complete problem context preserved

## Usage

Run the updated test:

```bash
cd @todo
python test_gurobi_timeout.py
```

Output will be saved to:
- `gurobi_timeout_verification/gurobi_timeout_test_TIMESTAMP.json` (full details)
- `gurobi_timeout_verification/gurobi_timeout_test_TIMESTAMP.csv` (summary table)

## Next Steps

Now you can:
1. Compare Gurobi solutions directly with hierarchical solver solutions
2. Verify constraint satisfaction across both methods
3. Analyze solution quality differences
4. Extract specific variable assignments for debugging

## Notes

- Binary variables are the decision variables (0 = not selected, 1 = selected)
- Area allocations = binary variable × farm area
- Validation checks both one-hot constraints (1-2 crops per farm-period) and rotation constraints (no consecutive same crop)
- First 10 violations are reported (if any) to keep output manageable
