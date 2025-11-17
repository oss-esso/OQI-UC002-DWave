# CQM vs PuLP Validation Fixes - Summary

## Issues Fixed

### 1. Food Group Constraint Count Mismatch (FIXED)

**Problem:**
- Validator reported: `PuLP=1, CQM=2` food group constraints
- Should have been: `PuLP=10, CQM=10` (2 constraints per food group × 5 groups)

**Root Cause:**
In `check_food_group_constraints()`, the code was using `_extract_food_group_id_from_name()` to create dictionary keys:

```python
# OLD CODE (BUGGY)
pulp_fg = {}
for name, constraint in self.pulp.constraints.items():
    if 'FoodGroup' in name:
        fg_id = self._extract_food_group_id_from_name(name)  # e.g., "Vegetables"
        pulp_fg[fg_id] = (name, constraint)  # OVERWRITES previous entry!
```

This caused:
- `MinFoodGroup_Global_Vegetables` → key: "Vegetables" → stored
- `MaxFoodGroup_Global_Vegetables` → key: "Vegetables" → **OVERWRITES** previous!

Result: Only 1 constraint per food group visible (the last one processed).

**Fix:**
Use the full constraint name as the dictionary key:

```python
# NEW CODE (FIXED)
pulp_fg = {}
for name, constraint in self.pulp.constraints.items():
    if 'FoodGroup' in name:
        pulp_fg[name] = (name, constraint)  # Use full name as key
```

Now all constraints are preserved:
- `MinFoodGroup_Global_Vegetables` → key: "MinFoodGroup_Global_Vegetables"
- `MaxFoodGroup_Global_Vegetables` → key: "MaxFoodGroup_Global_Vegetables"

### 2. Constraint Categorization Mismatch (FIXED)

**Problem:**
- Farm scenario showed: `PuLP=280 at_most_one, CQM=270 at_most_one`
- 10 constraint difference

**Root Cause:**
The `_categorize_constraint_name()` function was categorizing constraints incorrectly:

```python
# OLD CODE (BUGGY)
if 'max_area' in name_lower or 'maxarea' in name_lower:
    return 'at_most_one'  # WRONG!
```

This incorrectly categorized:
- **PuLP:** `Max_Area_{farm}` (land availability) → "at_most_one" ❌
- **PuLP:** `MaxArea_{farm}_{food}` (coupling) → "at_most_one" ❌
- **CQM:** `Land_Availability_{farm}` → "other" ✓
- **CQM:** `Max_Area_If_Selected_{farm}_{food}` → "at_most_one" ❌

**Fix:**
Improved categorization with proper priority order:

```python
# NEW CODE (FIXED)
def _categorize_constraint_name(self, name: str) -> str:
    name_lower = name.lower()
    
    # Check land availability FIRST
    if 'land_availability' in name_lower or \
       (name_lower.startswith('max_area_farm') and name_lower.count('_') == 2):
        return 'land_availability'
    
    # Check coupling constraints (linking binary to continuous)
    elif ('min_area_if_selected' in name_lower or
          'max_area_if_selected' in name_lower or
          'minarea_' in name_lower or
          'maxarea_' in name_lower) and name_lower.count('_') >= 3:
        return 'coupling'
    
    # Only pure "at most one" assignments
    elif ('max_assignment' in name_lower or 'atmostone' in name_lower) and \
         'if_selected' not in name_lower:
        return 'at_most_one'
    
    # ... other categories
```

## Results After Fix

### Before Fix:
```
❌ FAILED - Found 1 discrepancies
1. food_group_count_mismatch (error)
   Food group constraint count mismatch: PuLP=1, CQM=2
```

### After Fix:
```
✓ PASSED - CQM matches PuLP formulation
  Food group constraints: PuLP=10, CQM=10
  Land availability: PuLP=10, CQM=10
  Coupling constraints: PuLP=540, CQM=540
```

## Impact

The validator will now:
1. ✅ Correctly count all food group constraints (min + max for each group)
2. ✅ Properly categorize constraint types (land availability vs coupling vs at-most-one)
3. ✅ Allow benchmarks to proceed without false validation failures
4. ✅ Provide more accurate constraint comparison reports

## Files Modified

- `Utils/validate_cqm_vs_pulp.py`:
  - `check_food_group_constraints()`: Use full constraint names as dict keys
  - `_categorize_constraint_name()`: Improved categorization logic with priority order

## Validation Now Passes

The benchmark can now proceed without being blocked by false validation errors!
