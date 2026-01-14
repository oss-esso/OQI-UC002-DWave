# Crop Aggregation Standardization - Migration Plan

**Date:** 2026-01-14  
**Purpose:** Standardize all 27→6 crop family aggregation to use `Utils/crop_aggregation.py`

---

## Summary

Created a new standardized `aggregate_crops()` function in [`Utils/crop_aggregation.py`](Utils/crop_aggregation.py) to replace all inline crop family aggregation logic throughout the codebase.

---

## Standard Family Mapping

The canonical 6 families for Formulation B are:

1. **Fruits** - Banana, Orange, Mango, Apple, Grape
2. **Grains** - Rice, Wheat, Maize, Barley, Oats  
3. **Legumes** - Beans, Lentils, Chickpeas, Peas, Soybeans
4. **Leafy_Vegetables** - Spinach, Cabbage, Tomato, Broccoli, Lettuce
5. **Root_Vegetables** - Potatoes, Carrots, Cassava, Sweet Potatoes, Yams
6. **Proteins** - Beef, Chicken, Pork, Nuts, Egg

### Standard Attributes (Defaults)

```python
'Fruits':             {nutritional_value: 0.70, nutrient_density: 0.60, environmental_impact: 0.30, affordability: 0.80, sustainability: 0.70}
'Grains':             {nutritional_value: 0.80, nutrient_density: 0.70, environmental_impact: 0.40, affordability: 0.90, sustainability: 0.60}
'Legumes':            {nutritional_value: 0.90, nutrient_density: 0.80, environmental_impact: 0.20, affordability: 0.85, sustainability: 0.90}
'Leafy_Vegetables':   {nutritional_value: 0.75, nutrient_density: 0.90, environmental_impact: 0.25, affordability: 0.70, sustainability: 0.80}
'Root_Vegetables':    {nutritional_value: 0.65, nutrient_density: 0.60, environmental_impact: 0.35, affordability: 0.75, sustainability: 0.75}
'Proteins':           {nutritional_value: 0.95, nutrient_density: 0.85, environmental_impact: 0.60, affordability: 0.60, sustainability: 0.50}
```

---

## New Module API

### Main Function

```python
from Utils.crop_aggregation import aggregate_crops

result = aggregate_crops(
    crop_data={
        'food_names': [...],          # List of 27 crop names
        'food_benefits': {...},       # Dict {crop: benefit_score}
        'foods': {...}                # Dict {crop: {attr: value}} (optional)
    },
    aggregation_method='weighted_mean',  # 'weighted_mean', 'simple_mean', or 'max'
    include_metadata=True
)

# Returns:
# {
#     'families': ['Fruits', 'Grains', 'Legumes', 'Leafy_Vegetables', 'Root_Vegetables', 'Proteins'],
#     'family_benefits': {family: score},
#     'family_attributes': {family: {attr: value}},
#     'family_to_crops': {family: [crops]},
#     'crop_to_family': {crop: family},     # if include_metadata
#     'aggregation_method': str,            # if include_metadata
#     'original_crop_count': int            # if include_metadata
# }
```

### Helper Functions

```python
from Utils.crop_aggregation import (
    get_crop_family,              # Get family for a single crop
    get_crops_in_family,          # Get representative crops in a family
    compute_family_benefit_score, # Compute weighted benefit score
    validate_family_names,        # Check if family names are standard
    STANDARD_FAMILY_ORDER,        # List of 6 families
    CROP_TO_FAMILY_MAPPING,       # Complete 27→6 mapping
    FAMILY_DEFAULT_ATTRIBUTES     # Standard attributes
)
```

---

## Files Requiring Updates

### Priority 1: Core Implementation Files

These files contain inline aggregation logic that should be replaced:

#### 1. `src/scenarios.py` (Lines ~2094-2145)
**Current:** Inline `crop_families` dictionary defined in 4 places:
- `_load_rotation_micro_25_food_data()` - Line 2094
- `_load_rotation_small_50_food_data()` - Line 2168
- `_load_rotation_medium_100_food_data()` - Line 2238
- `_load_rotation_large_200_food_data()` - Line 2309

**Action:** Replace with:
```python
from Utils.crop_aggregation import FAMILY_DEFAULT_ATTRIBUTES, STANDARD_FAMILY_ORDER

crop_families = {
    family: FAMILY_DEFAULT_ATTRIBUTES[family]
    for family in STANDARD_FAMILY_ORDER
}
```

**Files affected:** 1 file, 4 locations

---

#### 2. `@todo/food_grouping.py`
**Current:** Uses different family names: `['Legumes', 'Grains', 'Vegetables', 'Roots', 'Fruits', 'Other']`

**Action:** 
- Mark as DEPRECATED or update to use `Utils/crop_aggregation.py`
- Add compatibility wrapper that maps old names to new names:
  ```python
  # Legacy compatibility
  from Utils.crop_aggregation import aggregate_crops as _new_aggregate
  
  def aggregate_foods_to_families(data: Dict) -> Dict:
      """DEPRECATED: Use Utils.crop_aggregation.aggregate_crops() instead."""
      result = _new_aggregate(data)
      # Map new names to old names if needed
      return result
  ```

**Files affected:** 1 file

---

#### 3. `@todo/hybrid_formulation.py` (Lines 36, 132-156)
**Current:** Imports from `food_grouping` and uses old family names

**Action:** Update imports:
```python
# OLD:
from food_grouping import FOOD_TO_FAMILY, get_family, FAMILY_ORDER

# NEW:
from Utils.crop_aggregation import (
    CROP_TO_FAMILY_MAPPING as FOOD_TO_FAMILY,
    get_crop_family as get_family,
    STANDARD_FAMILY_ORDER as FAMILY_ORDER
)
```

**Files affected:** 1 file

---

#### 4. `unified_benchmark/quantum_solvers.py` (Line ~509)
**Current:** Has aggregation logic for hierarchical solver

**Action:** Replace aggregation step with:
```python
from Utils.crop_aggregation import aggregate_crops

# Level 1: Aggregate 27 foods to 6 families
family_data = aggregate_crops(
    crop_data={
        'food_names': food_names,
        'food_benefits': food_benefits,
        'foods': foods
    },
    aggregation_method='weighted_mean'
)
```

**Files affected:** 1 file

---

#### 5. `Phase3Report/Scripts/hierarchical_statistical_test.py` (Lines 217, 479)
**Current:** Uses `food_grouping.aggregate_foods_to_families`

**Action:** Update imports:
```python
# OLD:
from food_grouping import aggregate_foods_to_families

# NEW:
from Utils.crop_aggregation import aggregate_crops as aggregate_foods_to_families
```

**Files affected:** 1 file

---

#### 6. `Phase3Report/Scripts/statistical_comparison_test.py` (Line 114)
**Current:** Inline `CROP_FAMILIES` definition

**Action:** Replace with import:
```python
from Utils.crop_aggregation import FAMILY_DEFAULT_ATTRIBUTES, STANDARD_FAMILY_ORDER

CROP_FAMILIES = {
    family: FAMILY_DEFAULT_ATTRIBUTES[family]
    for family in STANDARD_FAMILY_ORDER
}
```

**Files affected:** 1 file

---

### Priority 2: Test and Verification Files

These files use aggregation for testing and should be updated for consistency:

7. `@todo/test_3_real_qpu_datapoints.py` (Line 30)
8. `@todo/test_3_real_qpu_hard.py` (Line 27)
9. `@todo/test_hierarchical_system.py` (Line 28)
10. `@todo/test_hybrid_gurobi_comparison.py` (Line 12)
11. `@todo/run_validation_tests.py` (Line 33)
12. `@todo/verify_setup.py` (Line 81)
13. `@todo/investigate_gap_difference.py` (Line 187)
14. `@todo/unified_scaling_test_plan.py` (Line 31)

**Action for all:** Update imports to use `Utils.crop_aggregation`

---

### Priority 3: Documentation Files

These files mention crop aggregation in documentation and should reference the new module:

15. `verification_prompt.md` (Line 47-48)
16. `VERIFICATION_REPORT.md` (Lines 201, 336)
17. `unified_benchmark_prompt.md` (Line 48)
18. `RUTHLESS_ANALYSIS_AND_BENCHMARK_PLAN.md` (Lines 29, 95)
19. `QPU_PREPARATION_INSTRUCTIONS.md` (Line 12)

**Action:** Add note:
```markdown
**Note:** Crop aggregation is now standardized in `Utils/crop_aggregation.py`. 
See that module for the canonical 27→6 mapping and usage examples.
```

---

### Priority 4: Report/Paper Files (LaTeX)

These academic documents describe the aggregation but don't implement it:

20. `@todo/report/content_report_old.tex`
21. `@todo/report/content_report.tex`
22. `Phase3Report/Docs/problem_formulations.tex`
23. `Phase3Report/Docs/benchmark_scenario_analysis.tex`

**Action:** Update pseudocode and text descriptions to reference `aggregate_crops()` function

---

## Migration Steps

### Step 1: Update Core Files (Priority 1)
1. Update `src/scenarios.py` (4 functions)
2. Update `@todo/hybrid_formulation.py` imports
3. Update `unified_benchmark/quantum_solvers.py`
4. Update Phase3Report scripts

### Step 2: Update Test Files (Priority 2)
5. Update all test files to use new imports
6. Run tests to verify no breakage

### Step 3: Update Documentation (Priority 3)
7. Add references to new module in markdown docs
8. Update inline code examples

### Step 4: Mark Old Code as Deprecated
9. Add deprecation warnings to `@todo/food_grouping.py`
10. Add compatibility layer for old names

---

## Testing Checklist

After migration, verify:

- [ ] All rotation scenarios (`rotation_micro_25`, `rotation_small_50`, etc.) load correctly
- [ ] Family names are consistent: `['Fruits', 'Grains', 'Legumes', 'Leafy_Vegetables', 'Root_Vegetables', 'Proteins']`
- [ ] Benefit scores are similar to previous values (±5%)
- [ ] Hierarchical solver still works with new aggregation
- [ ] All test scripts pass
- [ ] No references to old family names (`'Vegetables'`, `'Roots'`, `'Other'`)

---

## Benefits of Standardization

1. **Single Source of Truth:** One module defines the 27→6 mapping
2. **Consistent Naming:** All files use the same 6 family names
3. **Reproducibility:** Standard defaults ensure consistent behavior
4. **Maintainability:** Changes to mapping only need to be made in one place
5. **Documentation:** Clear API with examples and docstrings
6. **Validation:** Built-in functions to check family name consistency
7. **Flexibility:** Supports multiple aggregation methods (weighted_mean, simple_mean, max)

---

## Example Usage

### Before (Inline Definition)
```python
# In scenarios.py
crop_families = {
    'Fruits': {'nutritional_value': 0.7, 'nutrient_density': 0.6, ...},
    'Grains': {'nutritional_value': 0.8, 'nutrient_density': 0.7, ...},
    # ... repeated in 4+ places
}
```

### After (Standardized Import)
```python
from Utils.crop_aggregation import FAMILY_DEFAULT_ATTRIBUTES, STANDARD_FAMILY_ORDER

crop_families = {
    family: FAMILY_DEFAULT_ATTRIBUTES[family]
    for family in STANDARD_FAMILY_ORDER
}
```

### For Full Aggregation
```python
from Utils.crop_aggregation import aggregate_crops

# Given 27-crop data
result = aggregate_crops(crop_data)

# Use aggregated families
families = result['families']                      # ['Fruits', 'Grains', ...]
family_benefits = result['family_benefits']        # {family: score}
family_attributes = result['family_attributes']    # {family: {attr: val}}
```

---

## Completion Status

- [x] Created `Utils/crop_aggregation.py` with standardized function
- [ ] Updated `src/scenarios.py` (4 functions)
- [ ] Updated `@todo/hybrid_formulation.py`
- [ ] Updated `unified_benchmark/quantum_solvers.py`
- [ ] Updated Phase3Report scripts
- [ ] Updated test files
- [ ] Updated documentation
- [ ] Added deprecation warnings to old code
- [ ] Ran full test suite
- [ ] Validated consistency

---

## Questions/Issues

If you encounter any issues during migration:

1. **Family name mismatch?** Check `validate_family_names()` function
2. **Different benefit scores?** Try different `aggregation_method` parameter
3. **Missing attributes?** Check `FAMILY_DEFAULT_ATTRIBUTES` for defaults
4. **Need old behavior?** See compatibility layer in `@todo/food_grouping.py`

---

**Next Steps:** Begin with Priority 1 files, starting with `src/scenarios.py`.
