# Crop Aggregation Quick Reference

## Import

```python
from Utils.crop_aggregation import (
    aggregate_crops,              # Main aggregation function
    get_crop_family,              # Single crop → family
    STANDARD_FAMILY_ORDER,        # List of 6 families
    FAMILY_DEFAULT_ATTRIBUTES,    # Default attributes
    CROP_TO_FAMILY_MAPPING        # Complete 27→6 mapping
)
```

## Standard Families

```python
STANDARD_FAMILY_ORDER = [
    'Fruits',
    'Grains',
    'Legumes',
    'Leafy_Vegetables',
    'Root_Vegetables',
    'Proteins'
]
```

## Quick Usage

### Get family for a single crop
```python
family = get_crop_family('Tomato')  # Returns: 'Leafy_Vegetables'
family = get_crop_family('rice')    # Returns: 'Grains' (case-insensitive)
```

### Full aggregation (27 crops → 6 families)
```python
# Input format
crop_data = {
    'food_names': ['Tomato', 'Banana', 'Rice', ...],  # 27 crops
    'food_benefits': {'Tomato': 0.8, 'Banana': 0.7, ...},
    'foods': {  # Optional detailed attributes
        'Tomato': {
            'nutritional_value': 0.9,
            'nutrient_density': 0.8,
            'environmental_impact': 0.3,
            'affordability': 0.85,
            'sustainability': 0.75
        },
        ...
    }
}

# Aggregate
result = aggregate_crops(crop_data, aggregation_method='weighted_mean')

# Access results
families = result['families']                    # List of 6 families
family_benefits = result['family_benefits']      # {family: benefit_score}
family_attributes = result['family_attributes']  # {family: {attr: value}}
family_to_crops = result['family_to_crops']      # {family: [crop_list]}
```

## For Scenarios (src/scenarios.py)

Replace inline definitions with:

```python
from Utils.crop_aggregation import FAMILY_DEFAULT_ATTRIBUTES, STANDARD_FAMILY_ORDER

# Old way (repeated 4+ times):
crop_families = {
    'Fruits': {'nutritional_value': 0.7, 'nutrient_density': 0.6, ...},
    'Grains': {'nutritional_value': 0.8, 'nutrient_density': 0.7, ...},
    ...
}

# New way (one line):
crop_families = {fam: FAMILY_DEFAULT_ATTRIBUTES[fam] for fam in STANDARD_FAMILY_ORDER}
```

## Crop-to-Family Mapping

| Family | Crops |
|--------|-------|
| **Fruits** | Banana, Orange, Mango, Apple, Grape, Avocado, Durian, Guava, Papaya, Watermelon, Pineapple, Kiwi |
| **Grains** | Rice, Wheat, Maize/Corn, Barley, Oats, Millet, Sorghum |
| **Legumes** | Beans, Lentils, Chickpeas, Peas, Soybeans, Groundnuts/Peanuts, Long bean, Tempeh, Tofu |
| **Leafy_Vegetables** | Spinach, Cabbage, Lettuce, Broccoli, Cauliflower, Celery, Tomato, Peppers, Onions, Cucumbers, Eggplant, Pumpkin |
| **Root_Vegetables** | Potatoes, Carrots, Cassava, Sweet Potatoes, Yams, Beets |
| **Proteins** | Beef, Chicken, Pork, Lamb, Egg, Nuts, Herbs, Spices, Coffee, Tea |

## Validation

Check if family names are standard:

```python
from Utils.crop_aggregation import validate_family_names

is_valid, mismatches = validate_family_names(['Fruits', 'Grains', 'Vegetables'])
# Returns: (False, ['Vegetables'])  # Should be 'Leafy_Vegetables'
```

## Common Patterns

### Pattern 1: Scenario Loader
```python
from Utils.crop_aggregation import FAMILY_DEFAULT_ATTRIBUTES, STANDARD_FAMILY_ORDER

def load_rotation_scenario():
    crop_families = {
        family: FAMILY_DEFAULT_ATTRIBUTES[family]
        for family in STANDARD_FAMILY_ORDER
    }
    return crop_families
```

### Pattern 2: Hierarchical Aggregation
```python
from Utils.crop_aggregation import aggregate_crops

# Aggregate 27 → 6 before solving
family_data = aggregate_crops(
    crop_data={'food_names': foods_27, 'food_benefits': benefits_27},
    aggregation_method='weighted_mean'
)

# Solve at family level
families = family_data['families']
benefits = family_data['family_benefits']
```

### Pattern 3: Quick Lookup
```python
from Utils.crop_aggregation import get_crop_family, get_crops_in_family

# Forward lookup
get_crop_family('Tomato')  # → 'Leafy_Vegetables'

# Reverse lookup
get_crops_in_family('Fruits')  # → ['Banana', 'Orange', 'Mango', 'Apple', 'Grape']
```

## Default Attribute Values

```python
FAMILY_DEFAULT_ATTRIBUTES = {
    'Fruits': {
        'nutritional_value': 0.70, 'nutrient_density': 0.60,
        'environmental_impact': 0.30, 'affordability': 0.80, 'sustainability': 0.70
    },
    'Grains': {
        'nutritional_value': 0.80, 'nutrient_density': 0.70,
        'environmental_impact': 0.40, 'affordability': 0.90, 'sustainability': 0.60
    },
    'Legumes': {
        'nutritional_value': 0.90, 'nutrient_density': 0.80,
        'environmental_impact': 0.20, 'affordability': 0.85, 'sustainability': 0.90
    },
    'Leafy_Vegetables': {
        'nutritional_value': 0.75, 'nutrient_density': 0.90,
        'environmental_impact': 0.25, 'affordability': 0.70, 'sustainability': 0.80
    },
    'Root_Vegetables': {
        'nutritional_value': 0.65, 'nutrient_density': 0.60,
        'environmental_impact': 0.35, 'affordability': 0.75, 'sustainability': 0.75
    },
    'Proteins': {
        'nutritional_value': 0.95, 'nutrient_density': 0.85,
        'environmental_impact': 0.60, 'affordability': 0.60, 'sustainability': 0.50
    }
}
```

## See Also

- **Full Documentation:** `Utils/crop_aggregation.py` (docstrings)
- **Migration Guide:** `CROP_AGGREGATION_MIGRATION_PLAN.md`
- **Original Implementation:** `@todo/food_grouping.py` (deprecated)
