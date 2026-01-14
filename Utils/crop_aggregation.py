#!/usr/bin/env python3
"""
Crop Aggregation Module - Standardized 27→6 Family Mapping for Formulation B

This module provides a unified, canonical way to aggregate 27 crops into 6 families
for use across all Formulation B implementations (rotation scenarios with family-level
aggregation).

**Standard Family Names:**
- Fruits
- Grains
- Legumes
- Leafy_Vegetables
- Root_Vegetables
- Proteins

**Design Goals:**
1. Single source of truth for crop→family mapping
2. Consistent family naming across all files
3. Standardized nutritional score aggregation
4. Support for both food benefits and attribute dictionaries

Author: OQI-UC002-DWave
Date: 2026-01-14
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ============================================================================
# CANONICAL CROP-TO-FAMILY MAPPING
# ============================================================================

# Standard family order (used everywhere in Formulation B)
STANDARD_FAMILY_ORDER = [
    'Fruits',
    'Grains',
    'Legumes',
    'Leafy_Vegetables',
    'Root_Vegetables',
    'Proteins'
]

# Complete mapping: 27 crops → 6 families
# Based on botanical classification, nutritional profiles, and agronomic rotation logic
CROP_TO_FAMILY_MAPPING = {
    # Fruits (tree and vine fruits, high in vitamins/fiber)
    'Banana': 'Fruits',
    'Bananas': 'Fruits',
    'Orange': 'Fruits',
    'Oranges': 'Fruits',
    'Mango': 'Fruits',
    'Mangoes': 'Fruits',
    'Apple': 'Fruits',
    'Apples': 'Fruits',
    'Grape': 'Fruits',
    'Grapes': 'Fruits',
    'Avocado': 'Fruits',
    'Durian': 'Fruits',
    'Guava': 'Fruits',
    'Papaya': 'Fruits',
    'Watermelon': 'Fruits',
    'Pineapple': 'Fruits',
    'Kiwi': 'Fruits',
    
    # Grains (cereals, high in carbohydrates)
    'Rice': 'Grains',
    'Wheat': 'Grains',
    'Maize': 'Grains',
    'Corn': 'Grains',
    'Barley': 'Grains',
    'Oats': 'Grains',
    'Millet': 'Grains',
    'Sorghum': 'Grains',
    
    # Legumes (nitrogen-fixing, high in protein)
    'Beans': 'Legumes',
    'Lentils': 'Legumes',
    'Chickpeas': 'Legumes',
    'Peas': 'Legumes',
    'Soybeans': 'Legumes',
    'Groundnuts': 'Legumes',
    'Peanuts': 'Legumes',
    'Long bean': 'Legumes',
    'Tempeh': 'Legumes',
    'Tofu': 'Legumes',
    
    # Leafy Vegetables (greens, high in micronutrients)
    'Spinach': 'Leafy_Vegetables',
    'Cabbage': 'Leafy_Vegetables',
    'Lettuce': 'Leafy_Vegetables',
    'Broccoli': 'Leafy_Vegetables',
    'Cauliflower': 'Leafy_Vegetables',
    'Celery': 'Leafy_Vegetables',
    'Tomato': 'Leafy_Vegetables',  # Botanically a fruit, but grouped here agronomically
    'Tomatoes': 'Leafy_Vegetables',
    'Peppers': 'Leafy_Vegetables',
    'Onions': 'Leafy_Vegetables',
    'Cucumbers': 'Leafy_Vegetables',
    'Cucumber': 'Leafy_Vegetables',
    'Eggplant': 'Leafy_Vegetables',
    'Pumpkin': 'Leafy_Vegetables',
    
    # Root Vegetables (tubers and roots, high in starch)
    'Potatoes': 'Root_Vegetables',
    'Potato': 'Root_Vegetables',
    'Carrots': 'Root_Vegetables',
    'Carrot': 'Root_Vegetables',
    'Cassava': 'Root_Vegetables',
    'Sweet Potatoes': 'Root_Vegetables',
    'Yams': 'Root_Vegetables',
    'Beets': 'Root_Vegetables',
    
    # Proteins (animal products, nuts, high in protein/fat)
    'Beef': 'Proteins',
    'Chicken': 'Proteins',
    'Pork': 'Proteins',
    'Lamb': 'Proteins',
    'Egg': 'Proteins',
    'Nuts': 'Proteins',
    'Herbs': 'Proteins',  # Grouped here for lack of better category
    'Spices': 'Proteins',
    'Coffee': 'Proteins',
    'Tea': 'Proteins',
}

# Reverse mapping: family → representative crops
FAMILY_TO_CROPS_MAPPING = {
    'Fruits': ['Banana', 'Orange', 'Mango', 'Apple', 'Grape'],
    'Grains': ['Rice', 'Wheat', 'Maize', 'Barley', 'Oats'],
    'Legumes': ['Beans', 'Lentils', 'Chickpeas', 'Peas', 'Soybeans'],
    'Leafy_Vegetables': ['Spinach', 'Cabbage', 'Tomato', 'Broccoli', 'Lettuce'],
    'Root_Vegetables': ['Potatoes', 'Carrots', 'Cassava', 'Sweet Potatoes', 'Yams'],
    'Proteins': ['Beef', 'Chicken', 'Pork', 'Nuts', 'Egg']
}

# Standard nutritional attributes for each family (used if no crop data available)
FAMILY_DEFAULT_ATTRIBUTES = {
    'Fruits': {
        'nutritional_value': 0.70,
        'nutrient_density': 0.60,
        'environmental_impact': 0.30,
        'affordability': 0.80,
        'sustainability': 0.70
    },
    'Grains': {
        'nutritional_value': 0.80,
        'nutrient_density': 0.70,
        'environmental_impact': 0.40,
        'affordability': 0.90,
        'sustainability': 0.60
    },
    'Legumes': {
        'nutritional_value': 0.90,
        'nutrient_density': 0.80,
        'environmental_impact': 0.20,
        'affordability': 0.85,
        'sustainability': 0.90
    },
    'Leafy_Vegetables': {
        'nutritional_value': 0.75,
        'nutrient_density': 0.90,
        'environmental_impact': 0.25,
        'affordability': 0.70,
        'sustainability': 0.80
    },
    'Root_Vegetables': {
        'nutritional_value': 0.65,
        'nutrient_density': 0.60,
        'environmental_impact': 0.35,
        'affordability': 0.75,
        'sustainability': 0.75
    },
    'Proteins': {
        'nutritional_value': 0.95,
        'nutrient_density': 0.85,
        'environmental_impact': 0.60,
        'affordability': 0.60,
        'sustainability': 0.50
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_crop_family(crop_name: str) -> str:
    """
    Get the family for a given crop name.
    
    Handles case-insensitive matching and common variations.
    
    Args:
        crop_name: Name of the crop (e.g., 'Tomato', 'tomatoes', 'RICE')
        
    Returns:
        Family name from STANDARD_FAMILY_ORDER
        
    Examples:
        >>> get_crop_family('Tomato')
        'Leafy_Vegetables'
        >>> get_crop_family('rice')
        'Grains'
        >>> get_crop_family('Unknown Crop')
        'Proteins'  # Default fallback
    """
    # Direct lookup
    if crop_name in CROP_TO_FAMILY_MAPPING:
        return CROP_TO_FAMILY_MAPPING[crop_name]
    
    # Case-insensitive lookup
    crop_lower = crop_name.lower()
    for crop, family in CROP_TO_FAMILY_MAPPING.items():
        if crop.lower() == crop_lower:
            return family
    
    # Partial match (e.g., "Green Beans" contains "Beans")
    for crop, family in CROP_TO_FAMILY_MAPPING.items():
        if crop.lower() in crop_lower or crop_lower in crop.lower():
            return family
    
    # Default fallback
    print(f"Warning: Unknown crop '{crop_name}', defaulting to 'Proteins' family")
    return 'Proteins'


def get_crops_in_family(family: str) -> List[str]:
    """Get list of representative crops in a family."""
    return FAMILY_TO_CROPS_MAPPING.get(family, [])


# ============================================================================
# MAIN AGGREGATION FUNCTION
# ============================================================================

def aggregate_crops(
    crop_data: Dict,
    aggregation_method: str = 'weighted_mean',
    include_metadata: bool = True
) -> Dict:
    """
    Aggregate 27 crops into 6 standardized families for Formulation B.
    
    This is the canonical function for crop→family aggregation. It handles:
    - Nutritional score aggregation (weighted average by area/benefit)
    - Attribute aggregation (environmental impact, affordability, etc.)
    - Metadata preservation for traceability
    
    Args:
        crop_data: Dictionary containing crop information with keys:
            - 'crops' or 'food_names': List of crop names (27 items)
            - 'crop_benefits' or 'food_benefits': Dict {crop: benefit_score}
            - 'crop_attributes' or 'foods': Dict {crop: {attr: value, ...}} (optional)
            - 'land_availability': Dict {farm: area} (optional, for weighting)
            
        aggregation_method: Method for aggregating scores
            - 'weighted_mean': Weight by benefit scores (default)
            - 'simple_mean': Unweighted average
            - 'max': Take maximum value in family
            
        include_metadata: If True, include mapping information in output
        
    Returns:
        Dictionary with aggregated family data:
            - 'families': List of 6 family names (STANDARD_FAMILY_ORDER)
            - 'family_benefits': Dict {family: aggregated_benefit}
            - 'family_attributes': Dict {family: {attr: value, ...}}
            - 'family_to_crops': Dict {family: [crop_names]}
            - 'crop_to_family': Dict {crop: family} (if include_metadata)
            - 'aggregation_method': str (if include_metadata)
            - 'original_crop_count': int (if include_metadata)
            
    Example:
        >>> crop_data = {
        ...     'food_names': ['Tomato', 'Banana', 'Rice', 'Beans', ...],
        ...     'food_benefits': {'Tomato': 0.8, 'Banana': 0.7, ...},
        ...     'foods': {
        ...         'Tomato': {'nutritional_value': 0.9, ...},
        ...         'Banana': {'nutritional_value': 0.7, ...},
        ...         ...
        ...     }
        ... }
        >>> result = aggregate_crops(crop_data)
        >>> result['families']
        ['Fruits', 'Grains', 'Legumes', 'Leafy_Vegetables', 'Root_Vegetables', 'Proteins']
        >>> result['family_benefits']['Fruits']
        0.72  # Aggregated from Banana, Orange, Mango, Apple, Grape
    """
    
    # Extract crop names (support multiple input formats)
    crop_names = crop_data.get('crops') or crop_data.get('food_names', [])
    if not crop_names:
        raise ValueError("crop_data must contain 'crops' or 'food_names' key with list of crop names")
    
    # Extract benefit scores (support multiple input formats)
    crop_benefits = crop_data.get('crop_benefits') or crop_data.get('food_benefits', {})
    
    # Extract detailed attributes if available
    crop_attributes = crop_data.get('crop_attributes') or crop_data.get('foods', {})
    
    # Initialize aggregation structures
    family_crops = {family: [] for family in STANDARD_FAMILY_ORDER}
    family_benefits_list = {family: [] for family in STANDARD_FAMILY_ORDER}
    family_attributes_lists = {family: defaultdict(list) for family in STANDARD_FAMILY_ORDER}
    
    # Group crops by family
    for crop in crop_names:
        family = get_crop_family(crop)
        family_crops[family].append(crop)
        
        # Collect benefit score if available
        if crop in crop_benefits:
            family_benefits_list[family].append(crop_benefits[crop])
        
        # Collect detailed attributes if available
        if crop in crop_attributes:
            attrs = crop_attributes[crop]
            if isinstance(attrs, dict):
                for attr_name, attr_value in attrs.items():
                    if isinstance(attr_value, (int, float)):
                        family_attributes_lists[family][attr_name].append(attr_value)
    
    # Aggregate benefits per family
    family_benefits = {}
    for family in STANDARD_FAMILY_ORDER:
        benefits = family_benefits_list[family]
        
        if len(benefits) == 0:
            # No crops in this family, use default
            family_benefits[family] = FAMILY_DEFAULT_ATTRIBUTES[family]['nutritional_value']
        elif aggregation_method == 'weighted_mean':
            # Weight by benefit values themselves (higher benefits get more weight)
            weights = np.array(benefits)
            family_benefits[family] = np.average(benefits, weights=weights)
        elif aggregation_method == 'simple_mean':
            family_benefits[family] = np.mean(benefits)
        elif aggregation_method == 'max':
            family_benefits[family] = np.max(benefits)
        else:
            raise ValueError(f"Unknown aggregation_method: {aggregation_method}")
    
    # Aggregate attributes per family
    family_attributes = {}
    for family in STANDARD_FAMILY_ORDER:
        attrs_lists = family_attributes_lists[family]
        
        if len(attrs_lists) == 0:
            # No attributes available, use defaults
            family_attributes[family] = FAMILY_DEFAULT_ATTRIBUTES[family].copy()
        else:
            # Aggregate each attribute
            family_attributes[family] = {}
            for attr_name, values in attrs_lists.items():
                if len(values) > 0:
                    if aggregation_method == 'weighted_mean':
                        # Weight by corresponding benefits
                        crops_with_attr = [c for c in family_crops[family] 
                                          if c in crop_attributes and attr_name in crop_attributes[c]]
                        weights = [crop_benefits.get(c, 1.0) for c in crops_with_attr]
                        family_attributes[family][attr_name] = np.average(values, weights=weights)
                    elif aggregation_method in ['simple_mean', 'max']:
                        family_attributes[family][attr_name] = (
                            np.mean(values) if aggregation_method == 'simple_mean' else np.max(values)
                        )
            
            # Fill in missing attributes with defaults
            for attr_name, default_value in FAMILY_DEFAULT_ATTRIBUTES[family].items():
                if attr_name not in family_attributes[family]:
                    family_attributes[family][attr_name] = default_value
    
    # Build output dictionary
    result = {
        'families': STANDARD_FAMILY_ORDER.copy(),
        'family_benefits': family_benefits,
        'family_attributes': family_attributes,
        'family_to_crops': {fam: family_crops[fam] for fam in STANDARD_FAMILY_ORDER}
    }
    
    # Add metadata if requested
    if include_metadata:
        result['crop_to_family'] = {crop: get_crop_family(crop) for crop in crop_names}
        result['aggregation_method'] = aggregation_method
        result['original_crop_count'] = len(crop_names)
        result['family_count'] = len(STANDARD_FAMILY_ORDER)
    
    return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def compute_family_benefit_score(family: str, weights: Dict[str, float]) -> float:
    """
    Compute overall benefit score for a family given attribute weights.
    
    Args:
        family: Family name (e.g., 'Fruits')
        weights: Dict of {attribute: weight}, e.g.:
            {'nutritional_value': 0.25, 'nutrient_density': 0.25, ...}
            
    Returns:
        Weighted benefit score [0, 1]
    """
    attrs = FAMILY_DEFAULT_ATTRIBUTES.get(family, {})
    score = sum(attrs.get(attr, 0.0) * weight for attr, weight in weights.items())
    return score


def validate_family_names(family_list: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if family names match the standard.
    
    Returns:
        (is_valid, list_of_mismatches)
    """
    mismatches = [f for f in family_list if f not in STANDARD_FAMILY_ORDER]
    return (len(mismatches) == 0, mismatches)


# ============================================================================
# TESTING/DEMO
# ============================================================================

if __name__ == '__main__':
    # Demo usage
    print("="*80)
    print("CROP AGGREGATION MODULE - Demo")
    print("="*80)
    
    # Example: 27 crops with benefits
    demo_crops = [
        'Tomato', 'Banana', 'Rice', 'Beans', 'Potato',
        'Spinach', 'Wheat', 'Lentils', 'Carrot', 'Orange',
        'Maize', 'Chickpeas', 'Cassava', 'Mango', 'Cabbage',
        'Barley', 'Peas', 'Sweet Potatoes', 'Apple', 'Lettuce',
        'Oats', 'Soybeans', 'Yams', 'Grape', 'Broccoli',
        'Beef', 'Chicken'
    ]
    
    demo_benefits = {crop: 0.5 + 0.5 * np.random.random() for crop in demo_crops}
    
    demo_attributes = {
        crop: {
            'nutritional_value': 0.5 + 0.5 * np.random.random(),
            'nutrient_density': 0.5 + 0.5 * np.random.random(),
            'environmental_impact': 0.2 + 0.3 * np.random.random(),
            'affordability': 0.6 + 0.4 * np.random.random(),
            'sustainability': 0.5 + 0.5 * np.random.random(),
        }
        for crop in demo_crops
    }
    
    crop_data = {
        'food_names': demo_crops,
        'food_benefits': demo_benefits,
        'foods': demo_attributes
    }
    
    # Aggregate
    result = aggregate_crops(crop_data, aggregation_method='weighted_mean')
    
    print("\n1. Standard Families:")
    print(f"   {result['families']}")
    
    print("\n2. Family Benefits:")
    for family in result['families']:
        print(f"   {family:20s}: {result['family_benefits'][family]:.4f}")
    
    print("\n3. Family Attributes (sample - Fruits):")
    for attr, value in result['family_attributes']['Fruits'].items():
        print(f"   {attr:25s}: {value:.4f}")
    
    print("\n4. Family Composition:")
    for family in result['families']:
        crops = result['family_to_crops'][family]
        print(f"   {family:20s}: {len(crops)} crops - {', '.join(crops[:3])}{'...' if len(crops) > 3 else ''}")
    
    print("\n5. Validation:")
    is_valid, mismatches = validate_family_names(result['families'])
    print(f"   Names valid: {is_valid}")
    
    print("\n" + "="*80)
    print("Demo complete! Use aggregate_crops() in your code.")
    print("="*80)
