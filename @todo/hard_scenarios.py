#!/usr/bin/env python3
"""
Create hard scenarios by replicating the pattern from rotation_medium_100
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from Utils.farm_sampler import generate_farms

def create_hard_scenario(n_farms, total_area, seed=None):
    """
    Create a hard scenario with high land variability.
    
    Uses generate_farms with specific parameters that create hard instances.
    Each scenario gets its own seed to create unique but consistently hard instances.
    
    Args:
        n_farms: Number of farms
        total_area: Total land area
        seed: Random seed (if None, will use n_farms-based seed)
    """
    if seed is None:
        # Use n_farms to generate consistent but different seeds
        seed = 10000 + n_farms
    
    # Generate farms with high variability (this is what makes it hard)
    land_availability = generate_farms(n_farms=n_farms, total_area=total_area, seed=seed)
    
    return land_availability

# Standard crop families
CROP_FAMILIES = {
    'Fruits': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.3, 
               'affordability': 0.8, 'sustainability': 0.7},
    'Grains': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.4,
               'affordability': 0.9, 'sustainability': 0.6},
    'Legumes': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.2,
                'affordability': 0.85, 'sustainability': 0.9},
    'Leafy_Vegetables': {'nutritional_value': 0.75, 'nutrient_density': 0.9, 'environmental_impact': 0.25,
                         'affordability': 0.7, 'sustainability': 0.8},
    'Root_Vegetables': {'nutritional_value': 0.65, 'nutrient_density': 0.6, 'environmental_impact': 0.35,
                        'affordability': 0.75, 'sustainability': 0.75},
    'Proteins': {'nutritional_value': 0.95, 'nutrient_density': 0.85, 'environmental_impact': 0.6,
                 'affordability': 0.6, 'sustainability': 0.5}
}

FOOD_GROUPS = {
    'Plant_Foods': ['Fruits', 'Grains', 'Legumes', 'Leafy_Vegetables', 'Root_Vegetables'],
    'Proteins': ['Proteins', 'Legumes']
}

# Hard scenario parameters (from rotation_medium_100)
HARD_PARAMS = {
    'frustration_ratio': 0.82,
    'negative_synergy_strength': -1.2,
    'rotation_gamma': 0.30,
    'one_hot_penalty': 2.0,
    'diversity_bonus': 0.22,
}

def get_scenario_config(n_farms, total_area):
    """Get complete scenario configuration"""
    land_availability = create_hard_scenario(n_farms, total_area)
    
    config = {
        'parameters': {
            'land_availability': land_availability,
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.25,
                'environmental_impact': 0.20,
                'affordability': 0.15,
                'sustainability': 0.15
            },
            'minimum_planting_area': {crop: 0.5 for crop in CROP_FAMILIES},
            'food_group_constraints': {
                'Plant_Foods': {'min': 3, 'max': 5},
                'Proteins': {'min': 1, 'max': 2}
            },
            'rotation_gamma': HARD_PARAMS['rotation_gamma'],
            'spatial_k_neighbors': 4,
            'frustration_ratio': HARD_PARAMS['frustration_ratio'],
            'negative_synergy_strength': HARD_PARAMS['negative_synergy_strength'],
            'use_soft_one_hot': True,
            'one_hot_penalty': HARD_PARAMS['one_hot_penalty'],
            'diversity_bonus': HARD_PARAMS['diversity_bonus']
        }
    }
    
    return list(land_availability.keys()), CROP_FAMILIES, FOOD_GROUPS, config

if __name__ == '__main__':
    print("="*80)
    print("HARD SCENARIOS - REPLICATED DISTRIBUTION PATTERN")
    print("="*80)
    
    for n_farms, total_area in [(20, 100), (50, 250), (90, 450), (225, 1125)]:
        farms, foods, food_groups, config = get_scenario_config(n_farms, total_area)
        params = config['parameters']
        land_availability = params['land_availability']
        
        areas = list(land_availability.values())
        
        print(f"\nScenario: {n_farms} farms")
        print(f"  Variables: {n_farms * 6 * 3}")
        print(f"  Total area: {sum(areas):.2f} ha")
        print(f"  Mean area: {np.mean(areas):.2f} ha")
        print(f"  Std dev: {np.std(areas):.2f} ha")
        print(f"  CoV: {np.std(areas)/np.mean(areas):.3f}")
        print(f"  Frustration: {params['frustration_ratio']}")
        print(f"  Neg strength: {params['negative_synergy_strength']}")
        print(f"  Gamma: {params['rotation_gamma']}")
        print(f"  Penalty: {params['one_hot_penalty']}")
