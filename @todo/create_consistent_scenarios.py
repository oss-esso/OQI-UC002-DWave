#!/usr/bin/env python3
"""
Create consistent hardness scenarios with constant plot area
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Utils.farm_sampler import generate_farms_constant_area

# Define scenarios with consistent parameters
SCENARIOS = {
    'hard_20': {
        'n_farms': 20,
        'area_per_farm': 5.0,
        'frustration_ratio': 0.82,
        'negative_synergy_strength': -1.2,
        'rotation_gamma': 0.30,
        'one_hot_penalty': 2.0,
        'seed': 10001,
    },
    'hard_50': {
        'n_farms': 50,
        'area_per_farm': 5.0,
        'frustration_ratio': 0.82,
        'negative_synergy_strength': -1.2,
        'rotation_gamma': 0.30,
        'one_hot_penalty': 2.0,
        'seed': 10002,
    },
    'hard_90': {
        'n_farms': 90,
        'area_per_farm': 5.0,
        'frustration_ratio': 0.82,
        'negative_synergy_strength': -1.2,
        'rotation_gamma': 0.30,
        'one_hot_penalty': 2.0,
        'seed': 10003,
    },
    'hard_225': {
        'n_farms': 225,
        'area_per_farm': 5.0,
        'frustration_ratio': 0.82,
        'negative_synergy_strength': -1.2,
        'rotation_gamma': 0.30,
        'one_hot_penalty': 2.0,
        'seed': 10004,
    },
}

def create_scenario_config(scenario_name):
    """Create a scenario configuration"""
    config = SCENARIOS[scenario_name]
    
    # Generate farms with constant area
    land_availability = generate_farms_constant_area(
        n_farms=config['n_farms'],
        area_per_farm=config['area_per_farm'],
        seed=config['seed']
    )
    
    total_area = sum(land_availability.values())
    
    # Standard 6 crop families
    crop_families = {
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
    
    food_groups = {
        'Plant_Foods': ['Fruits', 'Grains', 'Legumes', 'Leafy_Vegetables', 'Root_Vegetables'],
        'Proteins': ['Proteins', 'Legumes']
    }
    
    scenario_config = {
        'parameters': {
            'land_availability': land_availability,
            'weights': {
                'nutritional_value': 0.25,
                'nutrient_density': 0.25,
                'environmental_impact': 0.20,
                'affordability': 0.15,
                'sustainability': 0.15
            },
            'minimum_planting_area': {crop: 0.5 for crop in crop_families},
            'food_group_constraints': {
                'Plant_Foods': {'min': 3, 'max': 5},
                'Proteins': {'min': 1, 'max': 2}
            },
            'rotation_gamma': config['rotation_gamma'],
            'spatial_k_neighbors': 4,
            'frustration_ratio': config['frustration_ratio'],
            'negative_synergy_strength': config['negative_synergy_strength'],
            'use_soft_one_hot': True,
            'one_hot_penalty': config['one_hot_penalty'],
            'diversity_bonus': 0.22
        }
    }
    
    return list(land_availability.keys()), crop_families, food_groups, scenario_config, total_area

# Test each scenario
print("="*80)
print("CONSISTENT HARDNESS SCENARIOS - CONSTANT AREA PER FARM")
print("="*80)

for scenario_name in ['hard_20', 'hard_50', 'hard_90', 'hard_225']:
    config = SCENARIOS[scenario_name]
    farms, foods, food_groups, scenario_config, total_area = create_scenario_config(scenario_name)
    
    params = scenario_config['parameters']
    
    print(f"\n{scenario_name}:")
    print(f"  Farms: {config['n_farms']}")
    print(f"  Variables: {config['n_farms'] * 6 * 3}")
    print(f"  Area per farm: {config['area_per_farm']} ha")
    print(f"  Total area: {total_area:.2f} ha")
    print(f"  Frustration: {params['frustration_ratio']}")
    print(f"  Neg strength: {params['negative_synergy_strength']}")
    print(f"  Gamma: {params['rotation_gamma']}")
    print(f"  Penalty: {params['one_hot_penalty']}")
    print(f"  Seed: {config['seed']}")
