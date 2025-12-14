"""
Data loader utilities for benchmarking scripts.

Converts src.scenarios.load_food_data tuple format to dict format used by benchmarks.
"""
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.scenarios import load_food_data as _load_food_data_tuple

def load_food_data_as_dict(scenario_name: str) -> Dict:
    """
    Load food data and convert to dict format expected by benchmark scripts.
    
    Args:
        scenario_name: Scenario to load (e.g., 'rotation_micro_25', 'simple', etc.)
    
    Returns:
        Dict with keys:
            - farm_names: List[str]
            - food_names: List[str]
            - land_availability: Dict[str, float]
            - food_benefits: Dict[str, float]
            - total_area: float
            - food_groups: Dict[str, List[str]] (optional)
    """
    # Load tuple format
    farm_list, foods_dict, food_groups_dict, config = _load_food_data_tuple(scenario_name)
    
    # Convert to dict format
    data = {
        'farm_names': farm_list if isinstance(farm_list, list) else list(farm_list),
        'food_names': [],
        'land_availability': {},
        'food_benefits': {},
        'total_area': 0.0,
        'food_groups': food_groups_dict if food_groups_dict else {},
        'config': config if config else {},
    }
    
    # Extract food names and benefits
    if isinstance(foods_dict, dict):
        for food_name, food_data in foods_dict.items():
            data['food_names'].append(food_name)
            if isinstance(food_data, dict):
                # Extract benefit value
                if 'benefit' in food_data:
                    data['food_benefits'][food_name] = food_data['benefit']
                elif 'benefits' in food_data:
                    # Average if multiple benefits
                    benefits = food_data['benefits']
                    if isinstance(benefits, (list, tuple)):
                        data['food_benefits'][food_name] = sum(benefits) / len(benefits)
                    else:
                        data['food_benefits'][food_name] = benefits
                else:
                    # Use a default value
                    data['food_benefits'][food_name] = 1.0
            else:
                # Assume foods_dict values are benefits directly
                data['food_benefits'][food_name] = float(food_data)
    
    # Create land availability with VARIATION to prevent trivial problems
    total_area = 0.0
    import numpy as np
    
    # Use reproducible random seed for consistent results
    rng = np.random.RandomState(42)
    
    for idx, farm in enumerate(data['farm_names']):
        # Try to get area from config
        if config and 'farm_areas' in config and farm in config['farm_areas']:
            area = config['farm_areas'][farm]
        else:
            # IMPORTANT: Add heterogeneity - vary farm sizes 50-150 hectares
            # This prevents Gurobi from trivializing the problem
            base_area = 100.0
            variation = rng.uniform(-40, 40)  # ±40% variation
            area = max(50.0, min(150.0, base_area + variation))
        
        data['land_availability'][farm] = area
        total_area += area
    
    data['total_area'] = total_area
    
    # CRITICAL: Add heterogeneity to food benefits
    # If all benefits are the same (1.0), vary them to make problem non-trivial
    unique_benefits = set(data['food_benefits'].values())
    if len(unique_benefits) == 1 and list(unique_benefits)[0] == 1.0:
        # All benefits are 1.0 - add variation
        food_list = list(data['food_benefits'].keys())
        for idx, food in enumerate(food_list):
            # Vary benefits 0.5 to 1.5 with some foods more valuable
            variation = rng.uniform(-0.3, 0.5)
            data['food_benefits'][food] = max(0.5, min(1.5, 1.0 + variation))
    
    return data

def verify_data_structure(data: Dict) -> bool:
    """Verify data structure has all required fields."""
    required_keys = ['farm_names', 'food_names', 'land_availability', 'food_benefits', 'total_area']
    
    for key in required_keys:
        if key not in data:
            print(f"Error: Missing required key '{key}'")
            return False
    
    if len(data['farm_names']) == 0:
        print("Error: No farms in data")
        return False
    
    if len(data['food_names']) == 0:
        print("Error: No foods in data")
        return False
    
    return True

if __name__ == '__main__':
    # Test loading
    print("Testing data loader...")
    
    test_scenarios = ['simple', 'rotation_micro_25', '30farms']
    
    for scenario in test_scenarios:
        try:
            data = load_food_data_as_dict(scenario)
            if verify_data_structure(data):
                print(f"✓ {scenario}: {len(data['farm_names'])} farms, {len(data['food_names'])} foods")
            else:
                print(f"✗ {scenario}: Invalid data structure")
        except Exception as e:
            print(f"✗ {scenario}: {e}")
