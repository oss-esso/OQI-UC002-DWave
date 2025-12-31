"""
Scenario loading module for the unified benchmark.

Handles:
- Loading rotation_* scenarios from src/scenarios.py
- Equal area enforcement (mandatory per spec)
- Data validation
- Available scenario listing
"""

import os
import sys
import io
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from contextlib import redirect_stdout, redirect_stderr

# Add project root
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from existing codebase (suppress print statements)
_null_io = io.StringIO()
with redirect_stdout(_null_io), redirect_stderr(_null_io):
    from src.scenarios import load_food_data as _load_food_data_tuple_noisy

def _load_food_data_tuple(scenario_name: str):
    """Load food data while suppressing print statements from src.scenarios."""
    _null = io.StringIO()
    with redirect_stdout(_null), redirect_stderr(_null):
        return _load_food_data_tuple_noisy(scenario_name)

# Default area constant (all farms set to this)
DEFAULT_AREA_CONSTANT = 1.0

# Available rotation scenarios
ROTATION_SCENARIOS_6FAMILY = [
    "rotation_micro_25",      # 5 farms × 6 families
    "rotation_small_50",      # 10 farms × 6 families  
    "rotation_medium_100",    # 20 farms × 6 families
    "rotation_large_200",     # 40 farms × 6 families
]

ROTATION_SCENARIOS_27FOOD = [
    "rotation_250farms_27foods",   # 250 farms × 27 foods
    "rotation_350farms_27foods",   # 350 farms × 27 foods
    "rotation_500farms_27foods",   # 500 farms × 27 foods
    "rotation_1000farms_27foods",  # 1000 farms × 27 foods
]

# Extended scenario definitions for gap filling (synthetic)
SYNTHETIC_SCENARIOS = {
    # 6-family scenarios (for gap filling)
    "rotation_15farms_6foods": {"n_farms": 15, "n_foods": 6, "base": "rotation_medium_100"},
    "rotation_25farms_6foods": {"n_farms": 25, "n_foods": 6, "base": "rotation_medium_100"},
    "rotation_50farms_6foods": {"n_farms": 50, "n_foods": 6, "base": "rotation_large_200"},
    "rotation_75farms_6foods": {"n_farms": 75, "n_foods": 6, "base": "rotation_large_200"},
    "rotation_100farms_6foods": {"n_farms": 100, "n_foods": 6, "base": "rotation_large_200"},
    "rotation_150farms_6foods": {"n_farms": 150, "n_foods": 6, "base": "rotation_large_200"},
    
    # 27-food scenarios (for gap filling)
    "rotation_25farms_27foods": {"n_farms": 25, "n_foods": 27, "base": "rotation_250farms_27foods"},
    "rotation_50farms_27foods": {"n_farms": 50, "n_foods": 27, "base": "rotation_250farms_27foods"},
    "rotation_75farms_27foods": {"n_farms": 75, "n_foods": 27, "base": "rotation_250farms_27foods"},
    "rotation_100farms_27foods": {"n_farms": 100, "n_foods": 27, "base": "rotation_250farms_27foods"},
    "rotation_150farms_27foods": {"n_farms": 150, "n_foods": 27, "base": "rotation_250farms_27foods"},
    "rotation_200farms_27foods": {"n_farms": 200, "n_foods": 27, "base": "rotation_250farms_27foods"},
}


def get_available_scenarios() -> Dict[str, List[str]]:
    """
    Get list of available scenarios organized by type.
    
    Returns:
        Dict with keys:
            - "6_family": List of 6-family scenario names
            - "27_food": List of 27-food scenario names
            - "synthetic": List of synthetic gap-filling scenarios
    """
    return {
        "6_family": ROTATION_SCENARIOS_6FAMILY.copy(),
        "27_food": ROTATION_SCENARIOS_27FOOD.copy(),
        "synthetic": list(SYNTHETIC_SCENARIOS.keys()),
    }


def _normalize_areas(
    land_availability: Dict[str, float],
    area_constant: float
) -> Tuple[Dict[str, float], float]:
    """
    Normalize all farm areas to a constant value.
    
    Args:
        land_availability: Original land availability dict
        area_constant: Constant area for all farms
    
    Returns:
        Tuple of (normalized land dict, total area)
    """
    normalized = {farm: area_constant for farm in land_availability}
    total = area_constant * len(land_availability)
    return normalized, total


def _generate_food_benefits(
    food_names: List[str],
    seed: int = 42
) -> Dict[str, float]:
    """
    Generate heterogeneous food benefits.
    
    Uses deterministic random values for reproducibility.
    """
    rng = np.random.RandomState(seed)
    benefits = {}
    for food in food_names:
        # Benefits in range [0.5, 1.5] with some variation
        benefits[food] = rng.uniform(0.5, 1.5)
    return benefits


def _expand_farms(
    existing_farms: List[str],
    target_n: int,
    land_availability: Dict[str, float],
    area_constant: float
) -> Tuple[List[str], Dict[str, float]]:
    """
    Expand farm list to target size by duplication.
    
    Args:
        existing_farms: Existing farm names
        target_n: Target number of farms
        land_availability: Land availability dict
        area_constant: Area per farm
    
    Returns:
        Tuple of (expanded farm list, expanded land dict)
    """
    farms = existing_farms.copy()
    land = {f: area_constant for f in farms}
    
    while len(farms) < target_n:
        idx = len(farms) - len(existing_farms)
        base_farm = existing_farms[idx % len(existing_farms)]
        new_farm = f"{base_farm}_dup{idx}"
        farms.append(new_farm)
        land[new_farm] = area_constant
    
    return farms, land


def _shrink_farms(
    existing_farms: List[str],
    target_n: int,
    land_availability: Dict[str, float],
    area_constant: float
) -> Tuple[List[str], Dict[str, float]]:
    """
    Shrink farm list to target size.
    """
    farms = existing_farms[:target_n]
    land = {f: area_constant for f in farms}
    return farms, land


def load_scenario(
    scenario_name: str,
    area_constant: float = DEFAULT_AREA_CONSTANT,
    n_periods: int = 3,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Load a scenario with equal area enforcement.
    
    This is the ONLY way to load scenarios for the unified benchmark.
    Equal areas are mandatory to ensure fair comparisons.
    
    Args:
        scenario_name: Name of scenario to load
        area_constant: Area to set for all farms (default 1.0)
        n_periods: Number of rotation periods
        seed: Random seed for reproducible benefits
    
    Returns:
        Dict with:
            - farm_names: List[str]
            - food_names: List[str]
            - n_farms: int
            - n_foods: int
            - n_periods: int
            - n_vars: int
            - land_availability: Dict[str, float] (all equal to area_constant)
            - total_area: float
            - food_benefits: Dict[str, float]
            - food_groups: Dict[str, List[str]] (optional)
            - config: Dict (scenario config)
            - area_constant: float (recorded for logging)
    
    Raises:
        ValueError: If scenario not found or data invalid
    """
    # Check if this is a synthetic scenario
    if scenario_name in SYNTHETIC_SCENARIOS:
        spec = SYNTHETIC_SCENARIOS[scenario_name]
        base_scenario = spec["base"]
        target_n_farms = spec["n_farms"]
        target_n_foods = spec["n_foods"]
    else:
        base_scenario = scenario_name
        target_n_farms = None  # Use native size
        target_n_foods = None
    
    # Load base scenario
    try:
        farm_list, foods_dict, food_groups, config = _load_food_data_tuple(base_scenario)
    except Exception as e:
        raise ValueError(f"Failed to load scenario '{base_scenario}': {e}")
    
    # Convert to lists
    farm_names = list(farm_list) if not isinstance(farm_list, list) else farm_list
    
    # Extract food names and benefits
    food_names = []
    food_benefits = {}
    
    if isinstance(foods_dict, dict):
        for food_name, food_data in foods_dict.items():
            food_names.append(food_name)
            if isinstance(food_data, dict):
                # Try to extract benefit
                if 'benefit' in food_data:
                    food_benefits[food_name] = food_data['benefit']
                elif 'benefits' in food_data:
                    benefits = food_data['benefits']
                    if isinstance(benefits, (list, tuple)):
                        food_benefits[food_name] = sum(benefits) / len(benefits)
                    else:
                        food_benefits[food_name] = benefits
                else:
                    food_benefits[food_name] = 1.0
            else:
                food_benefits[food_name] = float(food_data)
    
    # Adjust farm count if synthetic scenario
    if target_n_farms is not None:
        if len(farm_names) < target_n_farms:
            farm_names, land_availability = _expand_farms(
                farm_names, target_n_farms, {}, area_constant
            )
        elif len(farm_names) > target_n_farms:
            farm_names, land_availability = _shrink_farms(
                farm_names, target_n_farms, {}, area_constant
            )
        else:
            land_availability = {f: area_constant for f in farm_names}
    else:
        land_availability = {f: area_constant for f in farm_names}
    
    # Adjust food count if specified (27→6 or vice versa)
    if target_n_foods is not None and len(food_names) != target_n_foods:
        if target_n_foods == 6 and len(food_names) > 6:
            # Aggregate to 6 families
            from food_grouping import FAMILY_ORDER
            food_names = FAMILY_ORDER.copy()
            food_benefits = _generate_food_benefits(food_names, seed)
        elif target_n_foods == 27 and len(food_names) == 6:
            # Expand to 27 foods - need to load a 27-food scenario
            try:
                _, foods_27, _, _ = _load_food_data_tuple("rotation_250farms_27foods")
                food_names = list(foods_27.keys())[:27]
                food_benefits = _generate_food_benefits(food_names, seed)
            except Exception:
                raise ValueError(f"Cannot expand to 27 foods from 6-family scenario")
    
    # Ensure benefits are heterogeneous
    unique_benefits = set(food_benefits.values())
    if len(unique_benefits) <= 1:
        food_benefits = _generate_food_benefits(food_names, seed)
    
    # Calculate totals
    n_farms = len(farm_names)
    n_foods = len(food_names)
    total_area = area_constant * n_farms
    n_vars = n_farms * n_foods * n_periods
    
    # Validate
    if n_farms == 0:
        raise ValueError(f"Scenario '{scenario_name}' has no farms")
    if n_foods == 0:
        raise ValueError(f"Scenario '{scenario_name}' has no foods")
    
    return {
        "farm_names": farm_names,
        "food_names": food_names,
        "n_farms": n_farms,
        "n_foods": n_foods,
        "n_periods": n_periods,
        "n_vars": n_vars,
        "land_availability": land_availability,
        "total_area": total_area,
        "food_benefits": food_benefits,
        "food_groups": food_groups or {},
        "config": config or {},
        "area_constant": area_constant,
        "scenario_name": scenario_name,
    }


def build_rotation_matrix(
    n_foods: int,
    frustration_ratio: float = 0.7,
    negative_strength: float = -0.8,
    seed: int = 42
) -> np.ndarray:
    """
    Build a rotation synergy matrix.
    
    From formulations.tex: R[c1,c2] represents the synergy/penalty
    for planting crop c2 after crop c1.
    
    Args:
        n_foods: Number of foods/crops
        frustration_ratio: Fraction of negative synergies
        negative_strength: Strength of negative synergies
        seed: Random seed
    
    Returns:
        n_foods × n_foods synergy matrix
    """
    rng = np.random.RandomState(seed)
    R = np.zeros((n_foods, n_foods))
    
    for i in range(n_foods):
        for j in range(n_foods):
            if i == j:
                # Same crop: strong negative (avoid monoculture)
                R[i, j] = negative_strength * 1.5
            elif rng.random() < frustration_ratio:
                # Most pairs: negative (frustration)
                R[i, j] = rng.uniform(negative_strength * 1.2, negative_strength * 0.3)
            else:
                # Some pairs: positive synergy
                R[i, j] = rng.uniform(0.02, 0.20)
    
    return R


def build_spatial_neighbors(
    farm_names: List[str],
    k_neighbors: int = 4,
    seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Build spatial neighbor graph.
    
    Arranges farms in a grid and connects each to its k nearest neighbors.
    
    Args:
        farm_names: List of farm names
        k_neighbors: Number of neighbors per farm
        seed: Random seed
    
    Returns:
        List of (farm_idx_1, farm_idx_2) neighbor pairs (undirected, no duplicates)
    """
    n_farms = len(farm_names)
    
    # Arrange farms in a grid
    side = int(np.ceil(np.sqrt(n_farms)))
    positions = {}
    for i, farm in enumerate(farm_names):
        row, col = i // side, i % side
        positions[i] = (row, col)
    
    # Build neighbor edges
    edges = set()
    for f1_idx in range(n_farms):
        # Calculate distances to all other farms
        distances = []
        for f2_idx in range(n_farms):
            if f1_idx != f2_idx:
                dist = np.sqrt(
                    (positions[f1_idx][0] - positions[f2_idx][0])**2 +
                    (positions[f1_idx][1] - positions[f2_idx][1])**2
                )
                distances.append((dist, f2_idx))
        
        # Take k nearest
        distances.sort()
        for _, f2_idx in distances[:k_neighbors]:
            # Store as sorted tuple to avoid duplicates
            edge = tuple(sorted([f1_idx, f2_idx]))
            edges.add(edge)
    
    return list(edges)


def validate_scenario_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate scenario data.
    
    Returns list of errors (empty if valid).
    """
    errors = []
    
    required = ["farm_names", "food_names", "land_availability", "food_benefits", "total_area"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if "farm_names" in data and len(data["farm_names"]) == 0:
        errors.append("No farms in scenario")
    
    if "food_names" in data and len(data["food_names"]) == 0:
        errors.append("No foods in scenario")
    
    # Check equal areas
    if "land_availability" in data:
        areas = list(data["land_availability"].values())
        if len(set(areas)) > 1:
            errors.append(f"Areas not equal: found {len(set(areas))} unique values")
    
    return errors


if __name__ == "__main__":
    # Test scenario loading
    print("Testing scenario loading...")
    
    test_scenarios = [
        "rotation_micro_25",
        "rotation_small_50",
        "rotation_25farms_27foods",
        "rotation_250farms_27foods",
    ]
    
    for name in test_scenarios:
        try:
            data = load_scenario(name)
            errors = validate_scenario_data(data)
            if errors:
                print(f"✗ {name}: {errors}")
            else:
                print(f"✓ {name}: {data['n_farms']} farms × {data['n_foods']} foods = {data['n_vars']} vars")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    print("\nAvailable scenarios:")
    for category, scenarios in get_available_scenarios().items():
        print(f"  {category}: {len(scenarios)} scenarios")
