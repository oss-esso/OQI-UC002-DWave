import os
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any, Union
import logging

logger = logging.getLogger(__name__)


def load_food_data(complexity_level: str = 'simple') -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Load food data based on specified complexity level.

    Args:
        complexity_level (str): One of 'simple', 'intermediate', or 'full'

    Returns:
        Tuple containing farms, foods, food_groups, and config
    """
    if complexity_level == 'simple':
        return _load_simple_food_data()
    elif complexity_level == 'intermediate':
        return _load_intermediate_food_data()
    elif complexity_level == 'custom':
        return _load_custom_food_data()
    elif complexity_level == '30farms':
        return _load_30farms_food_data()
    elif complexity_level == '60farms':
        return _load_60farms_food_data()
    elif complexity_level == '90farms':
        return _load_90farms_food_data()
    elif complexity_level == '250farms':
        return _load_250farms_food_data()
    elif complexity_level == '350farms':
        return _load_350farms_food_data()
    elif complexity_level == '1000farms_full':
        return _load_1000farms_full_food_data()
    elif complexity_level == '500farms_full':
        return _load_500farms_full_food_data()
    elif complexity_level == '2000farms_full':
        return _load_2000farms_full_food_data()
    elif complexity_level == 'full':
        return _load_full_food_data()
    elif complexity_level == 'full_family':
        return _load_full_family_food_data()
    # Synthetic small-scale scenarios for QPU embedding testing (6-160 variables)
    elif complexity_level == 'micro_6':
        return _load_micro_6_food_data()
    elif complexity_level == 'micro_12':
        return _load_micro_12_food_data()
    elif complexity_level == 'tiny_24':
        return _load_tiny_24_food_data()
    elif complexity_level == 'tiny_40':
        return _load_tiny_40_food_data()
    elif complexity_level == 'small_60':
        return _load_small_60_food_data()
    elif complexity_level == 'small_80':
        return _load_small_80_food_data()
    elif complexity_level == 'small_100':
        return _load_small_100_food_data()
    elif complexity_level == 'medium_120':
        return _load_medium_120_food_data()
    elif complexity_level == 'medium_160':
        return _load_medium_160_food_data()
    # Rotation scenarios with quantum-friendly characteristics
    elif complexity_level == 'rotation_micro_25':
        return _load_rotation_micro_25_food_data()
    elif complexity_level == 'rotation_small_50':
        return _load_rotation_small_50_food_data()
    elif complexity_level == 'rotation_medium_100':
        return _load_rotation_medium_100_food_data()
    elif complexity_level == 'rotation_large_200':
        return _load_rotation_large_200_food_data()
    # NEW: Large-scale rotation scenarios (for hierarchical quantum solver)
    elif complexity_level == 'rotation_250farms_27foods':
        return _load_rotation_250farms_27foods_food_data()
    elif complexity_level == 'rotation_350farms_27foods':
        return _load_rotation_350farms_27foods_food_data()
    elif complexity_level == 'rotation_500farms_27foods':
        return _load_rotation_500farms_27foods_food_data()
    elif complexity_level == 'rotation_1000farms_27foods':
        return _load_rotation_1000farms_27foods_food_data()
    else:
        raise ValueError(
            f"Invalid complexity level: {complexity_level}. Must be one of: simple, intermediate, custom, "
            f"30farms, 60farms, 90farms, 250farms, 350farms, 500farms_full, 1000farms_full, 2000farms_full, "
            f"full, full_family, micro_6, micro_12, tiny_24, tiny_40, small_60, small_80, small_100, medium_120, medium_160, "
            f"rotation_micro_25, rotation_small_50, rotation_medium_100, rotation_large_200, "
            f"rotation_250farms_27foods, rotation_350farms_27foods, rotation_500farms_27foods, rotation_1000farms_27foods")


def _load_30farms_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load a mid-scale scenario between 'custom' and 'full_family'.

    Design:
    - Farms: generated via farm_sampler (e.g., 30 farms)
    - Foods: up to 10 (sampled from Excel if available; fallback to intermediate foods)
    - Constraints: similar to full_family but with modest minimum areas to fit small farms
    """
    import sys as _sys
    import os as _os
    import pandas as _pd

    # Add project root to path to import farm_sampler
    _project_root = _os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)

    from Utils.farm_sampler import generate_farms

    # Generate a moderate number of farms (between custom=2 and full_family=125)
    L = generate_farms(n_farms=30, seed=123)
    farms = list(L.keys())

    print(f"Generated {len(farms)} farms for 30farms with farm_sampler")
    print(f"Total land (30farms): {sum(L.values()):.2f} ha")

    # Try to load foods similarly to full/full_family from Excel (2 per group), else fallback
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(
        project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    foods: Dict[str, Dict[str, float]]
    food_groups: Dict[str, List[str]]
    if _os.path.exists(excel_path):
        try:
            df = _pd.read_excel(excel_path)
            col_map = {
                'Food_Name': 'Food_Name',
                'food_group': 'Food_Group',
                'nutritional_value': 'nutritional_value',
                'nutrient_density': 'nutrient_density',
                'environmental_impact': 'environmental_impact',
                'affordability': 'affordability',
                'sustainability': 'sustainability'
            }
            grp_col = 'food_group'
            name_col = 'Food_Name'
            # Shuffle and take top-2 per group to avoid FutureWarning on groupby.apply
            df_shuffled = df.sample(frac=1, random_state=42)
            sampled = df_shuffled.groupby(
                grp_col, group_keys=False).head(2).reset_index(drop=True)
            foods_list = sampled[col_map['Food_Name']].tolist()

            filt = df[df[name_col].isin(
                foods_list)][list(col_map.keys())].copy()
            filt.rename(columns=col_map, inplace=True)

            # Clamp and coerce objective columns to [0,1]
            objectives = ['nutritional_value', 'nutrient_density',
                          'environmental_impact', 'affordability', 'sustainability']
            for obj in objectives:
                filt[obj] = _pd.to_numeric(
                    filt[obj], errors='coerce').fillna(0.5)
                filt[obj] = filt[obj].clip(0, 1)

            foods = {}
            for _, row in filt.iterrows():
                fname = row['Food_Name']
                foods[fname] = {obj: float(row[obj]) for obj in objectives}

            food_groups = {}
            for _, row in filt.iterrows():
                g = row['Food_Group'] or 'Unknown'
                food_groups.setdefault(g, []).append(row['Food_Name'])
        except Exception as e:
            print(f"Error loading Excel for 30farms: {e}")
            print("Using intermediate scenario foods as fallback...")
            _, foods, food_groups, _ = _load_intermediate_food_data()
    else:
        print(f"Excel file not found at: {excel_path}")
        print("Using intermediate scenario foods as fallback...")
        _, foods, food_groups, _ = _load_intermediate_food_data()

    # Minimum planting areas (modest, uniform) to fit generated farms
    min_areas = {food: 0.01 for food in foods.keys()}  # 0.01 ha per crop

    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': max(1, min(2, len(lst))), 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }

    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 100,
        'pulp_time_limit': 120,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded 30farms data for {len(farms)} farms (from sampler) and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_60farms_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load a larger mid-scale scenario approximately double 30farms's size.

    Target size: ~600 farm–food pairs (e.g., 60 farms × ~10 foods).
    Implementation chooses farm count based on sampled foods to hit ~600 pairs.
    """
    import sys as _sys
    import os as _os
    import math as _math
    import pandas as _pd

    # Add project root to path to import farm_sampler
    _project_root = _os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)

    from Utils.farm_sampler import generate_farms

    # Step 1: Sample foods like in full/30farms
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(
        project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    foods: Dict[str, Dict[str, float]]
    food_groups: Dict[str, List[str]]
    if _os.path.exists(excel_path):
        try:
            df = _pd.read_excel(excel_path)
            col_map = {
                'Food_Name': 'Food_Name',
                'food_group': 'Food_Group',
                'nutritional_value': 'nutritional_value',
                'nutrient_density': 'nutrient_density',
                'environmental_impact': 'environmental_impact',
                'affordability': 'affordability',
                'sustainability': 'sustainability'
            }
            grp_col = 'food_group'
            name_col = 'Food_Name'
            # Sample up to 2 per group without FutureWarning
            df_shuffled = df.sample(frac=1, random_state=321)
            sampled = df_shuffled.groupby(
                grp_col, group_keys=False).head(2).reset_index(drop=True)
            foods_list = sampled[col_map['Food_Name']].tolist()

            filt = df[df[name_col].isin(
                foods_list)][list(col_map.keys())].copy()
            filt.rename(columns=col_map, inplace=True)

            objectives = ['nutritional_value', 'nutrient_density',
                          'environmental_impact', 'affordability', 'sustainability']
            for obj in objectives:
                filt[obj] = _pd.to_numeric(
                    filt[obj], errors='coerce').fillna(0.5)
                filt[obj] = filt[obj].clip(0, 1)

            foods = {}
            for _, row in filt.iterrows():
                fname = row['Food_Name']
                foods[fname] = {obj: float(row[obj]) for obj in objectives}

            food_groups = {}
            for _, row in filt.iterrows():
                g = row['Food_Group'] or 'Unknown'
                food_groups.setdefault(g, []).append(row['Food_Name'])
        except Exception as e:
            print(f"Error loading Excel for 60farms: {e}")
            print("Using intermediate scenario foods as fallback...")
            _, foods, food_groups, _ = _load_intermediate_food_data()
    else:
        print(f"Excel file not found at: {excel_path}")
        print("Using intermediate scenario foods as fallback...")
        _, foods, food_groups, _ = _load_intermediate_food_data()

    # Step 2: Choose number of farms to target ~600 pairs
    target_pairs = 600
    n_foods = max(1, len(foods))
    n_farms = max(20, _math.ceil(target_pairs / n_foods))
    L = generate_farms(n_farms=n_farms, seed=321)
    farms = list(L.keys())

    print(
        f"Generated {len(farms)} farms for 60farms with farm_sampler (target pairs: {target_pairs}, foods: {n_foods})")
    print(f"Total land (60farms): {sum(L.values()):.2f} ha")

    # Constraints and weights similar to 30farms
    min_areas = {food: 0.01 for food in foods.keys()}
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': max(1, min(2, len(lst))), 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }

    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 100,
        'pulp_time_limit': 120,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded 60farms data for {len(farms)} farms (from sampler) and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_90farms_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load a larger mid-scale scenario targeting ~900 farm–food pairs.

    Strategy:
    - Sample foods first (Excel if available; fallback to intermediate)
    - Choose farms = ceil(900 / n_foods), with a minimum of 30
    - Constraints mirror 30farms/60farms for comparability
    """
    import sys as _sys
    import os as _os
    import math as _math
    import pandas as _pd

    _project_root = _os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms

    # Foods
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(
        project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    foods: Dict[str, Dict[str, float]]
    food_groups: Dict[str, List[str]]
    if _os.path.exists(excel_path):
        try:
            df = _pd.read_excel(excel_path)
            col_map = {
                'Food_Name': 'Food_Name',
                'food_group': 'Food_Group',
                'nutritional_value': 'nutritional_value',
                'nutrient_density': 'nutrient_density',
                'environmental_impact': 'environmental_impact',
                'affordability': 'affordability',
                'sustainability': 'sustainability'
            }
            grp_col = 'food_group'
            name_col = 'Food_Name'
            df_shuffled = df.sample(frac=1, random_state=456)
            sampled = df_shuffled.groupby(
                grp_col, group_keys=False).head(2).reset_index(drop=True)
            foods_list = sampled[col_map['Food_Name']].tolist()

            filt = df[df[name_col].isin(
                foods_list)][list(col_map.keys())].copy()
            filt.rename(columns=col_map, inplace=True)

            objectives = ['nutritional_value', 'nutrient_density',
                          'environmental_impact', 'affordability', 'sustainability']
            for obj in objectives:
                filt[obj] = _pd.to_numeric(
                    filt[obj], errors='coerce').fillna(0.5)
                filt[obj] = filt[obj].clip(0, 1)

            foods = {}
            for _, row in filt.iterrows():
                fname = row['Food_Name']
                foods[fname] = {obj: float(row[obj]) for obj in objectives}

            food_groups = {}
            for _, row in filt.iterrows():
                g = row['Food_Group'] or 'Unknown'
                food_groups.setdefault(g, []).append(row['Food_Name'])
        except Exception as e:
            print(f"Error loading Excel for 90farms: {e}")
            print("Using intermediate scenario foods as fallback...")
            _, foods, food_groups, _ = _load_intermediate_food_data()
    else:
        print(f"Excel file not found at: {excel_path}")
        print("Using intermediate scenario foods as fallback...")
        _, foods, food_groups, _ = _load_intermediate_food_data()

    # Farms to hit ~900 pairs
    target_pairs = 900
    n_foods = max(1, len(foods))
    n_farms = max(30, _math.ceil(target_pairs / n_foods))
    L = generate_farms(n_farms=n_farms, seed=456)
    farms = list(L.keys())

    print(
        f"Generated {len(farms)} farms for 90farms with farm_sampler (target pairs: {target_pairs}, foods: {n_foods})")
    print(f"Total land (90farms): {sum(L.values()):.2f} ha")

    # Constraints and weights consistent with 30farms/60farms
    min_areas = {food: 0.01 for food in foods.keys()}
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': max(1, min(2, len(lst))), 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }

    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 100,
        'pulp_time_limit': 120,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded 90farms data for {len(farms)} farms (from sampler) and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_250farms_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load a large scenario doubling full_family's problem size (~2500 pairs).

    Strategy:
    - Generate exactly 250 farms via farm_sampler
    - Sample 2 foods per group from Excel (fallback to intermediate foods)
    - Constraints mirror 30/60/90 farms scenarios for apples-to-apples comparison
    """
    import sys as _sys
    import os as _os
    import pandas as _pd

    _project_root = _os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms

    # Generate fixed number of farms
    L = generate_farms(n_farms=250, seed=250)
    farms = list(L.keys())
    print(f"Generated {len(farms)} farms for 250farms with farm_sampler")
    print(f"Total land (250farms): {sum(L.values()):.2f} ha")

    # Foods from Excel (2 per group), fallback to intermediate
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(
        project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    foods: Dict[str, Dict[str, float]]
    food_groups: Dict[str, List[str]]
    if _os.path.exists(excel_path):
        try:
            df = _pd.read_excel(excel_path)
            col_map = {
                'Food_Name': 'Food_Name',
                'food_group': 'Food_Group',
                'nutritional_value': 'nutritional_value',
                'nutrient_density': 'nutrient_density',
                'environmental_impact': 'environmental_impact',
                'affordability': 'affordability',
                'sustainability': 'sustainability'
            }
            grp_col = 'food_group'
            name_col = 'Food_Name'
            df_shuffled = df.sample(frac=1, random_state=250)
            sampled = df_shuffled.groupby(
                grp_col, group_keys=False).head(2).reset_index(drop=True)
            foods_list = sampled[col_map['Food_Name']].tolist()

            filt = df[df[name_col].isin(
                foods_list)][list(col_map.keys())].copy()
            filt.rename(columns=col_map, inplace=True)

            objectives = ['nutritional_value', 'nutrient_density',
                          'environmental_impact', 'affordability', 'sustainability']
            for obj in objectives:
                filt[obj] = _pd.to_numeric(
                    filt[obj], errors='coerce').fillna(0.5)
                filt[obj] = filt[obj].clip(0, 1)

            foods = {}
            for _, row in filt.iterrows():
                fname = row['Food_Name']
                foods[fname] = {obj: float(row[obj]) for obj in objectives}

            food_groups = {}
            for _, row in filt.iterrows():
                g = row['Food_Group'] or 'Unknown'
                food_groups.setdefault(g, []).append(row['Food_Name'])
        except Exception as e:
            print(f"Error loading Excel for 250farms: {e}")
            print("Using intermediate scenario foods as fallback...")
            _, foods, food_groups, _ = _load_intermediate_food_data()
    else:
        print(f"Excel file not found at: {excel_path}")
        print("Using intermediate scenario foods as fallback...")
        _, foods, food_groups, _ = _load_intermediate_food_data()

    # Constraints and weights consistent with other Nfarms scenarios
    min_areas = {food: 0.01 for food in foods.keys()}
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': max(1, min(2, len(lst))), 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }

    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 100,
        'pulp_time_limit': 120,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded 250farms data for {len(farms)} farms (from sampler) and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_350farms_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load a very large scenario targeting ~3500 pairs (350 farms × ~10 foods).

    Mirrors the 250farms flow but with n_farms=350. Foods are sampled from Excel
    (2 per group) with a fallback to intermediate if Excel is missing.
    """
    import sys as _sys
    import os as _os
    import pandas as _pd

    _project_root = _os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms

    # Farms
    L = generate_farms(n_farms=350, seed=350)
    farms = list(L.keys())
    print(f"Generated {len(farms)} farms for 350farms with farm_sampler")
    print(f"Total land (350farms): {sum(L.values()):.2f} ha")

    # Foods
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(
        project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    foods: Dict[str, Dict[str, float]]
    food_groups: Dict[str, List[str]]
    if _os.path.exists(excel_path):
        try:
            df = _pd.read_excel(excel_path)
            col_map = {
                'Food_Name': 'Food_Name',
                'food_group': 'Food_Group',
                'nutritional_value': 'nutritional_value',
                'nutrient_density': 'nutrient_density',
                'environmental_impact': 'environmental_impact',
                'affordability': 'affordability',
                'sustainability': 'sustainability'
            }
            grp_col = 'food_group'
            name_col = 'Food_Name'
            df_shuffled = df.sample(frac=1, random_state=350)
            sampled = df_shuffled.groupby(
                grp_col, group_keys=False).head(2).reset_index(drop=True)
            foods_list = sampled[col_map['Food_Name']].tolist()

            filt = df[df[name_col].isin(
                foods_list)][list(col_map.keys())].copy()
            filt.rename(columns=col_map, inplace=True)

            objectives = ['nutritional_value', 'nutrient_density',
                          'environmental_impact', 'affordability', 'sustainability']
            for obj in objectives:
                filt[obj] = _pd.to_numeric(
                    filt[obj], errors='coerce').fillna(0.5)
                filt[obj] = filt[obj].clip(0, 1)

            foods = {}
            for _, row in filt.iterrows():
                fname = row['Food_Name']
                foods[fname] = {obj: float(row[obj]) for obj in objectives}

            food_groups = {}
            for _, row in filt.iterrows():
                g = row['Food_Group'] or 'Unknown'
                food_groups.setdefault(g, []).append(row['Food_Name'])
        except Exception as e:
            print(f"Error loading Excel for 350farms: {e}")
            print("Using intermediate scenario foods as fallback...")
            _, foods, food_groups, _ = _load_intermediate_food_data()
    else:
        print(f"Excel file not found at: {excel_path}")
        print("Using intermediate scenario foods as fallback...")
        _, foods, food_groups, _ = _load_intermediate_food_data()

    # Constraints and weights (consistent with other Nfarms)
    min_areas = {food: 0.01 for food in foods.keys()}
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': max(1, min(2, len(lst))), 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }

    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 100,
        'pulp_time_limit': 120,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded 350farms data for {len(farms)} farms (from sampler) and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_1000farms_full_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load a very large scenario: 1000 farms with the full set of foods.

    Unlike the Nfarms scenarios that sample 2 per group (~10 foods), this uses
    ALL foods present in the Excel data (coerced to [0,1]), so expect ~27 foods
    based on your dataset. Falls back to intermediate foods if Excel is missing.
    """
    import sys as _sys
    import os as _os
    import pandas as _pd

    _project_root = _os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms

    # Generate 1000 farms
    L = generate_farms(n_farms=1000, seed=1000)
    farms = list(L.keys())
    print(f"Generated {len(farms)} farms for 1000farms_full with farm_sampler")
    print(f"Total land (1000farms_full): {sum(L.values()):.2f} ha")

    # Load ALL foods from Excel (no per-group sampling)
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(
        project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    foods: Dict[str, Dict[str, float]]
    food_groups: Dict[str, List[str]]
    if _os.path.exists(excel_path):
        try:
            df = _pd.read_excel(excel_path)
            col_map = {
                'Food_Name': 'Food_Name',
                'food_group': 'Food_Group',
                'nutritional_value': 'nutritional_value',
                'nutrient_density': 'nutrient_density',
                'environmental_impact': 'environmental_impact',
                'affordability': 'affordability',
                'sustainability': 'sustainability'
            }
            # Keep only needed columns; drop rows missing Food_Name or group
            filt = df[list(col_map.keys())].copy()
            filt.rename(columns=col_map, inplace=True)
            filt = filt.dropna(subset=['Food_Name', 'Food_Group'])

            objectives = ['nutritional_value', 'nutrient_density',
                          'environmental_impact', 'affordability', 'sustainability']
            for obj in objectives:
                filt[obj] = _pd.to_numeric(
                    filt[obj], errors='coerce').fillna(0.5).clip(0, 1)

            # Deduplicate foods by name keeping first occurrence
            filt = filt.drop_duplicates(subset=['Food_Name'])

            foods = {
                row['Food_Name']: {obj: float(row[obj]) for obj in objectives}
                for _, row in filt.iterrows()
            }

            food_groups = {}
            for _, row in filt.iterrows():
                g = row['Food_Group'] or 'Unknown'
                fname = row['Food_Name']
                food_groups.setdefault(g, []).append(fname)
        except Exception as e:
            print(f"Error loading Excel for 1000farms_full: {e}")
            print("Using intermediate scenario foods as fallback...")
            _, foods, food_groups, _ = _load_intermediate_food_data()
    else:
        print(f"Excel file not found at: {excel_path}")
        print("Using intermediate scenario foods as fallback...")
        _, foods, food_groups, _ = _load_intermediate_food_data()

    # Constraints (same family as other Nfarms) – scaled-friendly
    min_areas = {food: 0.01 for food in foods.keys()}
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': max(1, min(2, len(lst))), 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }

    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 100,
        'pulp_time_limit': 120,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded 1000farms_full data for {len(farms)} farms (from sampler) and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_500farms_full_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load a very large scenario: 500 farms with the full set of foods.

    Uses ALL foods present in the Excel data (no sampling). Falls back to the
    intermediate foods set if Excel is missing.
    """
    import sys as _sys
    import os as _os
    import pandas as _pd

    _project_root = _os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms

    # Generate 500 farms
    L = generate_farms(n_farms=500, seed=500)
    farms = list(L.keys())
    print(f"Generated {len(farms)} farms for 500farms_full with farm_sampler")
    print(f"Total land (500farms_full): {sum(L.values()):.2f} ha")

    # Load ALL foods from Excel (no per-group sampling)
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(
        project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    foods: Dict[str, Dict[str, float]]
    food_groups: Dict[str, List[str]]
    if _os.path.exists(excel_path):
        try:
            df = _pd.read_excel(excel_path)
            col_map = {
                'Food_Name': 'Food_Name',
                'food_group': 'Food_Group',
                'nutritional_value': 'nutritional_value',
                'nutrient_density': 'nutrient_density',
                'environmental_impact': 'environmental_impact',
                'affordability': 'affordability',
                'sustainability': 'sustainability'
            }
            filt = df[list(col_map.keys())].copy()
            filt.rename(columns=col_map, inplace=True)
            filt = filt.dropna(subset=['Food_Name', 'Food_Group'])

            objectives = ['nutritional_value', 'nutrient_density',
                          'environmental_impact', 'affordability', 'sustainability']
            for obj in objectives:
                filt[obj] = _pd.to_numeric(
                    filt[obj], errors='coerce').fillna(0.5).clip(0, 1)

            filt = filt.drop_duplicates(subset=['Food_Name'])
            foods = {row['Food_Name']: {obj: float(
                row[obj]) for obj in objectives} for _, row in filt.iterrows()}

            food_groups = {}
            for _, row in filt.iterrows():
                g = row['Food_Group'] or 'Unknown'
                fname = row['Food_Name']
                food_groups.setdefault(g, []).append(fname)
        except Exception as e:
            print(f"Error loading Excel for 500farms_full: {e}")
            print("Using intermediate scenario foods as fallback...")
            _, foods, food_groups, _ = _load_intermediate_food_data()
    else:
        print(f"Excel file not found at: {excel_path}")
        print("Using intermediate scenario foods as fallback...")
        _, foods, food_groups, _ = _load_intermediate_food_data()

    # Constraints consistent with the Nfarms family
    min_areas = {food: 0.01 for food in foods.keys()}
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': max(1, min(2, len(lst))), 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }

    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 100,
        'pulp_time_limit': 120,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded 500farms_full data for {len(farms)} farms (from sampler) and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_2000farms_full_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load an extra-large scenario: 2000 farms with the full set of foods.

    Same as 500/1000 farms full but with n_farms=2000. This is intended for
    scaling analysis; generation is lightweight but solving may be very heavy.
    """
    import sys as _sys
    import os as _os
    import pandas as _pd

    _project_root = _os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms

    # Generate 2000 farms
    L = generate_farms(n_farms=2000, seed=2000)
    farms = list(L.keys())
    print(f"Generated {len(farms)} farms for 2000farms_full with farm_sampler")
    print(f"Total land (2000farms_full): {sum(L.values()):.2f} ha")

    # Load ALL foods from Excel
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(
        project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    foods: Dict[str, Dict[str, float]]
    food_groups: Dict[str, List[str]]
    if _os.path.exists(excel_path):
        try:
            df = _pd.read_excel(excel_path)
            col_map = {
                'Food_Name': 'Food_Name',
                'food_group': 'Food_Group',
                'nutritional_value': 'nutritional_value',
                'nutrient_density': 'nutrient_density',
                'environmental_impact': 'environmental_impact',
                'affordability': 'affordability',
                'sustainability': 'sustainability'
            }
            filt = df[list(col_map.keys())].copy()
            filt.rename(columns=col_map, inplace=True)
            filt = filt.dropna(subset=['Food_Name', 'Food_Group'])

            objectives = ['nutritional_value', 'nutrient_density',
                          'environmental_impact', 'affordability', 'sustainability']
            for obj in objectives:
                filt[obj] = _pd.to_numeric(
                    filt[obj], errors='coerce').fillna(0.5).clip(0, 1)

            filt = filt.drop_duplicates(subset=['Food_Name'])
            foods = {row['Food_Name']: {obj: float(
                row[obj]) for obj in objectives} for _, row in filt.iterrows()}

            food_groups = {}
            for _, row in filt.iterrows():
                g = row['Food_Group'] or 'Unknown'
                fname = row['Food_Name']
                food_groups.setdefault(g, []).append(fname)
        except Exception as e:
            print(f"Error loading Excel for 2000farms_full: {e}")
            print("Using intermediate scenario foods as fallback...")
            _, foods, food_groups, _ = _load_intermediate_food_data()
    else:
        print(f"Excel file not found at: {excel_path}")
        print("Using intermediate scenario foods as fallback...")
        _, foods, food_groups, _ = _load_intermediate_food_data()

    # Constraints
    min_areas = {food: 0.01 for food in foods.keys()}
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': max(1, min(2, len(lst))), 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }

    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 100,
        'pulp_time_limit': 120,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded 2000farms_full data for {len(farms)} farms (from sampler) and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_simple_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load simplified food data for testing."""
    # Define farms
    farms = ['Farm1', 'Farm2', 'Farm3']

    # Define foods with nutritional values, etc.
    foods = {
        'Wheat': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.3,
            'affordability': 0.8,
            'sustainability': 0.7
        },
        'Corn': {
            'nutritional_value': 0.6,
            'nutrient_density': 0.5,
            'environmental_impact': 0.4,
            'affordability': 0.9,
            'sustainability': 0.6
        },
        'Rice': {
            'nutritional_value': 0.8,
            'nutrient_density': 0.7,
            'environmental_impact': 0.6,
            'affordability': 0.7,
            'sustainability': 0.5
        },
        'Soybeans': {
            'nutritional_value': 0.9,
            'nutrient_density': 0.8,
            'environmental_impact': 0.2,
            'affordability': 0.6,
            'sustainability': 0.8
        },
        'Potatoes': {
            'nutritional_value': 0.5,
            'nutrient_density': 0.4,
            'environmental_impact': 0.3,
            'affordability': 0.9,
            'sustainability': 0.7
        },
        'Apples': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.2,
            'affordability': 0.5,
            'sustainability': 0.8
        }
    }

    # Define food groups
    food_groups = {
        'Grains': ['Wheat', 'Corn', 'Rice'],
        'Legumes': ['Soybeans'],
        'Vegetables': ['Potatoes'],
        'Fruits': ['Apples']
    }

    # Set parameters
    parameters = {
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.25,
            'affordability': 0,
            'sustainability': 0,
            'environmental_impact': 0.5
        },
        'land_availability': {
            'Farm1': 75,
            'Farm2': 100,
            'Farm3': 50
        },
        'food_groups': food_groups
    }

    # Update config
    config = {
        'parameters': parameters
    }

    logger.info(
        f"Loaded simple data for {len(farms)} farms and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_intermediate_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load intermediate food data for testing."""
    # Define farms
    farms = ['Farm1', 'Farm2', 'Farm3']

    # Define foods with nutritional values, etc.
    foods = {
        'Wheat': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.3,
            'affordability': 0.8,
            'sustainability': 0.7
        },
        'Corn': {
            'nutritional_value': 0.6,
            'nutrient_density': 0.5,
            'environmental_impact': 0.4,
            'affordability': 0.9,
            'sustainability': 0.6
        },
        'Rice': {
            'nutritional_value': 0.8,
            'nutrient_density': 0.7,
            'environmental_impact': 0.6,
            'affordability': 0.7,
            'sustainability': 0.5
        },
        'Soybeans': {
            'nutritional_value': 0.9,
            'nutrient_density': 0.8,
            'environmental_impact': 0.2,
            'affordability': 0.6,
            'sustainability': 0.8
        },
        'Potatoes': {
            'nutritional_value': 0.5,
            'nutrient_density': 0.4,
            'environmental_impact': 0.3,
            'affordability': 0.9,
            'sustainability': 0.7
        },
        'Apples': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.2,
            'affordability': 0.5,
            'sustainability': 0.8
        }
    }

    # Define food groups
    food_groups = {
        'Grains': ['Wheat', 'Corn', 'Rice'],
        'Legumes': ['Soybeans'],
        'Vegetables': ['Potatoes'],
        'Fruits': ['Apples']
    }

    # Set parameters with updated configuration
    parameters = {
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'affordability': 0.15,
            'sustainability': 0.15,
            'environmental_impact': 0.25
        },
        'land_availability': {
            'Farm1': 75,
            'Farm2': 100,
            'Farm3': 50
        },
        'minimum_planting_area': {
            'Wheat': 10,
            'Corn': 8,
            'Rice': 12,
            'Soybeans': 7,
            'Potatoes': 5,
            'Apples': 15
        },
        'max_percentage_per_crop': {
            food: 0.4 for food in foods  # Updated to 40% max per crop
        },
        'social_benefit': {
            farm: 0.2 for farm in farms  # 20% minimum land utilization
        },
        'food_group_constraints': {
            group: {
                'min_foods': 1,  # At least 1 food from each group
                'max_foods': len(foods_in_group)  # Up to all foods in group
            }
            for group, foods_in_group in food_groups.items()
        }
    }

    # Update config with additional solver settings
    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,  # Tighter convergence tolerance
        'benders_max_iterations': 100,  # More iterations allowed
        'pulp_time_limit': 120,  # 2 minutes time limit for PuLP
        'use_multi_cut': True,  # Enable multi-cut Benders
        'use_trust_region': True,  # Enable trust region stabilization
        'use_anticycling': True,  # Enable anti-cycling measures
        'use_norm_cuts': True,  # Use normalized optimality cuts
        'quantum_settings': {  # Added quantum-specific settings
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded intermediate data for {len(farms)} farms and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_custom_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load custom food data for testing with 2 farms, 3 food groups, 2 foods per group."""
    # Define farms (reduced to 2)
    farms = ['Farm1', 'Farm2']

    # Define foods with nutritional values, etc. (3 food groups, 2 foods each)
    foods = {
        'Wheat': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.3,
            'affordability': 0.8,
            'sustainability': 0.7
        },
        'Rice': {
            'nutritional_value': 0.8,
            'nutrient_density': 0.7,
            'environmental_impact': 0.6,
            'affordability': 0.7,
            'sustainability': 0.5
        },
        'Soybeans': {
            'nutritional_value': 0.9,
            'nutrient_density': 0.8,
            'environmental_impact': 0.2,
            'affordability': 0.6,
            'sustainability': 0.8
        },
        'Potatoes': {
            'nutritional_value': 0.5,
            'nutrient_density': 0.4,
            'environmental_impact': 0.3,
            'affordability': 0.9,
            'sustainability': 0.7
        },
        'Apples': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.2,
            'affordability': 0.5,
            'sustainability': 0.8
        },
        'Tomatoes': {
            'nutritional_value': 0.6,
            'nutrient_density': 0.5,
            'environmental_impact': 0.2,
            'affordability': 0.7,
            'sustainability': 0.9
        }
    }

    # Define food groups (3 groups, 2 foods each)
    food_groups = {
        'Grains': ['Wheat', 'Rice'],
        # Treating potatoes as legumes for this scenario
        'Legumes': ['Soybeans', 'Potatoes'],
        # Treating tomatoes as fruits for this scenario
        'Fruits': ['Apples', 'Tomatoes']
    }

    # Set parameters with updated configuration (same as intermediate)
    parameters = {
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'affordability': 0.15,
            'sustainability': 0.15,
            'environmental_impact': 0.25
        },
        'land_availability': {
            'Farm1': 75,
            'Farm2': 100
        },
        'minimum_planting_area': {
            'Wheat': 10,
            'Rice': 12,
            'Soybeans': 7,
            'Potatoes': 5,
            'Apples': 15,
            'Tomatoes': 8
        },
        'max_percentage_per_crop': {
            food: 0.4 for food in foods  # 40% max per crop
        },
        'social_benefit': {
            farm: 0.2 for farm in farms  # 20% minimum land utilization
        },
        'food_group_constraints': {
            group: {
                'min_foods': 1,  # At least 1 food from each group
                'max_foods': len(foods_in_group)  # Up to all foods in group
            }
            for group, foods_in_group in food_groups.items()
        },
        # Additional parameters to match pulp_sim.py formulation
        # Minimum number of different food types selected globally
        'global_min_different_foods': 5,
        'min_foods_per_farm': 1,  # Minimum number of different foods per farm
        # Maximum number of different foods per farm (all 6 foods)
        'max_foods_per_farm': 6,
        'min_total_land_usage_percentage': 0.5  # Minimum 50% total land utilization
    }

    # Update config with additional solver settings (same as intermediate)
    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 100,
        'pulp_time_limit': 120,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded custom data for {len(farms)} farms and {len(foods)} foods")
    return farms, foods, food_groups, config


def _load_full_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load food data and configuration from Excel for optimization."""
    # Locate Excel file - look in the Inputs directory relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from src/
    excel_path = os.path.join(project_root, "Inputs",
                              "Combined_Food_Data.xlsx")

    # If file doesn't exist in project, check if it exists and offer helpful message
    if not os.path.exists(excel_path):
        print(f"Excel file not found at: {excel_path}")
        print(
            "The 'full' scenario requires Combined_Food_Data.xlsx in the Inputs/ directory.")
        print("Using 'intermediate' scenario as fallback...")
        return _load_intermediate_food_data()

    print(f"Loading food data from: {excel_path}")

    # Read Excel
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        print("Using 'intermediate' scenario as fallback...")
        return _load_intermediate_food_data()

    # Map columns
    col_map = {
        'Food_Name': 'Food_Name',
        'food_group': 'Food_Group',
        'nutritional_value': 'nutritional_value',
        'nutrient_density': 'nutrient_density',
        'environmental_impact': 'environmental_impact',
        'affordability': 'affordability',
        'sustainability': 'sustainability'
    }
    missing = [c for c in col_map if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in Excel: {missing}")

    # Select 2 samples per group
    grp_col = 'food_group'
    name_col = 'Food_Name'
    sampled = df.groupby(grp_col).apply(
        lambda x: x.sample(n=min(len(x), 2))
    ).reset_index(drop=True)
    foods_list = sampled[col_map['Food_Name']].tolist()

    # Filter and rename
    filt = df[df[name_col].isin(foods_list)][list(col_map.keys())].copy()
    filt.rename(columns=col_map, inplace=True)

    # Convert scores
    objectives = ['nutritional_value', 'nutrient_density',
                  'environmental_impact', 'affordability', 'sustainability']
    for obj in objectives:
        filt[obj] = pd.to_numeric(filt[obj], errors='coerce').fillna(0.0)

    # Build structures without profitability
    farms = ['Farm1', 'Farm2', 'Farm3', 'Farm4', 'Farm5']
    foods = {}
    for _, row in filt.iterrows():
        food_dict = {obj: float(row[obj]) for obj in objectives}
        # No longer adding profitability
        foods[row['Food_Name']] = food_dict

    food_groups: Dict[str, List[str]] = {}
    for _, row in filt.iterrows():
        fg = row['Food_Group'] or 'Unknown'
        food_groups.setdefault(fg, []).append(row['Food_Name'])

    # Default config parameters
    parameters = {
        'land_availability': {
            'Farm1': 50, 'Farm2': 75, 'Farm3': 100, 'Farm4': 80, 'Farm5': 50
        },
        'social_benefit': {
            'Farm1': 0.20,
            'Farm2': 0.25,
            'Farm3': 0.15,
            'Farm4': 0.20,
            'Farm5': 0.10,
        },
        'minimum_planting_area': {
            "Mango": 0.000929, "Papaya": 0.000400, "Orange": 0.005810, "Banana": 0.005950,
            "Guava": 0.000929, "Watermelon": 0.000334, "Apple": 0.003720, "Avocado": 0.008360,
            "Durian": 0.010000, "Corn": 0.000183, "Potato": 0.000090, "Tofu": 0.000010,
            "Tempeh": 0.000010, "Peanuts": 0.000030, "Chickpeas": 0.000020, "Pumpkin": 0.000100,
            "Spinach": 0.000090, "Tomatoes": 0.000105, "Long bean": 0.000090, "Cabbage": 0.000250,
            "Eggplant": 0.000360, "Cucumber": 0.000500, "Egg": 0.000019, "Beef": 0.728400,
            "Lamb": 0.025000, "Pork": 0.016200, "Chicken": 0.001000
        },
        'max_percentage_per_crop': {
            "Mango": 1.0, "Papaya": 1.0, "Orange": 1.0, "Banana": 1.0,
            "Guava": 1.0, "Watermelon": 1.0, "Apple": 1.0, "Avocado": 1.0,
            "Durian": 1.0, "Corn": 1.0, "Potato": 1.0, "Tofu": 1.0,
            "Tempeh": 1.0, "Peanuts": 1.0, "Chickpeas": 0.5, "Pumpkin": 1.0,
            "Spinach": 1.0, "Tomatoes": 1.0, "Long bean": 0.10,
            "Cabbage": 1.0, "Eggplant": 1.0, "Cucumber": 1.0, "Egg": 1.0,
            "Beef": 1.0, "Lamb": 1.0, "Pork": 1.0, "Chicken": 1.0
        },
        'food_group_constraints': {
            g: {'min_foods': 1, 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }

    # Add solver settings to config
    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,  # Tighter convergence tolerance
        'benders_max_iterations': 100,  # More iterations allowed
        'pulp_time_limit': 120,  # 2 minutes time limit for PuLP
        'use_multi_cut': True,  # Enable multi-cut Benders
        'use_trust_region': True,  # Enable trust region stabilization
        'use_anticycling': True,  # Enable anti-cycling measures
        'use_norm_cuts': True,  # Use normalized optimality cuts
        'quantum_settings': {  # Added quantum-specific settings
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded full data for {len(farms)} farms and {len(foods)} foods. Parameters generated.")

    return farms, foods, food_groups, config


def _load_full_family_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load full scenario data with 25 farms from Utils.farm_sampler and adjusted minimum areas."""
    import sys
    import os
    # Add project root to path to import farm_sampler
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from Utils.farm_sampler import generate_farms

    # Generate 15 farms using the sampler
    L = generate_farms(n_farms=125, seed=42)
    farms = list(L.keys())

    # Note: These farms are used as a default scenario template
    # Actual farm data should be provided by the calling code
    # print(f"Generated {len(farms)} farms with farm_sampler")
    # print(f"Total land: {sum(L.values()):.2f} ha")

    # Use the same food data as full scenario
    # Load from Excel (same as _load_full_food_data)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_excel = os.path.dirname(script_dir)
    excel_path = os.path.join(
        project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    if not os.path.exists(excel_path):
        print(f"Excel file not found at: {excel_path}")
        print("Using intermediate scenario foods as fallback...")
        # Use intermediate food data
        _, foods, food_groups, _ = _load_intermediate_food_data()
    else:
        print(f"Loading food data from: {excel_path}")
        try:
            df = pd.read_excel(excel_path)

            col_map = {
                'Food_Name': 'Food_Name',
                'food_group': 'Food_Group',
                'nutritional_value': 'nutritional_value',
                'nutrient_density': 'nutrient_density',
                'environmental_impact': 'environmental_impact',
                'affordability': 'affordability',
                'sustainability': 'sustainability'
            }

            # Load ALL foods from Excel (not just 2 per group)
            grp_col = 'food_group'
            name_col = 'Food_Name'
            # Use all foods, not just a sample
            foods_list = df[col_map['Food_Name']].tolist()

            filt = df[df[name_col].isin(
                foods_list)][list(col_map.keys())].copy()
            filt.rename(columns=col_map, inplace=True)

            objectives = ['nutritional_value', 'nutrient_density',
                          'environmental_impact', 'affordability', 'sustainability']
            for obj in objectives:
                filt[obj] = filt[obj].fillna(0.5)
                filt[obj] = filt[obj].clip(0, 1)

            # Build foods dict
            foods = {}
            for _, row in filt.iterrows():
                fname = row['Food_Name']
                foods[fname] = {
                    'nutritional_value': float(row['nutritional_value']),
                    'nutrient_density': float(row['nutrient_density']),
                    'environmental_impact': float(row['environmental_impact']),
                    'affordability': float(row['affordability']),
                    'sustainability': float(row['sustainability'])
                }

            # Build food groups
            food_groups = {}
            for _, row in filt.iterrows():
                g = row['Food_Group']
                fname = row['Food_Name']
                if g not in food_groups:
                    food_groups[g] = []
                food_groups[g].append(fname)

        except Exception as e:
            print(f"Error loading Excel: {e}")
            print("Using intermediate scenario foods as fallback...")
            _, foods, food_groups, _ = _load_intermediate_food_data()

    # Adjusted minimum planting areas (reduced to fit small farms)

    min_areas = {}
    for food in foods.keys():
        min_areas[food] = 0.01  # 0.01 ha (100 m²) minimum for all crops

    parameters = {
        'land_availability': L,  # from Utils.farm_sampler
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {
            food: 0.4 for food in foods
        },
        'social_benefit': {
            farm: 0.2 for farm in farms
        },
        'food_group_constraints': {
            g: {'min_foods': 2, 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }

    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 100,
        'pulp_time_limit': 120,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(
        f"Loaded full_family data for {len(farms)} farms (from sampler) and {len(foods)} foods")

    return farms, foods, food_groups, config


# ============================================================================
# SYNTHETIC SMALL-SCALE SCENARIOS FOR QPU EMBEDDING TESTING (6-160 variables)
# ============================================================================
# 
# These scenarios are designed to test direct QPU embedding at scales where
# embedding might succeed or fail on Pegasus/Zephyr architectures.
# 
# Variable count formula: n_vars = n_plots × n_foods + n_foods (for U variables)
# 
# Scenarios maintain meaningful constraints:
# - Food groups with min_foods constraints
# - Diversity requirements 
# - Realistic food benefit values
# ============================================================================


def _create_synthetic_foods(n_foods_per_group: Dict[str, int], seed: int = 42) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    """
    Create synthetic foods with realistic nutritional attributes.
    
    Args:
        n_foods_per_group: Dictionary mapping group name to number of foods
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (foods dict, food_groups dict)
    """
    import numpy as np
    np.random.seed(seed)
    
    # Base nutritional profiles by food group (realistic ranges)
    group_profiles = {
        'Grains': {
            'nutritional_value': (0.6, 0.8),
            'nutrient_density': (0.5, 0.7),
            'environmental_impact': (0.3, 0.5),
            'affordability': (0.7, 0.9),
            'sustainability': (0.5, 0.7)
        },
        'Legumes': {
            'nutritional_value': (0.7, 0.9),
            'nutrient_density': (0.7, 0.9),
            'environmental_impact': (0.2, 0.4),
            'affordability': (0.6, 0.8),
            'sustainability': (0.7, 0.9)
        },
        'Vegetables': {
            'nutritional_value': (0.5, 0.8),
            'nutrient_density': (0.6, 0.9),
            'environmental_impact': (0.2, 0.4),
            'affordability': (0.5, 0.8),
            'sustainability': (0.6, 0.8)
        },
        'Fruits': {
            'nutritional_value': (0.6, 0.8),
            'nutrient_density': (0.5, 0.8),
            'environmental_impact': (0.2, 0.4),
            'affordability': (0.4, 0.7),
            'sustainability': (0.6, 0.8)
        },
        'Proteins': {
            'nutritional_value': (0.7, 0.9),
            'nutrient_density': (0.6, 0.8),
            'environmental_impact': (0.4, 0.7),
            'affordability': (0.4, 0.7),
            'sustainability': (0.4, 0.7)
        }
    }
    
    # Food name templates by group
    food_names_templates = {
        'Grains': ['Wheat', 'Rice', 'Corn', 'Barley', 'Oats', 'Millet', 'Sorghum', 'Rye'],
        'Legumes': ['Soybeans', 'Lentils', 'Chickpeas', 'BlackBeans', 'PintoBeans', 'KidneyBeans'],
        'Vegetables': ['Potatoes', 'Carrots', 'Tomatoes', 'Onions', 'Cabbage', 'Spinach', 'Broccoli'],
        'Fruits': ['Apples', 'Oranges', 'Bananas', 'Grapes', 'Berries', 'Mangoes', 'Pears'],
        'Proteins': ['Eggs', 'Chicken', 'Fish', 'Beef', 'Pork', 'Turkey']
    }
    
    foods = {}
    food_groups = {}
    
    for group, n_foods in n_foods_per_group.items():
        if group not in group_profiles:
            # Use vegetables as default profile
            profile = group_profiles['Vegetables']
        else:
            profile = group_profiles[group]
        
        templates = food_names_templates.get(group, [f'{group}Food'])
        group_foods = []
        
        for i in range(n_foods):
            # Generate food name
            if i < len(templates):
                food_name = templates[i]
            else:
                food_name = f'{group}_{i+1}'
            
            # Generate realistic attributes within group ranges
            foods[food_name] = {
                'nutritional_value': round(np.random.uniform(*profile['nutritional_value']), 2),
                'nutrient_density': round(np.random.uniform(*profile['nutrient_density']), 2),
                'environmental_impact': round(np.random.uniform(*profile['environmental_impact']), 2),
                'affordability': round(np.random.uniform(*profile['affordability']), 2),
                'sustainability': round(np.random.uniform(*profile['sustainability']), 2)
            }
            group_foods.append(food_name)
        
        food_groups[group] = group_foods
    
    return foods, food_groups


def _create_synthetic_farms(n_farms: int, total_area: float = 100.0, seed: int = 42) -> Dict[str, float]:
    """
    Create synthetic farm land availability.
    
    Args:
        n_farms: Number of farms/plots
        total_area: Total area to distribute (hectares)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping farm names to land availability
    """
    import numpy as np
    np.random.seed(seed)
    
    # Generate variable farm sizes (log-normal distribution for realism)
    raw_sizes = np.random.lognormal(mean=1.0, sigma=0.5, size=n_farms)
    # Normalize to total_area
    sizes = raw_sizes / raw_sizes.sum() * total_area
    
    return {f'Plot{i+1}': round(float(s), 2) for i, s in enumerate(sizes)}


def _create_synthetic_config(farms: List[str], foods: Dict, food_groups: Dict[str, List[str]], 
                             land_availability: Dict[str, float], min_foods_per_group: int = 1) -> Dict:
    """
    Create configuration for synthetic scenarios.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary mapping groups to foods
        land_availability: Dictionary mapping farms to areas
        min_foods_per_group: Minimum foods required per group
        
    Returns:
        Configuration dictionary
    """
    # Food group constraints - require diversity
    food_group_constraints = {
        g: {'min_foods': min(min_foods_per_group, len(lst)), 'max_foods': len(lst)}
        for g, lst in food_groups.items()
    }
    
    parameters = {
        'land_availability': land_availability,
        'minimum_planting_area': {food: 0.01 for food in foods.keys()},
        'max_percentage_per_crop': {food: 0.5 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': food_group_constraints,
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }
    
    return {
        'parameters': parameters,
        'benders_tolerance': 1e-3,
        'benders_max_iterations': 50,
        'pulp_time_limit': 60,
        'use_multi_cut': True,
        'use_trust_region': True,
        'use_anticycling': True,
        'use_norm_cuts': True,
        'quantum_settings': {
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }


def _load_micro_6_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Micro scenario with ~6 variables for minimal QPU testing.
    
    Configuration:
    - 2 plots × 2 foods = 4 Y variables + 2 U variables = 6 total
    - 1 food group with 2 foods (min 1 required)
    
    This is the smallest meaningful problem that maintains constraints.
    """
    n_plots = 2
    foods_per_group = {'Grains': 2}
    
    foods, food_groups = _create_synthetic_foods(foods_per_group, seed=6)
    land_availability = _create_synthetic_farms(n_plots, total_area=100.0, seed=6)
    farms = list(land_availability.keys())
    
    config = _create_synthetic_config(farms, foods, food_groups, land_availability, min_foods_per_group=1)
    
    n_foods = len(foods)
    n_vars = n_plots * n_foods + n_foods  # Y vars + U vars
    logger.info(f"Loaded micro_6: {n_plots} plots × {n_foods} foods = {n_vars} variables")
    
    return farms, foods, food_groups, config


def _load_micro_12_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Micro scenario with ~12 variables.
    
    Configuration:
    - 3 plots × 3 foods = 9 Y variables + 3 U variables = 12 total
    - 2 food groups: Grains (2), Legumes (1) with min 1 each
    """
    n_plots = 3
    foods_per_group = {'Grains': 2, 'Legumes': 1}
    
    foods, food_groups = _create_synthetic_foods(foods_per_group, seed=12)
    land_availability = _create_synthetic_farms(n_plots, total_area=100.0, seed=12)
    farms = list(land_availability.keys())
    
    config = _create_synthetic_config(farms, foods, food_groups, land_availability, min_foods_per_group=1)
    
    n_foods = len(foods)
    n_vars = n_plots * n_foods + n_foods
    logger.info(f"Loaded micro_12: {n_plots} plots × {n_foods} foods = {n_vars} variables")
    
    return farms, foods, food_groups, config


def _load_tiny_24_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Tiny scenario with ~24 variables.
    
    Configuration:
    - 4 plots × 5 foods = 20 Y variables + 5 U variables = 25 total (~24)
    - 3 food groups: Grains (2), Legumes (2), Vegetables (1)
    """
    n_plots = 4
    foods_per_group = {'Grains': 2, 'Legumes': 2, 'Vegetables': 1}
    
    foods, food_groups = _create_synthetic_foods(foods_per_group, seed=24)
    land_availability = _create_synthetic_farms(n_plots, total_area=100.0, seed=24)
    farms = list(land_availability.keys())
    
    config = _create_synthetic_config(farms, foods, food_groups, land_availability, min_foods_per_group=1)
    
    n_foods = len(foods)
    n_vars = n_plots * n_foods + n_foods
    logger.info(f"Loaded tiny_24: {n_plots} plots × {n_foods} foods = {n_vars} variables (actual: {n_vars})")
    
    return farms, foods, food_groups, config


def _load_tiny_40_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Tiny scenario with ~40 variables.
    
    Configuration:
    - 5 plots × 6 foods = 30 Y variables + 6 U variables = 36 total (~40)
    - 4 food groups: Grains (2), Legumes (1), Vegetables (2), Fruits (1)
    """
    n_plots = 5
    foods_per_group = {'Grains': 2, 'Legumes': 1, 'Vegetables': 2, 'Fruits': 1}
    
    foods, food_groups = _create_synthetic_foods(foods_per_group, seed=40)
    land_availability = _create_synthetic_farms(n_plots, total_area=100.0, seed=40)
    farms = list(land_availability.keys())
    
    config = _create_synthetic_config(farms, foods, food_groups, land_availability, min_foods_per_group=1)
    
    n_foods = len(foods)
    n_vars = n_plots * n_foods + n_foods
    logger.info(f"Loaded tiny_40: {n_plots} plots × {n_foods} foods = {n_vars} variables")
    
    return farms, foods, food_groups, config


def _load_small_60_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Small scenario with ~60 variables.
    
    Configuration:
    - 6 plots × 8 foods = 48 Y variables + 8 U variables = 56 total (~60)
    - 4 food groups: Grains (3), Legumes (2), Vegetables (2), Fruits (1)
    """
    n_plots = 6
    foods_per_group = {'Grains': 3, 'Legumes': 2, 'Vegetables': 2, 'Fruits': 1}
    
    foods, food_groups = _create_synthetic_foods(foods_per_group, seed=60)
    land_availability = _create_synthetic_farms(n_plots, total_area=100.0, seed=60)
    farms = list(land_availability.keys())
    
    config = _create_synthetic_config(farms, foods, food_groups, land_availability, min_foods_per_group=1)
    
    n_foods = len(foods)
    n_vars = n_plots * n_foods + n_foods
    logger.info(f"Loaded small_60: {n_plots} plots × {n_foods} foods = {n_vars} variables")
    
    return farms, foods, food_groups, config


def _load_small_80_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Small scenario with ~80 variables.
    
    Configuration:
    - 7 plots × 10 foods = 70 Y variables + 10 U variables = 80 total
    - 5 food groups: Grains (3), Legumes (2), Vegetables (2), Fruits (2), Proteins (1)
    """
    n_plots = 7
    foods_per_group = {'Grains': 3, 'Legumes': 2, 'Vegetables': 2, 'Fruits': 2, 'Proteins': 1}
    
    foods, food_groups = _create_synthetic_foods(foods_per_group, seed=80)
    land_availability = _create_synthetic_farms(n_plots, total_area=100.0, seed=80)
    farms = list(land_availability.keys())
    
    config = _create_synthetic_config(farms, foods, food_groups, land_availability, min_foods_per_group=1)
    
    n_foods = len(foods)
    n_vars = n_plots * n_foods + n_foods
    logger.info(f"Loaded small_80: {n_plots} plots × {n_foods} foods = {n_vars} variables")
    
    return farms, foods, food_groups, config


def _load_small_100_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Small scenario with ~100 variables.
    
    Configuration:
    - 8 plots × 11 foods = 88 Y variables + 11 U variables = 99 total (~100)
    - 5 food groups: Grains (3), Legumes (2), Vegetables (3), Fruits (2), Proteins (1)
    """
    n_plots = 8
    foods_per_group = {'Grains': 3, 'Legumes': 2, 'Vegetables': 3, 'Fruits': 2, 'Proteins': 1}
    
    foods, food_groups = _create_synthetic_foods(foods_per_group, seed=100)
    land_availability = _create_synthetic_farms(n_plots, total_area=100.0, seed=100)
    farms = list(land_availability.keys())
    
    config = _create_synthetic_config(farms, foods, food_groups, land_availability, min_foods_per_group=1)
    
    n_foods = len(foods)
    n_vars = n_plots * n_foods + n_foods
    logger.info(f"Loaded small_100: {n_plots} plots × {n_foods} foods = {n_vars} variables")
    
    return farms, foods, food_groups, config


def _load_medium_120_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Medium scenario with ~120 variables.
    
    Configuration:
    - 9 plots × 12 foods = 108 Y variables + 12 U variables = 120 total
    - 5 food groups: Grains (4), Legumes (2), Vegetables (3), Fruits (2), Proteins (1)
    """
    n_plots = 9
    foods_per_group = {'Grains': 4, 'Legumes': 2, 'Vegetables': 3, 'Fruits': 2, 'Proteins': 1}
    
    foods, food_groups = _create_synthetic_foods(foods_per_group, seed=120)
    land_availability = _create_synthetic_farms(n_plots, total_area=100.0, seed=120)
    farms = list(land_availability.keys())
    
    config = _create_synthetic_config(farms, foods, food_groups, land_availability, min_foods_per_group=1)
    
    n_foods = len(foods)
    n_vars = n_plots * n_foods + n_foods
    logger.info(f"Loaded medium_120: {n_plots} plots × {n_foods} foods = {n_vars} variables")
    
    return farms, foods, food_groups, config


def _load_medium_160_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Medium scenario with ~160 variables (near embedding failure threshold).
    
    Configuration:
    - 10 plots × 14 foods = 140 Y variables + 14 U variables = 154 total (~160)
    - 5 food groups: Grains (4), Legumes (3), Vegetables (3), Fruits (2), Proteins (2)
    """
    n_plots = 10
    foods_per_group = {'Grains': 4, 'Legumes': 3, 'Vegetables': 3, 'Fruits': 2, 'Proteins': 2}
    
    foods, food_groups = _create_synthetic_foods(foods_per_group, seed=160)
    land_availability = _create_synthetic_farms(n_plots, total_area=100.0, seed=160)
    farms = list(land_availability.keys())
    
    config = _create_synthetic_config(farms, foods, food_groups, land_availability, min_foods_per_group=1)
    
    n_foods = len(foods)
    n_vars = n_plots * n_foods + n_foods
    logger.info(f"Loaded medium_160: {n_plots} plots × {n_foods} foods = {n_vars} variables")
    
    return farms, foods, food_groups, config


# ============================================================================
# ROTATION SCENARIOS WITH QUANTUM-FRIENDLY CHARACTERISTICS
# ============================================================================
# These scenarios are designed for 3-period rotation optimization with:
# - Reduced crop families (6-8 instead of 27 crops) for bounded degree
# - Enhanced rotation matrix with negative synergies (frustration)
# - Spatial neighbor structure (k-nearest for local interactions)
# - Target: 5-15% integrality gap, embeddable on QPU with chains ~2-3
# ============================================================================

def _load_rotation_micro_25_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Micro rotation scenario for quantum advantage testing.
    
    Configuration (3-period rotation):
    - 5 farms × 6 crop families × 3 periods = 90 Y variables + 18 U variables = 108 total
    - Crop families: Fruits, Grains, Legumes, Leafy-Veg, Root-Veg, Proteins (6 total)
    - Spatial structure: Grid layout with k=4 nearest neighbors
    - Enhanced rotation matrix: 40% negative synergies for frustration
    - Target variables: ~100 (embeddable on QPU)
    
    Quantum-friendly features:
    - Reduced choices per farm (6 families vs 27 crops)
    - Bounded max degree: (6-1) + 4×6 = 29 per variable
    - Frustrated interactions from negative rotation synergies
    - Sparse spatial interactions (k=4 neighbors only)
    """
    import sys as _sys
    import os as _os
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms
    
    # Generate farms with spatial structure (100ha total)
    n_farms = 5
    L = generate_farms(n_farms=n_farms, total_area=100.0, seed=2501)
    farms = list(L.keys())
    
    # Define crop families (aggregated from full crop list)
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
    
    # Food groups mapping
    food_groups = {
        'Plant_Foods': ['Fruits', 'Grains', 'Legumes', 'Leafy_Vegetables', 'Root_Vegetables'],
        'Proteins': ['Proteins', 'Legumes']
    }
    
    # Config with rotation-specific parameters
    config = {
        'parameters': {
            'land_availability': L,
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
            'rotation_gamma': 0.20,  # Synergy weight
            'spatial_k_neighbors': 4,  # k-nearest neighbors
            'frustration_ratio': 0.70,  # 70% negative synergies (HIGH)
            'negative_synergy_strength': -0.8,  # Strong penalties
            'use_soft_one_hot': True,  # Soft constraint for LP gap
            'one_hot_penalty': 3.0,  # Penalty weight (lower = harder)
            'diversity_bonus': 0.15  # Competing objective
        }
    }
    
    logger.info(f"Loaded rotation_micro_25: {n_farms} farms × {len(crop_families)} families × 3 periods")
    logger.info(f"  Variables: ~{n_farms * len(crop_families) * 3 + len(crop_families) * 3} (Y + U)")
    logger.info(f"  Max degree: ~{(len(crop_families)-1) + 4 * len(crop_families)} (bounded for QPU)")
    
    return farms, crop_families, food_groups, config


def _load_rotation_small_50_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Small rotation scenario for scaling analysis.
    
    Configuration (3-period rotation):
    - 10 farms × 6 crop families × 3 periods = 180 Y variables + 18 U variables = 198 total
    - Same crop families as micro_25
    - Spatial grid with k=4 neighbors
    - 40% frustration ratio
    """
    import sys as _sys
    import os as _os
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms
    
    n_farms = 10
    L = generate_farms(n_farms=n_farms, total_area=100.0, seed=5001)
    farms = list(L.keys())
    
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
    
    config = {
        'parameters': {
            'land_availability': L,
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
            'rotation_gamma': 0.25,
            'spatial_k_neighbors': 4,
            'frustration_ratio': 0.75,  # 75% negative
            'negative_synergy_strength': -1.0,  # Stronger
            'use_soft_one_hot': True,
            'one_hot_penalty': 2.5,
            'diversity_bonus': 0.18
        }
    }
    
    logger.info(f"Loaded rotation_small_50: {n_farms} farms × {len(crop_families)} families × 3 periods")
    
    return farms, crop_families, food_groups, config


def _load_rotation_medium_100_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Medium rotation scenario with increased frustration.
    
    Configuration (3-period rotation):
    - 20 farms × 6 crop families × 3 periods = 360 Y variables + 18 U variables = 378 total
    - Increased frustration: 50% negative synergies
    - Stronger negative coupling: -0.5
    - Target: >5% integrality gap
    """
    import sys as _sys
    import os as _os
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms
    
    n_farms = 20
    L = generate_farms(n_farms=n_farms, total_area=100.0, seed=10001)
    farms = list(L.keys())
    
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
    
    config = {
        'parameters': {
            'land_availability': L,
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
            'rotation_gamma': 0.30,  # Strong rotation effect
            'spatial_k_neighbors': 4,
            'frustration_ratio': 0.82,  # 82% negative synergies
            'negative_synergy_strength': -1.2,  # Very strong penalties
            'use_soft_one_hot': True,
            'one_hot_penalty': 2.0,
            'diversity_bonus': 0.22
        }
    }
    
    logger.info(f"Loaded rotation_medium_100: {n_farms} farms × {len(crop_families)} families × 3 periods")
    logger.info(f"  Frustration: 82% of rotation edges have negative synergies")
    
    return farms, crop_families, food_groups, config


def _load_rotation_large_200_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Large rotation scenario for classical hardness testing.
    
    Configuration (3-period rotation):
    - 50 farms × 6 crop families × 3 periods = 900 Y variables + 18 U variables = 918 total
    - High frustration: 60% negative synergies
    - Strong coupling: -0.7
    - Expected: Gurobi struggles (>10% gap or timeout)
    """
    import sys as _sys
    import os as _os
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms
    
    n_farms = 50
    L = generate_farms(n_farms=n_farms, total_area=100.0, seed=20001)
    farms = list(L.keys())
    
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
    
    config = {
        'parameters': {
            'land_availability': L,
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
            'rotation_gamma': 0.35,  # Maximum rotation effect
            'spatial_k_neighbors': 4,
            'frustration_ratio': 0.88,  # 88% negative synergies (extreme)
            'negative_synergy_strength': -1.5,  # Maximum penalties
            'use_soft_one_hot': True,
            'one_hot_penalty': 1.5,  # Weakest penalty = hardest
            'diversity_bonus': 0.25  # Strong competing objective
        }
    }
    
    logger.info(f"Loaded rotation_large_200: {n_farms} farms × {len(crop_families)} families × 3 periods")
    logger.info(f"  Frustration: 88% negative synergies (extreme classical hardness)")
    
    return farms, crop_families, food_groups, config


# ============================================================================
# LARGE-SCALE ROTATION SCENARIOS (27 foods, many farms, 3 periods)
# For hierarchical quantum solver testing
# ============================================================================

def _load_rotation_250farms_27foods_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Large-scale rotation scenario: 250 farms × 27 foods × 3 periods = 20,250 variables
    
    Design for hierarchical quantum solver:
    - Full food diversity (27 foods from Excel data)
    - 3-period rotation with synergies
    - Spatial interactions between neighboring farms
    - Diversity bonuses and soft one-hot constraints
    - Benefit scaling WITHOUT area normalization (rotation terms change scale)
    """
    import sys as _sys
    import os as _os
    import pandas as _pd

    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms

    # Generate farms
    n_farms = 250
    L = generate_farms(n_farms=n_farms, seed=250)
    farms = list(L.keys())
    total_area = sum(L.values())
    
    print(f"Generated {len(farms)} farms for rotation_250farms_27foods")
    print(f"Total land: {total_area:.2f} ha")

    # Load 27 foods from Excel (all foods for maximum diversity)
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    foods: Dict[str, Dict[str, float]]
    food_groups: Dict[str, List[str]]
    
    if _os.path.exists(excel_path):
        try:
            df = _pd.read_excel(excel_path)
            col_map = {
                'Food_Name': 'Food_Name',
                'food_group': 'Food_Group',
                'nutritional_value': 'nutritional_value',
                'nutrient_density': 'nutrient_density',
                'environmental_impact': 'environmental_impact',
                'affordability': 'affordability',
                'sustainability': 'sustainability'
            }
            
            # Get ALL foods (Excel should have exactly 27 foods)
            grp_col = 'food_group'
            name_col = 'Food_Name'
            
            # Take all unique foods from Excel (should be 27)
            filt = df[list(col_map.keys())].copy()
            filt = filt.drop_duplicates(subset=[name_col])
            filt.rename(columns=col_map, inplace=True)

            objectives = ['nutritional_value', 'nutrient_density',
                          'environmental_impact', 'affordability', 'sustainability']
            for obj in objectives:
                filt[obj] = _pd.to_numeric(filt[obj], errors='coerce').fillna(0.5)
                filt[obj] = filt[obj].clip(0, 1)

            foods = {}
            for _, row in filt.iterrows():
                fname = row['Food_Name']
                foods[fname] = {obj: float(row[obj]) for obj in objectives}

            food_groups = {}
            for _, row in filt.iterrows():
                g = row['Food_Group'] or 'Unknown'
                food_groups.setdefault(g, []).append(row['Food_Name'])
                
        except Exception as e:
            print(f"Error loading Excel: {e}")
            raise
    else:
        raise FileNotFoundError(f"Excel file required: {excel_path}")

    # Configuration for rotation optimization
    parameters = {
        'land_availability': L,
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        },
        'rotation_gamma': 0.20,  # Moderate rotation synergies
        'spatial_k_neighbors': 4,  # 4-neighbor grid
        'frustration_ratio': 0.70,  # 70% antagonistic pairs
        'negative_synergy_strength': -0.8,
        'use_soft_one_hot': True,
        'one_hot_penalty': 3.0,
        'diversity_bonus': 0.15,
        'benefit_scale': 1.0,  # No area normalization (rotation changes objective scale)
    }

    config = {
        'parameters': parameters,
        'quantum_settings': {
            'use_hierarchical_decomposition': True,
            'farms_per_cluster': 10,  # Target cluster size for QPU
            'num_reads': 100,
            'num_iterations': 3,
        }
    }

    logger.info(f"Loaded rotation_250farms_27foods: {n_farms} farms × {len(foods)} foods × 3 periods")
    logger.info(f"  Total variables: ~{n_farms * len(foods) * 3} (20,250 for 250×27×3)")
    
    return farms, foods, food_groups, config


def _load_rotation_350farms_27foods_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """350 farms × 27 foods × 3 periods = 28,350 variables"""
    import sys as _sys
    import os as _os
    import pandas as _pd

    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms

    n_farms = 350
    L = generate_farms(n_farms=n_farms, seed=350)
    farms = list(L.keys())
    
    # Load foods (same as 250farms version)
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    if _os.path.exists(excel_path):
        df = _pd.read_excel(excel_path)
        col_map = {
            'Food_Name': 'Food_Name',
            'food_group': 'Food_Group',
            'nutritional_value': 'nutritional_value',
            'nutrient_density': 'nutrient_density',
            'environmental_impact': 'environmental_impact',
            'affordability': 'affordability',
            'sustainability': 'sustainability'
        }
        
        grp_col = 'food_group'
        name_col = 'Food_Name'
        
        # Take all unique foods from Excel (should be 27)
        filt = df[list(col_map.keys())].copy()
        filt = filt.drop_duplicates(subset=[name_col])
        filt.rename(columns=col_map, inplace=True)

        objectives = ['nutritional_value', 'nutrient_density',
                      'environmental_impact', 'affordability', 'sustainability']
        for obj in objectives:
            filt[obj] = _pd.to_numeric(filt[obj], errors='coerce').fillna(0.5)
            filt[obj] = filt[obj].clip(0, 1)

        foods = {}
        for _, row in filt.iterrows():
            fname = row['Food_Name']
            foods[fname] = {obj: float(row[obj]) for obj in objectives}

        food_groups = {}
        for _, row in filt.iterrows():
            g = row['Food_Group'] or 'Unknown'
            food_groups.setdefault(g, []).append(row['Food_Name'])
    else:
        raise FileNotFoundError(f"Excel file required: {excel_path}")

    parameters = {
        'land_availability': L,
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        },
        'rotation_gamma': 0.22,
        'spatial_k_neighbors': 4,
        'frustration_ratio': 0.72,
        'negative_synergy_strength': -0.85,
        'use_soft_one_hot': True,
        'one_hot_penalty': 3.5,
        'diversity_bonus': 0.18,
        'benefit_scale': 1.0,
    }

    config = {
        'parameters': parameters,
        'quantum_settings': {
            'use_hierarchical_decomposition': True,
            'farms_per_cluster': 12,
            'num_reads': 100,
            'num_iterations': 3,
        }
    }

    logger.info(f"Loaded rotation_350farms_27foods: {n_farms} farms × {len(foods)} foods × 3 periods")
    return farms, foods, food_groups, config


def _load_rotation_500farms_27foods_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """500 farms × 27 foods × 3 periods = 40,500 variables"""
    import sys as _sys
    import os as _os
    import pandas as _pd

    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms

    n_farms = 500
    L = generate_farms(n_farms=n_farms, seed=500)
    farms = list(L.keys())
    
    # Load foods
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    if _os.path.exists(excel_path):
        df = _pd.read_excel(excel_path)
        col_map = {
            'Food_Name': 'Food_Name',
            'food_group': 'Food_Group',
            'nutritional_value': 'nutritional_value',
            'nutrient_density': 'nutrient_density',
            'environmental_impact': 'environmental_impact',
            'affordability': 'affordability',
            'sustainability': 'sustainability'
        }
        
        grp_col = 'food_group'
        name_col = 'Food_Name'
        
        # Take all unique foods from Excel (should be 27)
        filt = df[list(col_map.keys())].copy()
        filt = filt.drop_duplicates(subset=[name_col])
        filt.rename(columns=col_map, inplace=True)

        objectives = ['nutritional_value', 'nutrient_density',
                      'environmental_impact', 'affordability', 'sustainability']
        for obj in objectives:
            filt[obj] = _pd.to_numeric(filt[obj], errors='coerce').fillna(0.5)
            filt[obj] = filt[obj].clip(0, 1)

        foods = {}
        for _, row in filt.iterrows():
            fname = row['Food_Name']
            foods[fname] = {obj: float(row[obj]) for obj in objectives}

        food_groups = {}
        for _, row in filt.iterrows():
            g = row['Food_Group'] or 'Unknown'
            food_groups.setdefault(g, []).append(row['Food_Name'])
    else:
        raise FileNotFoundError(f"Excel file required: {excel_path}")

    parameters = {
        'land_availability': L,
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        },
        'rotation_gamma': 0.25,
        'spatial_k_neighbors': 4,
        'frustration_ratio': 0.75,
        'negative_synergy_strength': -0.9,
        'use_soft_one_hot': True,
        'one_hot_penalty': 4.0,
        'diversity_bonus': 0.20,
        'benefit_scale': 1.0,
    }

    config = {
        'parameters': parameters,
        'quantum_settings': {
            'use_hierarchical_decomposition': True,
            'farms_per_cluster': 15,
            'num_reads': 100,
            'num_iterations': 3,
        }
    }

    logger.info(f"Loaded rotation_500farms_27foods: {n_farms} farms × {len(foods)} foods × 3 periods")
    return farms, foods, food_groups, config


def _load_rotation_1000farms_27foods_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """1000 farms × 27 foods × 3 periods = 81,000 variables - Ultimate stress test!"""
    import sys as _sys
    import os as _os
    import pandas as _pd

    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _sys.path.insert(0, _project_root)
    from Utils.farm_sampler import generate_farms

    n_farms = 1000
    L = generate_farms(n_farms=n_farms, seed=1000)
    farms = list(L.keys())
    
    # Load foods
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    project_root_excel = _os.path.dirname(script_dir)
    excel_path = _os.path.join(project_root_excel, "Inputs", "Combined_Food_Data.xlsx")

    if _os.path.exists(excel_path):
        df = _pd.read_excel(excel_path)
        col_map = {
            'Food_Name': 'Food_Name',
            'food_group': 'Food_Group',
            'nutritional_value': 'nutritional_value',
            'nutrient_density': 'nutrient_density',
            'environmental_impact': 'environmental_impact',
            'affordability': 'affordability',
            'sustainability': 'sustainability'
        }
        
        grp_col = 'food_group'
        name_col = 'Food_Name'
        
        # Take all unique foods from Excel (should be 27)
        filt = df[list(col_map.keys())].copy()
        filt = filt.drop_duplicates(subset=[name_col])
        filt.rename(columns=col_map, inplace=True)

        objectives = ['nutritional_value', 'nutrient_density',
                      'environmental_impact', 'affordability', 'sustainability']
        for obj in objectives:
            filt[obj] = _pd.to_numeric(filt[obj], errors='coerce').fillna(0.5)
            filt[obj] = filt[obj].clip(0, 1)

        foods = {}
        for _, row in filt.iterrows():
            fname = row['Food_Name']
            foods[fname] = {obj: float(row[obj]) for obj in objectives}

        food_groups = {}
        for _, row in filt.iterrows():
            g = row['Food_Group'] or 'Unknown'
            food_groups.setdefault(g, []).append(row['Food_Name'])
    else:
        raise FileNotFoundError(f"Excel file required: {excel_path}")

    parameters = {
        'land_availability': L,
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        },
        'rotation_gamma': 0.30,  # Strong rotation effects
        'spatial_k_neighbors': 4,
        'frustration_ratio': 0.80,  # High frustration
        'negative_synergy_strength': -1.0,
        'use_soft_one_hot': True,
        'one_hot_penalty': 5.0,
        'diversity_bonus': 0.25,
        'benefit_scale': 1.0,
    }

    config = {
        'parameters': parameters,
        'quantum_settings': {
            'use_hierarchical_decomposition': True,
            'farms_per_cluster': 20,  # Larger clusters for efficiency
            'num_reads': 100,
            'num_iterations': 3,
        }
    }

    logger.info(f"Loaded rotation_1000farms_27foods: {n_farms} farms × {len(foods)} foods × 3 periods")
    logger.info(f"  🚀 ULTIMATE STRESS TEST: ~81,000 variables!")
    return farms, foods, food_groups, config


# Test scenario output
if __name__ == "__main__":
    complexity = 'rotation_micro_25'  # Test rotation scenario
    farms, foods, food_groups, config = load_food_data(complexity)

    # Display scenario details
    num_farms = len(farms)
    num_foods = len(foods)
    problem_size = num_farms * num_foods * 3 + num_foods * 3  # 3-period rotation
    print(f"Rotation Scenario Details:")
    print(f"  Farms: {num_farms} ({farms[:5]}...)")
    print(f"  Crop Families: {num_foods} ({list(foods.keys())})")
    print(f"  Problem Size: ~{problem_size} variables (3-period rotation)")
    print(f"  Rotation gamma: {config['parameters']['rotation_gamma']}")
    print(f"  Frustration ratio: {config['parameters']['frustration_ratio']}")
    print(f"  Spatial k-neighbors: {config['parameters']['spatial_k_neighbors']}")
