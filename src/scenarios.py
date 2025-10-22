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
    else:
        raise ValueError(
            f"Invalid complexity level: {complexity_level}. Must be one of: simple, intermediate, custom, 30farms, 60farms, 90farms, 250farms, 350farms, 500farms_full, 1000farms_full, 2000farms_full, full, full_family")


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

    from farm_sampler import generate_farms

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

    from farm_sampler import generate_farms

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
    from farm_sampler import generate_farms

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
    from farm_sampler import generate_farms

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
    from farm_sampler import generate_farms

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
    from farm_sampler import generate_farms

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
    from farm_sampler import generate_farms

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
    from farm_sampler import generate_farms

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
    """Load full scenario data with 25 farms from farm_sampler and adjusted minimum areas."""
    import sys
    import os
    # Add project root to path to import farm_sampler
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    from farm_sampler import generate_farms

    # Generate 15 farms using the sampler
    L = generate_farms(n_farms=125, seed=42)
    farms = list(L.keys())

    print(f"Generated {len(farms)} farms with farm_sampler")
    print(f"Total land: {sum(L.values()):.2f} ha")

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

            # Sample 2 per group
            grp_col = 'food_group'
            name_col = 'Food_Name'
            sampled = df.groupby(grp_col).apply(
                lambda x: x.sample(n=min(len(x), 2))
            ).reset_index(drop=True)
            foods_list = sampled[col_map['Food_Name']].tolist()

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
        'land_availability': L,  # From farm_sampler
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


# Test scenario output
if __name__ == "__main__":
    complexity = 'intermediate'
    farms, foods, food_groups, config = load_food_data(complexity)

    # Display scenario details
    num_farms = len(farms)
    num_foods = len(foods)
    problem_size = "N/A"  # Placeholder for problem size
    print(f"Scenario Details:")
    print(f"  Farms: {num_farms} ({farms})")
    print(f"  Foods: {num_foods} ({list(foods.keys())})")
    print(f"  Problem Size: {problem_size} variables")
