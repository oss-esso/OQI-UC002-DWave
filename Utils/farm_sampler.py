import numpy as np
import pandas as pd
from typing import Dict, List

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
# Number of farms to simulate
N_farms = 5
random_seed = 42
np.random.seed(random_seed)

# Distribution parameters
classes = [
    {"label": "<1",     "min": 0.1, "max": 1.0,  "farm_share": 0.45, "land_share": 0.10},
    {"label": "1–2",    "min": 1.0, "max": 2.0,  "farm_share": 0.20, "land_share": 0.10},
    {"label": "2–5",    "min": 2.0, "max": 5.0,  "farm_share": 0.15, "land_share": 0.20},
    {"label": "5–10",   "min": 5.0, "max": 10.0, "farm_share": 0.08, "land_share": 0.15},
    {"label": "10–20",  "min": 10., "max": 20.,  "farm_share": 0.05, "land_share": 0.20},
    {"label": ">20",    "min": 20., "max": 50.,  "farm_share": 0.07, "land_share": 0.25},
]

def generate_farms(n_farms: int, seed: int = 42, total_area: float = None) -> Dict[str, float]:
    """
    Generate farm land availability compatible with pulp_2.py format.
    
    Args:
        n_farms: Number of farms to generate
        seed: Random seed for reproducibility
        total_area: Optional total area in hectares to normalize to (default: use natural distribution)
        
    Returns:
        Dictionary mapping farm names to land availability (hectares)
    """
    np.random.seed(seed)
    
    # Assign farm counts per class
    for cls in classes:
        cls["n_farms"] = int(round(n_farms * cls["farm_share"]))
    
    # Ensure the total matches exactly
    diff = n_farms - sum(cls["n_farms"] for cls in classes)
    classes[0]["n_farms"] += diff
    
    # Sample farm sizes per class
    farm_records = []
    for cls in classes:
        n = cls["n_farms"]
        if n > 0:
            sizes = np.random.uniform(cls["min"], cls["max"], size=n)
            for s in sizes:
                farm_records.append({
                    "size_class": cls["label"],
                    "area_ha": s
                })
    
    farms = pd.DataFrame(farm_records)
    current_total = farms["area_ha"].sum()
    
    # Scale to match expected land shares
    expected_total_area = sum(cls["land_share"] for cls in classes)
    for cls in classes:
        cls["target_area"] = cls["land_share"] / expected_total_area
    
    current_shares = farms.groupby("size_class")["area_ha"].sum() / current_total
    
    scale_factors = {
        cls["label"]: cls["target_area"] / current_shares[cls["label"]]
        for cls in classes if cls["label"] in current_shares
    }
    
    farms["area_ha_scaled"] = farms.apply(
        lambda row: row["area_ha"] * scale_factors[row["size_class"]],
        axis=1
    )
    
    # If total_area specified, normalize to that total
    if total_area is not None:
        scaled_total = farms["area_ha_scaled"].sum()
        normalization_factor = total_area / scaled_total
        farms["area_ha_scaled"] = farms["area_ha_scaled"] * normalization_factor
    
    # Create pulp_2.py compatible format: {'Farm1': area, 'Farm2': area, ...}
    L = {}
    for i, area in enumerate(farms["area_ha_scaled"], 1):
        L[f'Farm{i}'] = round(area, 2)
    
    return L


def generate_farms_constant_area(n_farms: int, area_per_farm: float = 5.0, seed: int = 42) -> Dict[str, float]:
    """
    Generate farms with CONSTANT area per farm for consistent hardness scaling.
    
    This ensures that total land area scales linearly with number of farms,
    eliminating area-based instance difficulty variations.
    
    Args:
        n_farms: Number of farms to generate
        area_per_farm: Area per farm in hectares (default: 5.0 ha)
        seed: Random seed for reproducibility (not used for area, only for ordering)
        
    Returns:
        Dictionary mapping farm names to constant land availability (hectares)
    """
    np.random.seed(seed)
    
    # Create farms with constant area
    L = {}
    for i in range(1, n_farms + 1):
        L[f'Farm{i}'] = area_per_farm
    
    return L

# ------------------------------------------------------------
# MAIN: Display example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    print("="*80)
    print("FARM LAND AVAILABILITY GENERATOR")
    print("Compatible with pulp_2.py format")
    print("="*80)
    
    for n in [2, 5, 20]:
        print(f"\n{'='*80}")
        print(f"GENERATING {n} FARMS")
        print(f"{'='*80}")
        
        L = generate_farms(n, seed=42)
        farms_list = list(L.keys())
        
        print(f"\nFarm names: {farms_list}")
        print(f"\nLand availability (L):")
        for farm, area in L.items():
            print(f"  '{farm}': {area} ha")
        
        print(f"\nTotal land: {sum(L.values()):.2f} ha")
        print(f"Average: {sum(L.values())/len(L):.2f} ha per farm")
        print(f"Min: {min(L.values()):.2f} ha")
        print(f"Max: {max(L.values()):.2f} ha")
        
        # Show how to use in pulp_2.py
        print(f"\nTo use in pulp_2.py, replace:")
        print(f"  farms = {farms_list[:3]}...  # (showing first 3)")
        print(f"  L = {dict(list(L.items())[:3])}...  # (showing first 3)")

