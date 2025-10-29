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

# Distribution parameters (divided by 10 for patches)
classes = [
    {"label": "<0.1",   "min": 0.01, "max": 0.1,  "farm_share": 0.45, "land_share": 0.10},
    {"label": "0.1–0.2", "min": 0.1, "max": 0.2,  "farm_share": 0.20, "land_share": 0.10},
    {"label": "0.2–0.5", "min": 0.2, "max": 0.5,  "farm_share": 0.15, "land_share": 0.20},
    {"label": "0.5–1",   "min": 0.5, "max": 1.0, "farm_share": 0.08, "land_share": 0.15},
    {"label": "1–2",    "min": 1.0, "max": 2.0,  "farm_share": 0.05, "land_share": 0.20},
    {"label": ">2",     "min": 2.0, "max": 5.0,  "farm_share": 0.07, "land_share": 0.25},
]

def generate_farms(n_farms: int, seed: int = 42) -> Dict[str, float]:
    """
    Generate patch land availability (patches are 1/10 the size of farms).
    
    Args:
        n_farms: Number of patches to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping patch names to land availability (hectares)
    """
    np.random.seed(seed)
    
    # Assign patch counts per class
    for cls in classes:
        cls["n_farms"] = int(round(n_farms * cls["farm_share"]))
    
    # Ensure the total matches exactly
    diff = n_farms - sum(cls["n_farms"] for cls in classes)
    classes[0]["n_farms"] += diff
    
    # Sample patch sizes per class
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
    total_area = farms["area_ha"].sum()
    
    # Scale to match expected land shares
    expected_total_area = sum(cls["land_share"] for cls in classes)
    for cls in classes:
        cls["target_area"] = cls["land_share"] / expected_total_area
    
    current_shares = farms.groupby("size_class")["area_ha"].sum() / total_area
    
    scale_factors = {
        cls["label"]: cls["target_area"] / current_shares[cls["label"]]
        for cls in classes if cls["label"] in current_shares
    }
    
    farms["area_ha_scaled"] = farms.apply(
        lambda row: row["area_ha"] * scale_factors[row["size_class"]],
        axis=1
    )
    
    # Create patch format: {'Patch1': area, 'Patch2': area, ...}
    L = {}
    for i, area in enumerate(farms["area_ha_scaled"], 1):
        L[f'Patch{i}'] = round(area, 3)  # 3 decimals for smaller patch sizes
    
    return L


def generate_grid(n_farms: int, area: float,  seed: int = 42) -> Dict[str, float]:
    """
    Generate patch land availability (patches are 1/10 the size of farms).
    
    Args:
        n_farms: Number of patches to generate
        seed: Random seed for reproducibility
        area: Total land area to distribute among patches
    Returns:
        Dictionary mapping patch names to land availability (hectares)
    """
    np.random.seed(seed)

    # Generate even grid of patches
    L = {}
    area_per_patch = area / n_farms
    for i in range(1, n_farms + 1):
        L[f'Patch{i}'] = round(area_per_patch, 3)  # 3 decimals for smaller patch sizes
    
    return L










# ------------------------------------------------------------
# MAIN: Display example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    print("="*80)
    print("PATCH LAND AVAILABILITY GENERATOR")
    print("Patches are 1/10 the size of farms")
    print("="*80)
    
    for n in [2, 5, 20]:
        print(f"\n{'='*80}")
        print(f"GENERATING {n} PATCHES")
        print(f"{'='*80}")
        
        L = generate_farms(n, seed=42)
        patches_list = list(L.keys())
        
        print(f"\nPatch names: {patches_list}")
        print(f"\nLand availability (L):")
        for patch, area in L.items():
            print(f"  '{patch}': {area} ha")
        
        print(f"\nTotal land: {sum(L.values()):.3f} ha")
        print(f"Average: {sum(L.values())/len(L):.3f} ha per patch")
        print(f"Min: {min(L.values()):.3f} ha")
        print(f"Max: {max(L.values()):.3f} ha")
        
        # Show how to use in solver
        print(f"\nTo use in solver, replace:")
        print(f"  patches = {patches_list[:3]}...  # (showing first 3)")
        print(f"  L = {dict(list(L.items())[:3])}...  # (showing first 3)")

