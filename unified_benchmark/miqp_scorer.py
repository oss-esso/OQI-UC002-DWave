"""
MIQP Objective Scorer for the unified benchmark.

Implements the TRUE MIQP objective from formulations.tex (Formulation 1).
Used to recompute objectives on any solution, regardless of how it was obtained.

The objective is:
    maximize (Benefit + Temporal + Spatial - Penalty + Diversity)

where:
    - Benefit: base benefit weighted by area
    - Temporal: rotation synergies between consecutive years
    - Spatial: synergies between neighboring farms in same year
    - Penalty: soft one-hot violation penalty
    - Diversity: bonus for planting each crop at least once per farm
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .scenarios import build_rotation_matrix, build_spatial_neighbors
from .core import MIQP_PARAMS, ConstraintViolations


@dataclass
class MIQPObjectiveBreakdown:
    """Breakdown of MIQP objective components."""
    total: float
    benefit: float
    temporal_synergy: float
    spatial_synergy: float
    one_hot_penalty: float
    diversity_bonus: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "benefit": self.benefit,
            "temporal_synergy": self.temporal_synergy,
            "spatial_synergy": self.spatial_synergy,
            "one_hot_penalty": self.one_hot_penalty,
            "diversity_bonus": self.diversity_bonus,
        }


def compute_miqp_objective(
    solution: Dict[Tuple[int, int, int], int],  # (farm_idx, food_idx, period) -> 0/1
    scenario_data: Dict[str, Any],
    R: Optional[np.ndarray] = None,
    neighbor_edges: Optional[List[Tuple[int, int]]] = None,
    params: Optional[Dict[str, float]] = None,
    return_breakdown: bool = False
) -> float | Tuple[float, MIQPObjectiveBreakdown]:
    """
    Compute the TRUE MIQP objective for a solution.
    
    This implements Formulation 1 from formulations.tex exactly.
    
    Args:
        solution: Dict mapping (farm_idx, food_idx, period) -> binary value
                  Period is 1-indexed (1, 2, 3, ...)
        scenario_data: Scenario data from load_scenario()
        R: Rotation synergy matrix (n_foods × n_foods). If None, built from params.
        neighbor_edges: List of (farm_idx_1, farm_idx_2) pairs. If None, built.
        params: MIQP parameters. If None, uses MIQP_PARAMS.
        return_breakdown: If True, return (total, breakdown) tuple
    
    Returns:
        MIQP objective value (float) or (value, breakdown) tuple
    """
    if params is None:
        params = MIQP_PARAMS
    
    # Extract data
    farm_names = scenario_data["farm_names"]
    food_names = scenario_data["food_names"]
    land_availability = scenario_data["land_availability"]
    food_benefits = scenario_data["food_benefits"]
    total_area = scenario_data["total_area"]
    n_periods = scenario_data.get("n_periods", params.get("n_periods", 3))
    
    n_farms = len(farm_names)
    n_foods = len(food_names)
    
    # Get parameters
    rotation_gamma = params.get("rotation_gamma", 0.2)
    spatial_gamma = params.get("spatial_gamma", 0.1)
    one_hot_penalty = params.get("one_hot_penalty", 3.0)
    diversity_bonus = params.get("diversity_bonus", 0.15)
    
    # Build rotation matrix if not provided
    if R is None:
        R = build_rotation_matrix(
            n_foods,
            frustration_ratio=params.get("frustration_ratio", 0.7),
            negative_strength=params.get("negative_strength", -0.8),
            seed=42
        )
    
    # Build neighbor edges if not provided
    if neighbor_edges is None:
        neighbor_edges = build_spatial_neighbors(
            farm_names,
            k_neighbors=params.get("k_neighbors", 4)
        )
    
    # Helper to get Y value
    def Y(f_idx: int, c_idx: int, t: int) -> int:
        return solution.get((f_idx, c_idx, t), 0)
    
    # ========== OBJECTIVE COMPONENTS ==========
    
    # Part 1: Base Benefit (Linear)
    # Benefit = Σ_f Σ_c Σ_t (B_c * A_f / A_total) * Y_{f,c,t}
    benefit = 0.0
    for f_idx, farm in enumerate(farm_names):
        farm_area = land_availability[farm]
        for c_idx, food in enumerate(food_names):
            B_c = food_benefits.get(food, 1.0)
            for t in range(1, n_periods + 1):
                benefit += (B_c * farm_area / total_area) * Y(f_idx, c_idx, t)
    
    # Part 2: Temporal Synergy (Quadratic)
    # Temporal = γ_rot * Σ_f Σ_{t=2}^T Σ_{c1} Σ_{c2} (A_f / A_total) * R[c1,c2] * Y_{f,c1,t-1} * Y_{f,c2,t}
    temporal_synergy = 0.0
    for f_idx, farm in enumerate(farm_names):
        farm_area = land_availability[farm]
        area_frac = farm_area / total_area
        for t in range(2, n_periods + 1):
            for c1_idx in range(n_foods):
                y_prev = Y(f_idx, c1_idx, t - 1)
                if y_prev == 0:
                    continue  # Skip if not planted previous year
                for c2_idx in range(n_foods):
                    y_curr = Y(f_idx, c2_idx, t)
                    if y_curr == 0:
                        continue  # Skip if not planted current year
                    synergy = R[c1_idx, c2_idx]
                    temporal_synergy += rotation_gamma * area_frac * synergy * y_prev * y_curr
    
    # Part 3: Spatial Synergy (Quadratic)
    # Spatial = γ_spat * Σ_{(f1,f2)∈N} Σ_t Σ_{c1} Σ_{c2} (R[c1,c2] / A_total) * Y_{f1,c1,t} * Y_{f2,c2,t}
    spatial_synergy_val = 0.0
    for (f1_idx, f2_idx) in neighbor_edges:
        for t in range(1, n_periods + 1):
            for c1_idx in range(n_foods):
                y1 = Y(f1_idx, c1_idx, t)
                if y1 == 0:
                    continue
                for c2_idx in range(n_foods):
                    y2 = Y(f2_idx, c2_idx, t)
                    if y2 == 0:
                        continue
                    synergy = R[c1_idx, c2_idx]
                    spatial_synergy_val += spatial_gamma * (synergy / total_area) * y1 * y2
    
    # Part 4: Soft One-Hot Penalty (Quadratic)
    # Penalty = λ_oh * Σ_f Σ_t (Σ_c Y_{f,c,t} - 1)^2
    penalty = 0.0
    for f_idx in range(n_farms):
        for t in range(1, n_periods + 1):
            crop_count = sum(Y(f_idx, c_idx, t) for c_idx in range(n_foods))
            penalty += one_hot_penalty * (crop_count - 1) ** 2
    
    # Part 5: Diversity Bonus (Linear approximation)
    # Diversity = λ_div * Σ_f Σ_c I(Σ_t Y_{f,c,t} > 0)
    # I() is indicator function - we approximate as Σ_t Y_{f,c,t} / T
    diversity = 0.0
    for f_idx in range(n_farms):
        for c_idx in range(n_foods):
            # Count how many periods this crop is used on this farm
            uses = sum(Y(f_idx, c_idx, t) for t in range(1, n_periods + 1))
            if uses > 0:
                diversity += diversity_bonus
    
    # ========== TOTAL OBJECTIVE ==========
    total = benefit + temporal_synergy + spatial_synergy_val - penalty + diversity
    
    if return_breakdown:
        breakdown = MIQPObjectiveBreakdown(
            total=total,
            benefit=benefit,
            temporal_synergy=temporal_synergy,
            spatial_synergy=spatial_synergy_val,
            one_hot_penalty=penalty,
            diversity_bonus=diversity,
        )
        return total, breakdown
    
    return total


def check_constraints(
    solution: Dict[Tuple[int, int, int], int],
    scenario_data: Dict[str, Any],
    params: Optional[Dict[str, float]] = None
) -> ConstraintViolations:
    """
    Check constraint violations in a solution.
    
    Constraints (from formulations.tex):
    1. Max 2 crops per farm per period: Σ_c Y_{f,c,t} <= 2
    2. Min 1 crop per farm per period: Σ_c Y_{f,c,t} >= 1
    3. Rotation: No same crop in consecutive periods: Y_{f,c,t} + Y_{f,c,t+1} <= 1
    
    Args:
        solution: Dict mapping (farm_idx, food_idx, period) -> binary value
        scenario_data: Scenario data
        params: MIQP parameters
    
    Returns:
        ConstraintViolations with counts and details
    """
    if params is None:
        params = MIQP_PARAMS
    
    n_farms = scenario_data["n_farms"]
    n_foods = scenario_data["n_foods"]
    n_periods = scenario_data.get("n_periods", params.get("n_periods", 3))
    farm_names = scenario_data["farm_names"]
    food_names = scenario_data["food_names"]
    
    violations = ConstraintViolations()
    
    # Helper
    def Y(f_idx: int, c_idx: int, t: int) -> int:
        return solution.get((f_idx, c_idx, t), 0)
    
    # Check one-hot constraints
    for f_idx in range(n_farms):
        for t in range(1, n_periods + 1):
            crop_count = sum(Y(f_idx, c_idx, t) for c_idx in range(n_foods))
            
            if crop_count < 1:
                violations.one_hot_violations += 1
                violations.details.append({
                    "type": "min_crops",
                    "farm": farm_names[f_idx],
                    "period": t,
                    "count": crop_count,
                    "expected": ">=1"
                })
            elif crop_count > 2:
                violations.one_hot_violations += 1
                violations.details.append({
                    "type": "max_crops",
                    "farm": farm_names[f_idx],
                    "period": t,
                    "count": crop_count,
                    "expected": "<=2"
                })
    
    # Check rotation constraints
    for f_idx in range(n_farms):
        for c_idx in range(n_foods):
            for t in range(1, n_periods):
                y_t = Y(f_idx, c_idx, t)
                y_t1 = Y(f_idx, c_idx, t + 1)
                
                if y_t == 1 and y_t1 == 1:
                    violations.rotation_violations += 1
                    violations.details.append({
                        "type": "rotation",
                        "farm": farm_names[f_idx],
                        "food": food_names[c_idx],
                        "periods": f"t{t} and t{t+1}",
                        "message": "Same crop in consecutive periods"
                    })
    
    violations.total_violations = violations.one_hot_violations + violations.rotation_violations
    
    return violations


def convert_named_solution_to_indexed(
    named_solution: Dict[str, int],  # "Farm_Food_tN" -> value
    scenario_data: Dict[str, Any]
) -> Dict[Tuple[int, int, int], int]:
    """
    Convert a named solution to indexed format.
    
    Named format: "Farm1_Wheat_t1" -> 1
    Indexed format: (0, 5, 1) -> 1
    
    Args:
        named_solution: Dict with string keys
        scenario_data: Scenario data
    
    Returns:
        Dict with (farm_idx, food_idx, period) keys
    """
    farm_to_idx = {f: i for i, f in enumerate(scenario_data["farm_names"])}
    food_to_idx = {f: i for i, f in enumerate(scenario_data["food_names"])}
    
    indexed = {}
    for key, value in named_solution.items():
        if value == 0:
            continue  # Skip zeros for efficiency
        
        # Parse key: "Farm_Food_tN" or "Y_Farm_Food_tN"
        parts = key.split("_")
        
        # Handle various naming conventions
        if parts[0] == "Y":
            # Format: Y_Farm_Food_tN
            farm = parts[1]
            food = "_".join(parts[2:-1])  # Food name might contain underscores
            period_str = parts[-1]
        else:
            # Format: Farm_Food_tN
            # Find the period suffix
            period_str = parts[-1]
            remaining = "_".join(parts[:-1])
            
            # Try to match farm and food
            # This is tricky because both can have underscores
            farm = None
            food = None
            
            for test_farm in scenario_data["farm_names"]:
                if remaining.startswith(test_farm + "_"):
                    farm = test_farm
                    food = remaining[len(test_farm) + 1:]
                    break
            
            if farm is None:
                # Try simpler parsing
                farm = parts[0]
                food = "_".join(parts[1:-1])
        
        # Parse period
        if period_str.startswith("t"):
            period = int(period_str[1:])
        else:
            period = int(period_str)
        
        # Get indices
        f_idx = farm_to_idx.get(farm)
        c_idx = food_to_idx.get(food)
        
        if f_idx is not None and c_idx is not None:
            indexed[(f_idx, c_idx, period)] = int(value)
    
    return indexed


def convert_family_solution_to_27food(
    family_solution: Dict[Tuple[int, int, int], int],  # (farm_idx, family_idx, period) -> value
    scenario_data: Dict[str, Any],
    food_to_family: Dict[str, str],
    seed: int = 42
) -> Dict[Tuple[int, int, int], int]:
    """
    Refine a 6-family solution to 27-food solution.
    
    For each (farm, family, period) = 1, select one crop from that family.
    ENSURES: No same crop appears in consecutive periods (rotation constraint).
    
    Args:
        family_solution: Family-level solution (6 families)
        scenario_data: Scenario data (must have 27 foods)
        food_to_family: Mapping from food name to family name
        seed: Random seed
    
    Returns:
        27-food solution dict
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    
    # Build family to foods mapping
    family_to_foods = {}
    food_names = scenario_data["food_names"]
    
    for food in food_names:
        family = food_to_family.get(food, "Other")
        if family not in family_to_foods:
            family_to_foods[family] = []
        family_to_foods[family].append(food)
    
    # Get family order (assuming families are indexed 0-5)
    try:
        from food_grouping import FAMILY_ORDER
    except ImportError:
        FAMILY_ORDER = ["Legumes", "Grains", "Vegetables", "Roots", "Fruits", "Other"]
    
    food_to_idx = {f: i for i, f in enumerate(food_names)}
    
    # Group by farm and sort by period to handle rotation constraint
    farm_periods = {}
    for (f_idx, fam_idx, period), value in family_solution.items():
        if value == 0:
            continue
        if f_idx not in farm_periods:
            farm_periods[f_idx] = {}
        if period not in farm_periods[f_idx]:
            farm_periods[f_idx][period] = []
        farm_periods[f_idx][period].append(fam_idx)
    
    # Refine solution with rotation constraint enforcement
    refined = {}
    
    for f_idx, periods_data in farm_periods.items():
        # Track what foods were assigned in previous period for this farm
        prev_foods = set()
        
        for period in sorted(periods_data.keys()):
            current_foods = set()
            
            for fam_idx in periods_data[period]:
                # Get family name
                if fam_idx < len(FAMILY_ORDER):
                    family = FAMILY_ORDER[fam_idx]
                else:
                    family = "Other"
                
                # Get foods in this family
                foods = family_to_foods.get(family, [])
                if not foods:
                    # Fallback: use any food
                    foods = food_names
                
                # Filter out foods used in previous period (rotation constraint)
                available_foods = [f for f in foods if food_to_idx[f] not in prev_foods]
                
                if not available_foods:
                    # If all foods in family were used in prev period, use any from family
                    # This can happen when family has only 1 food
                    available_foods = foods
                
                # Select one food
                selected_food = rng.choice(available_foods)
                c_idx = food_to_idx[selected_food]
                
                refined[(f_idx, c_idx, period)] = 1
                current_foods.add(c_idx)
            
            # Update previous foods for next period
            prev_foods = current_foods
    
    return refined


if __name__ == "__main__":
    # Test MIQP scorer
    import sys
    sys.path.insert(0, str(__file__).replace("miqp_scorer.py", ""))
    
    from scenarios import load_scenario
    
    print("Testing MIQP scorer...")
    
    # Load a small scenario
    data = load_scenario("rotation_micro_25")
    print(f"Scenario: {data['n_farms']} farms × {data['n_foods']} foods")
    
    # Create a simple test solution (one crop per farm per period)
    solution = {}
    for f_idx in range(data["n_farms"]):
        for t in range(1, data["n_periods"] + 1):
            # Select crop based on farm and period to ensure rotation
            c_idx = (f_idx + t) % data["n_foods"]
            solution[(f_idx, c_idx, t)] = 1
    
    # Compute objective
    obj, breakdown = compute_miqp_objective(solution, data, return_breakdown=True)
    print(f"\nObjective breakdown:")
    print(f"  Total: {breakdown.total:.4f}")
    print(f"  Benefit: {breakdown.benefit:.4f}")
    print(f"  Temporal: {breakdown.temporal_synergy:.4f}")
    print(f"  Spatial: {breakdown.spatial_synergy:.4f}")
    print(f"  Penalty: {breakdown.one_hot_penalty:.4f}")
    print(f"  Diversity: {breakdown.diversity_bonus:.4f}")
    
    # Check constraints
    violations = check_constraints(solution, data)
    print(f"\nConstraint violations: {violations.total_violations}")
    print(f"  One-hot: {violations.one_hot_violations}")
    print(f"  Rotation: {violations.rotation_violations}")
