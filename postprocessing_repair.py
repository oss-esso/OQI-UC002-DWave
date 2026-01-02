#!/usr/bin/env python3
"""
Classical Post-Processing Repair for QPU Solutions

This module repairs constraint violations in QPU solutions from hierarchical decomposition.
The main violations are one-hot failures (farm-periods with no crop assigned).

Repair Strategy:
1. Load all cluster samplesets for a scenario
2. Reconstruct the global solution from cluster samples
3. Identify one-hot violations (farm-periods with no crop)
4. For each violation, greedily assign the best crop based on:
   - Crop benefit value
   - Rotation synergy with adjacent periods
   - Food group diversity
5. Recalculate objective with repaired solution

Author: Claudette
Date: 2026-01-02
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load food data and rotation matrix
def load_scenario_data():
    """Load food data and rotation synergy matrix."""
    # Food data
    food_data_path = project_root / 'Inputs' / 'food_data.csv'
    if food_data_path.exists():
        foods_df = pd.read_csv(food_data_path)
        foods = {}
        for _, row in foods_df.iterrows():
            foods[row['Food']] = {
                'nutritional_value': row.get('nutritional_value', 1.0),
                'environmental_impact': row.get('environmental_impact', 0.0),
                'family': row.get('Food Group', 'Unknown'),
            }
    else:
        # Default 6-family configuration
        families = ['Fruits', 'Grains', 'Leafy_Vegetables', 'Legumes', 'Root_Vegetables', 'Proteins']
        foods = {f: {'nutritional_value': 1.0, 'environmental_impact': 0.0, 'family': f} for f in families}
    
    # Rotation matrix
    rotation_path = project_root / 'Inputs' / 'rotation_crop_matrix.csv'
    if rotation_path.exists():
        rotation_df = pd.read_csv(rotation_path, index_col=0)
        rotation_matrix = rotation_df.to_dict('index')
    else:
        rotation_matrix = {}
    
    return foods, rotation_matrix


def load_cluster_samplesets(scenario_name: str, sampleset_dir: str = 'qpu_samplesets_all') -> Dict:
    """Load all cluster samplesets for a scenario."""
    sampleset_path = Path(sampleset_dir)
    
    # Find all files matching this scenario
    pattern = f"sampleset_hier_{scenario_name}_"
    files = [f for f in sampleset_path.glob('*.pkl') if pattern in f.name]
    
    if not files:
        print(f"No samplesets found for scenario: {scenario_name}")
        return {}
    
    clusters = {}
    for f in sorted(files):
        with open(f, 'rb') as fp:
            data = pickle.load(fp)
        
        iter_idx = data['iteration']
        cluster_idx = data['cluster_idx']
        key = (iter_idx, cluster_idx)
        
        # Get best sample
        sampleset = data['sampleset']
        best_sample = sampleset.first.sample
        energy = sampleset.first.energy
        
        clusters[key] = {
            'sample': best_sample,
            'energy': energy,
            'farms': data['cluster_farms'],
            'var_map': data['var_map'],
            'scenario': data['scenario_name'],
        }
    
    return clusters


def reconstruct_global_solution(clusters: Dict, n_periods: int = 3) -> Dict:
    """
    Reconstruct global solution from cluster solutions.
    
    Returns:
        Dict mapping (farm, food, period) -> 0/1
    """
    global_solution = {}
    
    for (iter_idx, cluster_idx), cluster_data in clusters.items():
        sample = cluster_data['sample']
        farms = cluster_data['farms']
        var_map = cluster_data['var_map']
        
        # Decode sample using var_map
        # var_map: (farm_local_idx, food_idx, period) -> var_name
        for (farm_local, food_idx, period), var_name in var_map.items():
            value = sample.get(var_name, 0)
            
            if farm_local < len(farms):
                farm = farms[farm_local]
                key = (farm, food_idx, period)
                
                # Take last iteration's value
                if iter_idx == max(k[0] for k in clusters.keys()):
                    global_solution[key] = value
    
    return global_solution


def identify_violations(solution: Dict, farms: List[str], n_foods: int = 6, n_periods: int = 3) -> List[Tuple]:
    """
    Identify one-hot constraint violations.
    
    A violation occurs when a farm-period has no crop assigned (sum = 0).
    
    Returns:
        List of (farm, period) tuples with violations
    """
    violations = []
    
    for farm in farms:
        for period in range(1, n_periods + 1):
            # Count crops assigned to this farm-period
            crop_count = sum(
                solution.get((farm, food_idx, period), 0)
                for food_idx in range(n_foods)
            )
            
            if crop_count == 0:
                violations.append((farm, period))
            elif crop_count > 1:
                # Multi-assignment (shouldn't happen with proper one-hot, but check)
                print(f"  Warning: {farm} period {period} has {crop_count} crops assigned")
    
    return violations


def repair_violation(solution: Dict, farm: str, period: int, 
                     n_foods: int, foods: Dict, rotation_matrix: Dict,
                     weights: Dict = None) -> int:
    """
    Repair a single one-hot violation by assigning the best crop.
    
    Strategy:
    1. Get crops assigned in adjacent periods (for rotation synergy)
    2. Score each possible crop based on:
       - Base benefit value
       - Rotation synergy with previous/next period
       - Penalty for same crop as adjacent (no rotation benefit)
    3. Select highest-scoring crop
    
    Returns:
        Best food_idx to assign
    """
    if weights is None:
        weights = {'nutritional_value': 1.0, 'environmental_impact': 0.3}
    
    food_names = list(foods.keys())
    n_periods = 3
    
    # Get crops in adjacent periods
    prev_crop = None
    next_crop = None
    
    for food_idx, food_name in enumerate(food_names):
        if period > 1 and solution.get((farm, food_idx, period - 1), 0) == 1:
            prev_crop = food_name
        if period < n_periods and solution.get((farm, food_idx, period + 1), 0) == 1:
            next_crop = food_name
    
    # Score each crop
    scores = {}
    for food_idx, food_name in enumerate(food_names):
        # Base benefit
        food_data = foods.get(food_name, {})
        score = (
            weights.get('nutritional_value', 1.0) * food_data.get('nutritional_value', 1.0) -
            weights.get('environmental_impact', 0.3) * food_data.get('environmental_impact', 0.0)
        )
        
        # Rotation synergy bonus
        if prev_crop and food_name in rotation_matrix and prev_crop in rotation_matrix.get(food_name, {}):
            synergy = rotation_matrix[food_name].get(prev_crop, 0)
            score += 0.5 * synergy  # Weight for rotation
        
        if next_crop and food_name in rotation_matrix and next_crop in rotation_matrix.get(food_name, {}):
            synergy = rotation_matrix[food_name].get(next_crop, 0)
            score += 0.5 * synergy
        
        # Penalty for same crop as adjacent (reduces diversity)
        if food_name == prev_crop or food_name == next_crop:
            score -= 0.3  # Small penalty
        
        scores[food_idx] = score
    
    # Return best scoring crop
    best_food_idx = max(scores.keys(), key=lambda k: scores[k])
    return best_food_idx


def calculate_objective(solution: Dict, farms: List[str], foods: Dict, 
                       rotation_matrix: Dict, land_availability: Dict,
                       n_periods: int = 3, weights: Dict = None) -> float:
    """
    Calculate the MIQP objective value for a solution.
    
    Objective = sum over all (farm, food, period):
        - Linear benefit: B_c * L_f * Y_{f,c,t}
        - Quadratic rotation synergy: R_{c1,c2} * L_f * Y_{f,c1,t} * Y_{f,c2,t+1}
    """
    if weights is None:
        weights = {'nutritional_value': 1.0, 'environmental_impact': 0.3}
    
    food_names = list(foods.keys())
    n_foods = len(food_names)
    
    total_area = sum(land_availability.values())
    objective = 0.0
    
    # Linear terms
    for farm in farms:
        L_f = land_availability.get(farm, 1.0)
        for food_idx, food_name in enumerate(food_names):
            food_data = foods.get(food_name, {})
            B_c = (
                weights.get('nutritional_value', 1.0) * food_data.get('nutritional_value', 1.0) -
                weights.get('environmental_impact', 0.3) * food_data.get('environmental_impact', 0.0)
            )
            
            for period in range(1, n_periods + 1):
                Y = solution.get((farm, food_idx, period), 0)
                objective += B_c * L_f * Y / total_area
    
    # Quadratic rotation synergy terms
    for farm in farms:
        L_f = land_availability.get(farm, 1.0)
        for period in range(1, n_periods):  # t to t+1
            for c1_idx, c1_name in enumerate(food_names):
                Y_c1_t = solution.get((farm, c1_idx, period), 0)
                if Y_c1_t == 0:
                    continue
                    
                for c2_idx, c2_name in enumerate(food_names):
                    Y_c2_t1 = solution.get((farm, c2_idx, period + 1), 0)
                    if Y_c2_t1 == 0:
                        continue
                    
                    # Get rotation synergy R_{c1, c2}
                    R = 0.0
                    if c1_name in rotation_matrix and c2_name in rotation_matrix.get(c1_name, {}):
                        R = rotation_matrix[c1_name].get(c2_name, 0)
                    
                    objective += R * L_f * Y_c1_t * Y_c2_t1 / total_area
    
    return objective


def repair_scenario(scenario_name: str, sampleset_dir: str = 'qpu_samplesets_all') -> Dict:
    """
    Repair all violations for a scenario and recalculate objective.
    
    Returns:
        Dict with original and repaired statistics
    """
    print(f"\n{'='*80}")
    print(f"Repairing scenario: {scenario_name}")
    print('='*80)
    
    # Load data
    foods, rotation_matrix = load_scenario_data()
    food_names = list(foods.keys())
    n_foods = len(food_names)
    n_periods = 3
    
    # Load cluster samplesets
    clusters = load_cluster_samplesets(scenario_name, sampleset_dir)
    if not clusters:
        return {'error': 'No samplesets found'}
    
    print(f"Loaded {len(clusters)} cluster samplesets")
    
    # Get all farms from clusters
    all_farms = set()
    for cluster_data in clusters.values():
        all_farms.update(cluster_data['farms'])
    farms = sorted(all_farms)
    print(f"Total farms: {len(farms)}")
    
    # Create land availability (default 1.0 per farm)
    land_availability = {f: 1.0 for f in farms}
    
    # Reconstruct global solution
    solution = reconstruct_global_solution(clusters, n_periods)
    print(f"Global solution has {len(solution)} variable assignments")
    
    # Calculate original objective
    original_obj = calculate_objective(solution, farms, foods, rotation_matrix, land_availability, n_periods)
    print(f"Original objective: {original_obj:.4f}")
    
    # Identify violations
    violations = identify_violations(solution, farms, n_foods, n_periods)
    print(f"Found {len(violations)} one-hot violations")
    
    if violations:
        print("\nViolations:")
        for farm, period in violations[:10]:
            print(f"  {farm}, period {period}: no crop assigned")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more")
    
    # Repair violations
    repaired_solution = solution.copy()
    repairs_made = []
    
    for farm, period in violations:
        best_food_idx = repair_violation(
            repaired_solution, farm, period, n_foods, foods, rotation_matrix
        )
        repaired_solution[(farm, best_food_idx, period)] = 1
        repairs_made.append((farm, period, food_names[best_food_idx]))
    
    print(f"\nRepairs made: {len(repairs_made)}")
    for farm, period, food in repairs_made[:10]:
        print(f"  {farm}, period {period}: assigned '{food}'")
    if len(repairs_made) > 10:
        print(f"  ... and {len(repairs_made) - 10} more")
    
    # Verify repairs
    remaining_violations = identify_violations(repaired_solution, farms, n_foods, n_periods)
    print(f"\nRemaining violations after repair: {len(remaining_violations)}")
    
    # Calculate repaired objective
    repaired_obj = calculate_objective(repaired_solution, farms, foods, rotation_matrix, land_availability, n_periods)
    print(f"Repaired objective: {repaired_obj:.4f}")
    
    # Compare
    improvement = repaired_obj - original_obj
    print(f"\nImprovement: {improvement:.4f} ({100*improvement/abs(original_obj):.1f}%)")
    
    return {
        'scenario': scenario_name,
        'n_farms': len(farms),
        'n_foods': n_foods,
        'original_objective': original_obj,
        'original_violations': len(violations),
        'repaired_objective': repaired_obj,
        'remaining_violations': len(remaining_violations),
        'repairs_made': len(repairs_made),
        'improvement': improvement,
        'improvement_pct': 100 * improvement / abs(original_obj) if original_obj != 0 else 0,
    }


def main():
    """Repair all scenarios and compare results."""
    # Find all scenarios from samplesets
    sampleset_dir = 'qpu_samplesets_all'
    sampleset_path = Path(sampleset_dir)
    
    if not sampleset_path.exists():
        print(f"Sampleset directory not found: {sampleset_dir}")
        return
    
    # Extract unique scenarios
    scenarios = set()
    for f in sampleset_path.glob('*.pkl'):
        # Parse filename: sampleset_hier_{scenario}_iter{i}_cluster{j}_{timestamp}.pkl
        parts = f.stem.split('_')
        # Find scenario name between 'hier' and 'iter'
        try:
            hier_idx = parts.index('hier')
            iter_idx = next(i for i, p in enumerate(parts) if p.startswith('iter'))
            scenario = '_'.join(parts[hier_idx + 1:iter_idx])
            scenarios.add(scenario)
        except (ValueError, StopIteration):
            continue
    
    print(f"Found {len(scenarios)} unique scenarios")
    print(f"Scenarios: {sorted(scenarios)}")
    
    # Repair each scenario
    results = []
    for scenario in sorted(scenarios):
        try:
            result = repair_scenario(scenario, sampleset_dir)
            results.append(result)
        except Exception as e:
            print(f"Error repairing {scenario}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 100)
    print("POST-PROCESSING REPAIR SUMMARY")
    print("=" * 100)
    
    print(f"\n{'Scenario':<35} {'Farms':>8} {'Orig Obj':>12} {'Viols':>8} {'Rep Obj':>12} {'Impr':>10}")
    print("-" * 95)
    
    for r in results:
        if 'error' not in r:
            print(f"{r['scenario']:<35} {r['n_farms']:>8} {r['original_objective']:>12.2f} "
                  f"{r['original_violations']:>8} {r['repaired_objective']:>12.2f} {r['improvement']:>10.2f}")
    
    # Save results
    output_path = Path('professional_plots') / 'postprocessing_repair_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    # Compare with Gurobi
    print("\n" + "=" * 100)
    print("COMPARISON WITH GUROBI (300s timeout)")
    print("=" * 100)
    
    gurobi_path = '@todo/gurobi_timeout_verification/gurobi_timeout_test_20251224_103144.json'
    if Path(gurobi_path).exists():
        with open(gurobi_path) as f:
            gurobi_data = json.load(f)
        
        gurobi_by_scenario = {}
        for entry in gurobi_data:
            if 'metadata' in entry:
                sc = entry['metadata']['scenario']
                gurobi_by_scenario[sc] = entry.get('result', {}).get('objective_value', 0)
        
        print(f"\n{'Scenario':<35} {'Gurobi':>12} {'QPU(Rep)':>12} {'Gap':>10} {'Gap %':>10}")
        print("-" * 95)
        
        total_gap_before = 0
        total_gap_after = 0
        
        for r in results:
            if 'error' not in r:
                sc = r['scenario']
                gurobi_obj = gurobi_by_scenario.get(sc, 0)
                repaired_obj = r['repaired_objective']
                original_obj = r['original_objective']
                
                if gurobi_obj > 0:
                    gap_after = abs(repaired_obj) - gurobi_obj
                    gap_pct = 100 * gap_after / gurobi_obj
                    gap_before = abs(original_obj) - gurobi_obj
                    
                    print(f"{sc:<35} {gurobi_obj:>12.2f} {repaired_obj:>12.2f} {gap_after:>10.2f} {gap_pct:>9.1f}%")
                    
                    total_gap_before += gap_before
                    total_gap_after += gap_after
        
        print("-" * 95)
        gap_reduction = total_gap_before - total_gap_after
        if total_gap_before != 0:
            print(f"Total gap reduced by: {gap_reduction:.2f} ({100*gap_reduction/total_gap_before:.1f}% improvement)")
        else:
            print(f"Total gap reduced by: {gap_reduction:.2f}")


if __name__ == '__main__':
    main()
