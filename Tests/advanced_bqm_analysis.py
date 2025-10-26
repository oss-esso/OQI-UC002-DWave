#!/usr/bin/env python3
"""
Advanced BQM Constraint Analysis for D-Wave Optimization

This script provides deeper analysis of constraint violations in BQM formulation,
specifically focusing on the "at most one crop per plot" constraint that's being
violated in 50-unit scenarios.

Investigates:
1. Manual BQM construction with explicit penalty terms
2. Constraint penalty magnitude analysis
3. Energy landscape analysis around constraint violations
4. Alternative constraint formulations for better enforcement
"""

import os
import sys
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dimod import BinaryQuadraticModel, cqm_to_bqm
from dwave.system import LeapHybridBQMSampler
from solver_runner_PATCH import validate_solution_constraints


class ManualBQMConstructor:
    """Constructs BQM manually with explicit constraint penalties."""
    
    def __init__(self, farms: List[str], foods: List[str], land_availability: Dict[str, float], 
                 food_data: Dict, weights: Dict, idle_penalty: float = 0.1):
        self.farms = farms
        self.foods = foods
        self.land_availability = land_availability
        self.food_data = food_data
        self.weights = weights
        self.idle_penalty = idle_penalty
        
    def construct_bqm(self, constraint_penalty_multiplier: float = 100.0) -> BinaryQuadraticModel:
        """
        Manually construct BQM with explicit constraint penalties.
        
        Variables:
        - X_{p,c}: Binary, 1 if plot p assigned to crop c
        - Y_c: Binary, 1 if crop c is grown
        
        Objective:
        - Maximize: sum_{p,c} (B_c + λ) * s_p * X_{p,c}
        - Minimize: -sum_{p,c} (B_c + λ) * s_p * X_{p,c}
        
        Constraints (as penalties):
        - At most one crop per plot: penalty * sum_p [sum_c X_{p,c} - 1]^2_+
        - X-Y linking: penalty * sum_{p,c} [X_{p,c} - Y_c]^2_+
        - Y activation: penalty * sum_c [Y_c - sum_p X_{p,c}]^2_+
        
        Args:
            constraint_penalty_multiplier: Penalty strength for constraint violations
        
        Returns:
            BinaryQuadraticModel with explicit penalties
        """
        print(f"\\n🔨 Constructing manual BQM...")
        print(f"   Constraint penalty multiplier: {constraint_penalty_multiplier}")
        
        bqm = BinaryQuadraticModel(vartype='BINARY')
        
        # Create all variables
        print("   Creating variables...")
        var_count = 0
        
        # X_{p,c} variables
        for plot in self.farms:
            for crop in self.foods:
                var_name = f"X_{plot}_{crop}"
                bqm.add_variable(var_name, 0.0)  # Start with zero bias
                var_count += 1
        
        # Y_c variables  
        for crop in self.foods:
            var_name = f"Y_{crop}"
            bqm.add_variable(var_name, 0.0)  # Start with zero bias
            var_count += 1
            
        print(f"   Created {var_count} variables")
        
        # Add objective terms
        print("   Adding objective terms...")
        obj_terms = 0
        
        for plot in self.farms:
            s_p = self.land_availability[plot]
            for crop in self.foods:
                # Calculate benefit B_c
                B_c = (
                    self.weights.get('nutritional_value', 0) * self.food_data[crop].get('nutritional_value', 0) +
                    self.weights.get('nutrient_density', 0) * self.food_data[crop].get('nutrient_density', 0) -
                    self.weights.get('environmental_impact', 0) * self.food_data[crop].get('environmental_impact', 0) +
                    self.weights.get('affordability', 0) * self.food_data[crop].get('affordability', 0) +
                    self.weights.get('sustainability', 0) * self.food_data[crop].get('sustainability', 0)
                )
                
                # Objective: minimize -(B_c + λ) * s_p * X_{p,c}
                var_name = f"X_{plot}_{crop}"
                objective_coeff = -(B_c + self.idle_penalty) * s_p
                bqm.add_variable(var_name, objective_coeff)
                obj_terms += 1
        
        print(f"   Added {obj_terms} objective terms")
        
        # Add constraint penalties
        print("   Adding constraint penalties...")
        penalty_terms = 0
        
        # 1. At most one crop per plot constraint
        # Penalty: M * sum_p [sum_c X_{p,c} - 1]^2_+ 
        # Expanded: M * sum_p [sum_c X_{p,c}^2 + sum_{c!=c'} X_{p,c} * X_{p,c'} - 2*sum_c X_{p,c} + 1]
        # Since X_{p,c}^2 = X_{p,c} for binary, this becomes:
        # M * sum_p [sum_c X_{p,c} + sum_{c!=c'} X_{p,c} * X_{p,c'} - 2*sum_c X_{p,c} + 1]
        # = M * sum_p [sum_{c!=c'} X_{p,c} * X_{p,c'} - sum_c X_{p,c} + 1]
        
        for plot in self.farms:
            # Linear penalty terms: -M * sum_c X_{p,c}
            for crop in self.foods:
                var_name = f"X_{plot}_{crop}"
                bqm.add_variable(var_name, -constraint_penalty_multiplier)
                penalty_terms += 1
            
            # Quadratic penalty terms: M * sum_{c!=c'} X_{p,c} * X_{p,c'}
            for i, crop1 in enumerate(self.foods):
                for j, crop2 in enumerate(self.foods):
                    if i < j:  # Avoid double counting
                        var1 = f"X_{plot}_{crop1}"
                        var2 = f"X_{plot}_{crop2}"
                        bqm.add_interaction(var1, var2, constraint_penalty_multiplier)
                        penalty_terms += 1
            
            # Constant term +M (doesn't affect optimization, but for completeness)
            bqm.offset += constraint_penalty_multiplier
        
        print(f"   Added {penalty_terms} penalty terms for plot constraints")
        
        # 2. X-Y linking constraints: X_{p,c} <= Y_c
        # Penalty: M * sum_{p,c} [X_{p,c} - Y_c]^2_+ = M * sum_{p,c} [X_{p,c} + Y_c - 2*X_{p,c}*Y_c]
        # Since we want to penalize X_{p,c} = 1, Y_c = 0, we use:
        # M * sum_{p,c} X_{p,c} * (1 - Y_c) = M * sum_{p,c} (X_{p,c} - X_{p,c}*Y_c)
        
        linking_terms = 0
        for plot in self.farms:
            for crop in self.foods:
                x_var = f"X_{plot}_{crop}"
                y_var = f"Y_{crop}"
                
                # Add penalty for X_{p,c} when Y_c = 0
                bqm.add_variable(x_var, constraint_penalty_multiplier)
                # Subtract penalty when both are 1
                bqm.add_interaction(x_var, y_var, -constraint_penalty_multiplier)
                linking_terms += 2
        
        print(f"   Added {linking_terms} penalty terms for X-Y linking")
        
        # 3. Y activation constraints: Y_c <= sum_p X_{p,c}
        # This is automatically satisfied in practice if we want to maximize the objective
        # and crops only contribute positively, so we'll skip this for simplicity
        
        print(f"   BQM construction complete:")
        print(f"     Variables: {len(bqm.variables)}")
        print(f"     Linear terms: {len(bqm.linear)}")
        print(f"     Quadratic terms: {len(bqm.quadratic)}")
        print(f"     Offset: {bqm.offset:.2f}")
        
        return bqm
    
    def analyze_bqm_properties(self, bqm: BinaryQuadraticModel) -> Dict:
        """Analyze properties of the constructed BQM."""
        
        linear_magnitudes = [abs(bias) for bias in bqm.linear.values()]
        quad_magnitudes = [abs(bias) for bias in bqm.quadratic.values()]
        
        analysis = {
            'n_variables': len(bqm.variables),
            'n_linear': len(bqm.linear),
            'n_quadratic': len(bqm.quadratic),
            'offset': bqm.offset,
            'linear_stats': {
                'min': min(linear_magnitudes) if linear_magnitudes else 0,
                'max': max(linear_magnitudes) if linear_magnitudes else 0,
                'mean': np.mean(linear_magnitudes) if linear_magnitudes else 0,
                'std': np.std(linear_magnitudes) if linear_magnitudes else 0
            },
            'quadratic_stats': {
                'min': min(quad_magnitudes) if quad_magnitudes else 0,
                'max': max(quad_magnitudes) if quad_magnitudes else 0,
                'mean': np.mean(quad_magnitudes) if quad_magnitudes else 0,
                'std': np.std(quad_magnitudes) if quad_magnitudes else 0
            }
        }
        
        return analysis


def run_manual_bqm_experiment(farms: List[str], foods: List[str], land_availability: Dict,
                            food_data: Dict, config: Dict, 
                            penalty_multipliers: List[float] = [1, 10, 100, 1000],
                            dwave_token: str = None,
                            food_groups: Dict = None):
    """
    Run experiments with different penalty multipliers in manual BQM construction.
    
    Args:
        farms: List of farm/plot names
        foods: List of food names  
        land_availability: Plot area mapping
        food_data: Food nutritional data
        config: Configuration parameters
        penalty_multipliers: List of penalty strengths to test
        dwave_token: D-Wave API token (optional)
        food_groups: Food group dictionary (optional)
    
    Returns:
        Dictionary with results for each penalty multiplier
    """
    print("\\n" + "="*80)
    print("🧪 MANUAL BQM CONSTRAINT PENALTY EXPERIMENT")
    print("="*80)
    
    if food_groups is None:
        food_groups = {}
    
    weights = config['parameters']['weights']
    idle_penalty = config['parameters']['idle_penalty_lambda']
    
    constructor = ManualBQMConstructor(farms, foods, land_availability, food_data, weights, idle_penalty)
    
    results = {}
    if not dwave_token:
        dwave_token = os.environ.get('DWAVE_API_TOKEN')
    
    if not dwave_token:
        print("⚠️  No D-Wave token found. Will analyze BQM structure only.")
        solve_with_dwave = False
    else:
        solve_with_dwave = True
        sampler = LeapHybridBQMSampler(token=dwave_token)
    
    for multiplier in penalty_multipliers:
        print(f"\\n🎯 Testing penalty multiplier: {multiplier}")
        
        # Construct BQM
        bqm = constructor.construct_bqm(constraint_penalty_multiplier=multiplier)
        
        # Analyze structure
        analysis = constructor.analyze_bqm_properties(bqm)
        
        print(f"   Structure analysis:")
        print(f"     Linear bias range: [{analysis['linear_stats']['min']:.2f}, {analysis['linear_stats']['max']:.2f}]")
        print(f"     Quadratic bias range: [{analysis['quadratic_stats']['min']:.2f}, {analysis['quadratic_stats']['max']:.2f}]")
        print(f"     Penalty/Objective ratio: {analysis['linear_stats']['max'] / (analysis['linear_stats']['mean'] + 1e-10):.1f}")
        
        result = {
            'penalty_multiplier': multiplier,
            'bqm_analysis': analysis,
            'solution': None,
            'validation': None,
            'timing': None
        }
        
        # Solve with D-Wave if token available
        if solve_with_dwave:
            try:
                print(f"   Solving with D-Wave...")
                start_time = time.time()
                sampleset = sampler.sample(bqm, label=f"Manual BQM - Penalty {multiplier}")
                solve_time = time.time() - start_time
                
                if len(sampleset) > 0:
                    best_sample = sampleset.first
                    solution = dict(best_sample.sample)
                    energy = best_sample.energy
                    
                    print(f"     Energy: {energy:.2f}")
                    print(f"     Solve time: {solve_time:.2f}s")
                    
                    # Validate solution
                    validation = validate_solution_constraints(
                        solution, farms, foods, food_groups, land_availability, config
                    )
                    
                    print(f"     Feasible: {validation['is_feasible']}")
                    print(f"     Violations: {validation['n_violations']}")
                    
                    result['solution'] = solution
                    result['energy'] = energy
                    result['validation'] = validation
                    result['timing'] = {'solve_time': solve_time}
                    
                else:
                    print("     No solutions returned")
                    
            except Exception as e:
                print(f"     ❌ Solving failed: {e}")
        
        results[multiplier] = result
    
    return results


def analyze_constraint_satisfaction_patterns(results: Dict):
    """Analyze patterns in constraint satisfaction across penalty multipliers."""
    
    print("\\n" + "="*80)  
    print("📊 CONSTRAINT SATISFACTION PATTERN ANALYSIS")
    print("="*80)
    
    feasible_multipliers = []
    violation_counts = []
    penalty_ratios = []
    
    for multiplier, result in results.items():
        if result['validation'] is not None:
            is_feasible = result['validation']['is_feasible']
            n_violations = result['validation']['n_violations']
            
            # Calculate penalty-to-objective ratio
            analysis = result['bqm_analysis']
            penalty_ratio = analysis['linear_stats']['max'] / (analysis['linear_stats']['mean'] + 1e-10)
            
            print(f"\\n📈 Penalty Multiplier {multiplier}:")
            print(f"   Feasible: {is_feasible}")
            print(f"   Violations: {n_violations}")
            print(f"   Penalty/Objective ratio: {penalty_ratio:.1f}")
            print(f"   Energy: {result.get('energy', 'N/A')}")
            
            if is_feasible:
                feasible_multipliers.append(multiplier)
            violation_counts.append((multiplier, n_violations))
            penalty_ratios.append((multiplier, penalty_ratio))
    
    # Find patterns
    print(f"\\n🎯 PATTERNS IDENTIFIED:")
    
    if feasible_multipliers:
        print(f"   ✅ Feasible solutions found at multipliers: {feasible_multipliers}")
        min_feasible = min(feasible_multipliers)
        print(f"   ✅ Minimum penalty for feasibility: {min_feasible}")
    else:
        print(f"   ❌ No feasible solutions found with any penalty multiplier")
        print(f"   ❌ This suggests the constraint formulation may need revision")
    
    # Analyze violation trend
    violation_trend = sorted(violation_counts, key=lambda x: x[0])
    print(f"\\n📉 Violation trend:")
    for multiplier, violations in violation_trend:
        print(f"     Penalty {multiplier:4.0f}: {violations:3d} violations")
    
    return {
        'feasible_multipliers': feasible_multipliers,
        'violation_trend': violation_trend,
        'penalty_ratios': penalty_ratios
    }


if __name__ == "__main__":
    print("🔬 Advanced BQM Constraint Analysis")
    print("   This script tests manual BQM construction with explicit penalties")
    print("   to investigate constraint enforcement in 50-unit scenarios")
    
    # Note: This script is designed to be called from the main test script
    # or run standalone with proper setup
    print("\\n⚠️  This script requires setup from test_patch_dwave_bqm_constraints.py")
    print("   Run the main investigation script first to generate test data")