#!/usr/bin/env python3
"""
Solution Validation for Decomposition Strategy Results

Compares decomposition strategy results against PuLP optimal solution:
- Checks constraint violations
- Identifies suboptimality
- Validates solution feasibility
- Compares objective values
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from result_formatter import validate_solution_constraints
from benchmark_utils_decomposed import generate_farm_data, create_config
from src.scenarios import load_food_data


def load_pulp_reference(config_size: int) -> Dict:
    """Load PuLP reference solution."""
    pulp_file = f"../Benchmarks/COMPREHENSIVE/Farm_PuLP/config_{config_size}_run_1.json"
    
    if not os.path.exists(pulp_file):
        print(f"⚠️  PuLP reference not found: {pulp_file}")
        return None
    
    with open(pulp_file, 'r') as f:
        return json.load(f)


def extract_solution_dict(result: Dict, strategy_name: str) -> Tuple[Dict, Dict]:
    """Extract A and Y solution dictionaries from result."""
    A_sol = {}
    Y_sol = {}
    
    # Try different result structures
    if 'result' in result and 'solution_areas' in result['result']:
        # Standard format
        areas = result['result']['solution_areas']
        selections = result['result'].get('solution_selections', {})
        
        for key, value in areas.items():
            parts = key.split('_', 1)
            if len(parts) == 2:
                farm, food = parts
                A_sol[(farm, food)] = value
        
        for key, value in selections.items():
            parts = key.split('_', 1)
            if len(parts) == 2:
                farm, food = parts
                Y_sol[(farm, food)] = value
    
    elif 'solution' in result:
        # Direct solution format
        solution = result['solution']
        for var_name, value in solution.items():
            if var_name.startswith('A_'):
                clean = var_name[2:]  # Remove 'A_'
                parts = clean.split('_', 1)
                if len(parts) == 2:
                    A_sol[tuple(parts)] = value
            elif var_name.startswith('Y_'):
                clean = var_name[2:]  # Remove 'Y_'
                parts = clean.split('_', 1)
                if len(parts) == 2:
                    Y_sol[tuple(parts)] = value
    
    return A_sol, Y_sol


def validate_strategy_solution(
    strategy_name: str,
    result: Dict,
    farms: Dict,
    foods: List[str],
    food_groups: Dict,
    config: Dict,
    pulp_obj: float
) -> Dict:
    """Validate a single strategy's solution."""
    
    validation = {
        'strategy': strategy_name,
        'status': result.get('solver_info', {}).get('status', 'Unknown'),
        'objective': 0.0,
        'pulp_objective': pulp_obj,
        'objective_ratio': 0.0,
        'is_feasible': False,
        'constraint_violations': [],
        'violation_count': 0,
        'suboptimality_gap': 0.0,
        'notes': []
    }
    
    # Get objective value
    if 'result' in result:
        validation['objective'] = result['result'].get('objective_value', 0.0)
    elif 'solution' in result:
        validation['objective'] = result.get('objective_value', 0.0)
    
    # Calculate objective ratio
    if pulp_obj > 0:
        validation['objective_ratio'] = validation['objective'] / pulp_obj
        validation['suboptimality_gap'] = (pulp_obj - validation['objective']) / pulp_obj * 100
    
    # Extract solution
    A_sol, Y_sol = extract_solution_dict(result, strategy_name)
    
    if not A_sol and not Y_sol:
        validation['notes'].append("Could not extract solution from result")
        return validation
    
    # Build full solution for validation
    full_solution = {}
    for (farm, food), value in A_sol.items():
        full_solution[f"A_{farm}_{food}"] = value
    for (farm, food), value in Y_sol.items():
        full_solution[f"Y_{farm}_{food}"] = value
    
    # Validate constraints
    try:
        constraint_val = validate_solution_constraints(
            full_solution, farms, foods, food_groups, farms, config, 'farm'
        )
        
        validation['is_feasible'] = constraint_val['is_feasible']
        validation['constraint_violations'] = constraint_val.get('violations', [])
        validation['violation_count'] = len(validation['constraint_violations'])
        
        # Add constraint check summary
        validation['constraint_checks'] = constraint_val.get('constraint_checks', {})
        
    except Exception as e:
        validation['notes'].append(f"Validation error: {e}")
    
    # Classification
    if validation['objective'] > pulp_obj * 1.01:  # More than 1% higher
        validation['notes'].append("⚠️  INFEASIBLE: Objective exceeds PuLP optimum (likely constraint violation)")
    elif validation['objective'] < pulp_obj * 0.95:  # More than 5% lower
        validation['notes'].append("⚠️  SUBOPTIMAL: Objective significantly below PuLP optimum")
    elif not validation['is_feasible']:
        validation['notes'].append("⚠️  INFEASIBLE: Constraint violations detected")
    elif abs(validation['objective'] - pulp_obj) < pulp_obj * 0.01:
        validation['notes'].append("✅ OPTIMAL: Matches PuLP solution within 1%")
    else:
        validation['notes'].append("✅ FEASIBLE: Valid solution, slightly suboptimal")
    
    return validation


def analyze_all_strategies(benchmark_file: str, config_size: int = 25) -> Dict:
    """Analyze all strategies from benchmark results."""
    
    # Load benchmark results
    with open(benchmark_file, 'r') as f:
        benchmark = json.load(f)
    
    # Load PuLP reference
    pulp_ref = load_pulp_reference(config_size)
    if not pulp_ref:
        print("❌ Cannot proceed without PuLP reference solution")
        return None
    
    pulp_obj = pulp_ref['objective_value']
    print(f"\n{'='*80}")
    print(f"SOLUTION VALIDATION AGAINST PULP REFERENCE")
    print(f"{'='*80}")
    print(f"PuLP Optimal Objective: {pulp_obj:.6f} (normalized)")
    print(f"Configuration: {config_size} farms")
    print(f"{'='*80}\n")
    
    # Generate problem data (same as benchmark)
    farm_data = generate_farm_data(config_size, 100.0)
    foods, food_groups, config_dict = create_config(farm_data['land_data'])
    
    # Add benefits
    _, _, _, base_config = load_food_data('full_family')
    config_dict['benefits'] = {}
    for food in foods:
        weights = config_dict.get('parameters', {}).get('weights', {})
        benefit = sum(
            base_config.get('nutrients', {}).get(food, {}).get(attr, 0) * weight
            for attr, weight in weights.items()
        )
        config_dict['benefits'][food] = benefit if benefit > 0 else 100.0
    
    farms = farm_data['land_data']
    
    # Validate each strategy
    validations = {}
    results = benchmark.get('results', {})
    
    for strategy_name, result in results.items():
        print(f"\n{'─'*80}")
        print(f"Validating: {strategy_name.upper()}")
        print(f"{'─'*80}")
        
        validation = validate_strategy_solution(
            strategy_name, result, farms, foods, food_groups, config_dict, pulp_obj
        )
        validations[strategy_name] = validation
        
        # Print summary
        print(f"Objective: {validation['objective']:.6f}")
        print(f"PuLP Objective: {validation['pulp_objective']:.6f}")
        print(f"Ratio: {validation['objective_ratio']:.4f} ({validation['suboptimality_gap']:+.2f}% gap)")
        print(f"Feasible: {'✅ Yes' if validation['is_feasible'] else '❌ No'}")
        print(f"Violations: {validation['violation_count']}")
        
        for note in validation['notes']:
            print(f"  {note}")
        
        if validation['violation_count'] > 0:
            print(f"\n  Constraint Violations ({validation['violation_count']}):")
            for i, viol in enumerate(validation['constraint_violations'][:5], 1):
                viol_type = viol.get('type', 'unknown')
                msg = viol.get('message', viol.get('constraint', ''))
                print(f"    {i}. {viol_type}: {msg}")
            if validation['violation_count'] > 5:
                print(f"    ... and {validation['violation_count'] - 5} more")
    
    return {
        'pulp_reference': {
            'objective': pulp_obj,
            'status': pulp_ref.get('status', 'Unknown')
        },
        'validations': validations,
        'summary': generate_summary(validations, pulp_obj)
    }


def generate_summary(validations: Dict, pulp_obj: float) -> Dict:
    """Generate summary statistics."""
    summary = {
        'total_strategies': len(validations),
        'feasible': 0,
        'infeasible': 0,
        'optimal': 0,
        'suboptimal': 0,
        'best_objective': 0.0,
        'worst_objective': float('inf'),
        'best_strategy': None,
        'worst_strategy': None
    }
    
    for strategy, val in validations.items():
        if val['is_feasible']:
            summary['feasible'] += 1
        else:
            summary['infeasible'] += 1
        
        if abs(val['objective'] - pulp_obj) < pulp_obj * 0.01:
            summary['optimal'] += 1
        elif val['is_feasible']:
            summary['suboptimal'] += 1
        
        if val['objective'] > summary['best_objective']:
            summary['best_objective'] = val['objective']
            summary['best_strategy'] = strategy
        
        if val['objective'] < summary['worst_objective']:
            summary['worst_objective'] = val['objective']
            summary['worst_strategy'] = strategy
    
    return summary


def print_comparison_table(analysis: Dict):
    """Print detailed comparison table."""
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    summary = analysis['summary']
    pulp_obj = analysis['pulp_reference']['objective']
    
    print(f"PuLP Reference Objective: {pulp_obj:.6f}")
    print(f"Total Strategies: {summary['total_strategies']}")
    print(f"Feasible: {summary['feasible']}")
    print(f"Infeasible: {summary['infeasible']}")
    print(f"Optimal (within 1%): {summary['optimal']}")
    print(f"Suboptimal: {summary['suboptimal']}")
    print(f"\nBest Strategy: {summary['best_strategy']} ({summary['best_objective']:.6f})")
    print(f"Worst Strategy: {summary['worst_strategy']} ({summary['worst_objective']:.6f})")
    
    print(f"\n{'='*80}")
    print("DETAILED COMPARISON")
    print(f"{'='*80}\n")
    
    header = f"{'Strategy':<25} {'Objective':<12} {'Ratio':<8} {'Gap %':<10} {'Feasible':<10} {'Violations':<12}"
    print(header)
    print("─" * 80)
    
    validations = analysis['validations']
    
    # Sort by objective descending
    sorted_strategies = sorted(
        validations.items(),
        key=lambda x: x[1]['objective'],
        reverse=True
    )
    
    for strategy, val in sorted_strategies:
        obj = val['objective']
        ratio = val['objective_ratio']
        gap = val['suboptimality_gap']
        feasible = '✅ Yes' if val['is_feasible'] else '❌ No'
        violations = val['violation_count']
        
        print(f"{strategy:<25} {obj:<12.6f} {ratio:<8.4f} {gap:>9.2f} {feasible:<10} {violations:<12}")
    
    print("─" * 80)


def main():
    """Main validation execution."""
    parser = argparse.ArgumentParser(
        description='Validate Decomposition Strategy Results'
    )
    
    parser.add_argument('--benchmark-file', type=str, required=True,
                       help='Path to benchmark results JSON file')
    parser.add_argument('--config', type=int, default=25,
                       help='Configuration size (number of farms)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for validation report (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.benchmark_file):
        print(f"❌ Benchmark file not found: {args.benchmark_file}")
        sys.exit(1)
    
    # Run analysis
    analysis = analyze_all_strategies(args.benchmark_file, args.config)
    
    if analysis:
        # Print comparison table
        print_comparison_table(analysis)
        
        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Validation report saved: {args.output}")


if __name__ == "__main__":
    main()
