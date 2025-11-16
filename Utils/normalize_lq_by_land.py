"""
Normalize LQ objectives using total available farm land area.

This script:
1. Uses generate_farms() with the same seed to recreate farm land availability
2. Calculates total available land for each configuration
3. Normalizes LQ objectives by dividing by total available land
4. Updates LQ result files with normalized values
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from .farm_sampler import generate_farms

def calculate_total_available_land():
    """Calculate total available land for each configuration using generate_farms."""
    
    configs = [5, 19, 72, 279]
    seed = 42  # Same seed used in benchmarks
    n_foods = 27  # Number of foods
    
    print("\n" + "="*80)
    print("CALCULATING TOTAL AVAILABLE LAND FROM FARM SAMPLING")
    print("="*80)
    print(f"Using seed={seed} (same as benchmarks)")
    print(f"Total variables = n_farms × n_foods (where n_foods = {n_foods})")
    print()
    
    total_lands = {}
    
    for config in configs:
        # Generate farms exactly as done in benchmarks
        # Note: config is the number of farms, not farms*27
        L = generate_farms(n_farms=config, seed=seed)
        
        # Calculate total available land
        total_land = sum(L.values())
        
        # Total problem size is config farms × 27 foods
        problem_size = config * n_foods
        
        total_lands[config] = total_land
        
        print(f"Config {config:>3} ({config:>4} farms, {problem_size:>4} variables): Total Available Land = {total_land:>10.2f} hectares")
    
    return total_lands

def normalize_lq_objectives(total_lands):
    """Normalize LQ objectives using total available land and update the files."""
    
    lq_dir = Path(__file__).parent / "Benchmarks" / "LQ"
    configs = [5, 19, 72, 279]
    solvers = ['PuLP', 'Pyomo', 'DWave']
    
    print("\n" + "="*80)
    print("NORMALIZING LQ OBJECTIVES WITH TOTAL AVAILABLE LAND")
    print("="*80)
    
    normalization_summary = {}
    
    for solver in solvers:
        normalization_summary[solver] = {}
        solver_dir = lq_dir / solver
        
        if not solver_dir.exists():
            print(f"\nWarning: {solver} directory not found in LQ")
            continue
        
        print(f"\n{solver} Solver:")
        print("-" * 80)
        
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            
            if not config_file.exists():
                print(f"  Warning: {config_file} not found")
                continue
            
            # Get total available land for this config
            if config not in total_lands:
                print(f"  Warning: No land data for config_{config}")
                continue
            
            total_land = total_lands[config]
            
            if total_land <= 1e-6:
                print(f"  Warning: Total land too small for config_{config}: {total_land}")
                continue
            
            # Load LQ result
            with open(config_file, 'r') as f:
                lq_data = json.load(f)
            
            result = lq_data.get('result', {})
            lq_objective = result.get('objective_value')
            
            if lq_objective is None:
                print(f"  Warning: No objective for config_{config}")
                continue
            
            # Calculate normalized objective
            normalized_objective = lq_objective / total_land
            
            # Update the result
            result['normalized_objective'] = normalized_objective
            result['total_available_land'] = total_land
            result['normalization_method'] = 'total_available_land'
            
            # Save updated file
            with open(config_file, 'w') as f:
                json.dump(lq_data, f, indent=2)
            
            print(f"  config_{config} ({config:>4} farms, {config * 27:>4} variables):")
            print(f"    LQ Objective:        {lq_objective:>12.4f}")
            print(f"    Total Land:          {total_land:>12.2f} ha")
            print(f"    Normalized:          {normalized_objective:>12.8f} (per hectare)")
            
            normalization_summary[solver][config] = {
                'lq_objective': lq_objective,
                'total_land': total_land,
                'normalized_objective': normalized_objective,
                'n_farms': config,
                'n_variables': config * 27
            }
    
    return normalization_summary

def print_comparison_table(summary, total_lands):
    """Print a comparison table showing the normalization results."""
    
    print("\n" + "="*80)
    print("NORMALIZATION COMPARISON TABLE")
    print("="*80)
    print(f"\n{'N_Farms':<10} {'N_Vars':<10} {'Solver':<10} {'LQ Objective':<15} {'Avail Land (ha)':<18} {'Normalized':<18}")
    print("-" * 100)
    
    configs = [5, 19, 72, 279]
    solvers = ['PuLP', 'Pyomo', 'DWave']
    
    for config in configs:
        n_farms = config
        n_vars = config * 27
        total_land = total_lands.get(config, 0)
        
        for i, solver in enumerate(solvers):
            if solver in summary and config in summary[solver]:
                data = summary[solver][config]
                farms_label = f"{n_farms}" if i == 0 else ""
                vars_label = f"{n_vars}" if i == 0 else ""
                land_label = f"{total_land:.2f}" if i == 0 else ""
                print(f"{farms_label:<10} {vars_label:<10} {solver:<10} {data['lq_objective']:<15.4f} "
                      f"{land_label:<18} {data['normalized_objective']:<18.8f}")
        if config < configs[-1]:
            print("-" * 100)
    
    print("="*100)
    
    # Print interpretation
    print("\nInterpretation:")
    print("  Normalized Objective = LQ Objective / Total Available Land")
    print("  Units: objective value per hectare")
    print("  Higher is better (more value generated per unit of available land)")
    print("  Note: N_Vars = N_Farms × 27 (foods)")
    print("="*100)

def verify_normalization():
    """Verify that normalization was applied correctly by reading the files back."""
    
    lq_dir = Path(__file__).parent / "Benchmarks" / "LQ"
    configs = [5, 19, 72, 279]
    solvers = ['PuLP', 'Pyomo', 'DWave']
    
    print("\n" + "="*80)
    print("VERIFICATION: Reading normalized values from files")
    print("="*80)
    
    all_valid = True
    
    for solver in solvers:
        solver_dir = lq_dir / solver
        if not solver_dir.exists():
            continue
        
        print(f"\n{solver}:")
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            if not config_file.exists():
                continue
            
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            result = data.get('result', {})
            
            if 'normalized_objective' in result and 'total_available_land' in result:
                n_farms = config
                n_vars = config * 27
                norm_obj = result['normalized_objective']
                total_land = result['total_available_land']
                print(f"  config_{config} ({n_farms:>3} farms, {n_vars:>4} vars): ✓ Normalized = {norm_obj:.8f} (Land = {total_land:.2f} ha)")
            else:
                print(f"  config_{config}: ✗ Missing normalized_objective or total_available_land")
                all_valid = False
    
    print("\n" + "="*80)
    if all_valid:
        print("✅ VERIFICATION PASSED: All LQ files have normalized objectives")
    else:
        print("❌ VERIFICATION FAILED: Some files missing normalized objectives")
    print("="*80 + "\n")
    
    return all_valid

def compare_across_solvers(summary):
    """Compare normalized objectives across solvers for each configuration."""
    
    print("\n" + "="*80)
    print("CROSS-SOLVER COMPARISON (Normalized Objectives)")
    print("="*80)
    
    configs = [5, 19, 72, 279]
    solvers = ['PuLP', 'Pyomo', 'DWave']
    
    for config in configs:
        n_farms = config
        n_vars = config * 27
        print(f"\nConfiguration: {n_farms} farms, {n_vars} variables (config_{config})")
        print("-" * 80)
        
        values = []
        for solver in solvers:
            if solver in summary and config in summary[solver]:
                norm_obj = summary[solver][config]['normalized_objective']
                values.append((solver, norm_obj))
                print(f"  {solver:<10}: {norm_obj:.8f}")
        
        if len(values) > 1:
            # Calculate differences
            best_solver, best_value = max(values, key=lambda x: x[1])
            print(f"\n  Best: {best_solver} ({best_value:.8f})")
            
            for solver, value in values:
                if solver != best_solver:
                    gap = ((best_value - value) / best_value) * 100
                    print(f"  Gap: {solver} is {gap:.4f}% worse than {best_solver}")
    
    print("="*80)

def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("LQ OBJECTIVE NORMALIZATION USING TOTAL AVAILABLE FARM LAND")
    print("="*80)
    print("\nThis script normalizes LQ objectives by dividing by the total available")
    print("farm land (sum of all farm sizes generated by farm_sampler).")
    print("="*80)
    
    # Step 1: Calculate total available land for each configuration
    total_lands = calculate_total_available_land()
    
    # Step 2: Normalize LQ objectives
    summary = normalize_lq_objectives(total_lands)
    
    # Step 3: Print comparison table
    print_comparison_table(summary, total_lands)
    
    # Step 4: Compare across solvers
    compare_across_solvers(summary)
    
    # Step 5: Verify
    verify_normalization()
    
    # Save summary
    summary_file = Path(__file__).parent / "lq_land_normalization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_available_lands': total_lands,
            'normalization_summary': summary
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_file}\n")

if __name__ == "__main__":
    main()
