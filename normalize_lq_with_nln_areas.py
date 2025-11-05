"""
Normalize LQ objectives using NLN total areas.

Since NLN is already normalized but LQ is not, we use NLN's total areas
as the normalization factor for LQ objectives to enable fair comparison.

This script:
1. Extracts total areas from NLN results (solver by solver)
2. Normalizes LQ objectives by dividing by corresponding NLN areas
3. Updates LQ result files with normalized values
"""

import json
import os
from pathlib import Path

def extract_nln_areas():
    """Extract total areas from NLN results for each config and solver."""
    
    nln_dir = Path(__file__).parent / "Benchmarks" / "NLN"
    configs = [5, 19, 72, 279]
    solvers = ['PuLP', 'Pyomo', 'DWave']
    
    nln_areas = {}
    
    print("\n" + "="*80)
    print("EXTRACTING NLN AREAS")
    print("="*80)
    
    for solver in solvers:
        nln_areas[solver] = {}
        solver_dir = nln_dir / solver
        
        if not solver_dir.exists():
            print(f"Warning: {solver} directory not found in NLN")
            continue
        
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            
            if not config_file.exists():
                print(f"Warning: {config_file} not found")
                continue
            
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            result = data.get('result', {})
            
            # Extract total area from NLN result
            total_area = 0.0
            
            if solver == 'DWave':
                # DWave stores areas in a flat dict
                areas = result.get('areas', {})
                for var_name, value in areas.items():
                    if value is not None and value > 1e-6:
                        total_area += value
            else:
                # PuLP and Pyomo store areas as nested dict or flat
                areas = result.get('areas', {})
                if isinstance(areas, dict):
                    for key, value in areas.items():
                        if isinstance(value, dict):
                            # Nested structure
                            for area_val in value.values():
                                if area_val is not None and area_val > 1e-6:
                                    total_area += area_val
                        elif value is not None and value > 1e-6:
                            # Flat structure
                            total_area += value
            
            nln_areas[solver][config] = total_area
            n_farms = config * 27
            print(f"{solver:<10} config_{config} ({n_farms:>4} farms): Total Area = {total_area:>10.2f}")
    
    return nln_areas

def normalize_lq_objectives(nln_areas):
    """Normalize LQ objectives using NLN areas and update the files."""
    
    lq_dir = Path(__file__).parent / "Benchmarks" / "LQ"
    configs = [5, 19, 72, 279]
    solvers = ['PuLP', 'Pyomo', 'DWave']
    
    print("\n" + "="*80)
    print("NORMALIZING LQ OBJECTIVES WITH NLN AREAS")
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
            
            # Check if we have NLN area for this config/solver
            if solver not in nln_areas or config not in nln_areas[solver]:
                print(f"  Warning: No NLN area for config_{config}")
                continue
            
            nln_area = nln_areas[solver][config]
            
            if nln_area <= 1e-6:
                print(f"  Warning: NLN area too small for config_{config}: {nln_area}")
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
            normalized_objective = lq_objective / nln_area
            
            # Update the result
            result['normalized_objective'] = normalized_objective
            result['normalization_area'] = nln_area
            result['normalization_source'] = f'NLN_{solver}'
            
            # Save updated file
            with open(config_file, 'w') as f:
                json.dump(lq_data, f, indent=2)
            
            n_farms = config * 27
            print(f"  config_{config} ({n_farms:>4} farms):")
            print(f"    LQ Objective:        {lq_objective:>12.4f}")
            print(f"    NLN Area:            {nln_area:>12.2f}")
            print(f"    Normalized:          {normalized_objective:>12.8f}")
            
            normalization_summary[solver][config] = {
                'lq_objective': lq_objective,
                'nln_area': nln_area,
                'normalized_objective': normalized_objective
            }
    
    return normalization_summary

def print_comparison_table(summary):
    """Print a comparison table showing the normalization results."""
    
    print("\n" + "="*80)
    print("NORMALIZATION COMPARISON TABLE")
    print("="*80)
    print(f"\n{'Config':<10} {'Solver':<10} {'LQ Obj':<15} {'NLN Area':<15} {'Normalized':<15}")
    print("-" * 80)
    
    configs = [5, 19, 72, 279]
    solvers = ['PuLP', 'Pyomo', 'DWave']
    
    for config in configs:
        n_farms = config * 27
        for i, solver in enumerate(solvers):
            if solver in summary and config in summary[solver]:
                data = summary[solver][config]
                config_label = f"{n_farms}" if i == 0 else ""
                print(f"{config_label:<10} {solver:<10} {data['lq_objective']:<15.4f} "
                      f"{data['nln_area']:<15.2f} {data['normalized_objective']:<15.8f}")
        if config < configs[-1]:
            print("-" * 80)
    
    print("="*80)

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
            
            if 'normalized_objective' in result:
                n_farms = config * 27
                print(f"  config_{config} ({n_farms} farms): ✓ Normalized objective = {result['normalized_objective']:.8f}")
            else:
                print(f"  config_{config}: ✗ Missing normalized_objective")
                all_valid = False
    
    print("\n" + "="*80)
    if all_valid:
        print("✅ VERIFICATION PASSED: All LQ files have normalized objectives")
    else:
        print("❌ VERIFICATION FAILED: Some files missing normalized objectives")
    print("="*80 + "\n")
    
    return all_valid

def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("LQ OBJECTIVE NORMALIZATION USING NLN AREAS")
    print("="*80)
    print("\nThis script normalizes LQ objectives by dividing by NLN total areas")
    print("(solver by solver) to enable fair comparison between formulations.")
    print("="*80)
    
    # Step 1: Extract NLN areas
    nln_areas = extract_nln_areas()
    
    # Step 2: Normalize LQ objectives
    summary = normalize_lq_objectives(nln_areas)
    
    # Step 3: Print comparison table
    print_comparison_table(summary)
    
    # Step 4: Verify
    verify_normalization()
    
    # Save summary
    summary_file = Path(__file__).parent / "lq_nln_normalization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'nln_areas': nln_areas,
            'normalization_summary': summary
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_file}\n")

if __name__ == "__main__":
    main()
