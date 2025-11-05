"""
Analysis script to calculate area normalization for LQ objectives.

This script:
1. Loads all LQ benchmark results
2. Extracts the total allocated area for each configuration
3. Calculates normalized objectives (objective / total_area)
4. Provides recommendations for updating the solver
"""

import json
import os
from pathlib import Path

def analyze_lq_results():
    """Analyze LQ benchmark results and calculate normalization factors."""
    
    benchmark_dir = Path(__file__).parent / "Benchmarks" / "LQ"
    configs = [5, 19, 72, 279]
    
    print("\n" + "="*80)
    print("LQ OBJECTIVE NORMALIZATION ANALYSIS")
    print("="*80)
    
    results = {
        'PuLP': {},
        'Pyomo': {},
        'DWave': {}
    }
    
    # Load results for each solver and configuration
    for solver in ['PuLP', 'Pyomo', 'DWave']:
        solver_dir = benchmark_dir / solver
        if not solver_dir.exists():
            print(f"Warning: {solver} directory not found")
            continue
        
        for config in configs:
            config_file = solver_dir / f"config_{config}_run_1.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    results[solver][config] = data
            else:
                print(f"Warning: {config_file} not found")
    
    # Analyze each configuration
    print("\n" + "-"*80)
    print(f"{'Config':<10} {'Solver':<10} {'Objective':<15} {'Total Area':<15} {'Normalized':<15}")
    print("-"*80)
    
    normalization_data = {}
    
    for config in configs:
        n_farms = config * 27
        normalization_data[config] = {}
        
        for solver in ['PuLP', 'Pyomo', 'DWave']:
            if config not in results[solver]:
                continue
            
            data = results[solver][config]
            result = data.get('result', {})
            
            # Get objective value
            obj_value = result.get('objective_value', None)
            if obj_value is None:
                continue
            
            # Calculate total area from solution
            total_area = 0
            
            if solver == 'DWave':
                # DWave stores areas differently
                areas = result.get('areas', {})
                for area_val in areas.values():
                    if area_val is not None:
                        total_area += area_val
            else:
                # PuLP and Pyomo store areas as nested dict
                areas = result.get('areas', {})
                for farm_areas in areas.values():
                    if isinstance(farm_areas, dict):
                        for area_val in farm_areas.values():
                            if area_val is not None and area_val > 1e-6:  # Skip near-zero values
                                total_area += area_val
                    elif farm_areas is not None:
                        total_area += farm_areas
            
            # Calculate normalized objective
            normalized_obj = obj_value / total_area if total_area > 0 else 0
            
            normalization_data[config][solver] = {
                'objective': obj_value,
                'total_area': total_area,
                'normalized': normalized_obj
            }
            
            print(f"{n_farms:<10} {solver:<10} {obj_value:<15.4f} {total_area:<15.2f} {normalized_obj:<15.6f}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("NORMALIZATION SUMMARY")
    print("="*80)
    
    for config in configs:
        n_farms = config * 27
        print(f"\nConfiguration: {n_farms} farms (config_{config})")
        
        if config not in normalization_data or not normalization_data[config]:
            print("  No data available")
            continue
        
        # Get average area across solvers for this config
        areas = [data['total_area'] for data in normalization_data[config].values() 
                if data['total_area'] > 0]
        
        if areas:
            avg_area = sum(areas) / len(areas)
            print(f"  Average total area: {avg_area:.2f}")
            print(f"  Area range: {min(areas):.2f} - {max(areas):.2f}")
            
            # Show how objectives compare when normalized
            print(f"\n  Original objectives:")
            for solver, data in normalization_data[config].items():
                print(f"    {solver}: {data['objective']:.4f}")
            
            print(f"\n  Normalized objectives (obj/area):")
            for solver, data in normalization_data[config].items():
                print(f"    {solver}: {data['normalized']:.6f}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
To normalize the LQ objective by total area:

1. In solver_runner_LQ.py:
   - After solving, calculate total_area = sum of all A[f,c] values
   - Divide objective_value by total_area
   - Store as 'normalized_objective' in results

2. In benchmark_scalability_LQ.py:
   - Update result extraction to use normalized_objective
   - Update plots to show "Normalized Objective Value"

3. Benefits:
   - Makes objectives comparable across different problem sizes
   - Represents "average value per unit area"
   - Better interpretability for decision makers

4. Implementation approach:
   - Keep both raw and normalized objectives in results
   - Use normalized for comparisons and plotting
   - Document the normalization clearly
""")
    
    return normalization_data

def main():
    """Run the analysis."""
    normalization_data = analyze_lq_results()
    
    # Save analysis results
    output_file = Path(__file__).parent / "lq_normalization_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(normalization_data, f, indent=2)
    
    print(f"\n✓ Analysis saved to: {output_file}\n")

if __name__ == "__main__":
    main()
