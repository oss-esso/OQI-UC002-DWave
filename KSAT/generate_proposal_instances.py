"""
Generate All Instances for Quantum Reserve Design Proposal

This script creates all instances referenced in the LaTeX proposal document,
including:
1. Small, medium, large real-world conservation instances
2. QAOA benchmark instances at various sizes
3. Comparison metrics and reports
4. Export in multiple formats (JSON, DIMACS, CSV)

Run this to generate all data for the proposal appendix.
"""

import json
import csv
import numpy as np
from datetime import datetime
from pathlib import Path

from real_world_instance import (
    create_solvable_real_world_instance,
    create_real_world_instance,
    MADAGASCAR_EASTERN_RAINFOREST,
    AMAZON_CORRIDOR,
    CORAL_TRIANGLE_MARINE
)
from qaoa_sat_instance import (
    generate_random_ksat,
    generate_planted_ksat,
    generate_hard_random_ksat,
    generate_qaoa_benchmark_suite
)
from hardness_metrics import compute_hardness_metrics
from instance_comparison import create_comprehensive_comparison


def generate_all_instances(output_dir: str = "proposal_instances"):
    """
    Generate all instances for the proposal
    
    Args:
        output_dir: Directory to save all outputs
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*70)
    print("GENERATING ALL INSTANCES FOR PROPOSAL")
    print("="*70)
    
    all_instances = {}
    all_metrics = {}
    
    # === 1. REAL-WORLD CONSERVATION INSTANCES ===
    print("\n1. Real-World Conservation Instances")
    print("-" * 70)
    
    # Small (QAOA-compatible)
    print("   Creating Small (6×6) instance...")
    small = create_solvable_real_world_instance('small', seed=42)
    all_instances['conservation_small'] = {
        'instance': small,
        'name': 'Madagascar Ranomafana Extension (Small)',
        'size': 'small'
    }
    
    # Medium
    print("   Creating Medium (10×10) instance...")
    medium = create_real_world_instance(MADAGASCAR_EASTERN_RAINFOREST, seed=123)
    all_instances['conservation_medium'] = {
        'instance': medium,
        'name': 'Madagascar Eastern Rainforest Corridor',
        'size': 'medium'
    }
    
    # Large
    print("   Creating Large (12×12) instance...")
    large = create_real_world_instance(AMAZON_CORRIDOR, seed=456)
    all_instances['conservation_large'] = {
        'instance': large,
        'name': 'Amazon-Cerrado Ecotone Corridor',
        'size': 'large'
    }
    
    # === 2. QAOA BENCHMARK INSTANCES ===
    print("\n2. QAOA Benchmark Instances")
    print("-" * 70)
    
    qaoa_configs = [
        (20, 3, 'small'),
        (30, 3, 'medium'),
        (50, 3, 'large')
    ]
    
    for n, k, size in qaoa_configs:
        print(f"   Creating QAOA {size} (n={n}, k={k})...")
        
        # Random
        random_inst = generate_hard_random_ksat(n=n, k=k, seed=42+n)
        all_instances[f'qaoa_random_{size}'] = {
            'instance': random_inst,
            'name': f'QAOA Random k-SAT (n={n}, k={k})',
            'size': size,
            'type': 'random'
        }
        
        # Planted
        planted_inst = generate_planted_ksat(n=n, k=k, alpha=4.27, seed=1000+n)
        all_instances[f'qaoa_planted_{size}'] = {
            'instance': planted_inst,
            'name': f'QAOA Planted k-SAT (n={n}, k={k})',
            'size': size,
            'type': 'planted'
        }
    
    # === 3. COMPUTE METRICS FOR ALL INSTANCES ===
    print("\n3. Computing Hardness Metrics")
    print("-" * 70)
    
    for key, data in all_instances.items():
        print(f"   {key}...")
        instance = data['instance']
        
        # Get clauses (different for conservation vs QAOA)
        if 'conservation' in key:
            # Conservation instance - would need SAT encoding
            # For now, use estimated values
            n_vars = instance.num_sites * 2
            n_clauses = instance.num_sites * 3
            all_metrics[key] = {
                'n_variables': n_vars,
                'n_clauses': n_clauses,
                'alpha': n_clauses / n_vars,
                'note': 'Estimated (requires SAT encoder)'
            }
        else:
            # QAOA instance
            metrics = compute_hardness_metrics(instance.n, instance.clauses)
            all_metrics[key] = {
                'n_variables': metrics.n_variables,
                'n_clauses': metrics.n_clauses,
                'alpha': metrics.alpha,
                'hardness_score': metrics.hardness_score,
                'expected_difficulty': metrics.expected_difficulty,
                'vcg_density': metrics.vcg_density,
                'vcg_clustering': metrics.vcg_clustering,
                'pos_neg_ratio': metrics.positive_negative_ratio
            }
    
    # === 4. EXPORT TO FILES ===
    print("\n4. Exporting to Files")
    print("-" * 70)
    
    # Summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'instances': {},
        'metrics': all_metrics
    }
    
    for key, data in all_instances.items():
        instance = data['instance']
        
        # Conservation instance
        if 'conservation' in key:
            summary['instances'][key] = {
                'name': data['name'],
                'type': 'conservation',
                'size': data['size'],
                'num_sites': instance.num_sites,
                'num_species': instance.num_species,
                'budget': float(instance.budget),
                'total_cost': float(np.sum(instance.costs)),
                'num_edges': len(instance.adjacency),
                'species_names': instance.species_names,
                'targets': instance.targets.tolist()
            }
            
            # Export to CSV
            csv_file = output_path / f"{key}.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Site', 'Cost'] + instance.species_names)
                for i in range(instance.num_sites):
                    row = [instance.site_names[i], instance.costs[i]]
                    row.extend(instance.presence[i, :].tolist())
                    writer.writerow(row)
            print(f"   Exported {csv_file}")
        
        # QAOA instance
        else:
            summary['instances'][key] = {
                'name': data['name'],
                'type': data.get('type', 'random'),
                'size': data['size'],
                'n_variables': instance.n,
                'n_clauses': instance.m,
                'k': instance.k,
                'alpha': instance.alpha,
                'is_planted': instance.is_planted
            }
            
            # Export to DIMACS
            dimacs_file = output_path / f"{key}.cnf"
            with open(dimacs_file, 'w') as f:
                f.write(instance.to_dimacs_cnf())
            print(f"   Exported {dimacs_file}")
    
    # Save summary JSON
    summary_file = output_path / "instance_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Exported {summary_file}")
    
    # === 5. CREATE COMPARISON TABLE ===
    print("\n5. Creating Comparison Table")
    print("-" * 70)
    
    table_file = output_path / "comparison_table.csv"
    with open(table_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Instance', 'Type', 'Size', 'Variables', 'Clauses', 'Alpha', 
            'Hardness', 'Difficulty'
        ])
        
        for key in sorted(all_instances.keys()):
            inst_data = summary['instances'][key]
            metrics = all_metrics.get(key, {})
            
            if 'conservation' in key:
                row = [
                    inst_data['name'],
                    'Conservation',
                    inst_data['size'],
                    metrics.get('n_variables', 'N/A'),
                    metrics.get('n_clauses', 'N/A'),
                    f"{metrics.get('alpha', 0):.2f}",
                    metrics.get('hardness_score', 'N/A'),
                    metrics.get('expected_difficulty', 'N/A')
                ]
            else:
                row = [
                    inst_data['name'],
                    f"QAOA {inst_data['type']}",
                    inst_data['size'],
                    inst_data['n_variables'],
                    inst_data['n_clauses'],
                    f"{inst_data['alpha']:.3f}",
                    f"{metrics.get('hardness_score', 0):.1f}",
                    metrics.get('expected_difficulty', 'N/A')
                ]
            
            writer.writerow(row)
    
    print(f"   Exported {table_file}")
    
    # === 6. FINAL SUMMARY ===
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nGenerated {len(all_instances)} instances:")
    print(f"  • {sum(1 for k in all_instances if 'conservation' in k)} conservation instances")
    print(f"  • {sum(1 for k in all_instances if 'qaoa' in k)} QAOA benchmark instances")
    print(f"\nFiles saved to: {output_path.absolute()}")
    print(f"  • {summary_file.name} - Complete summary (JSON)")
    print(f"  • {table_file.name} - Comparison table (CSV)")
    print(f"  • *.csv - Conservation instance data")
    print(f"  • *.cnf - QAOA instances (DIMACS format)")
    
    print("\n" + "="*70)
    print("✓ ALL INSTANCES GENERATED SUCCESSFULLY")
    print("="*70)
    
    return summary


if __name__ == "__main__":
    summary = generate_all_instances()
    
    print("\nQuick Stats:")
    print("-" * 70)
    for key, data in summary['instances'].items():
        if 'conservation' in key:
            print(f"{data['name']:50} | {data['num_sites']:3} sites | {data['num_species']:2} species")
        else:
            print(f"{data['name']:50} | {data['n_variables']:3} vars  | {data['n_clauses']:3} clauses")
