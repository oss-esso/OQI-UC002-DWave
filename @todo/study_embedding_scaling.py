#!/usr/bin/env python3
"""
Study Embedding Scaling with Problem Size

This script studies how embedding time and success rate scale with problem size
WITHOUT running actual QPU annealing (no QPU time billed).

Key metrics:
- Embedding time vs problem size
- Embedding success rate
- Number of physical qubits used
- Chain lengths
- Logical-to-physical qubit ratio

Based on the Benders master problem BQM structure.
"""
import time
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import statistics

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("="*80)
print("EMBEDDING SCALING STUDY")
print("="*80)
print("This script studies embedding WITHOUT running QPU annealing")
print("No QPU time will be billed!")
print("="*80)

# Step 1: Imports
print("\n[1/4] Importing libraries...")
import_start = time.time()

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, cqm_to_bqm
from dimod.generators import combinations
from dwave.system import DWaveSampler
import minorminer
import networkx as nx

print(f"      ✅ Imports done in {time.time() - import_start:.2f}s")

# Configuration
PROBLEM_SIZES = [5, 10, 15, 20, 25, 30]  # Number of farms to test
N_FOODS = 27  # Fixed number of foods (as in real problem)
EMBEDDING_ATTEMPTS = 3  # Attempts per problem size
EMBEDDING_TIMEOUT = 300  # 5 minutes max per attempt

# Get token
TOKEN = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')


def build_benders_master_bqm(n_farms: int, n_foods: int = 27) -> Tuple[BinaryQuadraticModel, Dict]:
    """
    Build BQM similar to Benders master problem.
    
    This mimics the CQM→BQM conversion used in decomposition_benders_qpu.py
    """
    # Create farm and food names
    farms = [f"Farm_{i}" for i in range(n_farms)]
    foods = [f"Food_{i}" for i in range(n_foods)]
    
    # Build CQM (same as Benders master)
    cqm = ConstrainedQuadraticModel()
    
    # Variables: Y[f,c] binary
    Y = {}
    for farm in farms:
        for food in foods:
            var_name = f"Y_{farm}_{food}"
            Y[(farm, food)] = Binary(var_name)
            cqm.add_variable('BINARY', var_name)
    
    # Simple objective: maximize selections
    objective = sum(Y[(f, c)] for f in farms for c in foods)
    cqm.set_objective(-objective)
    
    # Food group constraints (simplified - 3 groups)
    n_groups = 3
    foods_per_group = n_foods // n_groups
    
    for g in range(n_groups):
        group_foods = foods[g * foods_per_group:(g + 1) * foods_per_group]
        min_foods = 2  # Require at least 2 from each group
        
        total = sum(Y[(f, c)] for f in farms for c in group_foods)
        cqm.add_constraint(total >= min_foods, label=f"FG_Min_{g}")
    
    # Convert to BQM
    bqm, _ = cqm_to_bqm(cqm)
    
    info = {
        'n_farms': n_farms,
        'n_foods': n_foods,
        'n_logical_vars': len(bqm.variables),
        'n_linear': len(bqm.linear),
        'n_quadratic': len(bqm.quadratic),
        'density': 2 * len(bqm.quadratic) / (len(bqm.variables) * (len(bqm.variables) - 1)) if len(bqm.variables) > 1 else 0
    }
    
    return bqm, info


def get_pegasus_graph(sampler: DWaveSampler) -> nx.Graph:
    """Get the hardware graph from the sampler."""
    edges = sampler.edgelist
    nodes = sampler.nodelist
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    return G


def study_embedding(bqm: BinaryQuadraticModel, target_graph: nx.Graph, 
                    timeout: int = 300, attempts: int = 3) -> Dict:
    """
    Study embedding characteristics without running QPU.
    
    Returns embedding statistics.
    """
    # Get source graph from BQM
    source_edges = list(bqm.quadratic.keys())
    source_nodes = list(bqm.variables)
    
    source_graph = nx.Graph()
    source_graph.add_nodes_from(source_nodes)
    source_graph.add_edges_from(source_edges)
    
    results = {
        'attempts': [],
        'success_count': 0,
        'fail_count': 0,
        'best_embedding': None,
        'best_chain_stats': None
    }
    
    for attempt in range(attempts):
        print(f"        Attempt {attempt + 1}/{attempts}...", end=" ", flush=True)
        
        attempt_start = time.time()
        
        try:
            # Find embedding using minorminer directly
            embedding = minorminer.find_embedding(
                source_graph.edges(),
                target_graph.edges(),
                timeout=timeout,
                verbose=0
            )
            
            embed_time = time.time() - attempt_start
            
            if embedding:
                # Analyze embedding
                chain_lengths = [len(chain) for chain in embedding.values()]
                physical_qubits = sum(chain_lengths)
                
                attempt_result = {
                    'success': True,
                    'time': embed_time,
                    'physical_qubits': physical_qubits,
                    'chain_lengths': {
                        'min': min(chain_lengths),
                        'max': max(chain_lengths),
                        'mean': statistics.mean(chain_lengths),
                        'median': statistics.median(chain_lengths),
                        'stdev': statistics.stdev(chain_lengths) if len(chain_lengths) > 1 else 0
                    },
                    'qubit_ratio': physical_qubits / len(source_nodes)
                }
                
                results['success_count'] += 1
                
                # Track best embedding (shortest max chain)
                if results['best_embedding'] is None or attempt_result['chain_lengths']['max'] < results['best_chain_stats']['max']:
                    results['best_embedding'] = embedding
                    results['best_chain_stats'] = attempt_result['chain_lengths']
                
                print(f"✓ ({embed_time:.1f}s) - {physical_qubits} qubits, max chain: {max(chain_lengths)}")
            else:
                results['fail_count'] += 1
                attempt_result = {
                    'success': False,
                    'time': embed_time,
                    'error': 'Empty embedding returned'
                }
                print(f"✗ ({embed_time:.1f}s) - Empty embedding")
                
        except Exception as e:
            embed_time = time.time() - attempt_start
            results['fail_count'] += 1
            attempt_result = {
                'success': False,
                'time': embed_time,
                'error': str(e)
            }
            print(f"✗ ({embed_time:.1f}s) - {e}")
        
        results['attempts'].append(attempt_result)
    
    return results


def main():
    print("\n[2/4] Connecting to D-Wave (for hardware graph only)...")
    connect_start = time.time()
    
    sampler = DWaveSampler(token=TOKEN)
    target_graph = get_pegasus_graph(sampler)
    
    print(f"      Solver: {sampler.solver.name}")
    print(f"      Qubits: {len(target_graph.nodes())}")
    print(f"      Couplers: {len(target_graph.edges())}")
    print(f"      ✅ Connected in {time.time() - connect_start:.2f}s")
    
    print("\n[3/4] Studying embedding scaling...")
    print(f"      Problem sizes: {PROBLEM_SIZES} farms")
    print(f"      Foods per problem: {N_FOODS}")
    print(f"      Attempts per size: {EMBEDDING_ATTEMPTS}")
    print(f"      Timeout per attempt: {EMBEDDING_TIMEOUT}s")
    
    all_results = {}
    
    for n_farms in PROBLEM_SIZES:
        print(f"\n{'─'*80}")
        print(f"Testing {n_farms} farms × {N_FOODS} foods")
        print(f"{'─'*80}")
        
        # Build BQM
        print(f"    Building BQM...", end=" ", flush=True)
        bqm_start = time.time()
        bqm, bqm_info = build_benders_master_bqm(n_farms, N_FOODS)
        print(f"✓ ({time.time() - bqm_start:.2f}s)")
        
        print(f"    BQM Stats:")
        print(f"      Logical variables: {bqm_info['n_logical_vars']}")
        print(f"      Quadratic terms: {bqm_info['n_quadratic']}")
        print(f"      Graph density: {bqm_info['density']:.4f}")
        
        # Study embedding
        print(f"    Finding embeddings:")
        embed_results = study_embedding(
            bqm, target_graph, 
            timeout=EMBEDDING_TIMEOUT, 
            attempts=EMBEDDING_ATTEMPTS
        )
        
        # Summarize
        success_rate = embed_results['success_count'] / EMBEDDING_ATTEMPTS * 100
        
        all_results[n_farms] = {
            'bqm_info': bqm_info,
            'embedding_results': embed_results,
            'success_rate': success_rate
        }
        
        print(f"\n    Summary for {n_farms} farms:")
        print(f"      Success rate: {embed_results['success_count']}/{EMBEDDING_ATTEMPTS} ({success_rate:.0f}%)")
        
        if embed_results['best_chain_stats']:
            stats = embed_results['best_chain_stats']
            print(f"      Best embedding:")
            print(f"        Max chain length: {stats['max']}")
            print(f"        Mean chain length: {stats['mean']:.1f}")
            print(f"        Physical qubits: ~{int(stats['mean'] * bqm_info['n_logical_vars'])}")
    
    # Final summary
    print("\n" + "="*80)
    print("[4/4] SCALING SUMMARY")
    print("="*80)
    
    print(f"\n{'Farms':<8} {'Vars':<8} {'Quad':<10} {'Density':<10} {'Success':<10} {'Max Chain':<12} {'Phys Qubits':<12}")
    print("-"*80)
    
    for n_farms, data in all_results.items():
        bqm = data['bqm_info']
        embed = data['embedding_results']
        
        if embed['best_chain_stats']:
            max_chain = embed['best_chain_stats']['max']
            phys_qubits = int(embed['best_chain_stats']['mean'] * bqm['n_logical_vars'])
        else:
            max_chain = "FAILED"
            phys_qubits = "N/A"
        
        print(f"{n_farms:<8} {bqm['n_logical_vars']:<8} {bqm['n_quadratic']:<10} {bqm['density']:<10.4f} {data['success_rate']:<10.0f}% {str(max_chain):<12} {str(phys_qubits):<12}")
    
    # Save results
    output_file = f"embedding_scaling_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert results for JSON (remove embedding dict which has non-serializable keys)
    json_results = {}
    for n_farms, data in all_results.items():
        json_results[str(n_farms)] = {
            'bqm_info': data['bqm_info'],
            'success_rate': data['success_rate'],
            'embedding_attempts': [
                {k: v for k, v in attempt.items() if k != 'embedding'}
                for attempt in data['embedding_results']['attempts']
            ],
            'best_chain_stats': data['embedding_results']['best_chain_stats']
        }
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'solver': sampler.solver.name,
                'n_foods': N_FOODS,
                'embedding_attempts': EMBEDDING_ATTEMPTS,
                'embedding_timeout': EMBEDDING_TIMEOUT
            },
            'results': json_results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find max feasible size
    max_feasible = 0
    for n_farms, data in all_results.items():
        if data['success_rate'] >= 50:  # At least 50% success rate
            max_feasible = n_farms
    
    print(f"\n  Max feasible problem size (≥50% success): {max_feasible} farms")
    
    if max_feasible > 0:
        data = all_results[max_feasible]
        if data['embedding_results']['best_chain_stats']:
            max_chain = data['embedding_results']['best_chain_stats']['max']
            if max_chain > 10:
                print(f"  ⚠️  Long chains ({max_chain}) may reduce solution quality")
                print(f"      Consider using chain_strength parameter")
    
    # Scaling analysis
    print(f"\n  Scaling observations:")
    print(f"    - Variables scale as O(n_farms × n_foods) = O(n)")
    print(f"    - Quadratic terms scale as O(n²) due to CQM→BQM slack variables")
    print(f"    - Embedding difficulty increases rapidly with density")
    
    if max_feasible < 25:
        print(f"\n  💡 For larger problems (>{max_feasible} farms):")
        print(f"      - Use Hybrid solver (LeapHybridCQMSampler)")
        print(f"      - Or decompose problem further")
        print(f"      - Or reduce constraint complexity")
    
    print("\n" + "="*80)
    print("STUDY COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
