#!/usr/bin/env python
"""
Minimal test of Mohseni et al.'s coalition formation approach
Uses DWaveCliqueSampler on 2 small graph instances

Based on: external/Benchmark_Naeimeh/Solvers_Benchmarking/
"""

import sys
import os
import time
import numpy as np
from dwave.system.samplers import DWaveCliqueSampler
import dimod

# Simple coalition formation (adapted from their Dwave.py)
def solve_coalition_formation(num_agents, edges, num_reads=100):
    """
    Coalition formation via hierarchical splitting
    - Start with all agents in one coalition
    - Repeatedly try to split coalitions to maximize total value
    - Use DWaveCliqueSampler for each split decision
    """
    print(f"\n{'='*80}")
    print(f"COALITION FORMATION: {num_agents} agents")
    print(f"{'='*80}\n")
    
    coalitions = [list(range(num_agents))]
    total_qpu_time = 0
    total_embedding_time = 0
    num_splits = 0
    
    sampler = DWaveCliqueSampler()
    
    for iteration in range(num_agents):
        print(f"Iteration {iteration+1}: {len(coalitions)} coalitions")
        new_coalitions = coalitions.copy()
        improved = False
        
        for c in coalitions:
            if len(c) <= 1:
                continue
            
            # Build QUBO for splitting coalition c
            Q = {}
            for i in range(len(c)):
                for j in range(len(c)):
                    if i < j:
                        edge_weight = edges.get((c[i], c[j]), 0)
                        # Diagonal terms (benefit of keeping in same coalition)
                        Q[(i, i)] = Q.get((i, i), 0) + edge_weight
                        Q[(j, j)] = Q.get((j, j), 0) + edge_weight
                        # Off-diagonal (penalty for splitting)
                        Q[(i, j)] = Q.get((i, j), 0) - 2 * edge_weight
            
            # Convert to BQM
            bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
            
            print(f"  Coalition {c[:3]}{'...' if len(c) > 3 else ''} ({len(c)} agents, {len(bqm.variables)} vars)...", end=" ")
            
            # Solve with clique sampler
            try:
                start = time.time()
                sampleset = sampler.sample(bqm, num_reads=num_reads)
                solve_time = time.time() - start
                
                # Extract timing
                timing_info = sampleset.info.get('timing', {})
                qpu_time = timing_info.get('qpu_access_time', 0) / 1e6
                embed_time = sampleset.info.get('embedding_context', {}).get('embedding_time', 0) / 1e6
                
                total_qpu_time += qpu_time
                total_embedding_time += embed_time
                
                # Get solution
                best = sampleset.first
                solution = [best.sample[i] for i in range(len(c))]
                
                # Split into two coalitions
                c1 = [c[k] for k in range(len(c)) if solution[k] == 1]
                c2 = [c[k] for k in range(len(c)) if solution[k] == 0]
                
                # Calculate values
                def coalition_value(coalition, edges):
                    value = 0
                    for i in coalition:
                        for j in coalition:
                            if i < j:
                                value += edges.get((i, j), 0)
                    return value
                
                val_c = coalition_value(c, edges)
                val_c1 = coalition_value(c1, edges)
                val_c2 = coalition_value(c2, edges)
                
                # Accept split if it improves total value
                if val_c1 + val_c2 > val_c:
                    new_coalitions.remove(c)
                    if c1: new_coalitions.append(c1)
                    if c2: new_coalitions.append(c2)
                    improved = True
                    num_splits += 1
                    print(f"SPLIT! {val_c:.2f} → {val_c1:.2f} + {val_c2:.2f} (QPU: {qpu_time:.3f}s, embed: {embed_time:.4f}s)")
                else:
                    print(f"No improvement ({val_c:.2f} ≥ {val_c1:.2f} + {val_c2:.2f})")
                    
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        if not improved:
            print(f"\nConverged after {iteration+1} iterations\n")
            break
        
        coalitions = new_coalitions
    
    # Final statistics
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"Final coalitions: {coalitions}")
    print(f"Number of coalitions: {len(coalitions)}")
    print(f"Number of splits: {num_splits}")
    print(f"Total QPU time: {total_qpu_time:.3f}s")
    print(f"Total embedding time: {total_embedding_time:.4f}s")
    print(f"Avg QPU time per split: {total_qpu_time/num_splits if num_splits > 0 else 0:.3f}s")
    print(f"Avg embedding time per split: {total_embedding_time/num_splits if num_splits > 0 else 0:.4f}s")
    print(f"{'='*80}\n")
    
    return coalitions, {
        'total_qpu_time': total_qpu_time,
        'total_embedding_time': total_embedding_time,
        'num_splits': num_splits,
        'num_coalitions': len(coalitions)
    }


# Test case 1: Small balanced graph (10 nodes)
def test_case_1():
    print("\n" + "="*80)
    print("TEST CASE 1: Balanced Graph (10 nodes)")
    print("="*80)
    
    num_agents = 10
    edges = {}
    
    # Create a balanced graph with two natural communities
    # Community 1: nodes 0-4, Community 2: nodes 5-9
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            if (i < 5 and j < 5) or (i >= 5 and j >= 5):
                # Strong intra-community edges
                edges[(i, j)] = 1.0
            else:
                # Weak inter-community edges
                edges[(i, j)] = 0.1
    
    return solve_coalition_formation(num_agents, edges, num_reads=100)


# Test case 2: Slightly larger (15 nodes)
def test_case_2():
    print("\n" + "="*80)
    print("TEST CASE 2: Unbalanced Graph (15 nodes)")
    print("="*80)
    
    num_agents = 15
    edges = {}
    
    # Create an unbalanced graph with three natural communities
    # Community 1: nodes 0-4, Community 2: nodes 5-9, Community 3: nodes 10-14
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            comm_i = i // 5
            comm_j = j // 5
            if comm_i == comm_j:
                # Strong intra-community edges
                edges[(i, j)] = np.random.uniform(0.8, 1.2)
            else:
                # Weak inter-community edges
                edges[(i, j)] = np.random.uniform(0.0, 0.2)
    
    return solve_coalition_formation(num_agents, edges, num_reads=100)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MOHSENI ET AL. COALITION FORMATION TEST")
    print("Using DWaveCliqueSampler (zero embedding overhead)")
    print("="*80)
    
    # Run test cases
    results = []
    
    print("\n→ Running Test Case 1...")
    coalitions1, stats1 = test_case_1()
    results.append(('Test 1 (10 nodes)', stats1))
    
    print("\n→ Running Test Case 2...")
    coalitions2, stats2 = test_case_2()
    results.append(('Test 2 (15 nodes)', stats2))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL TESTS")
    print("="*80)
    for name, stats in results:
        print(f"\n{name}:")
        print(f"  Coalitions found: {stats['num_coalitions']}")
        print(f"  Splits performed: {stats['num_splits']}")
        print(f"  Total QPU time: {stats['total_qpu_time']:.3f}s")
        print(f"  Total embedding time: {stats['total_embedding_time']:.4f}s")
        print(f"  Avg QPU/split: {stats['total_qpu_time']/stats['num_splits'] if stats['num_splits'] > 0 else 0:.3f}s")
        print(f"  Avg embed/split: {stats['total_embedding_time']/stats['num_splits'] if stats['num_splits'] > 0 else 0:.4f}s")
    
    print("\n" + "="*80)
    print("✅ Tests complete!")
    print("="*80 + "\n")
