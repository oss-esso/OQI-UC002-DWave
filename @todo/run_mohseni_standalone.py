#!/usr/bin/env python
"""
Standalone version of Mohseni et al.'s coalition formation
Minimal dependencies - only uses DWaveCliqueSampler and their data
"""

import os
import pickle
import time
import copy
import numpy as np

# Set API token before importing DWave
os.environ['DWAVE_API_TOKEN'] = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'

from dwave.system.samplers import DWaveCliqueSampler
import dimod

print("="*80)
print("MOHSENI ET AL. COALITION FORMATION - STANDALONE VERSION")
print("="*80)
print()

# Load their data
data_path = '/Users/edoardospigarolo/Documents/OQI-UC002-DWave/external/Benchmark_Naeimeh/Data/data_forfig2.pkl'
print(f"Loading data: {data_path}")
with open(data_path, 'rb') as f:
    data = pickle.load(f)
print(f"✓ Data loaded: {sorted(data.keys())} agent sizes available\n")

# Helper functions (copied from their utils.py)
def transform(naeimeh_graph):
    """Transform their graph format to edges dict"""
    edges = {}
    for key in naeimeh_graph.keys():
        i, j = key.split(",")
        i, j = int(i)-1, int(j)-1
        edges[(i,j)] = naeimeh_graph[key]
    return edges

def add(Q, i, j, v):
    """Add value to QUBO matrix"""
    if (i,j) not in Q:
        Q[(i,j)] = v
    else:
        Q[(i,j)] += v

def value(coalition, edges):
    """Calculate coalition value"""
    v = 0
    for i in coalition:
        for j in coalition:
            if i < j:
                v += edges.get((i,j), 0)
    return v

def split_coalition(coalition, edges, sampler):
    """Split coalition into two groups using DWave"""
    # Build QUBO for graph bisection
    Q = {}
    for i in range(len(coalition)):
        for j in range(len(coalition)):
            if i < j:
                edge_weight = edges.get((coalition[i], coalition[j]), 0)
                add(Q, i, i, edge_weight)
                add(Q, j, j, edge_weight)
                add(Q, i, j, -2*edge_weight)
    
    # Convert to BQM and solve
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampleset = sampler.sample(bqm, num_reads=100)
    answer = sampleset.first.sample
    
    # Split based on solution
    c1 = [coalition[k] for k in range(len(coalition)) if answer.get(k, 0) == 1]
    c2 = [coalition[k] for k in range(len(coalition)) if answer.get(k, 0) == 0]
    
    return c1, c2, sampleset.info

def solve_coalition_formation(num_agents, edges, sampler):
    """
    Hierarchical coalition formation (Mohseni et al.'s algorithm)
    """
    coalitions = [list(range(num_agents))]
    all_qpu_info = []
    num_splits = 0
    
    for iteration in range(num_agents):
        new_coalitions = copy.deepcopy(coalitions)
        improved = False
        
        for c in coalitions:
            if len(c) > 1:
                # Try to split this coalition
                c1, c2, qpu_info = split_coalition(c, edges, sampler)
                all_qpu_info.append(qpu_info)
                
                # Check if split improves total value
                val_c = value(c, edges)
                val_c1 = value(c1, edges)
                val_c2 = value(c2, edges)
                
                if val_c1 + val_c2 > val_c:
                    # Accept split
                    new_coalitions.remove(c)
                    if c1: new_coalitions.append(c1)
                    if c2: new_coalitions.append(c2)
                    improved = True
                    num_splits += 1
        
        if not improved:
            break
        
        coalitions = new_coalitions
    
    return coalitions, all_qpu_info

# Configuration (use only available sizes from their data)
agent_list = [12, 16]  # Available: [4, 8, 12, 16, 20, 24, 28, 30, 35, 40, 45, 50]
num_instances = 2

print(f"Configuration:")
print(f"  Agent sizes: {agent_list}")
print(f"  Instances per size: {num_instances}")
print(f"  Total problems: {len(agent_list) * num_instances}")
print()

# Initialize sampler
print("Connecting to QPU...")
sampler = DWaveCliqueSampler()
print("✓ Connected\n")

# Run tests
all_results = []

for agent_count in agent_list:
    print(f"{'='*80}")
    print(f"AGENT COUNT: {agent_count}")
    print(f"{'='*80}\n")
    
    for inst_idx in range(num_instances):
        print(f"  Instance {inst_idx+1}/{num_instances}:")
        
        try:
            # Load graph
            naeimeh_graph = data[agent_count][inst_idx]
            edges = transform(naeimeh_graph)
            print(f"    Graph: {len(edges)} edges")
            
            # Solve
            start_time = time.time()
            coalitions, qpu_info = solve_coalition_formation(agent_count, edges, sampler)
            total_time = time.time() - start_time
            
            # Calculate value
            total_value = np.sum([value(c, edges) for c in coalitions])
            
            # Extract QPU timing
            qpu_time = 0
            if qpu_info:
                timing = qpu_info[0].get('timing', {})
                qpu_time = timing.get('qpu_access_time', 0) / 1e6
            
            print(f"    ✓ Success!")
            print(f"      Coalitions: {len(coalitions)} (sizes: {[len(c) for c in coalitions]})")
            print(f"      Total value: {total_value:.4f}")
            print(f"      Wall time: {total_time:.3f}s")
            print(f"      QPU calls: {len(qpu_info)}")
            if qpu_time > 0:
                print(f"      First QPU time: {qpu_time:.3f}s")
            print()
            
            all_results.append({
                'num_agents': agent_count,
                'instance': inst_idx,
                'coalitions': coalitions,
                'coalition_sizes': [len(c) for c in coalitions],
                'value': total_value,
                'time': total_time,
                'qpu_calls': len(qpu_info),
                'first_qpu_time': qpu_time
            })
            
        except Exception as e:
            print(f"    ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()

# Summary
print(f"\n{'='*80}")
print(f"FINAL RESULTS")
print(f"{'='*80}\n")

if all_results:
    print(f"Successful: {len(all_results)}/{len(agent_list) * num_instances}\n")
    
    print("Per-instance:")
    for r in all_results:
        print(f"  n={r['num_agents']}, inst={r['instance']}: "
              f"{len(r['coalitions'])} coalitions {r['coalition_sizes']}, "
              f"value={r['value']:.4f}, time={r['time']:.3f}s, "
              f"QPU calls={r['qpu_calls']}")
    
    print(f"\nAverages:")
    print(f"  Coalitions: {np.mean([len(r['coalitions']) for r in all_results]):.1f}")
    print(f"  Value: {np.mean([r['value'] for r in all_results]):.4f}")
    print(f"  Time: {np.mean([r['time'] for r in all_results]):.3f}s")
    print(f"  QPU calls: {np.mean([r['qpu_calls'] for r in all_results]):.1f}")

print(f"\n{'='*80}")
print("✅ Complete!")
print("="*80)
