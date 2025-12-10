#!/usr/bin/env python
"""
Minimal run of Mohseni et al.'s code with 2 small instances
Based on their Main.ipynb but with minimal data to conserve QPU access
"""

import sys
import os
import pickle
import time
import numpy as np

# Add their code to path
sys.path.insert(0, '/Users/edoardospigarolo/Documents/OQI-UC002-DWave/external/Benchmark_Naeimeh/Solvers_Benchmarking')

import utils
import Dwave
from dwave.system.samplers import DWaveCliqueSampler

# Set API token
os.environ['DWAVE_API_TOKEN'] = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'

print("="*80)
print("MOHSENI ET AL. CODE - MINIMAL RUN (2 instances only)")
print("="*80)
print()

# Load their data
data_path = '/Users/edoardospigarolo/Documents/OQI-UC002-DWave/external/Benchmark_Naeimeh/Data/data_forfig2.pkl'
print(f"Loading data from: {data_path}")
try:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Data loaded successfully")
    print(f"  Available agent counts: {list(data.keys())}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    sys.exit(1)

print()

# Test with just 2 small instances
agent_list = [10, 12]  # Small sizes to conserve QPU
num_instances_per_size = 2  # Just 2 instances per size

print(f"Testing {len(agent_list)} agent counts: {agent_list}")
print(f"Running {num_instances_per_size} instances per size")
print(f"Total: {len(agent_list) * num_instances_per_size} = {len(agent_list)} × {num_instances_per_size} problems")
print()

# Storage for results
all_results = []

# Initialize sampler once
print("Initializing DWaveCliqueSampler...")
try:
    sampler = DWaveCliqueSampler()
    print("✓ Connected to QPU")
    print()
except Exception as e:
    print(f"✗ Error connecting to QPU: {e}")
    sys.exit(1)

# Run tests
for agent_idx, num_agents in enumerate(agent_list):
    print(f"\n{'='*80}")
    print(f"AGENT COUNT: {num_agents}")
    print(f"{'='*80}\n")
    
    if num_agents not in data:
        print(f"✗ No data for {num_agents} agents, skipping...")
        continue
    
    for instance_idx in range(min(num_instances_per_size, len(data[num_agents]))):
        print(f"Instance {instance_idx + 1}/{num_instances_per_size} (n={num_agents}):")
        
        # Get graph instance
        naeimeh_graph = data[num_agents][instance_idx]
        edges = utils.transform(naeimeh_graph)
        
        print(f"  Graph: {len(edges)} edges")
        
        # Initialize tracking arrays (simplified from their code)
        Dwave_inf = [[[] for _ in range(1)] for _ in range(1)]
        rest_inf_dwave = [[[] for _ in range(1)] for _ in range(1)]
        
        # Run DWave coalition formation
        try:
            start_time = time.time()
            coalitions = Dwave.solve(
                num_agents, 
                edges, 
                timeout=10,  # 10 second timeout per split
                Dwave_inf=Dwave_inf,
                num=0,
                idx=0,
                rest_inf_dwave=rest_inf_dwave,
                sampler=sampler
            )
            total_time = time.time() - start_time
            
            # Calculate value
            total_value = np.sum([utils.value(c, edges) for c in coalitions])
            
            # Extract QPU info from first split if available
            qpu_info = Dwave_inf[0][0][0] if Dwave_inf[0][0] else {}
            timing = qpu_info.get('timing', {})
            qpu_time = timing.get('qpu_access_time', 0) / 1e6 if timing else 0
            
            # Count splits
            num_splits = len(coalitions) - 1
            
            print(f"  ✓ Success!")
            print(f"    Coalitions found: {len(coalitions)}")
            print(f"    Coalition sizes: {[len(c) for c in coalitions]}")
            print(f"    Total value: {total_value:.4f}")
            print(f"    Total time: {total_time:.3f}s")
            print(f"    Splits performed: {num_splits}")
            if qpu_time > 0:
                print(f"    First split QPU time: {qpu_time:.3f}s")
            
            all_results.append({
                'num_agents': num_agents,
                'instance': instance_idx,
                'coalitions': coalitions,
                'num_coalitions': len(coalitions),
                'coalition_sizes': [len(c) for c in coalitions],
                'total_value': total_value,
                'total_time': total_time,
                'num_splits': num_splits,
                'qpu_info': qpu_info,
                'success': True
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'num_agents': num_agents,
                'instance': instance_idx,
                'error': str(e),
                'success': False
            })

# Summary
print(f"\n{'='*80}")
print(f"FINAL SUMMARY")
print(f"{'='*80}\n")

successful = [r for r in all_results if r.get('success', False)]
failed = [r for r in all_results if not r.get('success', False)]

print(f"Total problems: {len(all_results)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")
print()

if successful:
    print("Successful runs:")
    for r in successful:
        print(f"  n={r['num_agents']}, instance={r['instance']}: "
              f"{r['num_coalitions']} coalitions (sizes: {r['coalition_sizes']}), "
              f"value={r['total_value']:.4f}, time={r['total_time']:.2f}s")
    
    print()
    avg_time = np.mean([r['total_time'] for r in successful])
    avg_coalitions = np.mean([r['num_coalitions'] for r in successful])
    avg_splits = np.mean([r['num_splits'] for r in successful])
    
    print(f"Averages:")
    print(f"  Time per problem: {avg_time:.3f}s")
    print(f"  Coalitions found: {avg_coalitions:.1f}")
    print(f"  Splits performed: {avg_splits:.1f}")

if failed:
    print()
    print("Failed runs:")
    for r in failed:
        print(f"  n={r['num_agents']}, instance={r['instance']}: {r.get('error', 'Unknown error')}")

print()
print("="*80)
print("✅ Test complete!")
print("="*80)
