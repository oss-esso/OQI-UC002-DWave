#!/usr/bin/env python
"""
Run Mohseni et al.'s original code with minimal modifications
Uses their exact Main.ipynb logic but with fewer instances
"""

import sys
import os
import pickle
import time
import numpy as np

# Add their code to path
sys.path.insert(0, '/Users/edoardospigarolo/Documents/OQI-UC002-DWave/external/Benchmark_Naeimeh/Solvers_Benchmarking')

# Set API token before importing dwave modules
os.environ['DWAVE_API_TOKEN'] = '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551'

import utils
import Dwave
from dwave.system.samplers import DWaveCliqueSampler

# Monkey-patch the global sampler into utils module (their code expects this)
utils.sampler = DWaveCliqueSampler()

print("="*80)
print("MOHSENI ET AL. ORIGINAL CODE - MINIMAL RUN")
print("Running their exact Main.ipynb logic with 2 agent sizes × 2 instances")
print("="*80)
print()

# Load their data
data_path = '/Users/edoardospigarolo/Documents/OQI-UC002-DWave/external/Benchmark_Naeimeh/Data/data_forfig2.pkl'
print(f"Loading data: {data_path}")
try:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Data loaded")
    print(f"  Available sizes: {sorted(data.keys())}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print()

# Minimal configuration (to save QPU time)
agent_list = [10, 12]  # Just 2 sizes
num_instances = 2      # Just 2 instances per size

print(f"Configuration:")
print(f"  Agent sizes: {agent_list}")
print(f"  Instances per size: {num_instances}")
print(f"  Total QPU calls: {len(agent_list)} × {num_instances} = {len(agent_list) * num_instances}")
print()

# Initialize storage (following their exact structure)
timeout = np.full((len(agent_list), num_instances), 10)
Dwave_inf = [[[] for _ in range(num_instances)] for _ in range(len(agent_list))]
rest_inf_dwave = [[[] for _ in range(num_instances)] for _ in range(len(agent_list))]
Value_Dwave = []
run_time_Dwave = []

print("Starting benchmark...\n")

# Main loop (their exact logic from Main.ipynb cell 2)
for num, value_agent in enumerate(agent_list):
    print(f"{'='*80}")
    print(f"Agent count: {value_agent} ({num+1}/{len(agent_list)})")
    print(f"{'='*80}")
    
    if value_agent not in data:
        print(f"  ✗ No data for size {value_agent}")
        continue
    
    for idx in range(num_instances):
        print(f"\n  Instance {idx+1}/{num_instances}:")
        
        try:
            # Get graph
            naeimeh_graph = data[value_agent][idx]
            edges = utils.transform(naeimeh_graph)
            print(f"    Graph loaded: {len(edges)} edges")
            
            # Run DWave (their exact code)
            print(f"    Running DWave coalition formation...")
            dwave_start_time = time.time()
            coalitions = Dwave.solve(
                value_agent, 
                edges, 
                timeout[num][idx],
                Dwave_inf,
                num,
                idx,
                rest_inf_dwave
            )
            dwave_end_time = time.time()
            
            # Calculate value
            value_Dwave = np.sum([utils.value(c, edges) for c in coalitions])
            Value_Dwave.append(value_Dwave)
            
            total_time = dwave_end_time - dwave_start_time
            run_time_Dwave.append(total_time)
            
            # Extract QPU timing if available
            qpu_info = Dwave_inf[num][idx][0] if Dwave_inf[num][idx] else {}
            timing = qpu_info.get('timing', {})
            qpu_time = timing.get('qpu_access_time', 0) / 1e6 if timing else 0
            
            print(f"    ✓ Success!")
            print(f"      Coalitions: {len(coalitions)} (sizes: {[len(c) for c in coalitions]})")
            print(f"      Value: {value_Dwave:.4f}")
            print(f"      Total time: {total_time:.3f}s")
            if qpu_time > 0:
                print(f"      First QPU time: {qpu_time:.3f}s")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

print(f"\n{'='*80}")
print(f"RESULTS SUMMARY")
print(f"{'='*80}\n")

if Value_Dwave:
    print(f"Successful runs: {len(Value_Dwave)}/{len(agent_list) * num_instances}")
    print(f"\nPer-instance results:")
    idx = 0
    for num, value_agent in enumerate(agent_list):
        for inst in range(min(num_instances, len(data.get(value_agent, [])))):
            if idx < len(Value_Dwave):
                print(f"  n={value_agent}, instance={inst}: "
                      f"value={Value_Dwave[idx]:.4f}, time={run_time_Dwave[idx]:.3f}s")
                idx += 1
    
    print(f"\nAverages:")
    print(f"  Value: {np.mean(Value_Dwave):.4f} ± {np.std(Value_Dwave):.4f}")
    print(f"  Runtime: {np.mean(run_time_Dwave):.3f}s ± {np.std(run_time_Dwave):.3f}s")
    
    # Analyze QPU usage
    total_qpu_calls = sum(len(Dwave_inf[i][j]) for i in range(len(agent_list)) 
                          for j in range(num_instances))
    print(f"  Total QPU calls made: {total_qpu_calls}")
    
else:
    print("No successful runs!")

print(f"\n{'='*80}")
print("✅ Benchmark complete!")
print("="*80)
