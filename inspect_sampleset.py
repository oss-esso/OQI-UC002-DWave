#!/usr/bin/env python3
"""Investigate sampleset structure."""

import pickle
import os

sampleset_dir = 'qpu_samplesets_all'
files = [f for f in os.listdir(sampleset_dir) if 'micro_25' in f]
print(f'Files for micro_25: {len(files)}')
for f in files:
    print(f'  {f}')

# Load both
for f in files:
    print(f'\n{"="*60}')
    print(f'File: {f}')
    print('='*60)
    
    with open(os.path.join(sampleset_dir, f), 'rb') as fp:
        data = pickle.load(fp)
    
    print(f'Cluster farms: {data["cluster_farms"]}')
    print(f'Iteration: {data["iteration"]}')
    print(f'Cluster idx: {data["cluster_idx"]}')
    
    # Check var_map
    var_map = data['var_map']
    print(f'\nVar map entries: {len(var_map)}')
    print('Sample keys:')
    for k, v in list(var_map.items())[:10]:
        print(f'  {k} -> {v}')
    
    # Get sample
    ss = data['sampleset']
    sample = ss.first.sample
    print(f'\nSample variables with value=1:')
    count = 0
    for k, v in sample.items():
        if v == 1:
            print(f'  {k}')
            count += 1
    print(f'Total selected: {count}')
