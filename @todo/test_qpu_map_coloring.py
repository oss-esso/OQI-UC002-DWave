#!/usr/bin/env python3
"""
Test QPU Connection with Map Coloring Example

This is a direct implementation of D-Wave's map coloring example
to verify QPU connectivity and measure actual timing.

Based on D-Wave documentation example.
"""
import time
import os

print("="*80)
print("QPU CONNECTION TEST - Map Coloring (Canada)")
print("="*80)

# Step 1: Import libraries
print("\n[1/6] Importing libraries...")
import_start = time.time()

import networkx as nx
from dimod.generators import combinations
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite

print(f"      ✅ Imports done in {time.time() - import_start:.2f}s")

# Step 2: Define the problem
print("\n[2/6] Defining map coloring problem...")
problem_start = time.time()

provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE',
             'QC', 'SK', 'YT']

neighbors = [('AB', 'BC'), ('AB', 'NT'), ('AB', 'SK'), ('BC', 'NT'), ('BC', 'YT'),
             ('MB', 'NU'), ('MB', 'ON'), ('MB', 'SK'), ('NB', 'NS'), ('NB', 'QC'),
             ('NL', 'QC'), ('NT', 'NU'), ('NT', 'SK'), ('NT', 'YT'), ('ON', 'QC')]

colors = ['y', 'g', 'r', 'b']  # yellow, green, red, blue

print(f"      Provinces: {len(provinces)}")
print(f"      Borders: {len(neighbors)}")
print(f"      Colors: {len(colors)}")
print(f"      ✅ Problem defined in {time.time() - problem_start:.2f}s")

# Step 3: Build the BQM
print("\n[3/6] Building BQM...")
bqm_start = time.time()

# Constraint 1: One color per province (one-hot)
bqm_one_color = BinaryQuadraticModel('BINARY')
for province in provinces:
    variables = [province + "_" + c for c in colors]
    bqm_one_color.update(combinations(variables, 1))

# Constraint 2: Different colors for neighbors
bqm_neighbors = BinaryQuadraticModel('BINARY')
for neighbor in neighbors:
    v, u = neighbor
    interactions = [(v + "_" + c, u + "_" + c) for c in colors]
    for interaction in interactions:
        bqm_neighbors.add_quadratic(interaction[0], interaction[1], 1)

# Combine constraints
bqm = bqm_one_color + bqm_neighbors

print(f"      Variables: {len(bqm.variables)}")
print(f"      Linear terms: {len(bqm.linear)}")
print(f"      Quadratic terms: {len(bqm.quadratic)}")
print(f"      ✅ BQM built in {time.time() - bqm_start:.2f}s")

# Step 4: Connect to QPU
print("\n[4/6] Connecting to D-Wave QPU...")
connect_start = time.time()

# Get token
token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
print(f"      Token: {token[:10]}...{token[-5:]}")

try:
    base_sampler = DWaveSampler(token=token)
    sampler = EmbeddingComposite(base_sampler)
    
    # Print solver info
    solver_name = base_sampler.solver.name
    num_qubits = len(base_sampler.nodelist)
    print(f"      Solver: {solver_name}")
    print(f"      Available qubits: {num_qubits}")
    print(f"      ✅ Connected in {time.time() - connect_start:.2f}s")
except Exception as e:
    print(f"      ❌ Connection failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Sample on QPU
print("\n[5/6] Sampling on QPU...")
print("      Parameters:")
print("        - num_reads: 100")
print("        - annealing_time: 20 µs")

sample_start = time.time()

try:
    sampleset = sampler.sample(
        bqm,
        num_reads=100,
        annealing_time=20,  # microseconds
        label='Test - Map Coloring QPU'
    )
    
    wall_time = time.time() - sample_start
    print(f"      ✅ Sampling complete in {wall_time:.2f}s (wall time)")
    
except Exception as e:
    print(f"      ❌ Sampling failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 6: Extract timing info
print("\n[6/6] Extracting timing information...")

timing_info = sampleset.info.get('timing', {})

print("\n" + "="*80)
print("TIMING RESULTS (from sampleset.info['timing'])")
print("="*80)

# All timing values are in MICROSECONDS
timing_fields = [
    ('qpu_sampling_time', 'Total sampling time'),
    ('qpu_anneal_time_per_sample', 'Anneal time per sample'),
    ('qpu_readout_time_per_sample', 'Readout time per sample'),
    ('qpu_access_time', 'QPU ACCESS TIME (BILLED)'),
    ('qpu_access_overhead_time', 'QPU access overhead'),
    ('qpu_programming_time', 'Programming time'),
    ('qpu_delay_time_per_sample', 'Delay time per sample'),
    ('total_post_processing_time', 'Post-processing time'),
    ('post_processing_overhead_time', 'Post-processing overhead'),
]

print(f"\n{'Field':<35} {'Value (µs)':<15} {'Value (ms)':<15} {'Value (s)':<10}")
print("-"*80)

for field, description in timing_fields:
    value_us = timing_info.get(field, 0)
    value_ms = value_us / 1000
    value_s = value_us / 1_000_000
    
    if field == 'qpu_access_time':
        print(f"{'>>> ' + description:<35} {value_us:<15.2f} {value_ms:<15.2f} {value_s:<10.4f} <<<")
    else:
        print(f"{description:<35} {value_us:<15.2f} {value_ms:<15.2f} {value_s:<10.4f}")

print("-"*80)

# Calculate derived metrics
qpu_access_time_us = timing_info.get('qpu_access_time', 0)
qpu_access_time_ms = qpu_access_time_us / 1000
qpu_access_time_s = qpu_access_time_us / 1_000_000

print(f"\n{'SUMMARY':-^80}")
print(f"  Wall clock time:     {wall_time:.3f} seconds")
print(f"  Actual QPU time:     {qpu_access_time_ms:.2f} ms ({qpu_access_time_s:.4f} s)")
print(f"  Overhead (network):  {(wall_time - qpu_access_time_s):.3f} seconds")
print(f"  QPU efficiency:      {100 * qpu_access_time_s / wall_time:.1f}%")

# Check solution
print(f"\n{'SOLUTION CHECK':-^80}")
best = sampleset.first
print(f"  Best energy: {best.energy}")
print(f"  Num samples: {len(sampleset)}")

if best.energy > 0:
    print("  ⚠️  Solution may be infeasible (energy > 0)")
else:
    print("  ✅ Found feasible solution (energy = 0)")
    
    # Show coloring
    color_map = {}
    for province in provinces:
        for c in colors:
            if best.sample.get(f"{province}_{c}", 0) == 1:
                color_map[province] = c
                break
    
    color_names = {'y': 'Yellow', 'g': 'Green', 'r': 'Red', 'b': 'Blue'}
    print("\n  Province colorings:")
    for province, color in sorted(color_map.items()):
        print(f"    {province}: {color_names.get(color, color)}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
