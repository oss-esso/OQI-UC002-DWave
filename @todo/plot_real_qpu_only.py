#!/usr/bin/env python3
"""
Plot ONLY REAL QPU results with comparable formulations.
All use: 6 families × 3 periods × N farms = 18N variables

Data sources:
- Dec 11: statistical_comparison_20251211_180707.json (clique_decomp method)
- Dec 15: hierarchical_results_20251215_172906.json (hierarchical_qpu method)
- Dec 24: summary_20251224_095741.csv (hierarchical_qpu method)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# =============================================================================
# Load REAL QPU Data
# =============================================================================

results = []

# --- Dec 11: Statistical Comparison (clique_decomp) ---
print("Loading Dec 11 statistical comparison...")
with open('statistical_comparison_results/statistical_comparison_20251211_180707.json') as f:
    d = json.load(f)

for size, data in d['results_by_size'].items():
    n_farms = data['n_farms']
    n_vars = data['n_variables']
    
    # Ground truth (Gurobi)
    gurobi_runs = data['methods']['ground_truth']['runs']
    gurobi_obj = np.mean([r['objective'] for r in gurobi_runs if r['success']])
    gurobi_time = np.mean([r['wall_time'] for r in gurobi_runs])
    
    # Clique decomp QPU
    clique_runs = data['methods']['clique_decomp']['runs']
    clique_obj = np.mean([r['objective'] for r in clique_runs if r['success']])
    clique_time = np.mean([r['wall_time'] for r in clique_runs])
    clique_qpu = np.mean([r['qpu_time'] for r in clique_runs])
    
    gap = (gurobi_obj - clique_obj) / gurobi_obj * 100 if gurobi_obj > 0 else 0
    speedup = gurobi_time / clique_time if clique_time > 0 else 0
    
    results.append({
        'source': 'Dec 11 - Clique QPU',
        'date': '2025-12-11',
        'method': 'clique_decomp',
        'n_farms': n_farms,
        'n_foods': 6,
        'n_vars': n_vars,
        'gurobi_obj': gurobi_obj,
        'gurobi_time': gurobi_time,
        'quantum_obj': clique_obj,
        'quantum_time': clique_time,
        'qpu_time': clique_qpu,
        'gap': gap,
        'speedup': speedup,
    })

# --- Dec 15: Hierarchical QPU ---
print("Loading Dec 15 hierarchical results...")
with open('hierarchical_statistical_results/hierarchical_results_20251215_172906.json') as f:
    d = json.load(f)

for scenario_name, scenario in d.items():
    data_info = scenario['data_info']
    n_farms = data_info['n_farms']
    n_foods = data_info['n_foods']
    n_vars = data_info['n_variables']
    
    # Skip 27-food scenario (different formulation)
    if n_foods != 6:
        print(f"  Skipping {scenario_name} (n_foods={n_foods}, not comparable)")
        continue
    
    # Gurobi
    gurobi_runs = scenario['gurobi']
    gurobi_obj = np.mean([r['objective'] for r in gurobi_runs if r['success']])
    gurobi_time = np.mean([r['solve_time'] for r in gurobi_runs])
    
    # Hierarchical QPU
    qpu_runs = scenario['hierarchical_qpu']
    qpu_obj = np.mean([r['objective'] for r in qpu_runs if r['success']])
    qpu_wall = np.mean([r['wall_time'] for r in qpu_runs])
    qpu_time = np.mean([r['qpu_time'] for r in qpu_runs])
    
    gap = (gurobi_obj - qpu_obj) / gurobi_obj * 100 if gurobi_obj > 0 else 0
    speedup = gurobi_time / qpu_wall if qpu_wall > 0 else 0
    
    results.append({
        'source': 'Dec 15 - Hierarchical QPU',
        'date': '2025-12-15',
        'method': 'hierarchical_qpu',
        'n_farms': n_farms,
        'n_foods': n_foods,
        'n_vars': n_vars,
        'gurobi_obj': gurobi_obj,
        'gurobi_time': gurobi_time,
        'quantum_obj': qpu_obj,
        'quantum_time': qpu_wall,
        'qpu_time': qpu_time,
        'gap': gap,
        'speedup': speedup,
    })

# --- Dec 24: Latest Hierarchical QPU ---
print("Loading Dec 24 hierarchical results...")
df_dec24 = pd.read_csv('hierarchical_statistical_results/summary_20251224_095741.csv')

for _, row in df_dec24.iterrows():
    if row['n_foods'] != 6:
        continue
    
    gap = row['gap_percent']
    speedup = row['speedup']
    
    results.append({
        'source': 'Dec 24 - Hierarchical QPU',
        'date': '2025-12-24',
        'method': 'hierarchical_qpu',
        'n_farms': row['n_farms'],
        'n_foods': row['n_foods'],
        'n_vars': row['n_vars'],
        'gurobi_obj': row['gurobi_obj'],
        'gurobi_time': row['gurobi_time'],
        'quantum_obj': row['quantum_obj'],
        'quantum_time': row['quantum_time'],
        'qpu_time': row['qpu_time'],
        'gap': gap,
        'speedup': speedup,
    })

# Create DataFrame
df = pd.DataFrame(results)
print(f"\nLoaded {len(df)} REAL QPU data points")
print(df[['source', 'n_farms', 'n_vars', 'gap', 'qpu_time']].to_string())

# =============================================================================
# Create Plots
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = {
    'Dec 11 - Clique QPU': '#1f77b4',
    'Dec 15 - Hierarchical QPU': '#2ca02c',
    'Dec 24 - Hierarchical QPU': '#d62728',
}

markers = {
    'Dec 11 - Clique QPU': 'o',
    'Dec 15 - Hierarchical QPU': '^',
    'Dec 24 - Hierarchical QPU': 's',
}

# --- Plot 1: Gap vs Variables ---
ax1 = axes[0, 0]
for source in df['source'].unique():
    subset = df[df['source'] == source].sort_values('n_vars')
    ax1.plot(subset['n_vars'], subset['gap'], 
             marker=markers[source], color=colors[source], 
             label=source, linewidth=2, markersize=10)
ax1.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='20% target')
ax1.set_xlabel('Number of Variables')
ax1.set_ylabel('Optimality Gap (%)')
ax1.set_title('REAL QPU: Optimality Gap vs Problem Size\n(6 families × 3 periods formulation)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# --- Plot 2: QPU Time vs Variables ---
ax2 = axes[0, 1]
for source in df['source'].unique():
    subset = df[df['source'] == source].sort_values('n_vars')
    ax2.plot(subset['n_vars'], subset['qpu_time'], 
             marker=markers[source], color=colors[source], 
             label=source, linewidth=2, markersize=10)
ax2.set_xlabel('Number of Variables')
ax2.set_ylabel('QPU Access Time (seconds)')
ax2.set_title('REAL QPU: QPU Access Time vs Problem Size')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Objective Comparison ---
ax3 = axes[1, 0]
for source in df['source'].unique():
    subset = df[df['source'] == source].sort_values('n_vars')
    ax3.plot(subset['n_vars'], subset['gurobi_obj'], 
             marker='x', color=colors[source], linestyle='--',
             linewidth=1.5, markersize=8, alpha=0.7)
    ax3.plot(subset['n_vars'], subset['quantum_obj'], 
             marker=markers[source], color=colors[source], 
             label=source, linewidth=2, markersize=10)
ax3.set_xlabel('Number of Variables')
ax3.set_ylabel('Objective Value')
ax3.set_title('REAL QPU: Solution Quality\n(dashed = Gurobi, solid = QPU)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Plot 4: Speedup ---
ax4 = axes[1, 1]
for source in df['source'].unique():
    subset = df[df['source'] == source].sort_values('n_vars')
    ax4.plot(subset['n_vars'], subset['speedup'], 
             marker=markers[source], color=colors[source], 
             label=source, linewidth=2, markersize=10)
ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Break-even')
ax4.set_xlabel('Number of Variables')
ax4.set_ylabel('Speedup Factor (Gurobi time / QPU time)')
ax4.set_title('REAL QPU: Speedup vs Classical Solver')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig('REAL_QPU_RESULTS_ONLY.png', dpi=150, bbox_inches='tight')
print(f"\nSaved to: REAL_QPU_RESULTS_ONLY.png")

# =============================================================================
# Print Summary Table
# =============================================================================

print("\n" + "="*100)
print("REAL QPU RESULTS SUMMARY (6 families × 3 periods formulation)")
print("="*100)
print(f"{'Source':<28} {'Farms':>6} {'Vars':>6} {'Gurobi Obj':>10} {'QPU Obj':>10} {'Gap%':>8} {'QPU Time':>10} {'Speedup':>8}")
print("-"*100)
for _, r in df.sort_values(['source', 'n_vars']).iterrows():
    print(f"{r['source']:<28} {r['n_farms']:>6} {r['n_vars']:>6} {r['gurobi_obj']:>10.2f} {r['quantum_obj']:>10.2f} {r['gap']:>8.1f} {r['qpu_time']:>9.3f}s {r['speedup']:>8.2f}x")
print("-"*100)
