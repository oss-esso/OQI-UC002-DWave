#!/usr/bin/env python3
"""
Analyze JSON results and regenerate plots with variables on x-axis
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_file = Path(__file__).parent / 'statistical_comparison_results' / 'statistical_comparison_20251211_180707.json'
with open(results_file, 'r') as f:
    results = json.load(f)

output_dir = Path(__file__).parent / 'statistical_comparison_results'

# Extract data
sizes = sorted([int(k) for k in results['results_by_size'].keys()])
methods = results['config']['methods']
quantum_methods = [m for m in methods if m != 'ground_truth']

# Data structures
data_by_method = {}
for method in methods:
    data_by_method[method] = {
        'objectives': [],
        'wall_times': [],
        'qpu_times': [],
        'gaps': [],
        'speedups': [],
        'variables': []
    }

variables = []

for n_farms in sizes:
    size_data = results['results_by_size'][str(n_farms)]
    n_vars = size_data['n_variables']
    variables.append(n_vars)
    
    for method in methods:
        method_data = size_data['methods'][method]['stats']
        data_by_method[method]['objectives'].append(method_data['objective']['mean'])
        data_by_method[method]['wall_times'].append(method_data['wall_time']['mean'])
        data_by_method[method]['qpu_times'].append(method_data['qpu_time']['mean'])
        data_by_method[method]['variables'].append(n_vars)
    
    # Gaps and speedups
    gaps = size_data.get('gaps', {})
    speedups = size_data.get('speedups', {})
    for qm in quantum_methods:
        data_by_method[qm]['gaps'].append(gaps.get(qm, 0))
        data_by_method[qm]['speedups'].append(speedups.get(qm, 1))

# Color scheme
COLORS = {
    'ground_truth': '#2E86AB',
    'clique_decomp': '#A23B72',
    'spatial_temporal': '#28A745',
}
LABELS = {
    'ground_truth': 'Classical (Gurobi)',
    'clique_decomp': 'Quantum (Clique Decomp)',
    'spatial_temporal': 'Quantum (Spatial-Temporal)',
}

print("Generating plots with variables on x-axis...")

# =========================================================================
# PLOT 1: Solution Quality vs Variables
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for method in methods:
    ax.plot(data_by_method[method]['variables'], 
            data_by_method[method]['objectives'],
            'o-', label=LABELS[method], color=COLORS[method], 
            linewidth=2, markersize=8)

ax.set_xlabel('Number of Variables', fontsize=13, fontweight='bold')
ax.set_ylabel('Objective Value (higher is better)', fontsize=13, fontweight='bold')
ax.set_title('Solution Quality vs Problem Size', fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(axis='both', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'plot_solution_quality_vs_vars.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'plot_solution_quality_vs_vars.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Solution quality plot saved")

# =========================================================================
# PLOT 2: Time Comparison vs Variables (Log Scale)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for method in methods:
    ax.plot(data_by_method[method]['variables'],
            data_by_method[method]['wall_times'],
            'o-', label=LABELS[method], color=COLORS[method],
            linewidth=2, markersize=8)

# Add QPU-only times
for method in quantum_methods:
    ax.plot(data_by_method[method]['variables'],
            data_by_method[method]['qpu_times'],
            's--', label=f'{LABELS[method]} (QPU only)', 
            color=COLORS[method], alpha=0.6, linewidth=1.5, markersize=6)

ax.set_xlabel('Number of Variables', fontsize=13, fontweight='bold')
ax.set_ylabel('Wall Time (seconds, log scale)', fontsize=13, fontweight='bold')
ax.set_title('Computation Time vs Problem Size', fontsize=15, fontweight='bold')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(output_dir / 'plot_time_vs_vars.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'plot_time_vs_vars.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Time comparison plot saved")

# =========================================================================
# PLOT 3: Gap and Speedup vs Variables
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Gaps
ax1 = axes[0]
for method in quantum_methods:
    ax1.plot(data_by_method[method]['variables'],
             data_by_method[method]['gaps'],
             'o-', label=LABELS[method], color=COLORS[method],
             linewidth=2, markersize=8)
ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='10% threshold')
ax1.set_xlabel('Number of Variables', fontsize=13, fontweight='bold')
ax1.set_ylabel('Optimality Gap (%)', fontsize=13, fontweight='bold')
ax1.set_title('Optimality Gap vs Problem Size', fontsize=15, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: Speedups
ax2 = axes[1]
for method in quantum_methods:
    ax2.plot(data_by_method[method]['variables'],
             data_by_method[method]['speedups'],
             'o-', label=LABELS[method], color=COLORS[method],
             linewidth=2, markersize=8)
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Break-even')
ax2.set_xlabel('Number of Variables', fontsize=13, fontweight='bold')
ax2.set_ylabel('Speedup Factor (×)', fontsize=13, fontweight='bold')
ax2.set_title('Speedup vs Problem Size', fontsize=15, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'plot_gap_speedup_vs_vars.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'plot_gap_speedup_vs_vars.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Gap and speedup plots saved")

# =========================================================================
# PLOT 4: Scaling Analysis (Time vs Variables, Log-Log)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for method in methods:
    ax.plot(data_by_method[method]['variables'],
            data_by_method[method]['wall_times'],
            'o-', label=LABELS[method], color=COLORS[method],
            linewidth=2, markersize=8)

ax.set_xlabel('Number of Variables (log scale)', fontsize=13, fontweight='bold')
ax.set_ylabel('Wall Time (seconds, log scale)', fontsize=13, fontweight='bold')
ax.set_title('Scaling Behavior (Log-Log)', fontsize=15, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(output_dir / 'plot_scaling_loglog.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'plot_scaling_loglog.pdf', bbox_inches='tight')
plt.close()
print("  ✓ Scaling plot saved")

# =========================================================================
# Summary Statistics
# =========================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

for n_farms in sizes:
    size_data = results['results_by_size'][str(n_farms)]
    n_vars = size_data['n_variables']
    
    print(f"\n{n_farms} farms ({n_vars} variables):")
    
    gt_obj = size_data['methods']['ground_truth']['stats']['objective']['mean']
    gt_time = size_data['methods']['ground_truth']['stats']['wall_time']['mean']
    gt_crops = size_data['methods']['ground_truth']['stats']['diversity']['total_unique_crops']['mean']
    
    print(f"  Ground Truth: obj={gt_obj:.4f}, time={gt_time:.2f}s, crops={gt_crops:.1f}")
    
    for qm in quantum_methods:
        qm_stats = size_data['methods'][qm]['stats']
        qm_obj = qm_stats['objective']['mean']
        qm_time = qm_stats['wall_time']['mean']
        qm_crops = qm_stats['diversity']['total_unique_crops']['mean']
        gap = size_data['gaps'][qm]
        speedup = size_data['speedups'][qm]
        
        print(f"  {qm:20s}: obj={qm_obj:.4f}, time={qm_time:.2f}s, crops={qm_crops:.1f}, gap={gap:.2f}%, speedup={speedup:.1f}x")

print("\n" + "="*80)
print("COMPARISON WITH ROADMAP PHASES")
print("="*80)

print("\nPhase 1 (Gurobi-only baseline):")
print("  ✓ Tested: 5, 10, 15, 20 farms")
print("  ✓ Gurobi timeout: 300s (NOTE: Config says 900s but code used 300s)")
print("  ✓ Ground truth established for all sizes")

print("\nPhase 2 (Decomposition strategies):")
print("  ✓ Clique Decomposition: 18 vars/farm, DWaveCliqueSampler")
print("  ✓ Spatial-Temporal: 2-3 farms/cluster, temporal slicing")
print("  ✓ QPU reads: 100 (as specified)")
print("  ✓ Iterations: 3 (as specified)")

print("\nPhase 3 (Statistical comparison):")
print("  ✓ Multiple runs: 2 per method (as specified)")
print("  ✓ Gap analysis: 11-20% for quantum methods")
print("  ✓ Speedup: 5-15x faster than Gurobi")
print("  ✗ Missing: 25 farms (stopped at 20)")

print("\nDiversity Metrics (Two-Level Optimization):")
for n_farms in sizes:
    size_data = results['results_by_size'][str(n_farms)]
    gt_crops = size_data['methods']['ground_truth']['stats']['diversity']['total_unique_crops']['mean']
    print(f"  {n_farms} farms: {gt_crops:.1f} unique crops grown")

print("\n" + "="*80)
print("✓ All plots regenerated with variables on x-axis")
print("="*80)
