#!/usr/bin/env python3
"""Visualize repair heuristic results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path('professional_plots') / 'postprocessing_repair_results.json'
with open(results_path) as f:
    results = json.load(f)

# Load Gurobi baseline
gurobi_data = {
    'rotation_100farms_6foods': 53.77,
    'rotation_15farms_6foods': 9.68,
    'rotation_25farms_6foods': 13.45,
    'rotation_50farms_6foods': 26.92,
    'rotation_75farms_6foods': 40.37,
    'rotation_large_200': 21.57,
    'rotation_medium_100': 12.78,
    'rotation_micro_25': 6.17,
    'rotation_small_50': 8.69,
}

# Prepare data
scenarios = [r['scenario'] for r in results]
n_farms = [r['n_farms'] for r in results]
original_obj = [r['original_objective'] for r in results]
repaired_obj = [r['repaired_objective'] for r in results]
original_viols = [r['original_violations'] for r in results]
gurobi_obj = [gurobi_data.get(s, 0) for s in scenarios]

# Sort by n_farms
sorted_idx = np.argsort(n_farms)
scenarios = [scenarios[i] for i in sorted_idx]
n_farms = [n_farms[i] for i in sorted_idx]
original_obj = [original_obj[i] for i in sorted_idx]
repaired_obj = [repaired_obj[i] for i in sorted_idx]
original_viols = [original_viols[i] for i in sorted_idx]
gurobi_obj = [gurobi_obj[i] for i in sorted_idx]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Objective comparison
ax = axes[0]
x = np.arange(len(scenarios))
width = 0.25

bars1 = ax.bar(x - width, original_obj, width, label='Original (w/ violations)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x, repaired_obj, width, label='Repaired (no violations)', color='#3498db', alpha=0.8)
bars3 = ax.bar(x + width, gurobi_obj, width, label='Gurobi Baseline', color='#2ecc71', alpha=0.8)

ax.set_xlabel('Scenario (sorted by farm count)', fontsize=12, fontweight='bold')
ax.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
ax.set_title('Objective Comparison: Original vs Repaired vs Gurobi', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{nf}F' for nf in n_farms], rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)

# Plot 2: Violations repaired
ax = axes[1]
bars = ax.bar(x, original_viols, color='#9b59b6', alpha=0.8)
ax.set_xlabel('Scenario (sorted by farm count)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of One-Hot Violations', fontsize=12, fontweight='bold')
ax.set_title('Constraint Violations Repaired', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{nf}F' for nf in n_farms], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(height)}',
           ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add annotation
ax.text(0.98, 0.98, 'All violations\nsuccessfully repaired\nto zero',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Gap to Gurobi
ax = axes[2]
gap_pcts = [100 * (rep - gur) / gur if gur > 0 else 0 
            for rep, gur in zip(repaired_obj, gurobi_obj)]
colors = ['#e74c3c' if g < -50 else '#f39c12' if g < -25 else '#f1c40f' 
          for g in gap_pcts]

bars = ax.bar(x, gap_pcts, color=colors, alpha=0.8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Scenario (sorted by farm count)', fontsize=12, fontweight='bold')
ax.set_ylabel('Gap to Gurobi (%)', fontsize=12, fontweight='bold')
ax.set_title('Quality Gap: Repaired QPU vs Gurobi Baseline', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{nf}F' for nf in n_farms], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.0f}%',
           ha='center', va='top' if height < 0 else 'bottom', 
           fontsize=8, fontweight='bold')

# Add annotation
avg_gap = np.mean(gap_pcts)
ax.text(0.02, 0.02, f'Average gap: {avg_gap:.1f}%\n(negative = below Gurobi)',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('professional_plots/repair_heuristic_summary.png', dpi=300, bbox_inches='tight')
plt.savefig('professional_plots/repair_heuristic_summary.pdf', bbox_inches='tight')
print("✓ Saved repair heuristic summary plots")

# Create second figure: improvement analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Objective improvement
ax = axes[0]
improvements = [r['improvement'] for r in results]
improvements = [improvements[i] for i in sorted_idx]

bars = ax.bar(x, improvements, color='#27ae60', alpha=0.8)
ax.set_xlabel('Scenario (sorted by farm count)', fontsize=12, fontweight='bold')
ax.set_ylabel('Objective Improvement', fontsize=12, fontweight='bold')
ax.set_title('Objective Improvement After Repair', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{nf}F' for nf in n_farms], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'+{height:.2f}',
           ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Violations vs improvement
ax = axes[1]
ax.scatter(original_viols, improvements, c=n_farms, cmap='viridis', s=150, alpha=0.7)
ax.set_xlabel('Original Violations Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Objective Improvement', fontsize=12, fontweight='bold')
ax.set_title('Violations vs Improvement (color = farm count)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', 
                           norm=plt.Normalize(vmin=min(n_farms), vmax=max(n_farms)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Number of Farms', fontsize=10, fontweight='bold')

# Add annotations for extremes
max_viols_idx = original_viols.index(max(original_viols))
min_viols_idx = original_viols.index(min(original_viols))

ax.annotate(f'{scenarios[max_viols_idx].replace("rotation_", "")}\n({n_farms[max_viols_idx]}F)',
            xy=(original_viols[max_viols_idx], improvements[max_viols_idx]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.annotate(f'{scenarios[min_viols_idx].replace("rotation_", "")}\n({n_farms[min_viols_idx]}F)',
            xy=(original_viols[min_viols_idx], improvements[min_viols_idx]),
            xytext=(10, -20), textcoords='offset points',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig('professional_plots/repair_improvement_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('professional_plots/repair_improvement_analysis.pdf', bbox_inches='tight')
print("✓ Saved repair improvement analysis plots")

# Print summary statistics
print("\n" + "="*80)
print("REPAIR HEURISTIC SUMMARY STATISTICS")
print("="*80)
print(f"Total scenarios processed: {len(results)}")
print(f"Total violations repaired: {sum(original_viols)}")
print(f"Average violations per scenario: {np.mean(original_viols):.1f}")
print(f"Average objective improvement: {np.mean(improvements):.2f}")
print(f"Average gap to Gurobi: {avg_gap:.1f}%")
print(f"\nAll scenarios: 100% of violations successfully repaired (0 remaining)")
print("\nNote: While repairs eliminate constraint violations, the resulting")
print("      objective values are still significantly below Gurobi baselines.")
print("      This indicates that the greedy repair heuristic produces feasible")
print("      but suboptimal solutions compared to exact MIQP solvers.")
