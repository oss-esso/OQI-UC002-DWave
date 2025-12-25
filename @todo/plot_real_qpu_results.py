#!/usr/bin/env python3
"""
Plot Real QPU Results - 3 Data Points
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results from the test
results = [
    {'name': 'Medium (20f)', 'gurobi_vars': 1620, 'quantum_vars': 360, 
     'gurobi_obj': 0.81, 'gurobi_time': 62.2, 'gurobi_status': 'timeout',
     'quantum_obj': 3.35, 'quantum_time': 20.6, 'qpu_time': 0.364, 'speedup': 3.02},
    {'name': 'Large (35f)', 'gurobi_vars': 2835, 'quantum_vars': 630,
     'gurobi_obj': -0.86, 'gurobi_time': 123.7, 'gurobi_status': 'timeout',
     'quantum_obj': 3.01, 'quantum_time': 30.0, 'qpu_time': 0.591, 'speedup': 4.12},
    {'name': 'XLarge (50f)', 'gurobi_vars': 4050, 'quantum_vars': 900,
     'gurobi_obj': -2.47, 'gurobi_time': 305.3, 'gurobi_status': 'timeout',
     'quantum_obj': 3.09, 'quantum_time': 36.8, 'qpu_time': 0.863, 'speedup': 8.30},
]

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Real QPU Results: Adaptive Hybrid vs Gurobi MIQP\n(27-Food Crop Rotation Problem)', 
             fontsize=14, fontweight='bold')

names = [r['name'] for r in results]
gurobi_vars = [r['gurobi_vars'] for r in results]
quantum_vars = [r['quantum_vars'] for r in results]
gurobi_obj = [r['gurobi_obj'] for r in results]
quantum_obj = [r['quantum_obj'] for r in results]
gurobi_time = [r['gurobi_time'] for r in results]
quantum_time = [r['quantum_time'] for r in results]
qpu_time = [r['qpu_time'] for r in results]
speedups = [r['speedup'] for r in results]

x = np.arange(len(names))
width = 0.35

# Plot 1: Objective Value Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(x - width/2, gurobi_obj, width, label='Gurobi MIQP', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x + width/2, quantum_obj, width, label='Quantum Hybrid', color='#3498db', alpha=0.8)
ax1.set_ylabel('Objective Value')
ax1.set_title('Solution Quality')
ax1.set_xticks(x)
ax1.set_xticklabels(names)
ax1.legend()
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_ylim(min(gurobi_obj) - 1, max(quantum_obj) + 1)

# Annotate timeout
for i, (bar, status) in enumerate(zip(bars1, [r['gurobi_status'] for r in results])):
    if status == 'timeout':
        ax1.annotate('TIMEOUT', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

# Plot 2: Solve Time Comparison
ax2 = axes[0, 1]
bars1 = ax2.bar(x - width/2, gurobi_time, width, label='Gurobi MIQP', color='#2ecc71', alpha=0.8)
bars2 = ax2.bar(x + width/2, quantum_time, width, label='Quantum Hybrid', color='#3498db', alpha=0.8)
ax2.set_ylabel('Solve Time (seconds)')
ax2.set_title('Solve Time Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(names)
ax2.legend()

# Plot 3: Speedup
ax3 = axes[0, 2]
colors = ['#e74c3c' if s > 1 else '#95a5a6' for s in speedups]
bars = ax3.bar(names, speedups, color=colors, alpha=0.8)
ax3.set_ylabel('Speedup Factor (x)')
ax3.set_title('Quantum Speedup vs Gurobi')
ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Break-even')
for bar, s in zip(bars, speedups):
    ax3.annotate(f'{s:.1f}x', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 4: Problem Size (Variables)
ax4 = axes[1, 0]
ax4.bar(x - width/2, gurobi_vars, width, label='Gurobi (27-food)', color='#2ecc71', alpha=0.8)
ax4.bar(x + width/2, quantum_vars, width, label='Quantum (6-family)', color='#3498db', alpha=0.8)
ax4.set_ylabel('Number of Variables')
ax4.set_title('Problem Size')
ax4.set_xticks(x)
ax4.set_xticklabels(names)
ax4.legend()

# Plot 5: QPU Time Only
ax5 = axes[1, 1]
ax5.bar(names, qpu_time, color='#9b59b6', alpha=0.8)
ax5.set_ylabel('QPU Time (seconds)')
ax5.set_title('Pure QPU Access Time')
ax5.set_ylim(0, 1)
for i, t in enumerate(qpu_time):
    ax5.annotate(f'{t*1000:.0f}ms', (i, t), ha='center', va='bottom', fontsize=10)

# Plot 6: Scaling - Variables vs Time
ax6 = axes[1, 2]
ax6.plot(gurobi_vars, gurobi_time, 'o-', label='Gurobi MIQP', color='#2ecc71', markersize=10, linewidth=2)
ax6.plot(gurobi_vars, quantum_time, 's-', label='Quantum Hybrid', color='#3498db', markersize=10, linewidth=2)
ax6.set_xlabel('Problem Size (Gurobi Variables)')
ax6.set_ylabel('Solve Time (seconds)')
ax6.set_title('Scaling Behavior')
ax6.legend()
ax6.set_ylim(0, max(gurobi_time) * 1.1)

# Add annotations for timeout
for i, (v, t, s) in enumerate(zip(gurobi_vars, gurobi_time, [r['gurobi_status'] for r in results])):
    if s == 'timeout':
        ax6.annotate('timeout', (v, t), xytext=(10, 10), textcoords='offset points',
                    fontsize=8, color='red')

plt.tight_layout()

# Save
output_path = Path(__file__).parent / 'real_qpu_results' / 'real_qpu_comparison_plot.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to: {output_path}")

plt.show()
