"""
Visualize CQM Partition Scaling Results
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark
plots = [10, 25, 50, 75, 100, 125, 150, 175, 200]
variables = [297, 702, 1377, 2052, 2727, 3402, 4077, 4752, 5427]
constraints = [290, 710, 1410, 2110, 2810, 3510, 4210, 4910, 5610]

# Ground truth times
gt_times = [0.002, 0.001, 0.003, 0.004, 0.005, 0.006, 0.006, 0.008, 0.009]

# Method times (partition + solve)
none_times = [0.005, 0.011, 0.022, 0.043, 0.045, 0.059, 0.068, 0.079, 0.168]
plotbased_times = [0.006, 0.017, 0.040, 0.078, 0.110, 0.195, 0.210, 0.274, 0.366]
spectral_times = [1.589, 0.088, 0.241, 0.486, 1.129, 1.575, 1.975, 2.347, 3.243]
mastersub_times = [0.012, 0.032, 0.086, 0.133, 0.209, 0.305, 0.410, 0.604, 0.745]

# Objectives
gt_obj = 0.429951
none_obj = [0.4300] * 9
plotbased_obj = [0.4300] * 9
spectral_obj = [0.4300] * 9
mastersub_obj = [0.2722] * 9

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Total Time vs Plots
ax1 = axes[0, 0]
ax1.plot(plots, gt_times, 'k-o', label='Ground Truth (Gurobi)', linewidth=2, markersize=8)
ax1.plot(plots, none_times, 'b-s', label='None (Full CQM)', linewidth=2, markersize=7)
ax1.plot(plots, plotbased_times, 'g-^', label='PlotBased', linewidth=2, markersize=7)
ax1.plot(plots, spectral_times, 'r-d', label='Spectral(4)', linewidth=2, markersize=7)
ax1.plot(plots, mastersub_times, 'm-v', label='MasterSubproblem', linewidth=2, markersize=7)
ax1.set_xlabel('Number of Plots', fontsize=12)
ax1.set_ylabel('Total Time (seconds)', fontsize=12)
ax1.set_title('Solve Time Scaling', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Time vs Variables
ax2 = axes[0, 1]
ax2.plot(variables, gt_times, 'k-o', label='Ground Truth', linewidth=2, markersize=8)
ax2.plot(variables, none_times, 'b-s', label='None', linewidth=2, markersize=7)
ax2.plot(variables, plotbased_times, 'g-^', label='PlotBased', linewidth=2, markersize=7)
ax2.plot(variables, spectral_times, 'r-d', label='Spectral(4)', linewidth=2, markersize=7)
ax2.plot(variables, mastersub_times, 'm-v', label='MasterSubproblem', linewidth=2, markersize=7)
ax2.set_xlabel('Number of Variables', fontsize=12)
ax2.set_ylabel('Total Time (seconds)', fontsize=12)
ax2.set_title('Time vs Problem Size (Variables)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: Objective Quality
ax3 = axes[1, 0]
gaps = {
    'None': [0] * 9,
    'PlotBased': [0] * 9,
    'Spectral(4)': [0] * 9,
    'MasterSubproblem': [36.7] * 9
}
x = np.arange(len(plots))
width = 0.2
ax3.bar(x - 1.5*width, gaps['None'], width, label='None', color='blue', alpha=0.7)
ax3.bar(x - 0.5*width, gaps['PlotBased'], width, label='PlotBased', color='green', alpha=0.7)
ax3.bar(x + 0.5*width, gaps['Spectral(4)'], width, label='Spectral(4)', color='red', alpha=0.7)
ax3.bar(x + 1.5*width, gaps['MasterSubproblem'], width, label='MasterSubproblem', color='magenta', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.set_xlabel('Number of Plots', fontsize=12)
ax3.set_ylabel('Optimality Gap (%)', fontsize=12)
ax3.set_title('Solution Quality (Gap from Optimal)', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(plots)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(-5, 45)

# Plot 4: Problem Size Scaling
ax4 = axes[1, 1]
ax4.plot(plots, variables, 'b-o', label='Variables', linewidth=2, markersize=8)
ax4.plot(plots, constraints, 'r-s', label='Constraints', linewidth=2, markersize=7)
ax4.set_xlabel('Number of Plots', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Problem Size Scaling', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add annotations for growth rates
ax4.annotate(f'~27 vars/plot\n(foods × plots + U)', 
             xy=(150, 4077), xytext=(100, 4500),
             arrowprops=dict(arrowstyle='->', color='blue'),
             fontsize=10, color='blue')
ax4.annotate(f'~28 constraints/plot', 
             xy=(150, 4210), xytext=(50, 3500),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

plt.suptitle('CQM Partition Scaling Benchmark (10-200 Plots, 27 Foods)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('cqm_partition_scaling_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('cqm_partition_scaling_plot.pdf', bbox_inches='tight')
print("Saved: cqm_partition_scaling_plot.png")
print("Saved: cqm_partition_scaling_plot.pdf")

# Create summary table
print("\n" + "="*100)
print("SUMMARY TABLE: CQM PARTITION SCALING BENCHMARK")
print("="*100)
print(f"{'Plots':>6} | {'Vars':>6} | {'Constr':>7} | {'GT':>8} | {'None':>8} | {'PlotBsd':>8} | {'Spectr':>8} | {'MstrSub':>8} | {'Best':>10}")
print("-"*100)

for i, p in enumerate(plots):
    times = {
        'GT': gt_times[i],
        'None': none_times[i],
        'PlotBased': plotbased_times[i],
        'Spectral': spectral_times[i],
        'MasterSub': mastersub_times[i]
    }
    # Best is the fastest that achieves optimal (gap=0)
    optimal_methods = ['GT', 'None', 'PlotBased', 'Spectral']
    best = min(optimal_methods, key=lambda m: times[m])
    
    print(f"{p:>6} | {variables[i]:>6} | {constraints[i]:>7} | {gt_times[i]:>7.3f}s | {none_times[i]:>7.3f}s | {plotbased_times[i]:>7.3f}s | {spectral_times[i]:>7.3f}s | {mastersub_times[i]:>7.3f}s | {best:>10}")

print("="*100)
print("\nKey Findings:")
print("  ✅ None, PlotBased, Spectral(4): ALL achieve OPTIMAL solution (0% gap)")
print("  ⚠️  MasterSubproblem: 36.7% gap (suboptimal due to sequential solving order)")
print("  🏆 Ground Truth (direct Gurobi) is fastest for this problem structure")
print("  📈 PlotBased scales better than Spectral (no expensive graph clustering)")
print("  💡 Partition methods work well but have overhead for small problems")
