#!/usr/bin/env python3
"""
Generate Publication-Quality Plots for OQI-UC002-DWave Paper

This script generates all figures needed for the results_and_conclusions.tex section.
Plots are designed for high-impact physics journal publication standards.

Output: All plots saved to paper_plots/ directory in PDF and PNG formats.

Author: OQI-UC002-DWave Project
Date: 2025-01
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_config import (
    setup_publication_style,
    get_color_palette,
    get_sequential_cmap,
    save_figure,
    METHOD_COLORS,
    QUALITATIVE_COLORS
)

# Initialize publication style
setup_publication_style()

# Override some settings for extra clarity in publication
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# Define colorblind-friendly palette (based on Wong, 2011)
COLORS = {
    'qpu': '#009E73',          # Bluish green
    'qpu_pure': '#0072B2',     # Blue
    'gurobi': '#D55E00',       # Vermillion/Orange
    'gurobi_optimal': '#E69F00',  # Orange
    'timeout': '#999999',       # Gray
    'classical': '#56B4E9',     # Sky blue
    'violation': '#CC79A7',     # Reddish purple
    'highlight': '#F0E442',     # Yellow
}

# Output directory
OUTPUT_DIR = Path(__file__).parent


def load_all_data():
    """Load all result files."""
    base_dir = Path(__file__).parent.parent
    
    data = {}
    
    # Main QPU results
    with open(base_dir / 'qpu_hier_repaired.json') as f:
        data['qpu_hier'] = json.load(f)
    
    # Gurobi baseline
    with open(base_dir / 'gurobi_baseline_60s.json') as f:
        data['gurobi_60s'] = json.load(f)
    
    # Try to load additional files if they exist
    try:
        with open(base_dir / 'test_gurobi_300s.json') as f:
            data['gurobi_300s'] = json.load(f)
    except FileNotFoundError:
        pass
    
    return data


def prepare_comparison_data(qpu_data, gurobi_data):
    """Prepare merged data for QPU vs Gurobi comparison."""
    qpu_by_scenario = {r['scenario_name']: r for r in qpu_data['runs']}
    gurobi_by_scenario = {r['scenario_name']: r for r in gurobi_data['runs']}
    
    records = []
    for scenario, q in qpu_by_scenario.items():
        g = gurobi_by_scenario.get(scenario, {})
        
        q_timing = q.get('timing', {})
        g_timing = g.get('timing', {})
        
        records.append({
            'scenario': scenario,
            'n_farms': q['n_farms'],
            'n_foods': q['n_foods'],
            'n_periods': q.get('n_periods', 3),
            'n_vars': q['n_vars'],
            # QPU metrics
            'qpu_total_time': q_timing.get('total_wall_time', 0),
            'qpu_access_time': q_timing.get('qpu_access_time', 0),
            'qpu_sampling_time': q_timing.get('qpu_sampling_time', 0),
            'qpu_objective': q.get('objective_miqp', 0),
            'qpu_feasible': q.get('feasible', False),
            'qpu_violations': q.get('constraint_violations', {}),
            # Gurobi metrics
            'gurobi_time': g_timing.get('total_wall_time', 0),
            'gurobi_objective': g.get('objective_miqp', 0),
            'gurobi_status': g.get('status', 'N/A'),
            'gurobi_timeout': g.get('status') == 'timeout',
            'gurobi_gap': g.get('mip_gap', 0) if g.get('status') == 'timeout' else 0,
        })
    
    # Sort by number of variables
    records.sort(key=lambda x: x['n_vars'])
    return records


# =============================================================================
# FIGURE 1: Time Comparison Bar Chart
# =============================================================================
def plot_fig1_time_comparison(data, output_dir):
    """
    Figure 1: Side-by-side bar chart comparing QPU vs Gurobi solve times.
    Highlights timeout cases and shows problem scale.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    scenarios = [d['scenario'].replace('rotation_', '').replace('_', '\n') for d in data]
    n_vars = [d['n_vars'] for d in data]
    qpu_times = [d['qpu_total_time'] for d in data]
    gurobi_times = [d['gurobi_time'] for d in data]
    timeouts = [d['gurobi_timeout'] for d in data]
    
    x = np.arange(len(scenarios))
    width = 0.38
    
    # QPU bars
    bars_qpu = ax.bar(x - width/2, qpu_times, width, 
                      label='D-Wave Hierarchical QPU', 
                      color=COLORS['qpu'], 
                      edgecolor='black', linewidth=0.8,
                      zorder=3)
    
    # Gurobi bars - different color for timeouts
    gurobi_colors = [COLORS['timeout'] if t else COLORS['gurobi'] for t in timeouts]
    bars_gurobi = ax.bar(x + width/2, gurobi_times, width,
                         label='Gurobi 12.0 (60s timeout)',
                         color=gurobi_colors,
                         edgecolor='black', linewidth=0.8,
                         zorder=3)
    
    # Add timeout markers
    for i, (bar, timeout) in enumerate(zip(bars_gurobi, timeouts)):
        if timeout:
            ax.annotate('TIMEOUT', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 2),
                       ha='center', va='bottom', fontsize=8, 
                       color='#8B0000', fontweight='bold',
                       rotation=90)
    
    # Add variable count labels below bars
    for i, nv in enumerate(n_vars):
        ax.annotate(f'{nv:,} vars', 
                   xy=(x[i], -0.15), 
                   ha='center', va='top',
                   fontsize=9, color='#444444',
                   xycoords=('data', 'axes fraction'))
    
    # Timeout reference line
    ax.axhline(y=60, color='#CC0000', linestyle='--', linewidth=1.5, 
               alpha=0.7, label='Gurobi timeout threshold', zorder=2)
    
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_ylabel('Solve Time (seconds)', fontweight='bold')
    ax.set_title('Solve Time Comparison: D-Wave QPU vs. Classical Gurobi Solver',
                fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(0.5, 200)
    ax.legend(loc='upper left', framealpha=0.95)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig1_time_comparison')
    plt.close()
    print("âœ“ Generated: fig1_time_comparison")


# =============================================================================
# FIGURE 2: QPU Time Breakdown (Stacked Bar)
# =============================================================================
def plot_fig2_qpu_breakdown(data, output_dir):
    """
    Figure 2: Stacked bar chart showing pure QPU time vs classical overhead.
    Emphasizes that quantum computation is a small fraction of total time.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    scenarios = [d['scenario'].replace('rotation_', '').replace('_', '\n') for d in data]
    n_vars = [d['n_vars'] for d in data]
    
    qpu_access = np.array([d['qpu_access_time'] for d in data])
    total_time = np.array([d['qpu_total_time'] for d in data])
    classical = total_time - qpu_access
    
    x = np.arange(len(scenarios))
    width = 0.65
    
    # Stacked bars
    bars1 = ax.bar(x, qpu_access, width, 
                   label='Pure QPU Access Time', 
                   color=COLORS['qpu_pure'],
                   edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x, classical, width, bottom=qpu_access,
                   label='Classical Overhead (embedding, coordination)',
                   color=COLORS['classical'],
                   edgecolor='black', linewidth=0.8)
    
    # Add percentage labels inside QPU portion
    for i, (qpu_t, tot_t) in enumerate(zip(qpu_access, total_time)):
        pct = (qpu_t / tot_t) * 100 if tot_t > 0 else 0
        if pct > 1:  # Only show if visible
            ax.annotate(f'{pct:.1f}%', 
                       xy=(i, qpu_t / 2), 
                       ha='center', va='center',
                       fontsize=9, color='white', fontweight='bold')
    
    # Add total time labels on top
    for i, tot in enumerate(total_time):
        ax.annotate(f'{tot:.1f}s', 
                   xy=(i, tot + 0.5),
                   ha='center', va='bottom',
                   fontsize=9, color='#333333')
    
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_title('QPU Time Breakdown: Quantum vs. Classical Computation\n'
                '(Percentages indicate pure QPU fraction)',
                fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig2_qpu_breakdown')
    plt.close()
    print("âœ“ Generated: fig2_qpu_breakdown")


# =============================================================================
# FIGURE 3: Scaling Analysis (Dual Log-Log)
# =============================================================================
def plot_fig3_scaling(data, output_dir):
    """
    Figure 3: Dual-panel log-log scaling analysis.
    Left: Total time vs variables. Right: Pure QPU time vs variables.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    n_vars = np.array([d['n_vars'] for d in data])
    qpu_total = np.array([d['qpu_total_time'] for d in data])
    qpu_access = np.array([d['qpu_access_time'] for d in data])
    gurobi_times = np.array([d['gurobi_time'] for d in data])
    timeouts = [d['gurobi_timeout'] for d in data]
    
    # Panel A: Total time scaling
    ax1 = axes[0]
    ax1.scatter(n_vars, qpu_total, s=120, c=COLORS['qpu'], 
                label='QPU Total Time', edgecolors='black', 
                linewidths=1, zorder=4, marker='o')
    
    # Gurobi points - different markers for timeout
    gurobi_normal = [(nv, gt) for nv, gt, to in zip(n_vars, gurobi_times, timeouts) if not to]
    gurobi_timeout = [(nv, gt) for nv, gt, to in zip(n_vars, gurobi_times, timeouts) if to]
    
    if gurobi_normal:
        nvn, gtn = zip(*gurobi_normal)
        ax1.scatter(nvn, gtn, s=120, c=COLORS['gurobi'], 
                   label='Gurobi (completed)', edgecolors='black',
                   linewidths=1, zorder=4, marker='s')
    if gurobi_timeout:
        nvt, gtt = zip(*gurobi_timeout)
        ax1.scatter(nvt, gtt, s=120, c=COLORS['timeout'],
                   label='Gurobi (timeout)', edgecolors='black',
                   linewidths=1, zorder=4, marker='s')
    
    # Fit line for QPU
    log_vars = np.log(n_vars)
    log_qpu = np.log(qpu_total)
    coef = np.polyfit(log_vars, log_qpu, 1)
    fit_line = np.exp(coef[1]) * n_vars ** coef[0]
    ax1.plot(n_vars, fit_line, '--', color=COLORS['qpu'], alpha=0.7, linewidth=2,
             label=f'QPU fit: $O(n^{{{coef[0]:.2f}}})$')
    
    # Timeout line
    ax1.axhline(y=60, color='#CC0000', linestyle=':', linewidth=1.5, 
                alpha=0.7, label='Timeout (60s)')
    
    ax1.set_xlabel('Number of Variables', fontweight='bold')
    ax1.set_ylabel('Total Solve Time (s)', fontweight='bold')
    ax1.set_title('(a) Total Time Scaling', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=18, 
             fontweight='bold', va='top')
    
    # Panel B: Pure QPU time scaling
    ax2 = axes[1]
    ax2.scatter(n_vars, qpu_access, s=120, c=COLORS['qpu_pure'],
                edgecolors='black', linewidths=1, zorder=4, marker='o')
    
    # Linear fit for pure QPU time
    log_qpu_access = np.log(qpu_access)
    coef_access = np.polyfit(log_vars, log_qpu_access, 1)
    fit_access = np.exp(coef_access[1]) * n_vars ** coef_access[0]
    ax2.plot(n_vars, fit_access, '--', color=COLORS['qpu_pure'], alpha=0.7, linewidth=2,
             label=f'Fit: $O(n^{{{coef_access[0]:.2f}}})$')
    
    # Calculate ms per variable
    ms_per_var = (qpu_access * 1000) / n_vars
    avg_ms = np.mean(ms_per_var)
    ax2.annotate(f'Average: {avg_ms:.2f} ms/var',
                xy=(0.95, 0.05), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('Number of Variables', fontweight='bold')
    ax2.set_ylabel('Pure QPU Access Time (s)', fontweight='bold')
    ax2.set_title('(b) Pure QPU Time Scaling', fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=18,
             fontweight='bold', va='top')
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig3_scaling_analysis')
    plt.close()
    print("âœ“ Generated: fig3_scaling_analysis")


# =============================================================================
# FIGURE 4: Quantum Advantage (Objective Comparison)
# =============================================================================
def plot_fig4_quantum_advantage(data, output_dir):
    """
    Figure 4: Objective value comparison showing quantum advantage.
    Bar chart with speedup/benefit ratio annotations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    scenarios = [d['scenario'].replace('rotation_', '').replace('_', '\n') for d in data]
    n_vars = [d['n_vars'] for d in data]
    
    qpu_obj = np.array([d['qpu_objective'] for d in data])
    gurobi_obj = np.array([d['gurobi_objective'] for d in data])
    timeouts = [d['gurobi_timeout'] for d in data]
    gurobi_time = np.array([d['gurobi_time'] for d in data])
    qpu_time = np.array([d['qpu_total_time'] for d in data])
    
    x = np.arange(len(scenarios))
    width = 0.38
    
    # Panel A: Objective values
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, qpu_obj, width,
                    label='D-Wave QPU', color=COLORS['qpu'],
                    edgecolor='black', linewidth=0.8)
    
    gurobi_colors = [COLORS['timeout'] if t else COLORS['gurobi'] for t in timeouts]
    bars2 = ax1.bar(x + width/2, gurobi_obj, width,
                    label='Gurobi', color=gurobi_colors,
                    edgecolor='black', linewidth=0.8)
    
    # Add timeout markers
    for i, (bar, timeout) in enumerate(zip(bars2, timeouts)):
        if timeout:
            ax1.annotate('TO', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color='#8B0000')
    
    ax1.set_xlabel('Scenario', fontweight='bold')
    ax1.set_ylabel('Objective Value (Benefit)', fontweight='bold')
    ax1.set_title('(a) Solution Quality Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=9)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=18,
             fontweight='bold', va='top')
    
    # Panel B: Benefit ratio (QPU/Gurobi) for timeout cases
    ax2 = axes[1]
    
    # Calculate benefit ratio (higher is better for QPU)
    benefit_ratio = np.where(gurobi_obj > 0, qpu_obj / gurobi_obj, np.inf)
    benefit_ratio = np.where(np.isinf(benefit_ratio), 10, benefit_ratio)  # Cap at 10x
    
    colors = [COLORS['qpu'] if r > 1 else COLORS['gurobi'] for r in benefit_ratio]
    bars = ax2.bar(x, benefit_ratio, width=0.7, color=colors,
                   edgecolor='black', linewidth=0.8)
    
    # Reference line at 1.0
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, 
                label='Parity (1.0Ã—)')
    
    # Add ratio labels
    for i, (bar, ratio) in enumerate(zip(bars, benefit_ratio)):
        label = f'{ratio:.2f}Ã—'
        ax2.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Scenario', fontweight='bold')
    ax2.set_ylabel('Benefit Ratio (QPU / Gurobi)', fontweight='bold')
    ax2.set_title('(b) Quantum Advantage: Benefit Ratio', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=9)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(benefit_ratio) * 1.2)
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=18,
             fontweight='bold', va='top')
    
    # Add summary annotation
    timeout_ratios = [r for r, t in zip(benefit_ratio, timeouts) if t]
    if timeout_ratios:
        avg_ratio = np.mean(timeout_ratios)
        ax2.annotate(f'Average advantage\n(timeout cases): {avg_ratio:.2f}Ã—',
                    xy=(0.98, 0.05), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig4_quantum_advantage')
    plt.close()
    print("âœ“ Generated: fig4_quantum_advantage")


# =============================================================================
# FIGURE 5: Comprehensive Summary (2x2 Grid)
# =============================================================================
def plot_fig5_summary(data, output_dir):
    """
    Figure 5: Comprehensive 2x2 summary figure.
    (a) Time scatter, (b) Success pie chart, (c) Scaling fit, (d) Statistics table
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    n_vars = np.array([d['n_vars'] for d in data])
    qpu_total = np.array([d['qpu_total_time'] for d in data])
    qpu_access = np.array([d['qpu_access_time'] for d in data])
    gurobi_times = np.array([d['gurobi_time'] for d in data])
    timeouts = np.array([d['gurobi_timeout'] for d in data])
    qpu_obj = np.array([d['qpu_objective'] for d in data])
    gurobi_obj = np.array([d['gurobi_objective'] for d in data])
    
    # Panel A: Time scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(gurobi_times, qpu_total, s=100, c=COLORS['qpu'],
                edgecolors='black', linewidths=1, zorder=3)
    
    max_time = max(max(gurobi_times), max(qpu_total)) * 1.1
    ax1.plot([0, max_time], [0, max_time], 'k--', alpha=0.5, label='Parity line')
    ax1.fill_between([0, max_time], [0, 0], [0, max_time], 
                     alpha=0.1, color=COLORS['qpu'], label='QPU faster')
    
    ax1.set_xlabel('Gurobi Time (s)', fontweight='bold')
    ax1.set_ylabel('QPU Time (s)', fontweight='bold')
    ax1.set_title('(a) Time Comparison Scatter', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=16,
             fontweight='bold', va='top')
    
    # Panel B: Pie chart - Gurobi outcomes
    ax2 = fig.add_subplot(gs[0, 1])
    n_timeout = np.sum(timeouts)
    n_complete = len(timeouts) - n_timeout
    
    sizes = [n_complete, n_timeout]
    labels = [f'Completed\n({n_complete})', f'Timeout\n({n_timeout})']
    colors_pie = [COLORS['gurobi'], COLORS['timeout']]
    explode = (0, 0.05)
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                        explode=explode, autopct='%1.0f%%',
                                        startangle=90, textprops={'fontsize': 11})
    ax2.set_title('(b) Gurobi Solver Outcomes (60s)', fontweight='bold')
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16,
             fontweight='bold', va='top')
    
    # Panel C: Pure QPU scaling
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(n_vars, qpu_access * 1000, s=100, c=COLORS['qpu_pure'],
                edgecolors='black', linewidths=1, zorder=3)
    
    # Linear fit
    coef = np.polyfit(n_vars, qpu_access * 1000, 1)
    fit_line = np.poly1d(coef)
    x_fit = np.linspace(min(n_vars), max(n_vars), 100)
    ax3.plot(x_fit, fit_line(x_fit), '--', color=COLORS['qpu_pure'], linewidth=2,
             label=f'Linear fit: {coef[0]:.3f} ms/var')
    
    ax3.set_xlabel('Number of Variables', fontweight='bold')
    ax3.set_ylabel('Pure QPU Time (ms)', fontweight='bold')
    ax3.set_title('(c) Pure QPU Time Scaling', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes, fontsize=16,
             fontweight='bold', va='top')
    
    # Panel D: Summary statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate statistics
    benefit_ratio = np.where(gurobi_obj > 0, qpu_obj / gurobi_obj, np.nan)
    timeout_benefit_ratio = benefit_ratio[timeouts]
    speedup = gurobi_times / qpu_total
    
    stats = [
        ['Total Scenarios', f'{len(data)}'],
        ['Gurobi Timeouts', f'{n_timeout} ({100*n_timeout/len(data):.0f}%)'],
        ['Avg. QPU Total Time', f'{np.mean(qpu_total):.1f} s'],
        ['Avg. Pure QPU Time', f'{np.mean(qpu_access)*1000:.1f} ms'],
        ['QPU Efficiency', f'{100*np.mean(qpu_access/qpu_total):.1f}%'],
        ['Avg. Benefit Ratio (timeout)', f'{np.nanmean(timeout_benefit_ratio):.2f}Ã—'],
        ['Max Benefit Ratio', f'{np.nanmax(benefit_ratio):.2f}Ã—'],
        ['Avg. Speedup', f'{np.mean(speedup):.2f}Ã—'],
    ]
    
    table = ax4.table(cellText=stats, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='left',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('(d) Summary Statistics', fontweight='bold', pad=20)
    ax4.text(0.02, 0.98, 'D', transform=ax4.transAxes, fontsize=16,
             fontweight='bold', va='top')
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig5_comprehensive_summary')
    plt.close()
    print("âœ“ Generated: fig5_comprehensive_summary")


# =============================================================================
# FIGURE 6: Formulation Split Analysis (6-Family vs 27-Food)
# =============================================================================
def plot_fig6_formulation_split(data, output_dir):
    """
    Figure 6: Side-by-side comparison of 6-family vs 27-food configurations.
    """
    # Separate by food count
    data_6food = [d for d in data if d['n_foods'] == 6]
    data_27food = [d for d in data if d['n_foods'] == 27]
    
    if not data_6food or not data_27food:
        print("âš  Skipping fig6: insufficient data for split analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    def plot_group(ax, group_data, title, panel_label):
        scenarios = [d['scenario'].replace('rotation_', '').replace('_', '\n') for d in group_data]
        qpu_times = [d['qpu_total_time'] for d in group_data]
        gurobi_times = [d['gurobi_time'] for d in group_data]
        timeouts = [d['gurobi_timeout'] for d in group_data]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, qpu_times, width, label='QPU',
                      color=COLORS['qpu'], edgecolor='black', linewidth=0.8)
        
        gurobi_colors = [COLORS['timeout'] if t else COLORS['gurobi'] for t in timeouts]
        bars2 = ax.bar(x + width/2, gurobi_times, width, label='Gurobi',
                      color=gurobi_colors, edgecolor='black', linewidth=0.8)
        
        # Timeout markers
        for i, (bar, timeout) in enumerate(zip(bars2, timeouts)):
            if timeout:
                ax.annotate('TO', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
                           ha='center', va='bottom', fontsize=8, fontweight='bold',
                           color='#8B0000')
        
        ax.axhline(y=60, color='#CC0000', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Scenario', fontweight='bold')
        ax.set_ylabel('Solve Time (s)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=9)
        ax.set_yscale('log')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='top')
        
        # Add timeout count
        n_timeout = sum(timeouts)
        ax.annotate(f'Gurobi timeouts: {n_timeout}/{len(group_data)}',
                   xy=(0.98, 0.05), xycoords='axes fraction',
                   ha='right', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plot_group(axes[0], data_6food, '(a) 6-Family Formulation', 'A')
    plot_group(axes[1], data_27food, '(b) 27-Food Formulation', 'B')
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig6_formulation_split')
    plt.close()
    print("âœ“ Generated: fig6_formulation_split")


# =============================================================================
# FIGURE 7: Violation Analysis
# =============================================================================
def plot_fig7_violation_analysis(data, output_dir):
    """
    Figure 7: Analysis of constraint violations and their impact on solution quality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    n_vars = np.array([d['n_vars'] for d in data])
    qpu_obj = np.array([d['qpu_objective'] for d in data])
    gurobi_obj = np.array([d['gurobi_objective'] for d in data])
    
    # Extract violation counts
    violations = []
    for d in data:
        v = d.get('qpu_violations', {})
        total_v = v.get('one_hot', 0) + v.get('min_crops', 0) + v.get('food_group', 0)
        violations.append(total_v)
    violations = np.array(violations)
    
    # Panel A: Violations vs Gap
    ax1 = axes[0]
    gap = np.where(gurobi_obj > 0, 100 * (gurobi_obj - qpu_obj) / gurobi_obj, 0)
    
    ax1.scatter(violations, gap, s=100, c=COLORS['violation'],
                edgecolors='black', linewidths=1, zorder=3)
    
    # Fit line if correlation exists
    if np.std(violations) > 0:
        coef = np.polyfit(violations, gap, 1)
        corr = np.corrcoef(violations, gap)[0, 1]
        fit_line = np.poly1d(coef)
        x_fit = np.linspace(0, max(violations), 100)
        ax1.plot(x_fit, fit_line(x_fit), '--', color=COLORS['violation'], 
                linewidth=2, label=f'Linear fit (r={corr:.2f})')
        ax1.legend(loc='upper left')
    
    ax1.set_xlabel('Total Constraint Violations', fontweight='bold')
    ax1.set_ylabel('Optimality Gap (%)', fontweight='bold')
    ax1.set_title('(a) Violations vs. Solution Gap', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=16,
             fontweight='bold', va='top')
    
    # Panel B: Violations by problem size
    ax2 = axes[1]
    ax2.scatter(n_vars, violations, s=100, c=COLORS['violation'],
                edgecolors='black', linewidths=1, zorder=3)
    
    # Fit line
    if np.std(n_vars) > 0:
        coef = np.polyfit(n_vars, violations, 1)
        fit_line = np.poly1d(coef)
        x_fit = np.linspace(min(n_vars), max(n_vars), 100)
        ax2.plot(x_fit, fit_line(x_fit), '--', color=COLORS['violation'],
                linewidth=2, label=f'Trend: {coef[0]:.3f} v/var')
        ax2.legend(loc='upper left')
    
    ax2.set_xlabel('Number of Variables', fontweight='bold')
    ax2.set_ylabel('Constraint Violations', fontweight='bold')
    ax2.set_title('(b) Violations vs. Problem Size', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16,
             fontweight='bold', va='top')
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'fig7_violation_analysis')
    plt.close()
    print("âœ“ Generated: fig7_violation_analysis")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Generate all publication plots."""
    print("="*60)
    print("Generating Publication-Quality Plots")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Load data
    print("Loading data files...")
    all_data = load_all_data()
    
    # Prepare comparison data
    data = prepare_comparison_data(all_data['qpu_hier'], all_data['gurobi_60s'])
    print(f"Loaded {len(data)} scenarios")
    print()
    
    # Generate all figures
    print("Generating figures...")
    print("-"*40)
    
    plot_fig1_time_comparison(data, OUTPUT_DIR)
    plot_fig2_qpu_breakdown(data, OUTPUT_DIR)
    plot_fig3_scaling(data, OUTPUT_DIR)
    plot_fig4_quantum_advantage(data, OUTPUT_DIR)
    plot_fig5_summary(data, OUTPUT_DIR)
    plot_fig6_formulation_split(data, OUTPUT_DIR)
    plot_fig7_violation_analysis(data, OUTPUT_DIR)
    
    print("-"*40)
    print("\nâœ“ All figures generated successfully!")
    print(f"  Output location: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

