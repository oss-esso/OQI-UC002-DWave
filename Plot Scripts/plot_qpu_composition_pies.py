#!/usr/bin/env python3
"""
QPU Benchmark Solution Composition Plots

Creates professional pie charts and histograms matching the style of:
- solution_composition_pies.pdf
- solution_composition_histograms.pdf  
- solution_quality_comparison.pdf

Author: OQI-UC002-DWave Project
Date: 2025-12-02
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns
from collections import defaultdict, Counter

# Use LaTeX-like rendering for professional look
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
QPU_RESULTS_DIR = PROJECT_ROOT / "@todo" / "qpu_benchmark_results"
OUTPUT_DIR = PROJECT_ROOT / "professional_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Food groups
FOOD_GROUPS = {
    'Meats': ['Beef', 'Chicken', 'Egg', 'Lamb', 'Pork'],
    'Fruits': ['Apple', 'Avocado', 'Banana', 'Durian', 'Guava', 'Mango', 'Orange', 'Papaya', 'Watermelon'],
    'Legumes': ['Chickpeas', 'Peanuts', 'Tempeh', 'Tofu'],
    'Grains': ['Corn', 'Potato'],
    'Vegetables': ['Cabbage', 'Cucumber', 'Eggplant', 'Long bean', 'Pumpkin', 'Spinach', 'Tomatoes']
}

FOOD_TO_GROUP = {}
for group, foods in FOOD_GROUPS.items():
    for food in foods:
        FOOD_TO_GROUP[food] = group

# Qualitative color palette (matching plot_benchmark_results.py)
QUALITATIVE_COLORS = [
    '#E63946', '#F4A261', '#2A9D8F', '#264653', '#E9C46A',
    '#8338EC', '#06FFA5', '#FF6B6B', '#4ECDC4', '#95E1D3',
    '#F38181', '#AA96DA', '#FCBAD3', '#A8D8EA', '#FFD93D',
]

# Method colors
METHOD_COLORS = {
    'Gurobi': '#E63946',
    'PuLP': '#E63946',
    'PlotBased_QPU': '#06FFA5',
    'Multilevel(5)_QPU': '#2EC4B6',
    'Multilevel(10)_QPU': '#20A39E',
    'Louvain_QPU': '#3DDC97',
    'Spectral(10)_QPU': '#5CDB95',
    'cqm_first_PlotBased': '#8338EC',
    'coordinated': '#FF6B6B',
}

METHOD_DISPLAY_NAMES = {
    'Gurobi': 'Gurobi (Optimal)',
    'PlotBased_QPU': 'PlotBased QPU',
    'Multilevel(5)_QPU': 'Multilevel(5) QPU',
    'Multilevel(10)_QPU': 'Multilevel(10) QPU',
    'Louvain_QPU': 'Louvain QPU',
    'Spectral(10)_QPU': 'Spectral(10) QPU',
    'cqm_first_PlotBased': 'CQM-First PlotBased',
    'coordinated': 'Coordinated',
}


def load_qpu_benchmark_data():
    """Load QPU benchmark results."""
    data = {'small_scale': None, 'large_scale': None}
    
    small_file = QPU_RESULTS_DIR / "qpu_benchmark_20251201_160444.json"
    large_file = QPU_RESULTS_DIR / "qpu_benchmark_20251201_200012.json"
    
    if small_file.exists():
        with open(small_file) as f:
            data['small_scale'] = json.load(f)
        print(f"  Loaded: {small_file.name}")
    
    if large_file.exists():
        with open(large_file) as f:
            data['large_scale'] = json.load(f)
        print(f"  Loaded: {large_file.name}")
    
    return data


def extract_compositions(qpu_data):
    """
    Extract solution compositions in format: {method: {scale: {crop: percentage}}}
    """
    compositions = defaultdict(dict)
    
    for scale_type in ['small_scale', 'large_scale']:
        if qpu_data[scale_type] is None:
            continue
        
        for result in qpu_data[scale_type]['results']:
            n_farms = result['n_farms']
            total_area = result['metadata'].get('total_area', 100.0)
            
            # Ground truth / Gurobi
            if 'ground_truth' in result and 'solution' in result['ground_truth']:
                sol = result['ground_truth']['solution']
                crop_areas = extract_crop_areas(sol, n_farms)
                if crop_areas:
                    compositions['Gurobi'][n_farms] = {
                        crop: (area / total_area) * 100 
                        for crop, area in crop_areas.items()
                    }
            
            # QPU methods
            for method_name, method_result in result.get('method_results', {}).items():
                if not method_result.get('success', False):
                    continue
                if 'solution' not in method_result:
                    continue
                
                sol = method_result['solution']
                display_name = method_name.replace('decomposition_', '')
                
                crop_areas = extract_crop_areas(sol, n_farms)
                if crop_areas:
                    compositions[display_name][n_farms] = {
                        crop: (area / total_area) * 100 
                        for crop, area in crop_areas.items()
                    }
    
    return dict(compositions)


def extract_crop_areas(solution, n_farms):
    """Extract crop area percentages from solution."""
    crop_counts = Counter()
    area_per_farm = 100.0 / n_farms  # Total 100 hectares
    
    if 'Y' in solution:
        for patch, crops in solution['Y'].items():
            for crop, val in crops.items():
                if val == 1:
                    crop_counts[crop] += area_per_farm
    elif 'allocations' in solution:
        for alloc_key, area in solution['allocations'].items():
            parts = alloc_key.rsplit('_', 1)
            if len(parts) == 2:
                crop = parts[1]
                crop_counts[crop] += area
    
    return dict(crop_counts)


def extract_quality_metrics(qpu_data):
    """
    Extract quality metrics in format: {method: {metric_name: [values per scale]}}
    """
    metrics = defaultdict(lambda: {
        'n_farms': [],
        'objective_value': [],
        'utilization': [],
        'n_crops': [],
        'violations': [],
        'pass_rate': [],
    })
    
    for scale_type in ['small_scale', 'large_scale']:
        if qpu_data[scale_type] is None:
            continue
        
        for result in qpu_data[scale_type]['results']:
            n_farms = result['n_farms']
            total_area = result['metadata'].get('total_area', 100.0)
            
            # Ground truth / Gurobi
            if 'ground_truth' in result:
                gt = result['ground_truth']
                sol = gt.get('solution', {})
                
                metrics['Gurobi']['n_farms'].append(n_farms)
                metrics['Gurobi']['objective_value'].append(gt.get('objective', 0))
                metrics['Gurobi']['violations'].append(gt.get('violations', 0))
                metrics['Gurobi']['pass_rate'].append(1.0 if gt.get('violations', 0) == 0 else 0.0)
                
                # Calculate utilization and diversity from solution
                crop_areas = extract_crop_areas(sol, n_farms)
                total_allocated = sum(crop_areas.values()) if crop_areas else 0
                metrics['Gurobi']['utilization'].append(total_allocated / total_area)
                metrics['Gurobi']['n_crops'].append(len(crop_areas))
            
            # QPU methods
            for method_name, method_result in result.get('method_results', {}).items():
                if not method_result.get('success', False):
                    continue
                
                display_name = method_name.replace('decomposition_', '')
                sol = method_result.get('solution', {})
                
                metrics[display_name]['n_farms'].append(n_farms)
                metrics[display_name]['objective_value'].append(method_result.get('objective', 0))
                violations = method_result.get('violations', 0)
                metrics[display_name]['violations'].append(violations)
                metrics[display_name]['pass_rate'].append(1.0 if violations == 0 else 0.0)
                
                crop_areas = extract_crop_areas(sol, n_farms)
                total_allocated = sum(crop_areas.values()) if crop_areas else 0
                metrics[display_name]['utilization'].append(total_allocated / total_area)
                metrics[display_name]['n_crops'].append(len(crop_areas))
    
    return dict(metrics)


def plot_solution_composition_pies(compositions, output_path):
    """
    Create pie charts showing solution composition across methods.
    Matches style of solution_composition_pies.pdf
    """
    methods = list(compositions.keys())
    
    # Find all problem sizes
    all_sizes = set()
    for method_compositions in compositions.values():
        all_sizes.update(method_compositions.keys())
    all_sizes = sorted(all_sizes)
    
    if not all_sizes:
        print("  Warning: No composition data for pie charts")
        return
    
    # Limit to 4 sizes for clarity
    display_sizes = all_sizes[:4] if len(all_sizes) > 4 else all_sizes
    
    # Create unified color mapping
    all_crops = set()
    for method_compositions in compositions.values():
        for composition in method_compositions.values():
            all_crops.update(composition.keys())
    all_crops = sorted(list(all_crops))
    crop_colors = {crop: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)] 
                   for i, crop in enumerate(all_crops)}
    
    fig, axes = plt.subplots(len(display_sizes), len(methods), 
                             figsize=(3.5 * len(methods), 3.5 * len(display_sizes)))
    fig.suptitle('Solution Composition Analysis (QPU Methods)', fontsize=14, fontweight='bold', y=0.98)
    
    # Ensure 2D array
    if len(display_sizes) == 1:
        axes = axes.reshape(1, -1)
    if len(methods) == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, method in enumerate(methods):
        method_compositions = compositions[method]
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        
        for row_idx, problem_size in enumerate(display_sizes):
            ax = axes[row_idx, col_idx]
            
            if problem_size not in method_compositions:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=11)
                ax.set_title(f'{problem_size} Farms', fontsize=10)
                ax.axis('off')
                continue
            
            composition = method_compositions[problem_size]
            
            # Sort and take top 8
            sorted_items = sorted(composition.items(), key=lambda x: -x[1])
            top_items = sorted_items[:8]
            other_pct = sum(item[1] for item in sorted_items[8:])
            
            labels = []
            sizes = []
            colors = []
            
            for crop, pct in top_items:
                labels.append(f'{crop}\n({pct:.1f}%)')
                sizes.append(pct)
                colors.append(crop_colors.get(crop, '#888888'))
            
            if other_pct > 0.5:
                labels.append(f'Other\n({other_pct:.1f}%)')
                sizes.append(other_pct)
                colors.append('#888888')
            
            if sizes:
                wedges, texts = ax.pie(sizes, labels=labels, colors=colors, 
                                       startangle=90, textprops={'fontsize': 7})
            
            # Title
            if row_idx == 0:
                ax.set_title(f'{display_name}\n{problem_size} Farms', fontsize=10, pad=10)
            else:
                ax.set_title(f'{problem_size} Farms', fontsize=10, pad=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_solution_composition_histograms(compositions, output_path):
    """
    Create histograms showing solution composition (log scale).
    Matches style of solution_composition_histograms.pdf
    """
    methods = list(compositions.keys())
    
    all_sizes = set()
    for method_compositions in compositions.values():
        all_sizes.update(method_compositions.keys())
    all_sizes = sorted(all_sizes)
    
    if not all_sizes:
        print("  Warning: No composition data for histograms")
        return
    
    display_sizes = all_sizes[:4] if len(all_sizes) > 4 else all_sizes
    
    # Unified colors
    all_crops = set()
    for method_compositions in compositions.values():
        for composition in method_compositions.values():
            all_crops.update(composition.keys())
    all_crops = sorted(list(all_crops))
    crop_colors = {crop: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)] 
                   for i, crop in enumerate(all_crops)}
    
    fig, axes = plt.subplots(len(display_sizes), len(methods), 
                             figsize=(4 * len(methods), 4 * len(display_sizes)))
    fig.suptitle('Solution Composition Analysis (Log Scale)', fontsize=14, fontweight='bold', y=0.98)
    
    if len(display_sizes) == 1:
        axes = axes.reshape(1, -1)
    if len(methods) == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, method in enumerate(methods):
        method_compositions = compositions[method]
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        
        for row_idx, problem_size in enumerate(display_sizes):
            ax = axes[row_idx, col_idx]
            
            if problem_size not in method_compositions:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=11)
                ax.set_title(f'{problem_size} Farms', fontsize=10)
                continue
            
            composition = method_compositions[problem_size]
            
            # Sort and take top 8
            sorted_items = sorted(composition.items(), key=lambda x: -x[1])
            top_items = sorted_items[:8]
            other_pct = sum(item[1] for item in sorted_items[8:])
            
            crops = []
            percentages = []
            colors = []
            
            for crop, pct in top_items:
                crops.append(crop)
                percentages.append(pct)
                colors.append(crop_colors.get(crop, '#888888'))
            
            if other_pct > 0.5:
                crops.append('Other')
                percentages.append(other_pct)
                colors.append('#888888')
            
            if percentages:
                x_pos = np.arange(len(crops))
                bars = ax.bar(x_pos, percentages, color=colors, edgecolor='black', linewidth=0.5)
                
                ax.set_yscale('log')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(crops, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Area (%)', fontsize=9)
                ax.grid(True, alpha=0.3, which='both', axis='y')
            
            if row_idx == 0:
                ax.set_title(f'{display_name}\n{problem_size} Farms', fontsize=10, pad=10)
            else:
                ax.set_title(f'{problem_size} Farms', fontsize=10, pad=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_solution_quality_comparison(metrics, output_path):
    """
    Create solution quality comparison.
    Matches style of solution_quality_comparison.pdf
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Solution Quality Comparison (QPU Methods)', fontsize=14, fontweight='bold', y=0.98)
    
    methods = list(metrics.keys())
    
    # Plot 1: Land Utilization
    ax = axes[0, 0]
    for method in methods:
        m = metrics[method]
        if not m['n_farms']:
            continue
        color = METHOD_COLORS.get(method, '#888888')
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        
        problem_size = [n * 27 for n in m['n_farms']]
        utilization_pct = [u * 100 for u in m['utilization']]
        ax.plot(problem_size, utilization_pct, 'o-', color=color, label=display_name, 
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Full Utilization')
    ax.set_xlabel('Problem Size (Farms × Foods)')
    ax.set_ylabel('Land Utilization (%)')
    ax.set_title('Resource Utilization', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.set_ylim(0, 150)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Crop Diversity
    ax = axes[0, 1]
    for method in methods:
        m = metrics[method]
        if not m['n_farms']:
            continue
        color = METHOD_COLORS.get(method, '#888888')
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        
        problem_size = [n * 27 for n in m['n_farms']]
        ax.plot(problem_size, m['n_crops'], 'o-', color=color, label=display_name,
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel('Problem Size (Farms × Foods)')
    ax.set_ylabel('Number of Crops')
    ax.set_title('Crop Diversity', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Constraint Satisfaction
    ax = axes[1, 0]
    for method in methods:
        m = metrics[method]
        if not m['n_farms']:
            continue
        color = METHOD_COLORS.get(method, '#888888')
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        
        problem_size = [n * 27 for n in m['n_farms']]
        pass_rates = [p * 100 for p in m['pass_rate']]
        ax.plot(problem_size, pass_rates, 'o-', color=color, label=display_name,
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Problem Size (Farms × Foods)')
    ax.set_ylabel('Constraint Pass Rate (%)')
    ax.set_title('Constraint Satisfaction', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Solution Value per Unit
    ax = axes[1, 1]
    for method in methods:
        m = metrics[method]
        if not m['n_farms'] or not m['objective_value']:
            continue
        color = METHOD_COLORS.get(method, '#888888')
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        
        problem_size = [n * 27 for n in m['n_farms']]
        obj_per_unit = [obj/n if n > 0 else 0 for obj, n in zip(m['objective_value'], m['n_farms'])]
        ax.plot(problem_size, obj_per_unit, 'o-', color=color, label=display_name,
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.set_xlabel('Problem Size (Farms × Foods)')
    ax.set_ylabel('Objective per Farm')
    ax.set_title('Solution Efficiency', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_land_utilization_pies(compositions, output_path):
    """
    Create pie charts showing land utilization by food group.
    """
    methods = list(compositions.keys())
    
    all_sizes = set()
    for method_compositions in compositions.values():
        all_sizes.update(method_compositions.keys())
    all_sizes = sorted(all_sizes)
    
    if not all_sizes:
        return
    
    # Use largest size for comparison
    display_size = all_sizes[-1]
    
    group_colors = {
        'Meats': '#E63946',
        'Fruits': '#F4A261',
        'Legumes': '#2A9D8F',
        'Grains': '#E9C46A',
        'Vegetables': '#264653',
        'Unallocated': '#CCCCCC',
    }
    
    n_methods = len(methods)
    cols = min(4, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(f'Land Utilization by Food Group ({display_size} Farms)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    axes = np.array(axes).flatten() if n_methods > 1 else [axes]
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        if display_size not in compositions[method]:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(METHOD_DISPLAY_NAMES.get(method, method))
            ax.axis('off')
            continue
        
        composition = compositions[method][display_size]
        
        # Aggregate by food group
        group_totals = defaultdict(float)
        for crop, pct in composition.items():
            group = FOOD_TO_GROUP.get(crop, 'Other')
            group_totals[group] += pct
        
        total_allocated = sum(group_totals.values())
        if total_allocated < 100:
            group_totals['Unallocated'] = 100 - total_allocated
        
        labels = []
        sizes = []
        colors = []
        
        for group in ['Vegetables', 'Grains', 'Legumes', 'Fruits', 'Meats', 'Unallocated']:
            if group in group_totals and group_totals[group] > 0.5:
                labels.append(f'{group}\n({group_totals[group]:.1f}%)')
                sizes.append(group_totals[group])
                colors.append(group_colors.get(group, '#888888'))
        
        if sizes:
            wedges, texts = ax.pie(sizes, labels=labels, colors=colors, startangle=90,
                                   textprops={'fontsize': 8})
        
        ax.set_title(METHOD_DISPLAY_NAMES.get(method, method), fontsize=10)
    
    # Hide unused axes
    for idx in range(len(methods), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("QPU Benchmark Solution Composition Plots")
    print("=" * 80)
    print()
    
    print("[1/3] Loading QPU benchmark data...")
    qpu_data = load_qpu_benchmark_data()
    
    print("\n[2/3] Extracting compositions and metrics...")
    compositions = extract_compositions(qpu_data)
    metrics = extract_quality_metrics(qpu_data)
    
    print(f"  Found {len(compositions)} methods with composition data")
    for method, data in compositions.items():
        print(f"    - {method}: scales {sorted(data.keys())}")
    
    print("\n[3/3] Creating plots...")
    
    # 1. Solution composition pie charts
    print("  Creating solution composition pie charts...")
    plot_solution_composition_pies(
        compositions, 
        OUTPUT_DIR / "qpu_solution_composition_pies.png"
    )
    
    # 2. Solution composition histograms (log scale)
    print("  Creating solution composition histograms...")
    plot_solution_composition_histograms(
        compositions,
        OUTPUT_DIR / "qpu_solution_composition_histograms.png"
    )
    
    # 3. Solution quality comparison
    print("  Creating solution quality comparison...")
    plot_solution_quality_comparison(
        metrics,
        OUTPUT_DIR / "qpu_solution_quality_comparison.png"
    )
    
    # 4. Land utilization pie charts
    print("  Creating land utilization pies...")
    plot_land_utilization_pies(
        compositions,
        OUTPUT_DIR / "qpu_land_utilization_pies.png"
    )
    
    print("\n" + "=" * 80)
    print("✅ All plots generated successfully!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("\n   Generated files:")
    print("   - qpu_solution_composition_pies.png/pdf")
    print("   - qpu_solution_composition_histograms.png/pdf")
    print("   - qpu_solution_quality_comparison.png/pdf")
    print("   - qpu_land_utilization_pies.png/pdf")
    print("=" * 80)


if __name__ == "__main__":
    main()
