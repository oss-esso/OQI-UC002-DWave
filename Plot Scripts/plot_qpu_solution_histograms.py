#!/usr/bin/env python3
"""
QPU Benchmark Solution Histograms

Creates professional histograms showing:
1. Crop allocation distribution by method
2. Food group composition comparison
3. Unique crops selected across methods
4. Land utilization patterns

Author: OQI-UC002-DWave Project
Date: 2025-12-02
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import unified plot configuration
from plot_config import (
    setup_publication_style, QUALITATIVE_COLORS, METHOD_COLORS,
    FOOD_GROUP_COLORS, FOOD_GROUPS, FOOD_TO_GROUP,
    save_figure, get_crop_color, get_method_color, METHOD_DISPLAY_NAMES
)

# Apply publication style
setup_publication_style()

# Paths
QPU_RESULTS_DIR = PROJECT_ROOT / "@todo" / "qpu_benchmark_results"
OUTPUT_DIR = PROJECT_ROOT / "professional_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Food groups and colors are imported from plot_config
    'Fruits': ['Apple', 'Avocado', 'Banana', 'Durian', 'Guava', 'Mango', 'Orange', 'Papaya', 'Watermelon'],
    'Legumes': ['Chickpeas', 'Peanuts', 'Tempeh', 'Tofu'],
    'Grains': ['Corn', 'Potato'],
    'Vegetables': ['Cabbage', 'Cucumber', 'Eggplant', 'Long bean', 'Pumpkin', 'Spinach', 'Tomatoes']
}

# Reverse lookup: food -> group
FOOD_TO_GROUP = {}
for group, foods in FOOD_GROUPS.items():
    for food in foods:
        FOOD_TO_GROUP[food] = group

# Color scheme
COLORS = {
    # Methods
    'Gurobi': '#E63946',
    'PuLP': '#E63946',
    'PlotBased_QPU': '#06FFA5',
    'Multilevel(5)_QPU': '#2EC4B6',
    'Multilevel(10)_QPU': '#20A39E',
    'Louvain_QPU': '#3DDC97',
    'Spectral(10)_QPU': '#5CDB95',
    'cqm_first_PlotBased': '#8338EC',
    'coordinated': '#FF6B6B',
    
    # Food groups
    'Meats': '#E63946',
    'Fruits': '#F4A261',
    'Legumes': '#2A9D8F',
    'Grains': '#E9C46A',
    'Vegetables': '#264653',
}

# Food-specific colors
FOOD_COLORS = {
    # Meats - reds
    'Beef': '#8B0000', 'Chicken': '#CD5C5C', 'Egg': '#F08080', 'Lamb': '#DC143C', 'Pork': '#FF6347',
    # Fruits - oranges/yellows
    'Apple': '#FF4500', 'Avocado': '#9ACD32', 'Banana': '#FFD700', 'Durian': '#DAA520', 
    'Guava': '#FF69B4', 'Mango': '#FFA500', 'Orange': '#FF8C00', 'Papaya': '#FFDAB9', 'Watermelon': '#FF6B6B',
    # Legumes - greens
    'Chickpeas': '#8FBC8F', 'Peanuts': '#D2B48C', 'Tempeh': '#BC8F8F', 'Tofu': '#F5F5DC',
    # Grains - browns/yellows
    'Corn': '#F0E68C', 'Potato': '#DEB887',
    # Vegetables - blues/greens
    'Cabbage': '#90EE90', 'Cucumber': '#98FB98', 'Eggplant': '#9370DB', 
    'Long bean': '#3CB371', 'Pumpkin': '#FF7F50', 'Spinach': '#228B22', 'Tomatoes': '#FF4500',
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


def extract_solution_data(qpu_data):
    """Extract solution compositions from benchmark data."""
    solutions = {}
    
    for scale_type in ['small_scale', 'large_scale']:
        if qpu_data[scale_type] is None:
            continue
        
        for result in qpu_data[scale_type]['results']:
            n_farms = result['n_farms']
            
            # Ground truth / Gurobi
            if 'ground_truth' in result and 'solution' in result['ground_truth']:
                sol = result['ground_truth']['solution']
                key = (n_farms, 'Gurobi')
                solutions[key] = extract_crop_counts(sol)
            
            # QPU methods
            for method_name, method_result in result.get('method_results', {}).items():
                if not method_result.get('success', False):
                    continue
                if 'solution' not in method_result:
                    continue
                
                sol = method_result['solution']
                display_name = method_name.replace('decomposition_', '')
                key = (n_farms, display_name)
                solutions[key] = extract_crop_counts(sol)
    
    return solutions


def extract_crop_counts(solution):
    """Extract crop counts from a solution dict."""
    crop_counts = Counter()
    
    if 'Y' in solution:
        # Structured format: Y[patch][crop] = 0/1
        for patch, crops in solution['Y'].items():
            for crop, val in crops.items():
                if val == 1:
                    crop_counts[crop] += 1
    elif 'allocations' in solution:
        # Flat format with allocations
        for alloc_key in solution['allocations']:
            # Format: Patch1_Spinach
            parts = alloc_key.rsplit('_', 1)
            if len(parts) == 2:
                crop = parts[1]
                crop_counts[crop] += 1
    
    # Also get unique crops from U if available
    unique_crops = []
    if 'U' in solution:
        unique_crops = [crop for crop, val in solution['U'].items() if val == 1]
    elif 'summary' in solution and 'foods_used' in solution['summary']:
        unique_crops = solution['summary']['foods_used']
    
    return {
        'crop_counts': dict(crop_counts),
        'unique_crops': unique_crops,
        'total_allocated': sum(crop_counts.values()),
    }


def get_food_group_counts(crop_counts):
    """Convert crop counts to food group counts."""
    group_counts = Counter()
    for crop, count in crop_counts.items():
        group = FOOD_TO_GROUP.get(crop, 'Other')
        group_counts[group] += count
    return dict(group_counts)


def plot_crop_distribution_comparison(solutions, scales, output_path):
    """
    Create histogram comparing crop distributions across methods for selected scales.
    """
    n_scales = len(scales)
    fig, axes = plt.subplots(n_scales, 1, figsize=(16, 5 * n_scales))
    if n_scales == 1:
        axes = [axes]
    
    fig.suptitle('Crop Allocation Distribution by Method', fontsize=18, fontweight='bold', y=1.02)
    
    # Get all crops that appear in any solution
    all_crops = set()
    for key, sol_data in solutions.items():
        all_crops.update(sol_data['crop_counts'].keys())
    all_crops = sorted(all_crops)
    
    for ax_idx, scale in enumerate(scales):
        ax = axes[ax_idx]
        
        # Find methods for this scale
        methods = [key[1] for key in solutions.keys() if key[0] == scale]
        methods = sorted(set(methods))
        
        if not methods:
            ax.text(0.5, 0.5, f'No data for scale {scale}', ha='center', va='center', fontsize=14)
            ax.set_title(f'{scale} Farms')
            continue
        
        x = np.arange(len(all_crops))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            key = (scale, method)
            if key not in solutions:
                continue
            
            counts = [solutions[key]['crop_counts'].get(crop, 0) for crop in all_crops]
            color = COLORS.get(method, '#888888')
            ax.bar(x + i * width, counts, width, label=method, color=color, alpha=0.8, edgecolor='white')
        
        ax.set_xlabel('Crop', fontsize=12)
        ax.set_ylabel('Number of Farms', fontsize=12)
        ax.set_title(f'{scale} Farms: Crop Allocation by Method', fontsize=14)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(all_crops, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_food_group_composition(solutions, scales, output_path):
    """
    Create stacked bar charts showing food group composition by method.
    """
    n_scales = len(scales)
    fig, axes = plt.subplots(1, n_scales, figsize=(6 * n_scales, 8))
    if n_scales == 1:
        axes = [axes]
    
    fig.suptitle('Food Group Composition by Method and Scale', fontsize=18, fontweight='bold', y=1.02)
    
    group_order = ['Vegetables', 'Grains', 'Legumes', 'Fruits', 'Meats']
    
    for ax_idx, scale in enumerate(scales):
        ax = axes[ax_idx]
        
        # Find methods for this scale
        methods = sorted(set([key[1] for key in solutions.keys() if key[0] == scale]))
        
        if not methods:
            ax.text(0.5, 0.5, f'No data', ha='center', va='center', fontsize=14)
            ax.set_title(f'{scale} Farms')
            continue
        
        # Build stacked bar data
        bottom = np.zeros(len(methods))
        
        for group in group_order:
            counts = []
            for method in methods:
                key = (scale, method)
                if key in solutions:
                    group_counts = get_food_group_counts(solutions[key]['crop_counts'])
                    counts.append(group_counts.get(group, 0))
                else:
                    counts.append(0)
            
            ax.bar(methods, counts, bottom=bottom, label=group, 
                   color=COLORS.get(group, '#888888'), alpha=0.85, edgecolor='white')
            bottom = bottom + np.array(counts)
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Number of Farms', fontsize=12)
        ax.set_title(f'{scale} Farms', fontsize=14)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        if ax_idx == 0:
            ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_unique_crops_heatmap(solutions, output_path):
    """
    Create a heatmap showing which crops are selected by each method at each scale.
    """
    # Get all scales and methods
    all_scales = sorted(set([key[0] for key in solutions.keys()]))
    all_methods = sorted(set([key[1] for key in solutions.keys()]))
    
    # Get all unique crops
    all_crops = set()
    for sol_data in solutions.values():
        all_crops.update(sol_data['unique_crops'])
    all_crops = sorted(all_crops)
    
    if not all_crops:
        print("  Warning: No unique crop data available for heatmap")
        return
    
    # Create figure with subplots for each scale
    n_scales = len(all_scales)
    fig, axes = plt.subplots(1, n_scales, figsize=(4 + 2 * n_scales, max(8, len(all_crops) * 0.4)))
    if n_scales == 1:
        axes = [axes]
    
    fig.suptitle('Unique Crops Selected: Method × Crop Heatmap', fontsize=16, fontweight='bold', y=1.02)
    
    for ax_idx, scale in enumerate(all_scales):
        ax = axes[ax_idx]
        
        # Get methods for this scale
        methods = [m for m in all_methods if (scale, m) in solutions]
        
        if not methods:
            ax.set_title(f'{scale} Farms\n(No data)')
            ax.axis('off')
            continue
        
        # Build matrix: rows = crops, cols = methods
        matrix = np.zeros((len(all_crops), len(methods)))
        
        for j, method in enumerate(methods):
            key = (scale, method)
            if key in solutions:
                unique = solutions[key]['unique_crops']
                for i, crop in enumerate(all_crops):
                    if crop in unique:
                        matrix[i, j] = 1
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([m[:12] for m in methods], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(np.arange(len(all_crops)))
        ax.set_yticklabels(all_crops, fontsize=9)
        ax.set_title(f'{scale} Farms', fontsize=12)
        
        # Add grid
        ax.set_xticks(np.arange(len(methods) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(all_crops) + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        
        # Add count annotations
        for i in range(len(all_crops)):
            for j in range(len(methods)):
                if matrix[i, j] == 1:
                    ax.text(j, i, '✓', ha='center', va='center', fontsize=10, color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_solution_quality_histograms(solutions, output_path):
    """
    Create histograms comparing solution characteristics across methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Solution Characteristics Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    # Organize data by method (aggregate across scales)
    method_data = defaultdict(lambda: {'unique_crops': [], 'total_allocated': [], 'crop_diversity': []})
    
    for (scale, method), sol_data in solutions.items():
        method_data[method]['unique_crops'].append(len(sol_data['unique_crops']))
        method_data[method]['total_allocated'].append(sol_data['total_allocated'])
        method_data[method]['crop_diversity'].append(len(sol_data['crop_counts']))
    
    methods = sorted(method_data.keys())
    colors = [COLORS.get(m, '#888888') for m in methods]
    
    # Plot 1: Average unique crops per method
    ax = axes[0, 0]
    avg_unique = [np.mean(method_data[m]['unique_crops']) for m in methods]
    std_unique = [np.std(method_data[m]['unique_crops']) for m in methods]
    bars = ax.bar(methods, avg_unique, yerr=std_unique, color=colors, alpha=0.8, 
                  edgecolor='white', capsize=5)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Unique Crops Selected', fontsize=12)
    ax.set_title('Average Unique Crops per Method', fontsize=14)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, avg_unique):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Total farms allocated per method (normalized by scale)
    ax = axes[0, 1]
    avg_alloc = [np.mean(method_data[m]['total_allocated']) for m in methods]
    bars = ax.bar(methods, avg_alloc, color=colors, alpha=0.8, edgecolor='white')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Farms with Crops', fontsize=12)
    ax.set_title('Average Farms Allocated per Method', fontsize=14)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Crop diversity (number of different crops used)
    ax = axes[1, 0]
    
    # Box plot for crop diversity across scales
    box_data = [method_data[m]['unique_crops'] for m in methods]
    bp = ax.boxplot(box_data, labels=methods, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Unique Crops', fontsize=12)
    ax.set_title('Crop Diversity Distribution', fontsize=14)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Comparison of Gurobi vs best QPU method at each scale
    ax = axes[1, 1]
    
    scales = sorted(set([key[0] for key in solutions.keys()]))
    gurobi_unique = []
    best_qpu_unique = []
    best_qpu_method = []
    
    for scale in scales:
        if (scale, 'Gurobi') in solutions:
            gurobi_unique.append(len(solutions[(scale, 'Gurobi')]['unique_crops']))
        else:
            gurobi_unique.append(0)
        
        # Find best QPU method (most unique crops matching Gurobi)
        best_count = 0
        best_method = 'N/A'
        for method in methods:
            if method == 'Gurobi':
                continue
            key = (scale, method)
            if key in solutions:
                count = len(solutions[key]['unique_crops'])
                if count > best_count:
                    best_count = count
                    best_method = method
        best_qpu_unique.append(best_count)
        best_qpu_method.append(best_method)
    
    x = np.arange(len(scales))
    width = 0.35
    ax.bar(x - width/2, gurobi_unique, width, label='Gurobi (Optimal)', color=COLORS['Gurobi'], alpha=0.8)
    ax.bar(x + width/2, best_qpu_unique, width, label='Best QPU Method', color='#06FFA5', alpha=0.8)
    
    ax.set_xlabel('Scale (# Farms)', fontsize=12)
    ax.set_ylabel('Unique Crops Selected', fontsize=12)
    ax.set_title('Gurobi vs Best QPU: Unique Crops', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_detailed_scale_comparison(solutions, scale, output_path):
    """
    Create detailed comparison for a specific scale showing crop allocations.
    """
    # Get methods for this scale
    methods = sorted([key[1] for key in solutions.keys() if key[0] == scale])
    
    if len(methods) < 2:
        print(f"  Skipping scale {scale}: not enough methods")
        return
    
    n_methods = len(methods)
    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(6 * ((n_methods + 1) // 2), 12))
    axes = axes.flatten()
    
    fig.suptitle(f'Detailed Crop Allocation: {scale} Farms', fontsize=18, fontweight='bold', y=0.98)
    
    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        key = (scale, method)
        
        if key not in solutions:
            ax.axis('off')
            continue
        
        crop_counts = solutions[key]['crop_counts']
        
        if not crop_counts:
            ax.text(0.5, 0.5, 'No allocations', ha='center', va='center')
            ax.set_title(method)
            continue
        
        # Sort by count descending
        sorted_crops = sorted(crop_counts.items(), key=lambda x: -x[1])
        crops = [c[0] for c in sorted_crops]
        counts = [c[1] for c in sorted_crops]
        colors = [FOOD_COLORS.get(c, '#888888') for c in crops]
        
        bars = ax.barh(crops, counts, color=colors, alpha=0.85, edgecolor='white')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{count}', va='center', fontsize=9)
        
        ax.set_xlabel('Number of Farms', fontsize=11)
        ax.set_title(f'{method}\n({len(crop_counts)} crops, {sum(counts)} farms)', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
    # Hide unused axes
    for ax_idx in range(len(methods), len(axes)):
        axes[ax_idx].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("QPU Benchmark Solution Histograms")
    print("=" * 80)
    print()
    
    print("[1/3] Loading QPU benchmark data...")
    qpu_data = load_qpu_benchmark_data()
    
    print("\n[2/3] Extracting solution data...")
    solutions = extract_solution_data(qpu_data)
    print(f"  Found {len(solutions)} solutions across scales and methods")
    
    # List available scales
    scales = sorted(set([key[0] for key in solutions.keys()]))
    print(f"  Scales: {scales}")
    
    print("\n[3/3] Creating histograms...")
    
    # 1. Crop distribution comparison (small scales)
    small_scales = [s for s in scales if s <= 100]
    if small_scales:
        print("  Creating crop distribution comparison (small scales)...")
        plot_crop_distribution_comparison(
            solutions, small_scales[:3],  # Limit to 3 scales for readability
            OUTPUT_DIR / "qpu_solution_crop_distribution_small.png"
        )
    
    # 2. Crop distribution comparison (large scales)
    large_scales = [s for s in scales if s >= 200]
    if large_scales:
        print("  Creating crop distribution comparison (large scales)...")
        plot_crop_distribution_comparison(
            solutions, large_scales,
            OUTPUT_DIR / "qpu_solution_crop_distribution_large.png"
        )
    
    # 3. Food group composition
    print("  Creating food group composition...")
    plot_food_group_composition(
        solutions, scales[:4] if len(scales) > 4 else scales,
        OUTPUT_DIR / "qpu_solution_food_groups.png"
    )
    
    # 4. Unique crops heatmap
    print("  Creating unique crops heatmap...")
    plot_unique_crops_heatmap(solutions, OUTPUT_DIR / "qpu_solution_unique_crops_heatmap.png")
    
    # 5. Solution quality histograms
    print("  Creating solution quality histograms...")
    plot_solution_quality_histograms(solutions, OUTPUT_DIR / "qpu_solution_quality_histograms.png")
    
    # 6. Detailed comparison for key scales
    for scale in [100, 500, 1000]:
        if scale in scales:
            print(f"  Creating detailed comparison for {scale} farms...")
            plot_detailed_scale_comparison(
                solutions, scale,
                OUTPUT_DIR / f"qpu_solution_detail_{scale}farms.png"
            )
    
    print("\n" + "=" * 80)
    print("✅ All histograms generated successfully!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
