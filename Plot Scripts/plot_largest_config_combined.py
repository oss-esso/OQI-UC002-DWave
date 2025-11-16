#!/usr/bin/env python3
"""
Reduced Visualization: Largest Configuration Only
Creates a single plot with 12 subplots (4 rows × 3 columns):
- Top 2 rows: 6 pie charts for each solver's largest configuration
- Bottom 2 rows: 6 histograms for each solver's largest configuration
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
from typing import Dict, List, Any

# =============================================================================
# CUSTOM STYLE DEFINITION
# =============================================================================

def configure_professional_style():
    """Configure professional matplotlib style with LaTeX integration."""
    
    rcParams.update({
        # Font configuration
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'serif'],
        'font.size': 11,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
        
        # Text rendering with LaTeX
        'text.usetex': True,
        'text.latex.preamble': r'''
            \usepackage{amsmath}
            \usepackage{amsfonts}
            \usepackage{amssymb}
            \usepackage{bm}
            \usepackage{xcolor}
        ''',
        
        # Figure layout
        'figure.figsize': (15, 16),
        'figure.dpi': 300,
        'figure.constrained_layout.use': True,
        
        # Axes configuration
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'axes.grid.axis': 'both',
        'axes.labelpad': 8,
        'axes.titlepad': 12,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        
        # Tick configuration
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        
        # Grid configuration
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Legend configuration
        'legend.fontsize': 8,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'legend.shadow': False,
        
        # Lines configuration
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1.2,
        
        # Savefig configuration
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.transparent': False,
    })

# =============================================================================
# COLOR AND STYLE DEFINITIONS
# =============================================================================

class ColorPalette:
    """Professional color palette for consistent visualization."""
    
    # Qualitative palette for crops/items
    QUALITATIVE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    # Semantic colors
    SEMANTIC = {
        'neutral': '#6c757d',
    }

# =============================================================================
# SOLVER CONFIGURATION
# =============================================================================

SOLVER_CONFIGS = {
    'Patch_PuLP': {
        'dir_name': 'Patch_PuLP',
        'display_name': r'$\text{Patch PuLP}$',
        'unit_label': 'Patches',
    },
    'Patch_GurobiQUBO': {
        'dir_name': 'Patch_GurobiQUBO',
        'display_name': r'$\text{Patch Gurobi QUBO}$',
        'unit_label': 'Patches',
    },
    'Patch_DWave': {
        'dir_name': 'Patch_DWave',
        'display_name': r'$\text{Patch D-Wave CQM}$',
        'unit_label': 'Patches',
    },
    'Patch_DWaveBQM': {
        'dir_name': 'Patch_DWaveBQM',
        'display_name': r'$\text{Patch D-Wave BQM}$',
        'unit_label': 'Patches',
    },
    'Farm_PuLP': {
        'dir_name': 'Farm_PuLP',
        'display_name': r'$\text{Farm PuLP}$',
        'unit_label': 'Farms',
    },
    'Farm_DWave': {
        'dir_name': 'Farm_DWave',
        'display_name': r'$\text{Farm D-Wave CQM}$',
        'unit_label': 'Farms',
    },
}

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_benchmark_data(benchmark_dir: Path, solver_key: str) -> Dict[int, Dict]:
    """Load benchmark data for a specific solver with error handling."""
    config = SOLVER_CONFIGS[solver_key]
    solver_dir = benchmark_dir / "COMPREHENSIVE" / config['dir_name']

    if not solver_dir.exists():
        raise FileNotFoundError(f"Solver directory not found: {solver_dir}")

    data = {}
    config_files = list(solver_dir.glob("config_*_run_*.json"))

    if not config_files:
        raise FileNotFoundError(f"No configuration files found in {solver_dir}")

    for config_file in sorted(config_files):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            n_units = config_data['n_units']
            data[n_units] = config_data

    return data

def extract_solution_composition(data: Dict[int, Dict], solver_key: str) -> Dict[int, Dict[str, float]]:
    """Extract solution composition for pie charts."""
    composition = {}

    for n_units, config_data in data.items():
        crop_areas = {}
        if 'solution_summary' in config_data:
            plot_assignments = config_data['solution_summary']['plot_assignments']
            for assignment in plot_assignments:
                crop = assignment['crop']
                area = assignment.get('area', assignment.get('total_area', 0))
                crop_areas[crop] = crop_areas.get(crop, 0) + area
        elif 'solution' in config_data and 'land_allocations' in config_data['solution']:
            # Fallback for Farm_PuLP (old format)
            land_allocations = config_data['solution']['land_allocations']
            for allocation in land_allocations:
                crop = allocation['crop']
                area = allocation['area']
                crop_areas[crop] = crop_areas.get(crop, 0) + area
        elif 'solution_areas' in config_data:
            # Farm_PuLP format: solution_areas has "Farm#_FoodName": area
            solution_areas = config_data['solution_areas']
            for farm_food, area in solution_areas.items():
                # Extract food name from "Farm#_FoodName"
                if '_' in farm_food:
                    parts = farm_food.split('_', 1)
                    if len(parts) == 2:
                        crop = parts[1]
                        # Include all non-zero allocations
                        if area > 0:
                            crop_areas[crop] = crop_areas.get(crop, 0) + area

        # Calculate percentages
        total_area = sum(crop_areas.values())
        if total_area > 0:
            crop_percentages = {crop: (area / total_area) * 100 for crop, area in crop_areas.items()}
            composition[n_units] = crop_percentages

    return composition

# =============================================================================
# MAIN PLOTTING FUNCTION
# =============================================================================

def plot_largest_config_combined(all_compositions: Dict[str, Dict], output_path: Path):
    """Create a single plot with 12 subplots (6 pie + 6 histogram) for largest config."""
    solvers = list(all_compositions.keys())
    
    if len(solvers) != 6:
        print(f"⚠ Warning: Expected 6 solvers, got {len(solvers)}")
    
    # Find largest configuration for each solver
    largest_configs = {}
    for solver, compositions in all_compositions.items():
        if compositions:
            largest_configs[solver] = max(compositions.keys())
    
    if not largest_configs:
        print("⚠ No composition data available")
        return
    
    # Create unified color mapping for crops across all plots
    all_crops = set()
    for solver, compositions in all_compositions.items():
        if solver in largest_configs:
            largest_size = largest_configs[solver]
            if largest_size in compositions:
                all_crops.update(compositions[largest_size].keys())
    all_crops = sorted(list(all_crops))
    
    crop_colors = {crop: ColorPalette.QUALITATIVE[i % len(ColorPalette.QUALITATIVE)] 
                  for i, crop in enumerate(all_crops)}
    
    # Create figure with 4 rows × 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    fig.suptitle(r'\textbf{Solution Composition - Largest Configuration}', fontsize=18, y=0.995)
    
    # Plot pie charts in first 2 rows (6 solvers)
    for idx, solver in enumerate(solvers):
        if solver not in largest_configs:
            continue
            
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        config = SOLVER_CONFIGS[solver]
        largest_size = largest_configs[solver]
        composition = all_compositions[solver][largest_size]
        
        # Prepare data for pie chart
        labels = []
        sizes = []
        colors = []
        
        # Sort by percentage (descending) and take top 8, group rest as "Other"
        sorted_items = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:8]
        other_percentage = sum(item[1] for item in sorted_items[8:])
        
        for crop, percentage in top_items:
            labels.append(f'{crop}\n({percentage:.1f}\\%)')
            sizes.append(percentage)
            colors.append(crop_colors.get(crop, ColorPalette.SEMANTIC['neutral']))
        
        if other_percentage > 0.5:  # Only show "Other" if significant
            labels.append(f'Other\n({other_percentage:.1f}\\%)')
            sizes.append(other_percentage)
            colors.append(ColorPalette.SEMANTIC['neutral'])
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='', startangle=90,
            textprops={'fontsize': 7, 'ha': 'center'}
        )
        
        # Improve text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(6)
        
        ax.set_title(
            f'{config["display_name"]}\n{largest_size} {config["unit_label"]}',
            fontsize=10, pad=10
        )
    
    # Plot histograms in last 2 rows (6 solvers)
    for idx, solver in enumerate(solvers):
        if solver not in largest_configs:
            continue
            
        row = 2 + (idx // 3)  # Start from row 2 (3rd row)
        col = idx % 3
        ax = axes[row, col]
        
        config = SOLVER_CONFIGS[solver]
        largest_size = largest_configs[solver]
        composition = all_compositions[solver][largest_size]
        
        # Prepare data for histogram
        crops = []
        percentages = []
        colors = []
        
        # Sort by percentage (descending) and take top 8, group rest as "Other"
        sorted_items = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:8]
        other_percentage = sum(item[1] for item in sorted_items[8:])
        
        for crop, percentage in top_items:
            crops.append(crop)
            percentages.append(percentage)
            colors.append(crop_colors.get(crop, ColorPalette.SEMANTIC['neutral']))
        
        if other_percentage > 0.5:  # Only show "Other" if significant
            crops.append('Other')
            percentages.append(other_percentage)
            colors.append(ColorPalette.SEMANTIC['neutral'])
        
        # Create histogram with log scale
        x_pos = np.arange(len(crops))
        bars = ax.bar(x_pos, percentages, color=colors, 
                     edgecolor='black', linewidth=1.0)
        
        # Set log scale on y-axis
        ax.set_yscale('log')
        
        # Set labels and title
        ax.set_xticks(x_pos)
        ax.set_xticklabels(crops, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(r'$\text{Area (\%)}$', fontsize=9)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, which='both', axis='y')
        
        ax.set_title(
            f'{config["display_name"]}\n{largest_size} {config["unit_label"]}',
            fontsize=10, pad=10
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined largest configuration plot saved to {output_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Reduced visualization: largest configuration only (6 pie + 6 histogram)')
    parser.add_argument('--solvers', nargs='+', 
                       choices=list(SOLVER_CONFIGS.keys()),
                       default=list(SOLVER_CONFIGS.keys()),
                       help='Solvers to analyze (default: all)')
    parser.add_argument('--benchmark-dir', type=str, default='Legacy',
                       help='Benchmark directory (default: Legacy)')
    parser.add_argument('--output-dir', type=str, default='professional_plots',
                       help='Output directory (default: professional_plots)')
    
    args = parser.parse_args()
    
    # Configure professional style
    configure_professional_style()
    
    # Set up paths
    script_dir = Path(__file__).parent
    benchmark_dir = script_dir / args.benchmark_dir
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("REDUCED VISUALIZATION: LARGEST CONFIGURATION ONLY")
    print("="*70)
    
    # Load data for all solvers
    print("\n📂 Loading benchmark data...")
    all_compositions = {}
    
    for solver in args.solvers:
        try:
            data = load_benchmark_data(benchmark_dir, solver)
            compositions = extract_solution_composition(data, solver)
            
            all_compositions[solver] = compositions
            
            if compositions:
                largest = max(compositions.keys())
                print(f"✓ {SOLVER_CONFIGS[solver]['display_name']}: Largest config = {largest}")
            else:
                print(f"⚠ {solver}: No composition data")
        except FileNotFoundError as e:
            print(f"✗ {solver}: {e}")
            continue
    
    if not all_compositions:
        print("❌ No valid solver data found!")
        return 1
    
    print(f"\n📊 Loaded data for {len(all_compositions)} solvers")
    
    # Generate combined plot
    print("\n📈 Generating combined visualization (6 pie + 6 histogram)...")
    
    if any(all_compositions.values()):
        plot_largest_config_combined(all_compositions, output_dir / "largest_config_combined.pdf")
    
    print(f"\n✅ Plot saved to: {output_dir}")
    print("\nGenerated file:")
    print("  - largest_config_combined.pdf : 12 plots (6 pie + 6 histogram) for largest configuration")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    exit(main())
