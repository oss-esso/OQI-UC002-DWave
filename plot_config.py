"""
Unified Plot Configuration for Publication-Quality Figures

This module provides consistent styling, colors, and formatting across all 
visualization scripts in the OQI-UC002-DWave project.

Author: OQI-UC002-DWave Project
Date: 2025-12-17
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# ============================================================================
# LaTeX-Style Configuration for Publication
# ============================================================================

def setup_publication_style():
    """
    Configure matplotlib for publication-quality plots.
    Uses native matplotlib rendering for better compatibility.
    Call this at the start of every plotting script.
    """
    plt.rcParams.update({
        # Use native matplotlib rendering (not LaTeX)
        # This avoids LaTeX compilation issues with special characters
        'text.usetex': False,
        
        # Font configuration
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        
        # Figure quality
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Grid and axes
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.axisbelow': True,
        
        # Line and marker properties
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'patch.linewidth': 0.5,
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': True,
        'legend.edgecolor': 'gray',
        
        # Spine visibility
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ============================================================================
# Unified Color Palette
# ============================================================================

# Primary qualitative palette (for different methods/categories)
QUALITATIVE_COLORS = [
    '#E63946',  # Red
    '#F4A261',  # Orange
    '#2A9D8F',  # Teal
    '#264653',  # Dark blue-gray
    '#E9C46A',  # Yellow
    '#8338EC',  # Purple
    '#06FFA5',  # Bright green
    '#FF6B6B',  # Light red
    '#4ECDC4',  # Cyan
    '#95E1D3',  # Mint
    '#F38181',  # Pink
    '#AA96DA',  # Lavender
    '#FCBAD3',  # Light pink
    '#A8D8EA',  # Light blue
    '#FFD93D',  # Bright yellow
]

# Sequential palette (for gradients/heatmaps)
SEQUENTIAL_PALETTE = [
    '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
    '#4292c6', '#2171b5', '#08519c', '#08306b'
]

# Diverging palette (for comparing positive/negative)
DIVERGING_PALETTE = [
    '#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'
]

# Method-specific colors (consistent across all plots)
METHOD_COLORS = {
    'Gurobi': '#E63946',
    'PuLP': '#E63946',
    'D-Wave Hybrid': '#3B82F6',
    'PlotBased_QPU': '#06FFA5',
    'Multilevel(5)_QPU': '#2EC4B6',
    'Multilevel(10)_QPU': '#20A39E',
    'Louvain_QPU': '#3DDC97',
    'Spectral(10)_QPU': '#5CDB95',
    'cqm_first_PlotBased': '#8338EC',
    'coordinated': '#FF6B6B',
    'HybridGrid(5,9)_QPU': '#06D6A0',
    'HybridGrid(10,9)_QPU': '#1B9AAA',
}

# Food group colors
FOOD_GROUP_COLORS = {
    'Vegetables': '#2A9D8F',  # Teal
    'Grains': '#E9C46A',      # Yellow
    'Legumes': '#06FFA5',     # Green
    'Fruits': '#F4A261',      # Orange
    'Meats': '#E63946',       # Red
}

# Food groups definition
FOOD_GROUPS = {
    'Meats': ['Beef', 'Chicken', 'Egg', 'Lamb', 'Pork'],
    'Fruits': ['Apple', 'Avocado', 'Banana', 'Durian', 'Guava', 'Mango', 
               'Orange', 'Papaya', 'Watermelon'],
    'Legumes': ['Chickpeas', 'Peanuts', 'Tempeh', 'Tofu'],
    'Grains': ['Corn', 'Potato'],
    'Vegetables': ['Cabbage', 'Cucumber', 'Eggplant', 'Long bean', 
                   'Pumpkin', 'Spinach', 'Tomatoes']
}

# Create reverse mapping
FOOD_TO_GROUP = {}
for group, foods in FOOD_GROUPS.items():
    for food in foods:
        FOOD_TO_GROUP[food] = group


# ============================================================================
# Display Name Mappings
# ============================================================================

METHOD_DISPLAY_NAMES = {
    'Gurobi': 'Gurobi (Optimal)',
    'PuLP': 'PuLP',
    'PlotBased_QPU': 'PlotBased QPU',
    'Multilevel(5)_QPU': 'Multilevel(5) QPU',
    'Multilevel(10)_QPU': 'Multilevel(10) QPU',
    'Louvain_QPU': 'Louvain QPU',
    'Spectral(10)_QPU': 'Spectral(10) QPU',
    'cqm_first_PlotBased': 'CQM-First PlotBased',
    'coordinated': 'Coordinated',
    'HybridGrid(5,9)_QPU': 'HybridGrid(5,9) QPU',
    'HybridGrid(10,9)_QPU': 'HybridGrid(10,9) QPU',
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_crop_color(crop_name, food_group_colors=True):
    """
    Get consistent color for a crop.
    
    Args:
        crop_name: Name of the crop
        food_group_colors: If True, color by food group; if False, use qualitative palette
    
    Returns:
        Hex color string
    """
    if food_group_colors and crop_name in FOOD_TO_GROUP:
        group = FOOD_TO_GROUP[crop_name]
        return FOOD_GROUP_COLORS[group]
    else:
        # Assign color based on crop name hash for consistency
        idx = hash(crop_name) % len(QUALITATIVE_COLORS)
        return QUALITATIVE_COLORS[idx]


def get_method_color(method_name):
    """Get consistent color for a method."""
    return METHOD_COLORS.get(method_name, QUALITATIVE_COLORS[0])


def format_large_number(x, pos=None):
    """Format large numbers with K/M suffixes for axes."""
    if abs(x) >= 1e6:
        return f'{x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'{x/1e3:.1f}K'
    else:
        return f'{x:.0f}'


def save_figure(fig, filepath, formats=['png', 'pdf']):
    """
    Save figure in multiple formats with consistent settings.
    
    Args:
        fig: Matplotlib figure object
        filepath: Path (without extension) where to save
        formats: List of formats to save (default: png and pdf)
    """
    from pathlib import Path
    
    filepath = Path(filepath)
    base_path = filepath.with_suffix('')
    
    for fmt in formats:
        output_path = base_path.with_suffix(f'.{fmt}')
        fig.savefig(output_path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"   Saved: {output_path}")


def create_legend_outside(ax, **kwargs):
    """
    Create a legend outside the plot area.
    
    Args:
        ax: Matplotlib axes object
        **kwargs: Additional arguments passed to ax.legend()
    """
    default_kwargs = {
        'loc': 'center left',
        'bbox_to_anchor': (1.02, 0.5),
        'frameon': True,
        'framealpha': 0.9,
        'edgecolor': 'gray',
    }
    default_kwargs.update(kwargs)
    return ax.legend(**default_kwargs)


def add_value_labels(ax, bars, format_str='{:.1f}', fontsize=8):
    """
    Add value labels on top of bars in a bar chart.
    
    Args:
        ax: Matplotlib axes object
        bars: Bar container from ax.bar()
        format_str: Format string for values
        fontsize: Font size for labels
    """
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   format_str.format(height),
                   ha='center', va='bottom', fontsize=fontsize)


# ============================================================================
# Initialization
# ============================================================================

# Apply publication style by default when module is imported
setup_publication_style()


if __name__ == "__main__":
    # Test the configuration
    print("Plot configuration loaded successfully!")
    print(f"Available colors: {len(QUALITATIVE_COLORS)}")
    print(f"Method colors defined: {len(METHOD_COLORS)}")
    print(f"Food groups: {list(FOOD_GROUPS.keys())}")
