"""
Unified Plot Configuration for Publication-Quality Figures

This module provides consistent styling, colors, and formatting across all 
visualization scripts in the OQI-UC002-DWave project.

GUARANTEED CONSISTENCY:
- All line plots use the same 20-color qualitative palette
- All heatmaps/gradients use 'Greens' sequential colormap
- All diverging plots use 'RdBu_r' colormap
- All method-specific colors are defined once and used everywhere
- Professional typography with bold 16pt titles, 14pt labels

USAGE:
```python
from plot_config import (
    setup_publication_style,
    get_sequential_cmap,    # Returns 'Greens' for all heatmaps
    get_diverging_cmap,     # Returns 'RdBu_r' for diverging data
    get_color_palette,      # Returns n colors from qualitative palette
    METHOD_COLORS,          # Dict of method-specific colors
    FOOD_GROUP_COLORS,      # Dict of food group colors
    save_figure             # Save in multiple formats
)

setup_publication_style()  # Call once at start of script

# Use consistent heatmap colormap
plt.imshow(data, cmap=get_sequential_cmap())

# Use consistent line colors
colors = get_color_palette(5)  # Get 5 distinct colors
```

Author: OQI-UC002-DWave Project
Date: 2025-12-18
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
    Professional style with enhanced visibility and clarity.
    Call this at the start of every plotting script.
    """
    plt.rcParams.update({
        # Use native matplotlib rendering (not LaTeX)
        # This avoids LaTeX compilation issues with special characters
        'text.usetex': False,
        
        # Font configuration - Larger and more prominent
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        
        # Figure quality
        'figure.dpi': 150,
        'figure.constrained_layout.use': True,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.transparent': False,
        
        # Grid configuration - More visible and professional
        'axes.grid': True,
        'axes.grid.which': 'major',
        'axes.grid.axis': 'both',
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'axes.axisbelow': True,
        
        # Axes configuration with proper arrows
        'axes.linewidth': 1.3,
        'axes.labelpad': 10,
        'axes.titlepad': 15,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        
        # Tick configuration - Enhanced visibility
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.3,
        'ytick.major.width': 1.3,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        
        # Line and marker properties - Thinner and more refined
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'lines.markeredgewidth': 1.0,
        'patch.linewidth': 0.8,
        
        # Legend - Professional appearance
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.shadow': False,
        
        # Error bars
        'errorbar.capsize': 3,
    })


# ============================================================================
# Unified Color Palette - Custom Professional Design
# ============================================================================

# Primary qualitative palette (20 colors for line plots, bar charts, etc.)
# Designed for maximum distinguishability and colorblind-friendliness
# Based on scientific visualization best practices
QUALITATIVE_COLORS = [
    '#1F77B4',  # Blue (primary)
    '#FF7F0E',  # Orange (secondary)
    '#2CA02C',  # Green
    '#D62728',  # Red
    '#9467BD',  # Purple
    '#8C564B',  # Brown
    '#E377C2',  # Pink
    '#7F7F7F',  # Gray
    '#BCBD22',  # Yellow-green
    '#17BECF',  # Cyan
    '#AEC7E8',  # Light blue
    '#FFBB78',  # Light orange
    '#98DF8A',  # Light green
    '#FF9896',  # Light red
    '#C5B0D5',  # Light purple
    '#C49C94',  # Light brown
    '#F7B6D2',  # Light pink
    '#C7C7C7',  # Light gray
    '#DBDB8D',  # Light yellow
    '#9EDAE5',  # Light cyan
]

# Sequential palette for heatmaps/gradients (light to dark blue-green)
# Professional gradient suitable for continuous data
SEQUENTIAL_PALETTE = [
    '#F7FCF5',  # Very light green (lightest)
    '#E5F5E0',  # Light green
    '#C7E9C0',  # Medium-light green
    '#A1D99B',  # Medium green
    '#74C476',  # Medium-dark green
    '#41AB5D',  # Dark green
    '#238B45',  # Darker green
    '#006D2C',  # Very dark green
    '#00441B',  # Darkest green
]

# Diverging palette for positive/negative or comparative data
# Balanced red-white-blue for showing deviation from center
DIVERGING_PALETTE = [
    '#B2182B',  # Dark red (negative extreme)
    '#D6604D',  # Red
    '#F4A582',  # Light red
    '#FDDBC7',  # Very light red
    '#F7F7F7',  # White (neutral)
    '#D1E5F0',  # Very light blue
    '#92C5DE',  # Light blue
    '#4393C3',  # Blue
    '#2166AC',  # Dark blue (positive extreme)
]

# Colormap names for matplotlib (for consistency across all heatmaps)
SEQUENTIAL_CMAP = 'Greens'      # Use 'Greens' colormap everywhere for sequential data
DIVERGING_CMAP = 'RdBu_r'       # Use 'RdBu_r' (reversed) for diverging data
QUALITATIVE_CMAP = 'tab20'      # Use 'tab20' for categorical data if needed

# Method-specific colors (consistent across all plots) - using professional palette
METHOD_COLORS = {
    'Gurobi': '#D95F02',        # Orange (optimal baseline)
    'PuLP': '#D95F02',          # Orange (optimal baseline)
    'D-Wave Hybrid': '#1B9E77', # Teal (hybrid approach)
    'PlotBased_QPU': '#7570B3', # Purple
    'Multilevel(5)_QPU': '#E7298A',   # Magenta
    'Multilevel(10)_QPU': '#66A61E',  # Green
    'Louvain_QPU': '#E6AB02',         # Gold
    'Spectral(10)_QPU': '#A6761D',    # Brown
    'cqm_first_PlotBased': '#377EB8', # Blue
    'coordinated': '#984EA3',         # Violet
    'HybridGrid(5,9)_QPU': '#4DAF4A', # Light green
    'HybridGrid(10,9)_QPU': '#FF7F00',# Bright orange
}

# Food group colors (using professional palette)
FOOD_GROUP_COLORS = {
    'Vegetables': '#1B9E77',  # Teal
    'Grains': '#E6AB02',      # Gold
    'Legumes': '#66A61E',     # Green
    'Fruits': '#D95F02',      # Orange
    'Meats': '#E7298A',       # Magenta
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


def get_qualitative_cmap():
    """
    Get the standard qualitative colormap for line plots and bar charts.
    Returns a matplotlib colormap object.
    """
    from matplotlib.colors import ListedColormap
    return ListedColormap(QUALITATIVE_COLORS)


def get_sequential_cmap():
    """
    Get the standard sequential colormap for heatmaps/gradients.
    Returns matplotlib colormap name (string).
    """
    return SEQUENTIAL_CMAP


def get_diverging_cmap():
    """
    Get the standard diverging colormap for comparative data.
    Returns matplotlib colormap name (string).
    """
    return DIVERGING_CMAP


def get_color_palette(n_colors):
    """
    Get n colors from the qualitative palette with cycling if needed.
    
    Args:
        n_colors: Number of colors needed
        
    Returns:
        List of hex color strings
    """
    if n_colors <= len(QUALITATIVE_COLORS):
        return QUALITATIVE_COLORS[:n_colors]
    else:
        # Cycle through colors if more are needed
        import itertools
        cycle = itertools.cycle(QUALITATIVE_COLORS)
        return [next(cycle) for _ in range(n_colors)]


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
