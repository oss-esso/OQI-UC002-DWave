# Choropleth Maps - Interactive Crop Allocation Visualizations

This directory contains 18 interactive choropleth maps visualizing crop allocation solutions for both Farm and Patch scenarios on Pulau Tidung Kecil, Indonesia.

## Quick Start

### View Individual Maps
Simply double-click any HTML file to open it in your default browser.

### Use the Viewer Script
From the parent directory, run:
```bash
python view_choropleths.py
```

This provides an interactive menu to:
- Open any specific map by number
- Open all farm maps
- Open all patch maps  
- Open all maps at once

## Map Files

### Farm Scenarios (6 maps)
Farm scenarios use **uneven grids** where each cell size is proportional to the actual farm area. Farms with multiple crops are split proportionally.

- **DWave Solver:**
  - `choropleth_Farm_DWave_config_10_run_1.html` - 10 farms
  - `choropleth_Farm_DWave_config_15_run_1.html` - 15 farms
  - `choropleth_Farm_DWave_config_25_run_1.html` - 25 farms

- **PuLP Solver:**
  - `choropleth_Farm_PuLP_config_10_run_1.html` - 10 farms
  - `choropleth_Farm_PuLP_config_15_run_1.html` - 15 farms
  - `choropleth_Farm_PuLP_config_25_run_1.html` - 25 farms

### Patch Scenarios (12 maps)
Patch scenarios use **even grids** where each cell represents a uniform plot. Cells are colored by the assigned crop.

- **DWave CQM Solver:**
  - `choropleth_Patch_DWave_config_10_run_1.html` - 10 plots
  - `choropleth_Patch_DWave_config_15_run_1.html` - 15 plots
  - `choropleth_Patch_DWave_config_25_run_1.html` - 25 plots

- **DWave BQM Solver:**
  - `choropleth_Patch_DWaveBQM_config_10_run_1.html` - 10 plots
  - `choropleth_Patch_DWaveBQM_config_15_run_1.html` - 15 plots
  - `choropleth_Patch_DWaveBQM_config_25_run_1.html` - 25 plots

- **Gurobi QUBO Solver:**
  - `choropleth_Patch_GurobiQUBO_config_10_run_1.html` - 10 plots
  - `choropleth_Patch_GurobiQUBO_config_15_run_1.html` - 15 plots
  - `choropleth_Patch_GurobiQUBO_config_25_run_1.html` - 25 plots

- **PuLP Solver:**
  - `choropleth_Patch_PuLP_config_10_run_1.html` - 10 plots
  - `choropleth_Patch_PuLP_config_15_run_1.html` - 15 plots
  - `choropleth_Patch_PuLP_config_25_run_1.html` - 25 plots

## Features

### Interactive Elements
- **Hover Tooltips**: Quick info showing farm/plot name and crop
- **Click Popups**: Detailed information including allocated area
- **Zoom/Pan**: Full map navigation
- **Legend**: Shows all crops used with consistent color coding

### Color Scheme
Crops are color-coded by category:
- **Proteins** (Browns/Reds): Chicken, Lamb, Pork, Beef
- **Fruits** (Yellows/Oranges): Apple, Mango, Orange, Durian, Guava
- **Vegetables** (Greens): Spinach, Long bean, Potato, Cabbage
- **Legumes** (Tans): Tempeh, Tofu, Chickpeas, Lentils
- **Grains** (Golden): Rice, Corn, Wheat

## Technical Details

### File Format
- Self-contained HTML files (no external dependencies)
- Based on Folium/Leaflet.js
- OpenStreetMap base layer
- Responsive design

### Geographic Location
All maps are centered on Pulau Tidung Kecil, Kepulauan Seribu, Jakarta, Indonesia.

### Generation
Maps generated using `choropleth_plo.py` from optimization results in `Legacy/COMPREHENSIVE/`.

## Troubleshooting

**Maps won't open?**
- Ensure you have a modern web browser installed
- Try right-click → "Open With" → select your browser
- Check that JavaScript is enabled

**Maps load slowly?**
- This is normal for larger configurations (25 units)
- Initial OSM tile loading may take a few seconds

**Colors hard to distinguish?**
- The color palette is designed for accessibility
- Adjacent crops use contrasting colors
- Refer to the legend for crop identification

## Regenerating Maps

To regenerate all maps:
```bash
cd /Users/edoardospigarolo/Documents/OQI-UC002-DWave
python choropleth_plo.py
```

This will:
1. Load all result files from `Legacy/COMPREHENSIVE/`
2. Query OpenStreetMap for island geometry
3. Generate all 18 maps
4. Save them to this directory

Execution time: ~2 minutes

## More Information

See `CHOROPLETH_IMPLEMENTATION_COMPLETE.md` in the parent directory for full implementation details.
