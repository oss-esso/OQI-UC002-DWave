# Instructions for `choropleth_plo.py`

## Objective
Modify the `choropleth_plo.py` script to generate a choropleth plot based on the benchmark results located in `@Legacy/COMPREHENSIVE/**`.

## Data Loading
- The script must load all the result files from the `@Legacy/COMPREHENSIVE/` directory.
- It should be able to distinguish between different scenarios (e.g., `Farm_DWave`, `Patch_PuLP`) based on the file path or content.

## Plotting Logic

The script should generate a choropleth grid based on the type of scenario. The location coordinates are already present in the `choropleth_plo.py` file and should be used as the base map.

### "Farm" Scenario
For results from "Farm" scenarios:
1.  **Uneven Grid**: The choropleth grid must be uneven, with the size of each cell corresponding to the size of the farm.
2.  **Farm Sizes**: If the farm sizes are not available in the result files, run the `farm_sampler.py` script to generate them.
3.  **Coloring**: Each farm cell should be colored.
4.  **Crop Splitting**: Each farm cell must be split and colored according to the crops assigned to it in the solution. The area of each split should be proportional to the area allocated to that crop on that farm.

### "Plot" Scenario
For results from "Plot" or "Patch" scenarios:
1.  **Even Grid**: The choropleth should be an even grid.
2.  **Coloring**: Each cell in the grid represents a plot and should be colored based on the crop assigned to it. If a plot has multiple crops assigned in the result, this should be indicated, for example, by splitting the cell or using a distinct color for multi-crop assignments.

## Color Palette
- **Standard Palettes**: Use industry-standard color palettes to ensure clarity and accessibility. Libraries like `ColorBrewer` or `matplotlib.cm.viridis` are recommended.
- **Crop-to-Color Mapping**: Create a consistent mapping of crops to colors. This mapping should be stable across different plots and scenarios. For example, "Corn" should always be yellow, "Spinach" should always be green, etc.
- **Distinct Colors**: Ensure that adjacent crops have visually distinct colors to make the plot easy to read.

## Legend
- **Comprehensive Legend**: The plot must include a comprehensive legend.
- **Content**: The legend should clearly map each crop name to its corresponding color.
- **Placement**: Position the legend in a non-obtrusive location on the map, such as the top-right or bottom-right corner.

## Interactivity (Tooltips)
To meet industry standards, the map should be interactive.
- **Hover Information**: When a user hovers over a farm or plot, a tooltip should appear.
- **Tooltip Content**: The tooltip should display key information, such as:
    - Farm/Plot Name
    - Total Area
    - Assigned Crop(s)
    - Area allocated to each crop (for "Farm" scenarios)

## Output Format
- **HTML File**: The final output should be a single, self-contained HTML file. This allows for easy sharing and viewing of the interactive map in any web browser.
- **Filename**: The output filename should be descriptive, incorporating the scenario and configuration details (e.g., `choropleth_farm_10_units.html`).
