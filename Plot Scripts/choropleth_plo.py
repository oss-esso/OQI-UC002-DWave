import math
import numpy as np
import geopandas as gp
from shapely.geometry import Point, Polygon, box, MultiPolygon
from shapely.affinity import translate, scale
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx
import json
import osmnx as ox
import folium
from folium import plugins
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Import farm_sampler to generate farm sizes if needed
from farm_sampler import generate_farms

# ============================================================
# CONFIGURATION
# ============================================================

# Color mapping for crops using ColorBrewer-inspired palette
CROP_COLORS = {
    # Proteins - Browns and Reds
    "Chicken": "#8B4513",  # SaddleBrown
    "Lamb": "#A0522D",     # Sienna
    "Pork": "#CD853F",     # Peru
    "Beef": "#D2691E",     # Chocolate
    
    # Fruits - Yellows, Oranges, Purples
    "Apple": "#FFD700",    # Gold
    "Mango": "#FFA500",    # Orange
    "Orange": "#FF8C00",   # DarkOrange
    "Durian": "#DAA520",   # GoldenRod
    "Guava": "#FFDAB9",    # PeachPuff
    "Papaya": "#FFB347",   # Light Orange
    "Banana": "#FFFF00",   # Yellow
    
    # Vegetables - Greens
    "Spinach": "#228B22",  # ForestGreen
    "Long bean": "#32CD32", # LimeGreen
    "Potato": "#8FBC8F",   # DarkSeaGreen
    "Cabbage": "#90EE90",  # LightGreen
    "Carrot": "#FF6347",   # Tomato
    "Eggplant": "#9370DB", # MediumPurple
    "Onion": "#F5DEB3",    # Wheat
    "Tomato": "#FF6347",   # Tomato
    
    # Legumes - Light browns/greens
    "Tempeh": "#D2B48C",   # Tan
    "Tofu": "#F5F5DC",     # Beige
    "Chickpeas": "#DEB887", # BurlyWood
    "Lentils": "#BC8F8F",  # RosyBrown
    "Peanuts": "#C19A6B",  # Camel
    
    # Grains - Golden/Tan
    "Rice": "#F0E68C",     # Khaki
    "Corn": "#FFDB58",     # Mustard
    "Wheat": "#F5DEB3",    # Wheat
    
    # Default
    "Other": "#A9A9A9",    # DarkGray
}

# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def parse_pulp_farm_solution(data: Dict) -> Dict:
    """
    Parse PuLP Farm solution format to standard format.
    PuLP uses flat structure: Farm{i}_{Crop} with solution_areas and solution_selections.
    
    Args:
        data: Raw result data
        
    Returns:
        Parsed data with solution_summary matching DWave format
    """
    solution_areas = data.get('solution_areas', {})
    solution_selections = data.get('solution_selections', {})
    
    if not solution_areas:
        return data
    
    # Group by farm and crop
    farm_crops = {}
    for key, area in solution_areas.items():
        if area > 0.0001:  # Only include non-zero allocations
            parts = key.split('_', 1)
            if len(parts) == 2:
                farm_name, crop_name = parts
                if farm_name not in farm_crops:
                    farm_crops[farm_name] = []
                farm_crops[farm_name].append({
                    'crop': crop_name,
                    'area': area
                })
    
    # Build plot_assignments in DWave format
    crop_assignments = {}
    for farm_name, crops in farm_crops.items():
        for crop_info in crops:
            crop = crop_info['crop']
            if crop not in crop_assignments:
                crop_assignments[crop] = {
                    'crop': crop,
                    'total_area': 0,
                    'n_plots': 0,
                    'plots': []
                }
            crop_assignments[crop]['plots'].append({
                'plot': farm_name,
                'area': crop_info['area']
            })
            crop_assignments[crop]['total_area'] += crop_info['area']
            crop_assignments[crop]['n_plots'] += 1
    
    plot_assignments = list(crop_assignments.values())
    crops_selected = list(crop_assignments.keys())
    
    # Add solution_summary
    data['solution_summary'] = {
        'crops_selected': crops_selected,
        'n_crops': len(crops_selected),
        'plot_assignments': plot_assignments
    }
    
    return data

def load_all_results(base_path: str = "Legacy/COMPREHENSIVE") -> Dict:
    """
    Load all result files from the Legacy/COMPREHENSIVE directory.
    
    Returns:
        Dictionary organized by scenario type and configuration
    """
    results = {
        "Farm": {},
        "Patch": {}
    }
    
    # Get all subdirectories
    scenario_dirs = glob.glob(f"{base_path}/*")
    
    for scenario_dir in scenario_dirs:
        if not os.path.isdir(scenario_dir):
            continue
            
        scenario_name = os.path.basename(scenario_dir)
        
        # Determine if this is Farm or Patch scenario
        if "Farm" in scenario_name:
            scenario_type = "Farm"
        elif "Patch" in scenario_name:
            scenario_type = "Patch"
        else:
            continue
        
        # Load all JSON files in this directory
        json_files = glob.glob(f"{scenario_dir}/*.json")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Parse PuLP Farm format if needed
                if scenario_type == "Farm" and "PuLP" in scenario_name:
                    data = parse_pulp_farm_solution(data)
                
                # Extract configuration info from filename
                filename = os.path.basename(json_file)
                
                # Store with scenario name and filename as key
                key = f"{scenario_name}_{filename}"
                results[scenario_type][key] = {
                    "data": data,
                    "scenario_name": scenario_name,
                    "filename": filename,
                    "path": json_file
                }
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return results

def get_farm_sizes(n_farms: int, seed: int = 42) -> Dict[str, float]:
    """
    Generate or retrieve farm sizes using farm_sampler.
    
    Args:
        n_farms: Number of farms
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping farm names to areas
    """
    return generate_farms(n_farms, seed=seed)

# ============================================================
# GRID GENERATION FUNCTIONS
# ============================================================

# ============================================================
# GRID GENERATION FUNCTIONS
# ============================================================

def physics_based_layout(bounds: Tuple[float, float, float, float],
                         polygons_dict: Dict[str, Tuple[float, float]],
                         max_iterations: int = 500,
                         crop_groups: Dict[str, str] = None) -> Dict[str, Polygon]:
    """
    Use physics simulation to arrange polygons without overlap.
    Polygons are attracted to center but repel each other.
    Same-crop polygons are attracted to each other to form clusters.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        polygons_dict: Dict mapping names to (width, height) tuples
        max_iterations: Number of simulation steps
        crop_groups: Optional dict mapping polygon names to crop names for clustering
        
    Returns:
        Dictionary mapping names to positioned Polygon objects
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    width = max_lon - min_lon
    height = max_lat - min_lat
    center_x = (min_lon + max_lon) / 2
    center_y = (min_lat + max_lat) / 2
    
    # Initialize positions randomly
    np.random.seed(42)
    positions = {}  # {name: [x, y]}
    velocities = {}  # {name: [vx, vy]}
    
    for name, (w, h) in polygons_dict.items():
        # Start near center with some randomness
        positions[name] = np.array([
            center_x + (np.random.random() - 0.5) * width * 0.3,
            center_y + (np.random.random() - 0.5) * height * 0.3
        ])
        velocities[name] = np.array([0.0, 0.0])
    
    # Physics parameters - much stronger forces for better separation
    center_attraction = 0.00002  # Gentle pull towards center
    repulsion_strength = 0.011  # Strong push away from overlaps (20x stronger)
    same_farm_attraction = 0.00005  # Attraction between same-farm plots
    damping = 0.6  # Lower damping for faster movement
    min_separation = 0.0012  # Larger minimum distance between polygons
    
    # Simulation loop
    for iteration in range(max_iterations):
        forces = {name: np.array([0.0, 0.0]) for name in polygons_dict.keys()}
        
        # Apply center attraction (only after things settle)
        if iteration > max_iterations * 0.3:  # Start attracting after initial separation
            for name, pos in positions.items():
                to_center = np.array([center_x - pos[0], center_y - pos[1]])
                dist = np.linalg.norm(to_center)
                if dist > 0.0001:
                    forces[name] += to_center * center_attraction
        
        # Apply same-farm attraction if crop_groups is provided (for Farm scenarios)
        if crop_groups:
            names_list = list(polygons_dict.keys())
            for i, name1 in enumerate(names_list):
                crop1 = crop_groups.get(name1)
                if not crop1:
                    continue
                
                pos1 = positions[name1]
                
                for name2 in names_list[i+1:]:
                    crop2 = crop_groups.get(name2)
                    if not crop2 or crop1 != crop2:
                        continue
                    
                    # Same farm - attract them together
                    pos2 = positions[name2]
                    delta = pos2 - pos1
                    dist = np.linalg.norm(delta)
                    
                    if dist > 0.0001:
                        direction = delta / dist
                        attraction = direction * same_farm_attraction * dist
                        forces[name1] += attraction
                        forces[name2] -= attraction
        
        # Apply strong repulsion between overlapping/close polygons
        names_list = list(polygons_dict.keys())
        for i, name1 in enumerate(names_list):
            w1, h1 = polygons_dict[name1]
            pos1 = positions[name1]
            
            for name2 in names_list[i+1:]:
                w2, h2 = polygons_dict[name2]
                pos2 = positions[name2]
                
                # Vector from 2 to 1
                delta = pos1 - pos2
                dist = np.linalg.norm(delta)
                
                # Check if bounding boxes overlap or are too close
                overlap_x = (w1 + w2) / 2 + min_separation - abs(delta[0])
                overlap_y = (h1 + h2) / 2 + min_separation - abs(delta[1])
                
                if overlap_x > 0 and overlap_y > 0:
                    # Calculate repulsion force
                    if dist > 0.00001:
                        direction = delta / dist
                    else:
                        # Random direction if perfectly overlapping
                        direction = np.array([np.random.random() - 0.5, 
                                             np.random.random() - 0.5])
                        direction = direction / (np.linalg.norm(direction) + 0.00001)
                    
                    # Much stronger repulsion for overlaps
                    overlap_magnitude = max(overlap_x, overlap_y)
                    # Exponential repulsion - stronger for larger overlaps
                    repulsion = direction * repulsion_strength * (overlap_magnitude ** 2) * 100
                    
                    forces[name1] += repulsion
                    forces[name2] -= repulsion
        
        # Update velocities and positions
        for name in polygons_dict.keys():
            velocities[name] += forces[name]
            velocities[name] *= damping  # Apply damping
            positions[name] += velocities[name]
    
    # Create final polygons from positions
    result = {}
    for name, (w, h) in polygons_dict.items():
        pos = positions[name]
        # Center the polygon on its position
        result[name] = box(pos[0] - w/2, pos[1] - h/2, 
                          pos[0] + w/2, pos[1] + h/2)
    
    return result

def create_even_grid(bounds: Tuple[float, float, float, float], 
                     n_plots: int) -> List[Polygon]:
    """
    Create scattered plot polygons using physics-based layout.
    Plots are attracted to center but repel each other.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        n_plots: Number of plots to create
        
    Returns:
        List of Polygon objects
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    width = max_lon - min_lon
    height = max_lat - min_lat
    
    # Calculate plot size (assuming equal area for patches)
    plot_width = width / np.sqrt(n_plots * 2.0)
    plot_height = height / np.sqrt(n_plots * 2.0)
    
    # Create dict for physics simulation
    polygons_dict = {f"Patch{i}": (plot_width, plot_height) 
                     for i in range(1, n_plots + 1)}
    
    # Run physics simulation
    positioned_polygons = physics_based_layout(bounds, polygons_dict, max_iterations=1000)
    
    # Return as list in order
    return [positioned_polygons[f"Patch{i}"] for i in range(1, n_plots + 1)]

def create_uneven_grid(bounds: Tuple[float, float, float, float],
                       farm_sizes: Dict[str, float],
                       farm_groups: Dict[str, str] = None) -> Dict[str, Polygon]:
    """
    Create scattered farm polygons using physics-based layout.
    Each farm size is proportional to its area in hectares.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        farm_sizes: Dictionary mapping farm names to areas (in hectares)
        farm_groups: Optional dict mapping farm names to farm groupings (e.g., "Farm1" for clustering)
        
    Returns:
        Dictionary mapping farm names to Polygon objects
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    width = max_lon - min_lon
    height = max_lat - min_lat
    
    total_farm_area = sum(farm_sizes.values())
    
    # Scale factor: farms should fit in the available coordinate space
    coord_area = width * height
    scale_factor = coord_area / (total_farm_area * 1.5)  # 1.5 for spacing
    
    # Calculate dimensions for each farm
    polygons_dict = {}
    for farm_name, farm_area in farm_sizes.items():
        farm_coord_area = farm_area * scale_factor
        # Approximate square shape
        farm_width = np.sqrt(farm_coord_area)
        farm_height = farm_coord_area / farm_width
        polygons_dict[farm_name] = (farm_width, farm_height)
    
    # Run physics simulation with farm groupings
    return physics_based_layout(bounds, polygons_dict, max_iterations=800, crop_groups=farm_groups)

def split_polygon_by_crops(polygon: Polygon, 
                           crop_allocations: List[Dict]) -> List[Tuple[Polygon, str, float]]:
    """
    Split a polygon into sections based on crop allocations.
    
    Args:
        polygon: The polygon to split
        crop_allocations: List of dicts with 'crop' and 'area' keys
        
    Returns:
        List of tuples (polygon_section, crop_name, area)
    """
    if len(crop_allocations) == 1:
        return [(polygon, crop_allocations[0]['crop'], crop_allocations[0]['area'])]
    
    # Calculate proportions
    total_area = sum(c['area'] for c in crop_allocations)
    
    bounds = polygon.bounds
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    
    # Split horizontally based on proportions
    sections = []
    current_y = min_y
    
    for crop_alloc in crop_allocations:
        proportion = crop_alloc['area'] / total_area
        section_height = height * proportion
        
        section_poly = box(min_x, current_y, max_x, min(current_y + section_height, max_y))
        sections.append((section_poly, crop_alloc['crop'], crop_alloc['area']))
        
        current_y += section_height
    
    return sections

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def get_crop_color(crop_name: str) -> str:
    """Get color for a crop, with fallback to default."""
    return CROP_COLORS.get(crop_name, CROP_COLORS["Other"])

def create_choropleth_map(result_data: Dict, scenario_type: str, 
                         scenario_name: str, filename: str, island_cache=None) -> folium.Map:
    """
    Create a choropleth map for the given result data.
    
    Args:
        result_data: Loaded result JSON data
        scenario_type: "Farm" or "Patch"
        scenario_name: Name of the scenario (e.g., "Farm_DWave")
        filename: Original filename
        island_cache: Cached island GeoDataFrame to avoid repeated OSM queries
        
    Returns:
        Folium map object
    """
    # Use cached island if provided, otherwise create from provided coordinates
    if island_cache is None:
        print("  Creating island geometry from provided coordinates...")
        # Use the same coordinates as the main cache: 5°55'51.4"S, 106°09'11.8"E
        center_lat = -5.975236
        center_lon = 107.058068
        delta_lon = 0.008  # Match main bbox size
        delta_lat = 0.006  # Match main bbox size
        minx = center_lon - delta_lon
        maxx = center_lon + delta_lon
        miny = center_lat - delta_lat
        maxy = center_lat + delta_lat
        island_poly = box(minx, miny, maxx, maxy)
        island = gp.GeoDataFrame({'geometry': [island_poly]}, crs='EPSG:4326')
    else:
        island = island_cache
    
    # Center coordinates
    # Suppress warning by converting to projected CRS temporarily for centroid calculation
    island_projected = island.to_crs('EPSG:3857')  # Web Mercator
    centroid_projected = island_projected.geometry.centroid
    centroid_latlon = centroid_projected.to_crs(island.crs)
    center = [centroid_latlon.y.values[0], centroid_latlon.x.values[0]]
    
    # Get bounds for grid creation
    bounds = island.total_bounds  # (minx, miny, maxx, maxy)
    
    # Create folium map
    m = folium.Map(location=center, zoom_start=15, tiles='OpenStreetMap')
    
    # Do not draw the bounding-box rectangle; keep island bounds for layout only
    # (Previously we added a GeoJson box around the island here.)
    
    # Extract solution data
    solution = result_data.get('solution_summary', {})
    plot_assignments = solution.get('plot_assignments', [])
    n_units = result_data.get('n_units', 10)
    total_area = result_data.get('total_area', 0)
    
    if scenario_type == "Farm":
        # Generate farm sizes
        farm_sizes = get_farm_sizes(n_units, seed=42)
        
        # Create a mapping of farm name -> farm group (e.g., Farm1, Farm2, etc.)
        # This is used to cluster plots with the same farm together
        farm_groups = {name: name for name in farm_sizes.keys()}
        
        # Create uneven grid
        farm_polygons = create_uneven_grid(bounds, farm_sizes, farm_groups=farm_groups)
        
        # Create a dictionary to map farm names to crop allocations
        farm_crops = {}
        for assignment in plot_assignments:
            crop = assignment['crop']
            for plot_info in assignment['plots']:
                farm_name = plot_info['plot']
                area = plot_info['area']
                
                if farm_name not in farm_crops:
                    farm_crops[farm_name] = []
                farm_crops[farm_name].append({'crop': crop, 'area': area})
        
        # Draw farms with crop splits
        for farm_name, polygon in farm_polygons.items():
            if farm_name in farm_crops:
                crop_allocations = farm_crops[farm_name]
                sections = split_polygon_by_crops(polygon, crop_allocations)
                
                for section_poly, crop_name, area in sections:
                    color = get_crop_color(crop_name)
                    
                    # Convert shapely polygon to GeoJSON
                    geo_json = folium.GeoJson(
                        section_poly.__geo_interface__,
                        style_function=lambda x, c=color: {
                            'fillColor': c,
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.7
                        }
                    )
                    
                    # Add tooltip
                    tooltip_text = f"""
                    <b>Farm:</b> {farm_name}<br>
                    <b>Crop:</b> {crop_name}<br>
                    <b>Area:</b> {area:.2f} ha
                    """
                    folium.Popup(tooltip_text, max_width=200).add_to(geo_json)
                    folium.Tooltip(f"{farm_name}: {crop_name}").add_to(geo_json)
                    
                    geo_json.add_to(m)
            else:
                # Farm not used - show in gray
                geo_json = folium.GeoJson(
                    polygon.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'lightgray',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.3
                    }
                )
                folium.Tooltip(f"{farm_name}: Unused").add_to(geo_json)
                geo_json.add_to(m)
    
    else:  # Patch scenario
        # Create even grid
        plot_polygons = create_even_grid(bounds, n_units)
        
        # Create mapping of plot names to crops
        plot_crops = {}
        for assignment in plot_assignments:
            crop = assignment['crop']
            for plot_info in assignment['plots']:
                plot_name = plot_info['plot']
                area = plot_info['area']
                
                if plot_name not in plot_crops:
                    plot_crops[plot_name] = []
                plot_crops[plot_name].append({'crop': crop, 'area': area})
        
        # Draw plots
        for i, polygon in enumerate(plot_polygons, 1):
            plot_name = f"Patch{i}"
            
            if plot_name in plot_crops:
                crop_info = plot_crops[plot_name]
                
                # If multiple crops, split the cell
                if len(crop_info) > 1:
                    sections = split_polygon_by_crops(polygon, crop_info)
                    
                    for section_poly, crop_name, area in sections:
                        color = get_crop_color(crop_name)
                        
                        geo_json = folium.GeoJson(
                            section_poly.__geo_interface__,
                            style_function=lambda x, c=color: {
                                'fillColor': c,
                                'color': 'black',
                                'weight': 1,
                                'fillOpacity': 0.7
                            }
                        )
                        
                        tooltip_text = f"""
                        <b>Plot:</b> {plot_name}<br>
                        <b>Crop:</b> {crop_name}<br>
                        <b>Area:</b> {area:.3f} ha
                        """
                        folium.Popup(tooltip_text, max_width=200).add_to(geo_json)
                        folium.Tooltip(f"{plot_name}: {crop_name}").add_to(geo_json)
                        
                        geo_json.add_to(m)
                else:
                    # Single crop
                    crop_name = crop_info[0]['crop']
                    area = crop_info[0]['area']
                    color = get_crop_color(crop_name)
                    
                    geo_json = folium.GeoJson(
                        polygon.__geo_interface__,
                        style_function=lambda x, c=color: {
                            'fillColor': c,
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.7
                        }
                    )
                    
                    tooltip_text = f"""
                    <b>Plot:</b> {plot_name}<br>
                    <b>Crop:</b> {crop_name}<br>
                    <b>Area:</b> {area:.3f} ha
                    """
                    folium.Popup(tooltip_text, max_width=200).add_to(geo_json)
                    folium.Tooltip(f"{plot_name}: {crop_name}").add_to(geo_json)
                    
                    geo_json.add_to(m)
            else:
                # Plot not used
                geo_json = folium.GeoJson(
                    polygon.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'lightgray',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.3
                    }
                )
                folium.Tooltip(f"{plot_name}: Unused").add_to(geo_json)
                geo_json.add_to(m)
    
    # Add legend
    add_legend(m, plot_assignments)
    
    return m

def add_legend(m: folium.Map, plot_assignments: List[Dict]):
    """Add a comprehensive legend to the map."""
    # Collect all crops used
    crops_used = set()
    for assignment in plot_assignments:
        crops_used.add(assignment['crop'])
    
    # Sort crops alphabetically
    crops_used = sorted(list(crops_used))
    
    # Create legend HTML
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <h4 style="margin-top:0;">Crop Legend</h4>
    '''
    
    for crop in crops_used:
        color = get_crop_color(crop)
        legend_html += f'''
        <p style="margin: 5px 0;">
            <span style="background-color:{color}; 
                        width: 20px; height: 20px; 
                        display: inline-block; 
                        border: 1px solid black;"></span>
            {crop}
        </p>
        '''
    
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function."""
    print("="*80)
    print("CHOROPLETH PLOT GENERATOR")
    print("="*80)
    
    # Load all results
    print("\nLoading result files...")
    results = load_all_results()
    
    print(f"Found {len(results['Farm'])} Farm scenarios")
    print(f"Found {len(results['Patch'])} Patch scenarios")
    
    # Use provided coordinates for near Jakarta (avoids geocoding mismatches)
    print("\nUsing provided coordinates for near Jakarta (no geocoding)...")
    try:
        # Coordinates from user: 5°55'51.4"S, 106°09'11.8"E
        center_lat = -5.975236
        center_lon = 107.058068
        # Larger bounding box for scattered farm placement
        delta_lon = 0.008  # Increased for more space
        delta_lat = 0.006  # Increased for more space

        minx = center_lon - delta_lon
        maxx = center_lon + delta_lon
        miny = center_lat - delta_lat
        maxy = center_lat + delta_lat

        island_poly = box(minx, miny, maxx, maxy)
        island = gp.GeoDataFrame({'geometry': [island_poly]}, crs='EPSG:4326')
        print(f"  ✓ Created island bbox centered at ({center_lat:.6f}, {center_lon:.6f})")
    except Exception as e:
        print(f"  ✗ Error creating island geometry: {e}")
        island = None
    
    # Generate maps for each result
    output_dir = "choropleth_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario_type in ['Farm', 'Patch']:
        print(f"\n{'='*80}")
        print(f"Processing {scenario_type} scenarios")
        print(f"{'='*80}")
        
        for key, result_info in results[scenario_type].items():
            print(f"\nGenerating map for: {key}")
            
            try:
                m = create_choropleth_map(
                    result_info['data'],
                    scenario_type,
                    result_info['scenario_name'],
                    result_info['filename'],
                    island_cache=island
                )
                
                # Save map
                output_filename = f"{output_dir}/choropleth_{key.replace('.json', '')}.html"
                m.save(output_filename)
                print(f"  ✓ Saved to: {output_filename}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print(f"Maps saved to: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()