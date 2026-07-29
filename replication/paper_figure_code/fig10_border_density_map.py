import geopandas as gpd
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from matplotlib.lines import Line2D
import math

def add_scale_bar(ax, x_min, x_max, y_min, y_max):
    """Add a scale bar to the map in the lower right"""
    # Calculate approximate distance for scale bar (in degrees)
    # At this latitude (~37°), 1 degree longitude ≈ 89 km
    lat_center = (y_min + y_max) / 2
    lon_per_km = 1 / (111.32 * math.cos(math.radians(lat_center)))
    
    # Choose appropriate scale (50 km)
    scale_km = 50
    scale_degrees = scale_km * lon_per_km
    
    # Position scale bar in bottom right (moved even higher up)
    scale_x_start = x_max - 0.25 * (x_max - x_min)
    scale_y = y_min + 0.25 * (y_max - y_min)  # Moved up from 0.15 to 0.25
    
    # Draw scale bar
    ax.plot([scale_x_start, scale_x_start + scale_degrees], 
            [scale_y, scale_y], 'k-', linewidth=3)
    
    # Add scale bar labels
    ax.text(scale_x_start, scale_y - 0.02 * (y_max - y_min), '0', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(scale_x_start + scale_degrees, scale_y - 0.02 * (y_max - y_min), f'{scale_km} km', 
            ha='center', va='top', fontsize=10, fontweight='bold')

def add_north_arrow(ax, x_min, x_max, y_min, y_max):
    """Add a north arrow to the map in the lower right"""
    # Position north arrow next to scale bar in bottom right (moved even higher up)
    arrow_x = x_max - 0.08 * (x_max - x_min)  # Position to the right of scale bar
    arrow_y_base = y_min + 0.21 * (y_max - y_min)  # Moved up from 0.11 to 0.21
    arrow_y_top = y_min + 0.29 * (y_max - y_min)   # Moved up from 0.19 to 0.29
    
    # Draw north arrow (longer body)
    ax.annotate('', xy=(arrow_x, arrow_y_top), 
                xytext=(arrow_x, arrow_y_base),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add 'N' label
    ax.text(arrow_x, arrow_y_top + 0.02 * (y_max - y_min), 'N', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Use only the Syrian data
gdf_frame_overlayed_destinations_sinira_giden_suriyeliler = gdf_frame_overlayed_destinations[gdf_frame_overlayed_destinations["customer_id"].isin(sinira_giden_suriyeliler)]

# Set the coordinate reference systems
gdf_frame_overlayed_destinations_sinira_giden_suriyeliler.crs = "EPSG:32636"
gdf_tower_unique.crs = "EPSG:32636"

# Define the border cities of interest
border_cities = ["SANLIURFA", "GAZIANTEP", "KILIS", "HATAY"]

# Extract city column if it doesn't exist
if 'city' not in gdf_tower_unique.columns:
    gdf_tower_unique[['city', 'district']] = gdf_tower_unique['city_district'].str.split('_', expand=True)

# Filter towers to include only those in the border cities
filtered_towers = gdf_tower_unique[gdf_tower_unique["city"].isin(border_cities)].copy()

# Create a unary union of ALL border cities combined into a single shape
border_region_union = unary_union(filtered_towers.geometry)
border_region_gdf = gpd.GeoDataFrame(geometry=[border_region_union], crs="EPSG:32636")

# Create individual city unions for labeling purposes
city_unions = {}
for city in border_cities:
    city_districts = filtered_towers[filtered_towers["city"] == city]
    if not city_districts.empty:
        city_union = unary_union(city_districts.geometry)
        city_unions[city] = gpd.GeoDataFrame(geometry=[city_union], crs="EPSG:32636")

# Convert to WGS84 for visualization
border_region_wgs84 = border_region_gdf.to_crs("EPSG:4326")
city_unions_wgs84 = {city: union.to_crs("EPSG:4326") for city, union in city_unions.items()}

# Define border gates
border_gates = [
    {"name": "Karkamış", "lat": 36.8345, "lon": 37.9983, "city": "GAZIANTEP"},
    {"name": "Yayladağı", "lat": 35.9025, "lon": 36.0606, "city": "HATAY"},
    {"name": "Cilvegözü", "lat": 36.2338, "lon": 36.6797, "city": "HATAY"},
    {"name": "Öncüpınar", "lat": 36.6439, "lon": 37.0872, "city": "KILIS"},
    {"name": "Çobanbey", "lat": 36.6325, "lon": 37.4728, "city": "KILIS"},
    {"name": "Akçakale", "lat": 36.7072, "lon": 38.9491, "city": "SANLIURFA"},
    {"name": "Ceylanpınar", "lat": 36.8461, "lon": 40.0489, "city": "SANLIURFA"}
]

# Convert city centers and refugee camps to WGS84
gdf_city_centers_wgs84 = gdf_city_centers.to_crs("EPSG:4326")
refugee_camps_wgs84 = refugee_camps.to_crs("EPSG:4326")

# Get the bounds for the visualization
x_min, y_min, x_max, y_max = border_region_wgs84.total_bounds

# Create the figure
fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')

# Create a smooth gradient with 100 different shades
# Define key transition points for smooth color progression
import matplotlib.colors as mcolors

# Define color transition points: white -> light blue -> medium blue -> dark blue -> purple -> dark purple
color_points = [
    (0.0, '#ffffff'),    # 0% - white
    (0.2, '#cce7ff'),    # 20% - very light blue
    (0.4, '#66ccff'),    # 40% - light blue
    (0.6, '#0099ff'),    # 60% - medium blue
    (0.8, '#0066cc'),    # 80% - dark blue
    (0.9, '#6600cc'),    # 90% - purple
    (1.0, '#330066')     # 100% - dark purple
]

# Create smooth colormap with 100 segments for gradual transitions
all_colors = []
for i in range(101):  # 0 to 100%
    ratio = i / 100.0
    # Find the appropriate color segment
    for j in range(len(color_points) - 1):
        if color_points[j][0] <= ratio <= color_points[j + 1][0]:
            # Interpolate between the two colors
            start_ratio = color_points[j][0]
            end_ratio = color_points[j + 1][0]
            start_color = color_points[j][1]
            end_color = color_points[j + 1][1]
            
            # Calculate interpolation factor
            if end_ratio == start_ratio:
                interp_factor = 0
            else:
                interp_factor = (ratio - start_ratio) / (end_ratio - start_ratio)
            
            # Interpolate colors
            start_rgb = mcolors.hex2color(start_color)
            end_rgb = mcolors.hex2color(end_color)
            
            interpolated_rgb = [
                start_rgb[0] + (end_rgb[0] - start_rgb[0]) * interp_factor,
                start_rgb[1] + (end_rgb[1] - start_rgb[1]) * interp_factor,
                start_rgb[2] + (end_rgb[2] - start_rgb[2]) * interp_factor
            ]
            
            all_colors.append(mcolors.rgb2hex(interpolated_rgb))
            break

# Set background color
ax.set_facecolor('white')

# Plot the combined border region boundary with transparency
border_region_wgs84.boundary.plot(
    ax=ax,
    color='black',
    linewidth=1.5,
    alpha=0.6  # Made more transparent from 1.0 to 0.6
)

# Plot individual city boundaries with thinner lines and transparency
for city, city_gdf in city_unions_wgs84.items():
    city_gdf.boundary.plot(
        ax=ax,
        color='gray',
        linewidth=0.8,
        alpha=0.4  # Made more transparent from 0.6 to 0.4
    )

# Convert Syrian data to WGS84
gdf_data_wgs84 = gdf_frame_overlayed_destinations_sinira_giden_suriyeliler.to_crs("EPSG:4326")
if 'city' not in gdf_data_wgs84.columns and 'city_district' in gdf_data_wgs84.columns:
    gdf_data_wgs84[['city', 'district']] = gdf_data_wgs84['city_district'].str.split('_', expand=True)

border_data = gdf_data_wgs84[gdf_data_wgs84["city"].isin(border_cities)]

# Count unique customer_ids across all border cities
unique_customers = border_data['customer_id'].nunique() if not border_data.empty else 0

# Extract centroids of polygons for KDE processing
if not border_data.empty:
    centroids = border_data.geometry.centroid
    points_x = np.array([p.x for p in centroids])
    points_y = np.array([p.y for p in centroids])

    # Create a finer grid for the KDE
    grid_size = 400  # Higher resolution
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Apply KDE with adjusted bandwidth
    kde = gaussian_kde(np.vstack([points_x, points_y]), bw_method='silverman')
    kde.set_bandwidth(kde.factor * 0.3)  # Consistent bandwidth adjustment
    
    Z = kde(positions)
    Z = Z.reshape(X.shape)
    
    # Normalize Z values to 0-100% scale
    Z_normalized = (Z / Z.max()) * 100
    
    # Create custom color map with 100 smooth transitions
    cmap = ListedColormap(all_colors)
    
    # Plot the KDE map with 100 discrete levels for clear differentiation
    contour = ax.contourf(X, Y, Z_normalized, levels=100, cmap=cmap, alpha=0.8)
    
    # Add contour lines with normalized values
    contour_lines = ax.contour(X, Y, Z_normalized, levels=15, colors='black', linewidths=0.3, alpha=0.5)
    
    # Add a colorbar with proper sizing to match plot height
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02, aspect=30)
    
    # Create tick locations at 0%, 20%, 40%, 60%, 80%, 100%
    tick_locations = [0, 20, 40, 60, 80, 100]
    cbar.set_ticks(tick_locations)
    
    # Format the labels
    tick_labels = [f"{val}%" for val in tick_locations]
    cbar.set_ticklabels(tick_labels)
    
    # Set the colorbar label
    cbar.set_label('Syrian DP concentration (% of maximum density)', fontsize=10)
else:
    ax.text(0.5, 0.5, f"No data available\nTotal number: {unique_customers}", 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)

# Add border gates with red frame and increased transparency
for gate in border_gates:
    # Plot a square with red frame (no black lines)
    ax.scatter(gate["lon"], gate["lat"], 
               marker='s',  # Square for gate
               facecolor='none',  # No fill
               edgecolor='red',  # Red frame instead of orange
               linewidths=2,  # Thinner line, reduced from 5 to 2
               s=150,  # Larger squares
               alpha=0.6,  # Added transparency
               zorder=5)  # Make sure it's on top of other elements
    
    # Special case for Yayladağı to avoid overlap with refugee camp
    if gate["name"] == "Yayladağı":
        label_offset = (20, -15)  # Move label to the right and down
    else:
        label_offset = (0, -20)  # Move all other labels below
    
    # Add the label
    ax.annotate(gate["name"], 
                xy=(gate["lon"], gate["lat"]), 
                xytext=label_offset,
                textcoords="offset points",
                fontsize=8,
                ha='center',
                va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                zorder=5)

# Filter refugee camps to only include those within the border region
refugee_camps_in_border = gpd.sjoin(
    refugee_camps_wgs84, 
    border_region_wgs84,
    how="inner", 
    predicate="within"
)

# If no camps are within the exact border, use a small buffer to find nearby camps
if len(refugee_camps_in_border) == 0:
    border_region_buffer = border_region_wgs84.buffer(0.05)  # Small buffer in degrees
    border_gdf_buffer = gpd.GeoDataFrame(geometry=border_region_buffer, crs="EPSG:4326")
    refugee_camps_in_border = gpd.sjoin(
        refugee_camps_wgs84, 
        border_gdf_buffer,
        how="inner", 
        predicate="within"
    )

# Add refugee camps as larger yellow stars with transparency
for idx, camp in refugee_camps_in_border.iterrows():
    # Use the original positions without shifting
    ax.scatter(camp.geometry.x, camp.geometry.y, 
               marker='*', 
               color='gold', 
               edgecolor='black',
               s=200,  # Larger stars
               alpha=0.7,  # Added transparency
               zorder=5)

# Filter city centers to only include the specified cities
border_city_list = ['Hatay', 'Gaziantep', 'Kilis', 'Şanlıurfa']
filtered_city_centers = gdf_city_centers_wgs84[gdf_city_centers_wgs84['City'].isin(border_city_list)]

# Add filtered city centers as red points
filtered_city_centers.plot(ax=ax, color='red', markersize=70, zorder=4)

# Add city labels consistently at upper left
for idx, row in filtered_city_centers.iterrows():
    # Place all city labels consistently at upper left
    x_offset, y_offset = -25, -25
    
    ax.annotate(
        text=row['City'],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        fontsize=10,
        color='red',
        weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7)
    )

# Add scale bar and north arrow in the lower right
add_scale_bar(ax, x_min, x_max, y_min, y_max)
add_north_arrow(ax, x_min, x_max, y_min, y_max)

# Set title with total count
ax.set_title(f'Distribution of Syrian DPs Across Syrian Border\nTotal number: {unique_customers}', 
          fontsize=14, pad=10)
ax.set_xlabel('Longitude', fontsize=10)
ax.set_ylabel('Latitude', fontsize=10)

# Set the extent to the border region
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])

# Enforce aspect ratio for more balanced appearance
ax.set_aspect('equal')

# Show grid
ax.grid(True, linestyle='--', alpha=0.3)

# Update legend to show red square
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='red', 
           markeredgewidth=2.5, markersize=10, label='Border Gates'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', 
           markersize=12, label='Temporary Accommodation Centers (TACs)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
           markersize=10, label='City Centers')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Adjust layout
plt.tight_layout()

# Save the figure with higher DPI for better quality
plt.savefig("/Desktop/syrian_border_analysis.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()