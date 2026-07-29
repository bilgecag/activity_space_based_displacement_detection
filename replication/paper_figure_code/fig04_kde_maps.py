# Set the coordinate reference systems (keeping your original data setup)
gdf_origins.crs = "EPSG:32636"
gdf_destinations.crs = "EPSG:32636"
gdf_tower_unique.crs = "EPSG:32636"
eq_cities_unary.crs = "EPSG:32636"
gdf_city_centers.crs = "EPSG:32636"  # Add city centers CRS
refugee_camps.crs = "EPSG:32636"  # Add refugee camps CRS

# Convert all datasets to WGS84 for visualization
gdf_origins_wgs84 = gdf_origins.to_crs("EPSG:4326")
gdf_destinations_wgs84 = gdf_destinations.to_crs("EPSG:4326")
gdf_tower_unique_wgs84 = gdf_tower_unique.to_crs("EPSG:4326")
eq_cities_unary_wgs84 = eq_cities_unary.to_crs("EPSG:4326")
gdf_city_centers_wgs84 = gdf_city_centers.to_crs("EPSG:4326")  # Convert city centers to WGS84
refugee_camps_wgs84 = refugee_camps.to_crs("EPSG:4326")  # Convert refugee camps to WGS84

# Filter out rows with no night duration
gdf_origins_wgs84 = gdf_origins_wgs84[gdf_origins_wgs84["night_duration_aggregated"] > 0].reset_index(drop=True)
gdf_destinations_wgs84 = gdf_destinations_wgs84[gdf_destinations_wgs84["night_duration_aggregated"] > 0].reset_index(drop=True)
gdf_origins_wgs84=gdf_origins_wgs84.merge(df_affected_customers,on="customer_id",how="left")
gdf_destinations_wgs84=gdf_destinations_wgs84.merge(df_affected_customers,on="customer_id",how="left")

# Get the bounds for consistent visualization
x_min, y_min, x_max, y_max = gdf_tower_unique_wgs84[gdf_tower_unique_wgs84["is_in_earthquake_area"] == 1].total_bounds

# Define segments and color schemes
segments = [2, 1]  # 2 for Syrian (left), 1 for Turkish (right)
segment_names = {1: "Turkish", 2: "Syrian"}

# Define density threshold for color change
density_threshold = 0.5  # Values above this will shift towards purple

# Store KDE results
all_z_values = {
    "origins": {"Syrian": None, "Turkish": None},
    "destinations": {"Syrian": None, "Turkish": None}
}

# First pass: Generate all KDE data
for dataset_idx, dataset in enumerate([
    {"data": gdf_origins_wgs84, "title_prefix": "Origin"},
    {"data": gdf_destinations_wgs84, "title_prefix": "Destination"}
]):
    data = dataset["data"]
    dataset_type = "origins" if dataset_idx == 0 else "destinations"
    
    # Process each segment (Syrian and Turkish)
    for i, segment in enumerate(segments):
        segment_name = "Syrian" if segment == 2 else "Turkish"
        
        # Filter data for the current segment
        segment_data = data[data['segment'] == segment]
        
        # Skip processing if no data for this segment
        if len(segment_data) == 0:
            all_z_values[dataset_type][segment_name] = np.zeros((400, 400))
            continue
        
        # Extract centroids of polygons for faster KDE processing
        centroids = segment_data.centroid
        points_x = np.array([p.x for p in centroids])
        points_y = np.array([p.y for p in centroids])
        
        # Get weights from night_duration_aggregated
        weights = segment_data['night_duration_aggregated'].values
        
        # Create a finer grid for the KDE
        x_grid = np.linspace(x_min, x_max, 400)
        y_grid = np.linspace(y_min, y_max, 400)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])

        
        # WITH THE WEIGHTS 

        
        kde = gaussian_kde(np.vstack([points_x, points_y]), weights=weights/np.sum(weights), bw_method='silverman')

        # WITHOUT THE WEIGHTS 
        

        #kde = gaussian_kde(np.vstack([points_x, points_y]), bw_method='silverman')


        
        kde.set_bandwidth(kde.factor * 0.3)
        
        Z = kde(positions)
        Z = Z.reshape(X.shape)
        
        # Store Z values
        all_z_values[dataset_type][segment_name] = Z

# Calculate max value across all datasets for consistent scaling
max_all = np.max([
    np.max(all_z_values["origins"]["Syrian"]), 
    np.max(all_z_values["origins"]["Turkish"]),
    np.max(all_z_values["destinations"]["Syrian"]), 
    np.max(all_z_values["destinations"]["Turkish"])
])
# ADD THESE 8 PRINTS HERE:
print("=== KDE Value Ranges Across All Subplots ===")
print(f"Syrian Origin - Lowest: {np.min(all_z_values['origins']['Syrian']):.6f}")
print(f"Syrian Origin - Highest: {np.max(all_z_values['origins']['Syrian']):.6f}")
print(f"Turkish Origin - Lowest: {np.min(all_z_values['origins']['Turkish']):.6f}")
print(f"Turkish Origin - Highest: {np.max(all_z_values['origins']['Turkish']):.6f}")
print(f"Syrian Destination - Lowest: {np.min(all_z_values['destinations']['Syrian']):.6f}")
print(f"Syrian Destination - Highest: {np.max(all_z_values['destinations']['Syrian']):.6f}")
print(f"Turkish Destination - Lowest: {np.min(all_z_values['destinations']['Turkish']):.6f}")
print(f"Turkish Destination - Highest: {np.max(all_z_values['destinations']['Turkish']):.6f}")
print("=" * 50)

# Calculate percentiles for annotation
all_values = np.concatenate([
    all_z_values["origins"]["Syrian"].flatten(),
    all_z_values["origins"]["Turkish"].flatten(),
    all_z_values["destinations"]["Syrian"].flatten(), 
    all_z_values["destinations"]["Turkish"].flatten()
])
all_values = all_values[all_values > 0]

# Find which percentile corresponds to our density threshold
threshold_percentile = 100 * (np.sum(all_values <= density_threshold) / len(all_values))

# Function to add scale bar
def add_scale_bar(ax, x_min, x_max, y_min, y_max):
    """Add a scale bar to the map"""
    import math
    
    # Calculate approximate distance for scale bar (in degrees)
    # At this latitude (~37°), 1 degree longitude ≈ 89 km
    lat_center = (y_min + y_max) / 2
    lon_per_km = 1 / (111.32 * math.cos(math.radians(lat_center)))
    
    # Choose appropriate scale (50 km)
    scale_km = 50
    scale_degrees = scale_km * lon_per_km
    
    # Position scale bar in bottom left
    scale_x_start = x_min + 0.05 * (x_max - x_min)
    scale_y = y_min + 0.08 * (y_max - y_min)
    
    # Draw scale bar
    ax.plot([scale_x_start, scale_x_start + scale_degrees], 
            [scale_y, scale_y], 'k-', linewidth=3)
    
    # Add scale bar labels
    ax.text(scale_x_start, scale_y - 0.02 * (y_max - y_min), '0', 
            ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(scale_x_start + scale_degrees, scale_y - 0.02 * (y_max - y_min), f'{scale_km} km', 
            ha='center', va='top', fontsize=10, fontweight='bold')

# Function to add north arrow
def add_north_arrow(ax, x_min, x_max, y_min, y_max):
    """Add a north arrow to the map"""
    # Position north arrow next to scale bar in bottom left
    arrow_x = x_min + 0.18 * (x_max - x_min)  # Position to the right of scale bar
    arrow_y_base = y_min + 0.04 * (y_max - y_min)  # Lower base position
    arrow_y_top = y_min + 0.12 * (y_max - y_min)   # Higher top position for longer arrow
    
    # Draw north arrow (longer body)
    ax.annotate('', xy=(arrow_x, arrow_y_top), 
                xytext=(arrow_x, arrow_y_base),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add 'N' label
    ax.text(arrow_x, arrow_y_top + 0.02 * (y_max - y_min), 'N', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Create a single 2x2 figure with all plots
fig, axs = plt.subplots(2, 2, figsize=(20, 16), facecolor='white')
axs = axs.flatten()  # Flatten to easily index

# Positioning for the plots
plot_positions = [
    {"dataset_type": "origins", "segment": 2, "ax_idx": 0, "title": "Syrian Origin Stay Locations", "data": gdf_origins_wgs84},
    {"dataset_type": "origins", "segment": 1, "ax_idx": 1, "title": "Turkish Origin Stay Locations", "data": gdf_origins_wgs84},
    {"dataset_type": "destinations", "segment": 2, "ax_idx": 2, "title": "Syrian Destination Stay Locations", "data": gdf_destinations_wgs84},
    {"dataset_type": "destinations", "segment": 1, "ax_idx": 3, "title": "Turkish Destination Stay Locations", "data": gdf_destinations_wgs84}
]

# Create custom colormaps for low density with more granular colors
blue_colors = [
    '#ffffff', '#f7fbff', '#e3f2fd', '#d1eafd', 
    '#b3e0fc', '#90d1fb', '#6ec2fa', '#4bb3f9', 
    '#29a4f8', '#0295f7', '#0084e0', '#0063a7'
]

# Use the original green colors but expand to create a more granular transition
original_green_colors = ['#ffffff', '#f1f8e9', '#dcedc8', '#c5e1a5', '#aed581', '#9ccc65', '#8bc34a', '#7cb342', '#689f38', '#558b2f']
# Add intermediate greens to create a more granular palette
green_colors = [
    '#ffffff',  # White
    '#f1f8e9',  # Very light green
    '#e8f5db',  # Additional light green (interpolated)
    '#dcedc8',  # Light green
    '#d1e9b6',  # Additional light-medium green (interpolated)
    '#c5e1a5',  # Light-medium green
    '#b9dc93',  # Additional medium green (interpolated)
    '#aed581',  # Medium green
    '#9fd06e',  # Additional medium-dark green (interpolated)
    '#8bc34a',  # Medium-dark green
    '#7cb342',  # Dark green
    '#689f38',  # Darker green
    '#558b2f'   # Darkest green
]

blue_cmap = LinearSegmentedColormap.from_list('granular_blue', blue_colors, N=256)
green_cmap = LinearSegmentedColormap.from_list('granular_green', green_colors, N=256)

# Create a more granular purple colormap for high density areas
purple_colors = [
    '#9370DB', '#8A6BBE', '#8066A2', '#7661A0', 
    '#6C5C9E', '#62579D', '#59529C', '#4F4D9B', 
    '#46489A', '#3D4399', '#333E98', '#2A3997', 
    '#213496', '#182F95', '#0F2A94', '#062593'
]
purple_cmap = ListedColormap(purple_colors)

# Process each plot
for position in plot_positions:
    dataset_type = position["dataset_type"]
    segment = position["segment"]
    ax = axs[position["ax_idx"]]
    title = position["title"]
    data = position["data"]
    
    segment_name = "Syrian" if segment == 2 else "Turkish"
    Z = all_z_values[dataset_type][segment_name]
    
    # Count unique customer_ids for this segment
    segment_data = data[data['segment'] == segment]
    unique_customers = segment_data['customer_id'].nunique() if len(segment_data) > 0 else 0
    
    # Update title with count
    #title_with_count = f"{title} (N={unique_customers})"
    
    # Skip processing if no data for this segment
    if np.all(Z == 0):
        ax.text(0.5, 0.5, f"No data for {segment_names[segment]}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=20)
        ax.set_title(title_with_count, fontsize=24)
        continue
    
    # Create grid for contour plotting
    x_grid = np.linspace(x_min, x_max, 400)
    y_grid = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create mask for different density ranges
    low_mask = Z <= density_threshold
    high_mask = Z > density_threshold
    
    # Create visualization arrays
    Z_low = np.copy(Z)
    Z_low[high_mask] = np.nan
    
    Z_high = np.copy(Z)
    Z_high[low_mask] = np.nan
    
    # Choose the correct colormap for low values
    low_cmap_choice = blue_cmap if segment == 2 else green_cmap
    
    # Set levels for low density - more granular
    # Determine the range of our low values
    low_min = np.min(Z_low[~np.isnan(Z_low)]) if np.any(~np.isnan(Z_low)) else 0
    low_max = density_threshold
    
    # Create more granular levels for the blue/green range
    num_low_levels = len(green_colors) if segment == 1 else len(blue_colors)  # Match number of colors in our colormaps
    low_levels = np.linspace(low_min, low_max, num_low_levels)
    
    # Plot the low density areas
    if np.any(~np.isnan(Z_low)):
        contour_low = ax.contourf(X, Y, Z_low, levels=low_levels, cmap=low_cmap_choice, alpha=0.8, extend='max')
    
    # Set levels for high density (purple) - more granular
    if np.any(high_mask):
        high_min = density_threshold
        high_max = np.max(Z_high[~np.isnan(Z_high)])
        
        # Create more granular levels for the purple range
        num_purple_colors = len(purple_colors)
        high_levels = np.linspace(high_min, high_max, num_purple_colors + 1)
        
        # Plot the high density areas
        contour_high = ax.contourf(X, Y, Z_high, levels=high_levels, cmap=purple_cmap, alpha=0.8, extend='max')
    
    # Add contour lines for visual clarity
    contour_lines = ax.contour(X, Y, Z, levels=[density_threshold], colors='black', linewidths=1.5, alpha=0.7)
    
    # Add more contour lines in both low and high density regions for more granular detail
    if np.any(high_mask):
        # More contour lines in high density regions
        high_level_count = 6
        high_contour_levels = np.linspace(density_threshold, high_max, high_level_count + 1)[1:]
        contour_high_lines = ax.contour(X, Y, Z, levels=high_contour_levels, colors='black', linewidths=0.3, alpha=0.5)
    
    if np.any(~np.isnan(Z_low)):
        # Add some contour lines in low density regions too
        low_level_count = 5
        low_contour_levels = np.linspace(low_min + 0.1*(low_max-low_min), low_max, low_level_count)
        contour_low_lines = ax.contour(X, Y, Z, levels=low_contour_levels, colors='black', linewidths=0.2, alpha=0.3)
    
    # Add the tower district boundaries
    gdf_tower_unique_wgs84.boundary.plot(
        ax=ax,
        color='black',
        linewidth=0.7,
        alpha=0.9
    )
    
    # Add the earthquake city boundaries with thicker lines
    eq_cities_unary_wgs84.boundary.plot(
        ax=ax,
        color='black',
        linewidth=3.0,
        alpha=1.0
    )
    
    # Add city centers as red points to ALL plots with enhanced styling
    gdf_city_centers_wgs84.plot(
        ax=ax,
        color='red',
        markersize=50,
        alpha=0.7,
        edgecolor='black',  # Add black edge to red circles
        linewidth=1.5,     # Make the edge line visible
        zorder=10  # Ensure points are drawn on top of boundaries
    )
    
    # Add city labels for Origin plots (top row) and Turkish Destination plot (lower right)
    if dataset_type == "origins" or (dataset_type == "destinations" and segment == 1):
        # Add city labels with adjusted positions and enhanced bounding boxes
        for idx, row in gdf_city_centers_wgs84.iterrows():
            # Special positioning for Hatay (lower right)
            if row['City'] == 'Hatay':
                x_offset, y_offset = 10, 10  # Move to lower right
            else:
                x_offset, y_offset = 15, -15  # Move to upper right for all others
                
            ax.annotate(
                text=row['City'],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(x_offset, y_offset),  # Custom offset based on city
                textcoords="offset points",
                fontsize=10,
                color='red',
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8, edgecolor='black', linewidth=1)  # Enhanced bbox
            )
    
    # Add refugee camps as yellow stars for Syrian Origin and Syrian Destination plots
    if (dataset_type == "destinations" and segment == 2) or (dataset_type == "origins" and segment == 2):  
        # Plot refugee camps as yellow stars with enhanced black outline
        refugee_camps_wgs84.plot(
            ax=ax,
            color='yellow',
            marker='*',
            markersize=150,  # Keep the size at 150
            alpha=0.9,       # Slightly increase alpha for better visibility
            edgecolor='black',
            linewidth=1.5,   # Increase linewidth for more prominent star definition
            zorder=15  # Higher zorder to ensure stars are on top of everything
        )
        
        # Add camp labels ONLY for Syrian Destination plot (lower left)
        if dataset_type == "destinations":
            for idx, row in refugee_camps_wgs84.iterrows():
                ax.annotate(
                    text=row['cmp_tr'],  # Using cmp_tr as the label
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(5, 5),  # Small offset from the star
                    textcoords="offset points",
                    fontsize=10,
                    color='black',  # Black text
                    weight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8, edgecolor='black', linewidth=1),  # Enhanced bbox
                    zorder=16  # Make sure labels are on top of stars
                )
    
    # Add scale bar and north arrow to each subplot
    add_scale_bar(ax, x_min, x_max, y_min, y_max)
    add_north_arrow(ax, x_min, x_max, y_min, y_max)
    
    # Set title and labels
    ax.set_title(title, fontsize=24)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    
    # Set consistent extent for all plots
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    # Show grid
    ax.grid(True, linestyle='--', alpha=0.4)

# Add a legend for city centers and refugee camps
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=20, 
           markeredgecolor='black', markeredgewidth=1.5, label='City Centers'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', markersize=20, 
           markeredgecolor='black', markeredgewidth=1.5, label='Temporary Accommodation Centers (TACs)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=15, bbox_to_anchor=(0.5, 0.02))

# Add threshold information below main plot area
#plt.figtext(0.5, 0.01, 
#            f"Density threshold at 1.0 represents the {threshold_percentile:.1f}th percentile of all non-zero density values.",
#            ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.7, "pad":5})

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust bottom margin to accommodate the legend

# Save the figure with high resolution
plt.savefig("/Desktop/combined-density-plot-with-scale.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()