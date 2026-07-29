import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Assuming plot_damage, syrian_data, turkish_data are already created as in your first code snippet
# The following creates a 2x2 figure with:
# - Top row: Distance distribution KDE plots (from first code)
# - Bottom row: Cumulative displacement plots (from second code)

# Set up the figure
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Define professional color palette
professional_colors = {
    1: '#1f77b4',  # Blue - Low damage
    2: '#ff7f0e',  # Orange - Medium low damage
    3: '#d62728',  # Red - High damage
    4: '#9467bd'  # Purple - Very high damage
}

damage_colors = {
    'Low': '#1f77b4',        # Blue
    'Medium': '#ff7f0e', # Orange
    'High': '#d62728',       # Red
    'Very High ': '#9467bd'   # Purple
}

# Create damage labels dictionary
damage_labels = {
    1: 'Low',
    2: 'Medium',
    3: 'High',#,
    4: 'Very High'
}

#######################################
# TOP ROW - Distance KDE Plots (from first code)
#######################################

x_min = 0
x_max = max(plot_damage['distance_km'].max() * 1.05, 10)  # Add 5% buffer or at least show up to 10 km

# Syrian plot - top left
for i, damage_cat in enumerate([1, 2, 3, 4]):
    cat_data = syrian_data[syrian_data['damage_category'] == damage_cat]['distance_km']
    if len(cat_data) >= 2:  # Need at least 2 points for KDE
        label = damage_labels[damage_cat]
        sns.kdeplot(cat_data, 
                   label=label, 
                   fill=False, 
                   alpha=0.7, 
                   color=professional_colors[damage_cat], 
                   linewidth=2, 
                   ax=axes[0,0])

# Turkish plot - top right
for i, damage_cat in enumerate([1, 2, 3, 4]):
    cat_data = turkish_data[turkish_data['damage_category'] == damage_cat]['distance_km']
    if len(cat_data) >= 2:  # Need at least 2 points for KDE
        label = damage_labels[damage_cat]
        sns.kdeplot(cat_data, 
                   label=label, 
                   fill=False, 
                   alpha=0.7, 
                   color=professional_colors[damage_cat], 
                   linewidth=2, 
                   ax=axes[0,1])

# Explicitly set x-axis limits to prevent negative values
axes[0,0].set_xlim(x_min, x_max)
axes[0,1].set_xlim(x_min, x_max)

# Fix the x-axis ticks for distance plots
max_tick = int(np.ceil(x_max / 100) * 100)
x_ticks = np.arange(0, max_tick + 1, 100)  # 0, 100, 200, etc.
x_labels = [f"{int(x)}" for x in x_ticks]

# Apply the cleaned ticks
for ax in [axes[0,0], axes[0,1]]:
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Distance (km)", fontsize=12)

axes[0,0].set_title("Distance distribution by damage category - Syrian DPs", fontsize=20)
axes[0,0].set_ylabel("Density", fontsize=12)
# Move legend to lower right
axes[0,0].legend(title="Damage Category", fontsize=9, loc='lower right')

axes[0,1].set_title("Distance distribution by damage category - Turkish DPs", fontsize=20)
axes[0,1].set_ylabel("Density", fontsize=12)
# Move legend to lower right
axes[0,1].legend(title="Damage Category", fontsize=9, loc='lower right')

#######################################
# BOTTOM ROW - Cumulative Displacement Plots (from second code)
#######################################

# Lists to collect line objects and labels for ordered legend
syrian_lines = []
syrian_labels = []
turkish_lines = []
turkish_labels = []

# Process data and plot for Syrian DPs - bottom left
for damage_cat in [1, 2, 3, 4]:  # Plot in order from low to very high
    if damage_cat in syrian_data['damage_category'].unique():
        label = damage_labels[damage_cat]
        # Filter data for this damage category
        cat_data = syrian_data[syrian_data['damage_category'] == damage_cat]
        
        # Get total count for this damage category
        cat_total = len(cat_data)
        
        if cat_total == 0:
            continue  # Skip if no data for this category
        
        # Group by day to get count per day for this category
        daily_counts = cat_data.groupby('displacement_date').size().reset_index(name='count')
        
        # Convert to percentage of total for this damage category
        daily_counts['percentage'] = daily_counts['count'] / cat_total * 100
        
        # Make sure we have all days up to max_day (fill missing with zeros)
        max_day = 25  # As specified in your original code
        all_days = pd.DataFrame({'displacement_date': range(1, max_day + 1)})
        daily_counts = all_days.merge(daily_counts, on='displacement_date', how='left').fillna(0)
        
        # Calculate cumulative percentage
        daily_counts['cumulative'] = daily_counts['percentage'].cumsum()
        
        # Plot this damage category with updated label format
        line, = axes[1,0].plot(
            daily_counts['displacement_date'], 
            daily_counts['cumulative'], 
            marker='o',
            linewidth=2,
            color=professional_colors[damage_cat]
        )
        
        # Store line and label for ordered legend
        syrian_lines.append(line)
        syrian_labels.append(f"{label} (Number of DPs = {cat_total})")

# Process data and plot for Turkish DPs - bottom right
for damage_cat in [1, 2, 3, 4]:  # Plot in order from low to very high
    if damage_cat in turkish_data['damage_category'].unique():
        label = damage_labels[damage_cat]
        # Filter data for this damage category
        cat_data = turkish_data[turkish_data['damage_category'] == damage_cat]
        
        # Get total count for this damage category
        cat_total = len(cat_data)
        
        if cat_total == 0:
            continue  # Skip if no data for this category
        
        # Group by day to get count per day for this category
        daily_counts = cat_data.groupby('displacement_date').size().reset_index(name='count')
        
        # Convert to percentage of total for this damage category
        daily_counts['percentage'] = daily_counts['count'] / cat_total * 100
        
        # Make sure we have all days up to max_day (fill missing with zeros)
        max_day = 25  # As specified in your original code
        all_days = pd.DataFrame({'displacement_date': range(1, max_day + 1)})
        daily_counts = all_days.merge(daily_counts, on='displacement_date', how='left').fillna(0)
        
        # Calculate cumulative percentage
        daily_counts['cumulative'] = daily_counts['percentage'].cumsum()
        
        # Plot this damage category with updated label format
        line, = axes[1,1].plot(
            daily_counts['displacement_date'], 
            daily_counts['cumulative'], 
            marker='o',
            linewidth=2,
            color=professional_colors[damage_cat]
        )
        
        # Store line and label for ordered legend
        turkish_lines.append(line)
        turkish_labels.append(f"{label} (Number of DPs = {cat_total})")

# Configure cumulative displacement plot aesthetics
# Add custom ordered legends
axes[1,0].legend(syrian_lines, syrian_labels, loc='lower right', fontsize=9, title="Damage Category")
axes[1,1].legend(turkish_lines, turkish_labels, loc='lower right', fontsize=9, title="Damage Category")

for i, title in enumerate(['Cumulative displacement patterns of Syrian DPs', 
                          'Cumulative displacement patterns of Turkish DPs']):
    axes[1,i].set_title(title, fontsize=20)
    axes[1,i].set_xlabel('Date', fontsize=12)
    axes[1,i].set_xlim(1, 25)
    axes[1,i].set_ylim(0, 100)  # Percentage from 0-100%
    axes[1,i].set_xticks(range(1, 26, 2))
    axes[1,i].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[1,i].set_yticks(range(0, 101, 10))
    axes[1,i].grid(True, linestyle='--', alpha=0.7)
    axes[1,i].set_axisbelow(True)

# Adjust layout 
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.subplots_adjust(hspace=0.3)

# Save the figure
plt.savefig("/Desktop/distance-displacement-datef.png", dpi=300, bbox_inches='tight')
plt.show()