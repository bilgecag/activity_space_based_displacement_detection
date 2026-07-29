from scipy import stats

# Start with the merged data
plot_damage = origins_grouped.merge(pd.concat([turkish, syrians]))
plot_damage['distance'] = pd.to_numeric(plot_damage['distance'], errors='coerce')
plot_damage = plot_damage[plot_damage['distance'].notna() & (plot_damage['distance'] >= 0)]
plot_damage['distance_km'] = plot_damage['distance'] / 1000

# Separate by segment (1=Turkish, 2=Syrian)
syrian_data = plot_damage[plot_damage['segment'] == 2]
turkish_data = plot_damage[plot_damage['segment'] == 1]

# Create damage category based on weighted_damage_index_origin
def create_damage_category(df):
    conditions = [
        (df['weighted_damage_index_origin'] <= 0.01),
        (df['weighted_damage_index_origin'] > 0.01)& (df['weighted_damage_index_origin']<= 0.1), 
        (df['weighted_damage_index_origin'] > 0.1)  & (df['weighted_damage_index_origin']<= 0.2),
        (df['weighted_damage_index_origin'] > 0.2)
    ]
    values = [1, 2, 3, 4]
    labels = ['Low', 'Medium', 'High', 'Very High']#, 'High'
    
    df['damage_category'] = np.select(conditions, values, default=np.nan)
    df['damage_category_label'] = np.select(conditions, labels, default='Unknown')
    return df

syrian_data = create_damage_category(syrian_data)
turkish_data = create_damage_category(turkish_data)

# Combine the datasets again for plotting
plot_damage = pd.concat([syrian_data, turkish_data])

# Define a professional color palette
professional_colors = [
    '#1f77b4',  # Blue - Low damage
    '#ff7f0e',  # Orange - Medium low damage
    '#d62728',  # Red - High damage
    '#9467bd',  # Purple - Very high damage
]

# Create a labels dictionary
damage_labels = {
    1: 'Low',
    2: 'Medium Low',
    3: 'High',
    4: 'Very High'
}

# Set up the figure
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Set explicit x-range to ensure no negative values appear
x_min = 0
x_max = max(plot_damage['distance_km'].max() * 1.05, 10)  # Add 5% buffer or at least show up to 10 km

# Top row - Probability Distribution of Distance by damage category
# Syrian plot
for i, damage_cat in enumerate([1, 2, 3, 4]):
    cat_data = syrian_data[syrian_data['damage_category'] == damage_cat]['distance_km']
    if len(cat_data) >= 2:  # Need at least 2 points for KDE
        label = damage_labels[damage_cat]
        sns.kdeplot(cat_data, 
                   label=label, 
                   fill=False, 
                   alpha=0.7, 
                   color=professional_colors[i % len(professional_colors)], 
                   linewidth=2, 
                   ax=axes[0,0])

# Turkish plot
for i, damage_cat in enumerate([1, 2, 3, 4]):
    cat_data = turkish_data[turkish_data['damage_category'] == damage_cat]['distance_km']
    if len(cat_data) >= 2:  # Need at least 2 points for KDE
        label = damage_labels[damage_cat]
        sns.kdeplot(cat_data, 
                   label=label, 
                   fill=False, 
                   alpha=0.7, 
                   color=professional_colors[i % len(professional_colors)], 
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
axes[0,0].legend(title="Damage Category", fontsize=9)

axes[0,1].set_title("Distance distribution by damage category - Turkish DPs", fontsize=20)
axes[0,1].set_ylabel("Density", fontsize=12)
axes[0,1].legend(title="Damage Category", fontsize=9)

# Bottom row - KDE plots for displacement date by damage category
if 'displacement_date' in syrian_data.columns:
    # Syrian KDE plot for displacement date
    for i, damage_cat in enumerate([1, 2, 3, 4]):
        cat_data = syrian_data[syrian_data['damage_category'] == damage_cat]['displacement_date']
        if len(cat_data) >= 2:  # Need at least 2 points for KDE
            label = damage_labels[damage_cat]
            sns.kdeplot(cat_data, 
                       label=label, 
                       fill=False, 
                       alpha=0.7, 
                       color=professional_colors[i % len(professional_colors)], 
                       linewidth=2, 
                       ax=axes[1,0])
    
    # Create day-based x-ticks for the displacement period
    min_day = 1  # Start from day 1
    max_day = 25  # Up to day 25
    day_ticks = np.arange(min_day, max_day + 1, 1)  # Every day
    
    axes[1,0].set_xlim(min_day, max_day)
    axes[1,0].set_xticks(day_ticks)
    axes[1,0].set_xticklabels([str(day) for day in day_ticks])
    
    axes[1,0].set_title("Displacement date distribution by damage category - Syrian DPs", fontsize=20)
    axes[1,0].set_xlabel("Displacement Period (Days)", fontsize=12)
    axes[1,0].set_ylabel("Density", fontsize=12)
    axes[1,0].legend(title="Damage Category", fontsize=9)
else:
    axes[1,0].text(0.5, 0.5, "displacement_date column not found in data", ha='center', va='center', fontsize=20)

if 'displacement_date' in turkish_data.columns:
    # Turkish KDE plot for displacement date
    for i, damage_cat in enumerate([1, 2, 3, 4]):
        cat_data = turkish_data[turkish_data['damage_category'] == damage_cat]['displacement_date']
        if len(cat_data) >= 2:  # Need at least 2 points for KDE
            label = damage_labels[damage_cat]
            sns.kdeplot(cat_data, 
                       label=label, 
                       fill=False, 
                       alpha=0.7, 
                       color=professional_colors[i % len(professional_colors)], 
                       linewidth=2, 
                       ax=axes[1,1])
    
    # Use the same day ticks for consistency
    axes[1,1].set_xlim(min_day, max_day)
    axes[1,1].set_xticks(day_ticks)
    axes[1,1].set_xticklabels([str(day) for day in day_ticks])
    
    axes[1,1].set_title("Displacement date distribution by damage category - Turkish DPs", fontsize=20)
    axes[1,1].set_xlabel("Displacement Period (Days)", fontsize=12)
    axes[1,1].set_ylabel("Density", fontsize=12)
    axes[1,1].legend(title="Damage Category", fontsize=9)
else:
    axes[1,1].text(0.5, 0.5, "displacement_date column not found in data", ha='center', va='center', fontsize=20)
%matplotlib inline
plt.tight_layout()
#plt.savefig("/Desktop/damage_category_plot.png", dpi=300)
plt.show()