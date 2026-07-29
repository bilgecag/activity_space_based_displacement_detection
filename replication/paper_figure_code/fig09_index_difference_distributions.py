def plot_distributions(diff_syrians, diff_turkish):
    sns.set(rc={"figure.dpi": 300})
    # Set the style
    sns.set(style="whitegrid")
    
    # Create a 3x2 figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # Define the columns to plot
    columns = [
     #   'distance',
     #   'displacement_date',
        'weighted_damage_index_diff',
        'weighted_urbanization_index_diff',
        'weighted_city_type_diff',  # This is used for 'Population (difference)'
        'weighted_syria_border_diff'  # This is used for 'Border Index (difference)'
    ]
    
    # Define the pretty names for the columns
    pretty_names = [
     #   'Distance',
     #   'Displacement Date',
        'Damage Index (difference)',
        'Urbanization Index (difference)',
        'Population (difference)',
        'Border Index (difference)'
    ]
    
    # Create clean copies of the data to filter
    syrian_df = diff_syrians.copy()
    turkish_df = diff_turkish.copy()
    
    # Ensure distance and displacement_date are non-negative in both datasets
    syrian_df['distance'] = pd.to_numeric(syrian_df['distance'], errors='coerce')
    turkish_df['distance'] = pd.to_numeric(turkish_df['distance'], errors='coerce')
    
    syrian_df['displacement_date'] = pd.to_numeric(syrian_df['displacement_date'], errors='coerce')
    turkish_df['displacement_date'] = pd.to_numeric(turkish_df['displacement_date'], errors='coerce')
    
    # Plot each column
    for i, (column, name) in enumerate(zip(columns, pretty_names)):
        # For non-negative values only (distance and displacement_date)
        if column in ['distance', 'displacement_date']:
            # Filter out negative and NaN values
            syrian_plot_data = syrian_df[syrian_df[column].notna() & (syrian_df[column] >= 0)]
            turkish_plot_data = turkish_df[turkish_df[column].notna() & (turkish_df[column] >= 0)]
            
            # Set x-axis minimum to 0 for these columns
            x_min = 0
            
            # Plot the filtered data
            sns.kdeplot(
                data=syrian_plot_data, 
                x=column, 
                color='blue',
                fill=True,
                alpha=0.3,
                label='Syrian',
                ax=axes[i],
                cut=0  # This prevents the density from extending below the actual data range
            )
            
            sns.kdeplot(
                data=turkish_plot_data, 
                x=column, 
                color='green',
                fill=True,
                alpha=0.3,
                label='Turkish',
                ax=axes[i],
                cut=0  # This prevents the density from extending below the actual data range
            )
            
            # Explicitly set the x-axis minimum to 0
            x_max = max(
                syrian_plot_data[column].max() if not syrian_plot_data.empty else 0,
                turkish_plot_data[column].max() if not turkish_plot_data.empty else 0,
            ) * 1.05  # Add 5% buffer
            
            axes[i].set_xlim(x_min, x_max)
            
            # Add appropriate tick formatting for distance
            if column == 'distance':
                # If the distance is large, we might want to show in km
                if x_max > 1000:
                    # Convert to km for better readability
                    ticks = np.arange(0, x_max + 1, max(500, x_max//5))  # Reasonable number of ticks
                    axes[i].set_xticks(ticks)
                    axes[i].set_xticklabels([f"{int(x)}" for x in ticks])
                    axes[i].set_xlabel('Distance (m)', fontsize=12)
                else:
                    axes[i].set_xlabel('Distance (m)', fontsize=12)
            
            # Add appropriate tick formatting for displacement date
            if column == 'displacement_date':
                # Assuming displacement_date is in days
                ticks = np.arange(0, x_max + 1, max(1, int(x_max)//10))  # Reasonable number of ticks
                axes[i].set_xticks(ticks)
                axes[i].set_xticklabels([f"{int(x)}" for x in ticks])
                axes[i].set_xlabel('Displacement Date (days)', fontsize=12)
        else:
            # For difference metrics (can be negative)
            sns.kdeplot(
                data=syrian_df, 
                x=column, 
                color='blue',
                fill=True,
                alpha=0.3,
                label='Syrian',
                ax=axes[i]
            )
            
            sns.kdeplot(
                data=turkish_df, 
                x=column, 
                color='green',
                fill=True,
                alpha=0.3,
                label='Turkish',
                ax=axes[i]
            )
            
            # Get the current x limits
            xlim = axes[i].get_xlim()
            
            # Ensure 0 is in the middle third of the plot for difference metrics
            range_size = xlim[1] - xlim[0]
            if xlim[0] > -range_size/3:  # If 0 is too close to the right edge
                axes[i].set_xlim(-range_size/3, xlim[1])
            if xlim[1] < range_size/3:  # If 0 is too close to the left edge
                axes[i].set_xlim(xlim[0], range_size/3)
            
            # Add a vertical line at x=0 for reference
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.9, linewidth=2)
            axes[i].set_xlabel(name, fontsize=12)
        
        # Set the title and labels for all plots
        axes[i].set_title(name, fontsize=14)
        axes[i].set_ylabel('Density', fontsize=12)
        axes[i].legend()
    
    # Adjust the layout to ensure no overlap
    plt.tight_layout()
    
    # Show the plot
    #plt.savefig('/Desktop/turkish_syrian_distributions2.png', dpi=300)
    plt.show()

# Example usage
plot_distributions(diff_syrians, diff_turkish)