def plot_migration_side_by_side(df_adjusted_tmb, df_adjusted_as, figsize=(18, 8)):
    from matplotlib import pyplot as plt
    import copy
    
    fig = plt.figure(figsize=figsize, dpi=300)
    
    # Create first subplot
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Modified plot_migration_correlation for subplot 1
    affected_cities=['ADANA', 'KILIS', 'GAZIANTEP', 'SANLIURFA', 'HATAY', 'DIYARBAKIR', 'MALATYA', 'ADIYAMAN', 'OSMANIYE', 'KAHRAMANMARAS']
    
    # Unaffected cities - inflow
    plot_df_tmb1 = df_adjusted_tmb.copy()
    plot_df_as1 = df_adjusted_as.copy()
    
    # Filter for unaffected cities
    plot_df_tmb1 = plot_df_tmb1[plot_df_tmb1['city'].isin(affected_cities)==False].reset_index(drop=True)
    plot_df_as1 = plot_df_as1[plot_df_as1['city'].isin(affected_cities)==False].reset_index(drop=True)
    
    x_column1 = 'turkish_adjusted_inflow_rate'
    y_column1 = 'other_inflow'
    
    plt.sca(ax1)
    sns.scatterplot(data=plot_df_tmb1, 
                x=x_column1, 
                y=y_column1,
                color='red',
                label='TMB (city-level)',
                s=100)  # Increased point size here
    
    sns.scatterplot(data=plot_df_as1, 
                x=x_column1, 
                y=y_column1,
                color='blue',
                label='ASA',
                s=100)  # Increased point size here
    
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    lims = [max(xlim[0], ylim[0]), min(xlim[1], ylim[1])]
    plt.plot(lims, lims, 'k--', alpha=0.5, label='45° line')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Calculate Pearson correlations and p-values
    from scipy.stats import pearsonr
    corr_tmb1, p_tmb1 = pearsonr(plot_df_tmb1[x_column1], plot_df_tmb1[y_column1])
    corr_as1, p_as1 = pearsonr(plot_df_as1[x_column1], plot_df_as1[y_column1])
    
    # Add significance markers
    sig_tmb1 = ''
    if p_tmb1 < 0.001: sig_tmb1 = '***'
    elif p_tmb1 < 0.01: sig_tmb1 = '**'
    elif p_tmb1 < 0.05: sig_tmb1 = '*'
        
    sig_as1 = ''
    if p_as1 < 0.001: sig_as1 = '***'
    elif p_as1 < 0.01: sig_as1 = '**'
    elif p_as1 < 0.05: sig_as1 = '*'
    
    # Add correlation text
    plt.text(0.05, 0.85, f'Pearson Correlations:\nTMB: {corr_tmb1:.2f}{sig_tmb1}\nASA: {corr_as1:.2f}{sig_as1}', 
             transform=plt.gca().transAxes, 
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', linewidth=1, pad=5))
    
    texts1 = []
    
    for idx, row in plot_df_tmb1.iterrows():
        if ((row[y_column1] > 10000) or (row[x_column1] > 10000)):
            # Special case for Istanbul in TMB approach - position it lower left
            if row.city == 'ISTANBUL':
                texts1.append(plt.text(row[x_column1] - 2000, 
                                    row[y_column1] - 2000, 
                                    row.city,
                                    fontsize=12))
            else:
                texts1.append(plt.text(row[x_column1], 
                                    row[y_column1], 
                                    row.city,
                                    fontsize=12))
    
    for idx, row in plot_df_as1.iterrows():
        if ((row[y_column1] > 10000) or (row[x_column1] > 10000)):
            texts1.append(plt.text(row[x_column1], 
                                row[y_column1], 
                                row.city,
                                fontsize=12))
    
    from adjustText import adjust_text
    adjust_text(texts1, 
               expand_points=(1.5, 1.5),
               arrowprops=dict(arrowstyle="-", color='none'))
    
    plt.title('DP inflow to non-affected cities', fontsize=18, pad=15)
    plt.xlabel("CDR estimates (TMB vs ASA)", fontsize=14)
    plt.ylabel("TURKSTAT estimates", fontsize=14)
    plt.legend(loc='lower right')
    
    # Create second subplot
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Affected cities - outflow
    plot_df_tmb2 = df_adjusted_tmb.copy()
    plot_df_as2 = df_adjusted_as.copy()
    
    # Filter for affected cities
    plot_df_tmb2 = plot_df_tmb2[plot_df_tmb2['city'].isin(affected_cities)==True].reset_index(drop=True)
    plot_df_as2 = plot_df_as2[plot_df_as2['city'].isin(affected_cities)==True].reset_index(drop=True)
    
    x_column2 = 'turkish_adjusted_outflow_rate'
    y_column2 = 'other_outflow'
    
    plt.sca(ax2)
    sns.scatterplot(data=plot_df_tmb2, 
                x=x_column2, 
                y=y_column2,
                color='red',
                label='TMB (city-level)',
                s=100)  # Increased point size here
    
    sns.scatterplot(data=plot_df_as2, 
                x=x_column2, 
                y=y_column2,
                color='blue',
                label='ASA',
                s=100)  # Increased point size here
    
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    lims = [max(xlim[0], ylim[0]), min(xlim[1], ylim[1])]
    plt.plot(lims, lims, 'k--', alpha=0.5, label='45° line')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Calculate Pearson correlations and p-values
    corr_tmb2, p_tmb2 = pearsonr(plot_df_tmb2[x_column2], plot_df_tmb2[y_column2])
    corr_as2, p_as2 = pearsonr(plot_df_as2[x_column2], plot_df_as2[y_column2])
    
    # Add significance markers
    sig_tmb2 = ''
    if p_tmb2 < 0.001: sig_tmb2 = '***'
    elif p_tmb2 < 0.01: sig_tmb2 = '**'
    elif p_tmb2 < 0.05: sig_tmb2 = '*'
        
    sig_as2 = ''
    if p_as2 < 0.001: sig_as2 = '***'
    elif p_as2 < 0.01: sig_as2 = '**'
    elif p_as2 < 0.05: sig_as2 = '*'
    
    # Add correlation text
    plt.text(0.05, 0.85, f'Pearson Correlations:\nTMB: {corr_tmb2:.2f}{sig_tmb2}\nASA: {corr_as2:.2f}{sig_as2}', 
             transform=plt.gca().transAxes, 
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', linewidth=1, pad=5))
    
    texts2 = []
    
    for idx, row in plot_df_tmb2.iterrows():
        if ((row[y_column2] > 0) or (row[x_column2] > 0)):
            texts2.append(plt.text(row[x_column2], 
                                row[y_column2], 
                                row.city,
                                fontsize=12))  # Increased fontsize from 8 to 12
    
    for idx, row in plot_df_as2.iterrows():
        if ((row[y_column2] > 0) or (row[x_column2] > 0)):
            texts2.append(plt.text(row[x_column2], 
                                row[y_column2], 
                                row.city,
                                fontsize=12))  # Increased fontsize from 8 to 12
    
    adjust_text(texts2, 
               expand_points=(1.5, 1.5),
               arrowprops=dict(arrowstyle="-", color='none'))
    
    plt.title('DP outflow from earthquake region', fontsize=18, pad=15)
    plt.xlabel("CDR estimates (TMB vs ASA)", fontsize=14)
    plt.ylabel("TURKSTAT estimates", fontsize=14)
    #plt.ylabel(f'{y_column2.replace("_", " ").title()}', fontsize=14)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    
    return fig
%matplotlib inline
plot_migration_side_by_side(df_adjusted_tmb, df_adjusted_as, figsize=(18, 8))
#plt.savefig("/Desktop/validation.png",dpi=100)
plt.show()