# import graphlab as gl
#import turicreate as gl
import pandas as pd
import numpy as np
import os
from .core import TrajRecord
import pandas as pd
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from datetime import timedelta


def to_csv(result, result_path='result', file_name='migration_event.csv'):
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    save_file = os.path.join(result_path, file_name)
    result.select_columns(
            ['user_id', 'home', 'destination', 'migration_date',
         'uncertainty', 'num_error_day',
         'home_start', 'home_end',
         'destination_start', 'destination_end',
         'home_start_date', 'home_end_date',
         'destination_start_date', 'destination_end_date']
    ).export_csv(save_file)


def read_csv(file_path):
    # Read data and process dates
    user_daily_loc_count = pd.read_csv(file_path)
    user_daily_loc_count['user_id'] = user_daily_loc_count['user_id'].astype(str)

    # Create date ranges and mappings
    start_date = pd.to_datetime(str(user_daily_loc_count['date'].min()), format='%Y%m%d')
    end_date = pd.to_datetime(str(user_daily_loc_count['date'].max()), format='%Y%m%d')

    all_date = pd.date_range(start=start_date, end=end_date)
    all_date_new = [int(date.strftime('%Y%m%d')) for date in all_date]
    date2index = dict(zip(all_date_new, range(len(all_date_new))))
    index2date = dict(zip(range(len(all_date_new)), all_date_new))

    # Create extended date range for analysis
    end_date_long = end_date + pd.Timedelta(days=200)
    all_date_long = pd.date_range(start=start_date, end=end_date_long)
    all_date_long_new = [int(date.strftime('%Y%m%d')) for date in all_date_long]
    date_num_long = pd.DataFrame({
        'date': all_date_long_new,
        'date_num': range(len(all_date_long_new))
    })

    # Process migration data
    migration_df = user_daily_loc_count.copy()
    migration_df['date_num'] = migration_df['date'].map(date2index)

    # Group data by user and location
    user_loc_date_agg = migration_df.groupby(['user_id', 'location'])['date_num'].agg(list).reset_index()
    user_loc_agg = user_loc_date_agg.groupby('user_id').apply(
        lambda x: dict(zip(x['location'], x['date_num']))
    ).reset_index(name='all_record')

    return TrajRecord(user_loc_agg, migration_df, index2date, date_num_long)


def plot_traj_common(traj, user_id, start_day, end_day, date_num_long):
    duration = end_day - start_day + 1

    # Get dates for axis
    start_date = str(date_num_long[date_num_long['date_num'] == start_day]['date'].iloc[0])
    end_date = str(date_num_long[date_num_long['date_num'] == end_day]['date'].iloc[0])

    month_start = pd.date_range(
        start=pd.to_datetime(start_date, format='%Y%m%d'),
        end=pd.to_datetime(end_date, format='%Y%m%d'),
        freq='MS'
    )

    # Filter daily records
    daily_record = traj[
        (traj['user_id'] == user_id) &
        (traj['date_num'].between(start_day, end_day))
        ].copy()

    daily_record['date_count'] = 1
    appear_loc = sorted(daily_record['location'].unique())

    # Create template for heatmap
    date_plot = np.tile(np.arange(start_day, end_day + 1), len(appear_loc))
    loc_plot = np.repeat(appear_loc, duration)

    template_df = pd.DataFrame({
        'location': loc_plot,
        'date_num': date_plot
    })

    # Create heatmap data
    heatmap_df = template_df.merge(
        daily_record[['location', 'date_count', 'date_num']],
        on=['date_num', 'location'],
        how='left'
    )
    heatmap_df['date_count'] = heatmap_df['date_count'].fillna(0)
    heatmap_pivot = heatmap_df.pivot("location", "date_num", "date_count")

    # Plot
    height = len(appear_loc)
    fig_width = 28.0 / 365 * duration
    fig, ax = plt.subplots(dpi=300, figsize=(fig_width, height))
    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.97, top=0.95)

    # Plot heatmap
    cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True)
    sns.heatmap(heatmap_pivot, cmap=cmap, cbar=False, linewidths=1)

    # Add grid lines
    for xline in range(duration):
        plt.axvline(xline, color='lightgray', alpha=0.5)
    for yline in range(len(appear_loc) + 1):
        plt.axhline(yline, color='lightgray', alpha=0.5)

    # Format axis
    location_y_order = dict(zip(appear_loc, range(len(appear_loc))))

    month_labels = [d.strftime('%Y-%m-%d') for d in month_start]
    plt.xticks(fontsize=22, rotation=30)
    plt.yticks(fontsize=25, rotation='horizontal')

    plt.tick_params(axis='both', which='both', bottom='on', top='off',
                    labelbottom='on', right='off', left='off',
                    labelleft='on')
    plt.ylabel('Location', fontsize=22)
    plt.xlabel('Date', fontsize=22)

    return fig, ax, location_y_order, appear_loc