from __future__ import division
# import graphlab as gl
# import turicreate as gl
import os
import copy
from array import array
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
from .traj_utils import * #edited by David to correct path "."
#from .sequence_analysis import *
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from sklearn.metrics import silhouette_score, calinski_harabasz_score

class TrajRecord():
    # Input data: user_id, date(int: YYYYMMDD), location(int)
    # Output data:
    #     user_id, migration_date, home, destination, home_start, home_end,
    #     destiantion_start, destination_end,
    #     uncertainty, num_error_day
    def __init__(self, user_traj, raw_traj, index2date, date_num_long):
        """
        Attributes
        ----------
        user_traj : gl.dataframe
            Trajector of users after aggregation
        raw_traj : gl.dataframe
            Raw dataset of users' trajectory
        index2date: dict
            Convert from date index to real date
        date_num_long : gl.SFrame
            Date and num: 'date', 'date_num'
        """
        self.user_traj = user_traj
        self.raw_traj = raw_traj
        self.index2date = index2date
        self.date_num_long = date_num_long

    def plot_trajectory(self, user_id, start_date=None, end_date=None, if_save=True, fig_path='figure'):
        date_min = self.raw_traj[self.raw_traj['user_id'] == user_id]['date'].min()
        date_max = self.raw_traj[self.raw_traj['user_id'] == user_id]['date'].max()

        if start_date:
            assert int(start_date) >= date_min
            start_day = self.date_num_long[self.date_num_long['date'] == int(start_date)]['date_num'].iloc[0]
        else:
            start_day = self.date_num_long[self.date_num_long['date'] == date_min]['date_num'].iloc[0]
            start_date = str(self.date_num_long[self.date_num_long['date_num'] == start_day]['date'].iloc[0])

        if end_date:
            assert int(end_date) <= date_max
            end_day = self.date_num_long[self.date_num_long['date'] == int(end_date)]['date_num'].iloc[0]
        else:
            end_day = self.date_num_long[self.date_num_long['date'] == date_max]['date_num'].iloc[0]
            end_date = str(self.date_num_long[self.date_num_long['date_num'] == end_day]['date'].iloc[0])

        fig, ax, _, _ = plot_traj_common(self.raw_traj, user_id, start_day, end_day, self.date_num_long)

        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        save_path = os.path.join(fig_path, f"{user_id}_{start_date}-{end_date}_trajectory")
        if if_save:
            fig.savefig(save_path, bbox_inches="tight")
    #def plot_trajectory(self, user_id, start_date=None, end_date=None, if_save=True, fig_path='figure'):
    #    """
    #    Plot an individual's trajectory.

    #    Attributes
    #    ----------
    #    user_id : string
    #        user id
    #    start_date : str
    #        start date of the figure in the format of 'YYYYMMDD'
    #    end_date : str
    #        end date of the figure in the format of 'YYYYMMDD'
    #    if_save : boolean
    #        if save the figure
    #    fig_path : str
    #        the path to save figures
    #    """
    #    date_min = self.raw_traj.filter_by(user_id, 'user_id')['date'].min()
    #    date_max = self.raw_traj.filter_by(user_id, 'user_id')['date'].max()
    #    if start_date:
    #        assert int(start_date) >= date_min, "start date must be later than the first day of this user's records, which is " + str(date_min)
    #        start_day = self.date_num_long.filter_by(int(start_date), 'date')['date_num'][0]
    #    else:
    #        start_day = self.date_num_long.filter_by(date_min, 'date')['date_num'][0]
    #        start_date = str(self.date_num_long.filter_by(start_day, 'date_num')['date'][0])
    #    if end_date:
    #        assert int(end_date) <= date_max, "end date must be earlier than the last day of this user's records, which is " + str(date_max)
    #        end_day = self.date_num_long.filter_by(int(end_date), 'date')['date_num'][0]
    #    else:
    #        end_day = self.date_num_long.filter_by(date_max, 'date')['date_num'][0]
    #        end_date = str(self.date_num_long.filter_by(end_day, 'date_num')['date'][0])
    #    fig, ax, _, _ = plot_traj_common(self.raw_traj, user_id, start_day, end_day, self.date_num_long)
    #    if not os.path.isdir(fig_path):
    #        os.makedirs(fig_path)
    #    save_path = os.path.join(fig_path, user_id  + '_' + start_date + '-' + end_date + '_trajectory')
    #    if if_save:
    #        fig.savefig(save_path, bbox_inches="tight")

    def find_migrants(self, num_stayed_days_migrant=90, num_days_missing_gap=7,
                      small_seg_len=30, seg_prop=0.6, min_overlap_part_len=0,
                      max_gap_home_des=30):

        """
        Find migrants step by step

        - step 1: Fill the small missing gaps
                  fill_missing_day('all_record', num_days_missing_gap) -> 'filled_record'

        - step 2: Group consecutive days in the same location together into segments
                  and find segments over certain length.
                  find_segment('filled_record',small_seg_len) -> 'segment_dict'

        - step 3: Find segments in which the user appeared more than prop*len(segment)
                  number of days for that segment.
                  filter_seg_appear_prop(x, 'segment_dict',seg_prop)
                  -> 'segment_over_prop'

        - step 4: Merge neighboring segments together if there are no segments
                  in other districts between the neighboring segments.
                  join_segment_if_no_gap('segment_over_prop') -> 'medium_segment'

        - step 5: Remove overlap parts between any segments who have overlapping
                  and keep segments >= num_stayed_days_migrant days.
                  change_overlap_segment(x, 'medium_segment',
                  min_overlap_part_len, num_stayed_days_migrant) -> 'long_seg'

        - step 6: Find migration: home, destination
                  user_loc_agg['long_seg_num'] = user_loc_agg['long_seg'].apply(lambda x: len(x))
                  user_long_seg = user_loc_agg.filter_by([0,1],'long_seg_num',exclude=True)
                  find_migration_by_segment('long_seg',min_overlap_part_len) -> 'migration_result'

        - step 7: Find migration day
                  find_migration_day_segment(x)

        - step 8: Filter migration segment
                  a) The gap between home segment and destination segment <= 31 days.
                  'seg_diff' <= 31  -> seg_migr_filter
                  b) For short-term migration: Restriction on the length of home segment
                  and destination segment.
                  filter_migration_segment_len('migration_list', hmin, hmax, dmin, dmax)
                  -> 'flag_home_des_len' (0 or 1)

        Attributes
        ----------
        num_stayed_days_migrant : int
            Number of stayed days in home/destination to consider as a migrant
        num_days_missing_gap : int
            Fill the small missing gaps (radius/epsilon)
        small_seg_len : int
            First threshold to filter small segment (minPts)
        seg_prop : float
            Only keep segments that appear >= prop*len(segment).
        min_overlap_part_len : int
            Overlap: 0 days
        max_gap_home_des : int
            Gaps beteen home segment and destination segment
        """

        # Step 1: Fill small missing gaps
        self.user_traj['filled_record'] = self.user_traj['all_record'].progress_apply(
            lambda x: fill_missing_day(x, num_days_missing_gap)
        )

        # Step 2: Find segments over certain length
        self.user_traj['segment_dict'] = self.user_traj['filled_record'].progress_apply(
            lambda x: find_segment(x, small_seg_len)
        )

        # Step 3: Filter segments by appearance proportion
        self.user_traj['segment_over_prop'] = self.user_traj.apply(
            lambda x: filter_seg_appear_prop(x, 'segment_dict', seg_prop),
            axis=1
        )

        # Step 4: Join neighboring segments
        self.user_traj['medium_segment'] = self.user_traj['segment_over_prop'].apply(
            lambda x: join_segment_if_no_gap(x)
        )

        print('Start: Detecting migration')

        # Step 5: Change overlap segments
        self.user_traj['long_seg'] = self.user_traj.apply(
            lambda x: change_overlap_segment(x['medium_segment'], min_overlap_part_len, num_stayed_days_migrant),
            axis=1
        )

        # Calculate segment numbers
        self.user_traj['long_seg_num'] = self.user_traj['long_seg'].apply(len)

        # Filter users with multiple locations
        user_long_seg = self.user_traj[self.user_traj['long_seg_num'] > 1].copy()
        if user_long_seg.empty:
            print('No migrants found.')
            return None

        # Find migration results
        user_long_seg['migration_result'] = user_long_seg['long_seg'].progress_apply(
            lambda x: find_migration_by_segment(x, min_overlap_part_len)
        )

        migration_rows = []
        for _, row in user_long_seg.iterrows():
            for mig in row['migration_result']:
                migration_rows.append({
                    'user_id': row['user_id'],
                    'migration_list': mig,
                    'all_record': row['all_record']
                })

        user_seg_migr = pd.DataFrame(migration_rows)

        # Process migrations with all required data
        user_seg_migr['migration_segment'] = user_seg_migr['migration_list'].apply(create_migration_dict)
        user_seg_migr['home'] = user_seg_migr['migration_list'].apply(lambda x: x[2])
        user_seg_migr['destination'] = user_seg_migr['migration_list'].apply(lambda x: x[3])

        # Ensure all required data is passed to find_migration_day_segment
        user_seg_migr['migration_day_result'] = user_seg_migr.progress_apply(find_migration_day_segment, axis=1)

        if user_seg_migr.empty:
            print('No migrants found.')
            return None

        # Extract migration day results
        user_seg_migr['migration_day'] = user_seg_migr['migration_day_result'].apply(lambda x: int(x[0]))
        user_seg_migr['num_error_day'] = user_seg_migr['migration_day_result'].apply(lambda x: x[1])

        # Add dates
        user_seg_migr['migration_date'] = user_seg_migr['migration_day'].apply(lambda x: self.index2date[x])

        # Extract segment dates
        user_seg_migr['home_start'] = user_seg_migr['migration_list'].apply(lambda x: x[0][0])
        user_seg_migr['destination_end'] = user_seg_migr['migration_list'].apply(lambda x: x[1][1])
        user_seg_migr['home_end'] = user_seg_migr['migration_list'].apply(lambda x: x[0][1])
        user_seg_migr['destination_start'] = user_seg_migr['migration_list'].apply(lambda x: x[1][0])

        # Convert to dates
        for col in ['home_start', 'home_end', 'destination_start', 'destination_end']:
            user_seg_migr[f'{col}_date'] = user_seg_migr[col].apply(lambda x: self.index2date[x])

        # Calculate gaps and filter
        user_seg_migr['seg_diff'] = user_seg_migr['destination_start'] - user_seg_migr['home_end']
        seg_migr_filter = user_seg_migr[user_seg_migr['seg_diff'] <= max_gap_home_des].copy()
        seg_migr_filter['uncertainty'] = seg_migr_filter['seg_diff'] - 1

        print('Done')
        return seg_migr_filter

    def output_segments(self, result_path='result', segment_file='segments.csv', which_step=3):
        """
        Output segments after step 1, 2, or 3
        step 1: Identify contiguous segments
        step 2: Merge segments
        step 3: Remove overlap

        Attributes
        ----------
        segment_file : string
            File name of the outputed segment
        which_step : int
            Output segments in which step
        """
        if which_step == 3:
            user_seg = self.user_traj[self.user_traj['long_seg_num'] > 0]
            user_seg['seg_selected'] = user_seg['long_seg']
        elif which_step == 2:
            user_seg = self.user_traj[self.user_traj['medium_segment_num'] > 0]
            user_seg['seg_selected'] = user_seg['medium_segment']
        elif which_step == 1:
            user_seg = self.user_traj[self.user_traj['segment_over_prop_num'] > 0]
            user_seg['seg_selected'] = user_seg['segment_over_prop']
        user_seg_migr = user_seg.stack(
            'seg_selected',
            new_column_name=['location', 'migration_list']
        )
        user_seg_migr = user_seg_migr.stack(
            'migration_list',
            new_column_name='segment'
        )
        user_seg_migr['segment_start'] = user_seg_migr['segment'].apply(
            lambda x: x[0]
        )
        user_seg_migr['segment_end'] = user_seg_migr['segment'].apply(
            lambda x: x[1]
        )
        user_seg_migr['segment_start_date'] = user_seg_migr.apply(
            lambda x: self.index2date[x['segment_start']]
        )
        user_seg_migr['segment_end_date'] = user_seg_migr.apply(
            lambda x: self.index2date[x['segment_end']]
        )
        user_seg_migr['segment_length'] = (user_seg_migr['segment_end'] -
                                           user_seg_migr['segment_start'])
        user_seg_migr = user_seg_migr.sort(['user_id', 'segment_start_date'], ascending=True)
        if not os.path.isdir(result_path):
            os.makedirs(result_path)
        save_file = os.path.join(result_path, segment_file)
        user_seg_migr.select_columns(
            ['user_id', 'location',
             'segment_start_date', 'segment_end_date', 'segment_length']
        ).export_csv(save_file)

    def plot_segment(self, user_result, if_migration=False,
                     start_date=None, end_date=None,
                     segment_which_step=3,
                     if_save=True, fig_path='figure'):
        """
        Plot migrant daily records in a year and highlight the segments.

        Attributes
        ----------
        user_result : dict
            a user with all the attributes
        if_migration : boolean
            if this record contains a migration event.
            if so, a migration date line will be added
        start_date : str
            start date of the figure in the format of 'YYYYMMDD'
        end_date : str
            end date of the figure in the format of 'YYYYMMDD'
        segment_which_step : int
            1: 'segment_over_prop'
            2: 'medium_segment'
            3: 'long_seg'
        if_save : boolean
            if save the figure
        fig_path : str
            the path to save figures
        """
        segment_which_step_dict = {
            1: 'segment_over_prop',
            2: 'medium_segment',
            3: 'long_seg'
        }
        user_id = user_result['user_id']
        plot_segment = user_result[segment_which_step_dict[segment_which_step]]
        if if_migration:
            migration_day = user_result['migration_day']
            home_start = int(user_result['home_start'])
            des_end = int(user_result['destination_end'])
            start_day = home_start
            end_day = des_end
            # only plot one year's trajectory if the des_end - home_start is longer than one year
            if end_day - start_day > 365 - 1:
                if migration_day <= 180:
                    start_day = home_start
                    end_day = home_start + 365 - 1
                else:
                    start_day = migration_day - 180
                    end_day = migration_day + 184
            start_date = str(self.date_num_long.filter_by(start_day, 'date_num')['date'][0])
            end_date = str(self.date_num_long.filter_by(end_day, 'date_num')['date'][0])
        else:
            date_min = self.raw_traj.filter_by(user_id, 'user_id')['date'].min()
            date_max = self.raw_traj.filter_by(user_id, 'user_id')['date'].max()
            if start_date:
                assert int(start_date) >= date_min, "start date must be later than the first day of this user's records, which is " + str(date_min)
                start_day = self.date_num_long.filter_by(int(start_date), 'date')['date_num'][0]
            else:
                start_day = self.date_num_long.filter_by(date_min, 'date')['date_num'][0]
                start_date = str(self.date_num_long.filter_by(start_day, 'date_num')['date'][0])
            if end_date:
                assert int(end_date) <= date_max, "end date must be earlier than the last day of this user's records, which is " + str(date_max)
                end_day = self.date_num_long.filter_by(int(end_date), 'date')['date_num'][0]
            else:
                end_day = self.date_num_long.filter_by(date_max, 'date')['date_num'][0]
                end_date = str(self.date_num_long.filter_by(end_day, 'date_num')['date'][0])

        duration = end_day - start_day + 1
        fig, ax, location_y_order_loc_appear, appear_loc = plot_traj_common(self.raw_traj, user_id, start_day, end_day, self.date_num_long)
        plot_appear_segment = {k: v for k, v in plot_segment.items() if k in appear_loc}
        for location, value in plot_appear_segment.items():
            y_min = location_y_order_loc_appear[location]
            for segment in value:
                seg_start = segment[0]
                seg_end = segment[1]
                ax.add_patch(
                    patches.Rectangle((seg_start - start_day, y_min),
                                      seg_end - seg_start + 1, 1,
                                      linewidth=4,
                                      edgecolor='red',
                                      facecolor='none')
                )
        if if_migration:
            ax.axvline(migration_day + 0.5 - start_day, color='orange', linewidth=4)
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        save_file = os.path.join(fig_path, user_id + '_' + start_date + '-' + end_date + '_segment')
        if if_save:
            fig.savefig(save_file, bbox_inches="tight")

    def augment_location_data(self, affected_areas, disaster_date, lookback_days, min_observations):
        """
        Augment location data for users in affected areas and identify their home locations.

        Parameters
        ----------
        affected_areas : list
            List of location IDs that were affected by the disaster
        disaster_date : str
            Date of the disaster in 'YYYYMMDD' format
        lookback_days : int
            Number of days before disaster to look for establishing home location
        min_observations : int
            Minimum number of observations required in lookback period to establish home

        Returns
        -------
        tuple
            (augmented_data: pandas.DataFrame, home_locations: dict)
            - augmented_data: DataFrame with augmented trajectory data
            - home_locations: Dictionary mapping user_ids to their home locations
        """
        # Convert disaster date to internal date number
        disaster_day = self.date_num_long[self.date_num_long['date'] == int(disaster_date)]['date_num'].iloc[0]
        lookback_start_day = disaster_day - lookback_days

        # Only filter by time period for home location identification
        lookback_data = self.raw_traj[
            (self.raw_traj['date_num'].between(lookback_start_day, disaster_day))
        ]

        # First find users who were in affected areas
        affected_users = lookback_data[
            lookback_data['location'].isin(affected_areas)
        ]['user_id'].unique()

        # Then get all locations for these users during lookback
        lookback_data = lookback_data[
            lookback_data['user_id'].isin(affected_users)
        ]

        # Calculate location frequencies and find home locations in one operation
        home_locations_df = (lookback_data
                             .groupby(['user_id', 'location'])
                             .agg(visits=('date_num', 'count'))
                             .reset_index()
                             .sort_values(['user_id', 'visits'], ascending=[True, False])
                             .loc[lambda x: x['visits'] >= min_observations]
                             .groupby('user_id')
                             .first()
                             .reset_index()
                             )[['user_id', 'location']]

        # Convert to dictionary for home locations
        home_locations_dict = dict(zip(home_locations_df['user_id'], home_locations_df['location']))

        if home_locations_df.empty:
            return pd.DataFrame(columns=['user_id', 'all_record']), {}

        # Rest of the function remains exactly the same as before
        full_date_range = pd.DataFrame({
            'date_num': range(lookback_start_day, disaster_day + 1)
        })

        user_dates = (home_locations_df[['user_id']]
                      .assign(key=1)
                      .merge(full_date_range.assign(key=1), on='key')
                      .drop('key', axis=1)
                      )

        existing_records = self.raw_traj[
            (self.raw_traj['user_id'].isin(home_locations_df['user_id'])) &
            (self.raw_traj['date_num'].between(lookback_start_day, disaster_day))
            ][['user_id', 'date_num', 'location']]

        complete_records = (user_dates
                            .merge(existing_records, on=['user_id', 'date_num'], how='left')
                            .merge(home_locations_df, on='user_id', suffixes=('_existing', '_home'))
                            )

        complete_records['location'] = complete_records['location_existing'].fillna(
            complete_records['location_home']
        )

        augmented_records = (complete_records
                             .groupby(['user_id', 'location'])['date_num']
                             .agg(list)
                             .groupby(level=0)
                             .agg(lambda x: dict(zip(x.index.get_level_values(1), x.values)))
                             .reset_index()
                             .rename(columns={'date_num': 'all_record'})
                             )

        original_records = self.user_traj[self.user_traj['user_id'].isin(home_locations_df['user_id'])]

        def merge_records(aug_record, orig_record):
            if pd.isna(orig_record):
                return aug_record

            merged = dict(orig_record)
            for loc, dates in aug_record.items():
                if loc not in merged:
                    merged[loc] = []
                merged[loc] = sorted(list(set(merged[loc] + dates)))
            return merged

        final_records = pd.merge(
            augmented_records,
            original_records[['user_id', 'all_record']],
            on='user_id',
            how='left'
        )

        final_records['all_record'] = final_records.progress_apply(
            lambda x: merge_records(x['all_record_x'], x['all_record_y']),
            axis=1
        )

        final_records = final_records[['user_id', 'all_record']]

        return final_records, home_locations_dict


    def plot_cluster_patterns(self, cluster_analysis: pd.DataFrame, figsize=(15, 10)):
        """
        Create visualizations to compare movement patterns between clusters.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        sns.boxplot(data=cluster_analysis, x='cluster', y='num_transitions', ax=axes[0, 0])
        axes[0, 0].set_title('Number of Transitions by Cluster')

        sns.boxplot(data=cluster_analysis, x='cluster', y='avg_segment_duration', ax=axes[0, 1])
        axes[0, 1].set_title('Average Stay Duration by Cluster')

        sns.boxplot(data=cluster_analysis, x='cluster', y='home_time_ratio', ax=axes[1, 0])
        axes[1, 0].set_title('Proportion of Time at Home by Cluster')

        sns.boxplot(data=cluster_analysis, x='cluster', y='distinct_locations', ax=axes[1, 1])
        axes[1, 1].set_title('Number of Distinct Locations by Cluster')

        plt.tight_layout()
        return fig





