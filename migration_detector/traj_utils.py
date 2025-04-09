import pandas as pd
import numpy as np
from array import array
# import graphlab as gl
#import turicreate as gl
import matplotlib.pyplot as plt
import seaborn as sns



def fill_missing_day(all_loc_rec, k):
    """
    For any location (L) in any day, see the next k days.
    If in these k days, there is >= 1 day when this person appears in L,
    then fill the gaps with L.
    """
    result_dict = {}
    for loc in all_loc_rec.keys():
        loc_date = all_loc_rec[loc]
        loc_date.sort()
        new_loc_date = list(loc_date)
        if len(loc_date) > 1:
            for i, date in enumerate(loc_date[:-1]):
                close_date = loc_date[i+1]
                date_diff = close_date - date
                if 1 < date_diff <= k:
                    new_loc_date += range(date+1, close_date)
        new_loc_date.sort()
        result_dict[loc] = new_loc_date
    return result_dict


def find_segment(all_fill_rec, k):
    """
    Group consecutive days in the same location together into segments.
    A segment need to have consecutive days >= k days. (default k = 30)
    """
    result_dict = {}
    for loc in all_fill_rec.keys():
        loc_date = all_fill_rec[loc]
        if len(loc_date) >= k:
            loc_date_next = loc_date[1:]
            loc_date_next.append(0)
            diff_next = np.array(loc_date_next) - np.array(loc_date)

            # the index of date that is not consecutive
            # i.e., the next element is not the next date for today
            # but today could be the next date for the former element
            split_idx = np.where(diff_next != 1)[0]
            segment_idx = []

            if len(split_idx) == 1:
                # the last element in diff_next must <0
                # so all the date in loc_date are consective
                segment_idx.append([loc_date[0], loc_date[-1]])
            else:
                # the days before split_idx[0]
                if split_idx[0] >= k-1:
                    segment_idx.append([loc_date[0], loc_date[split_idx[0]]])

                for i, index in enumerate(split_idx[:-1]):
                    next_idx = split_idx[i+1]
                    # Find the start and end index of each period(consecutive days)
                    # For each period, the start and end index are defined by two
                    # nearby elements in split_idx(a list),
                    # and the two elements, which are also indexes in loc_date
                    # (the date list of loc L),
                    # should have a difference >=k,
                    # because we want the period have at least k day
                    # so in loc_date, a period must statisfy:
                    # [split_idx1, start_idx,...(k-2 days), end_idx(split_idx2)]
                    if next_idx - index >= k:
                        seg_start_idx = index + 1
                        seg_end_idx = next_idx
                        seg_start_date = loc_date[seg_start_idx]
                        seg_end_date = loc_date[seg_end_idx]
                        segment_idx.append([seg_start_date, seg_end_date])

            if len(segment_idx) > 0:
                result_dict[loc] = segment_idx
    return result_dict


def join_segment_if_no_gap(old_segment):
    """Modified for Python 3 and pandas compatibility"""
    if not old_segment:
        return {}

    loc_list = list(old_segment.keys())
    new_segment = {}

    if len(loc_list) > 1:
        for location in loc_list:
            loc_record = old_segment[location]
            new_loc_record = []

            if len(loc_record) == 1:
                new_loc_record.append(loc_record[0])
            elif len(loc_record) > 1:
                current_segment = loc_record[0]
                other_loc_record = [value for key, value in old_segment.items()
                                    if key != location]
                other_loc_record = [j for l in other_loc_record for j in l]
                other_loc_exist_days = [range(int(x[0]), int(x[1]) + 1)
                                        for x in other_loc_record]
                other_loc_exist_days = set([j for l in other_loc_exist_days
                                            for j in l])

                for i in range(1, len(loc_record)):
                    next_segment = loc_record[i]
                    gap = [current_segment[1] + 1, next_segment[0] - 1]
                    gap_days_list = range(int(gap[0]), int(gap[1]) + 1)
                    gap_other_loc_interset = set(gap_days_list) & other_loc_exist_days

                    if len(gap_other_loc_interset) == 0:
                        current_segment = [current_segment[0], next_segment[1]]
                    else:
                        new_loc_record.append(current_segment)
                        current_segment = next_segment
                new_loc_record.append(current_segment)

            if new_loc_record:
                new_segment[location] = new_loc_record

    return new_segment


def change_overlap_segment(segments, k, d):

    """Modified to accept segments directly"""
    if not segments:
        return {}

    loc_list = list(segments.keys())
    remove_overlap_date_dict = {}

    for location in loc_list:
        other_loc = list(set(loc_list) - {location})
        current_loc_changed_date = []

        for current_segment in segments[location]:
            other_segment_list = [value for key, value in segments.items()
                                    if key in other_loc]
            other_segment_list = [j for l in other_segment_list for j in l]
            current_segment_date = set(range(int(current_segment[0]),
                                                int(current_segment[1]) + 1))

            for other_segment in other_segment_list:
                other_segment_date = range(int(other_segment[0]),
                                            int(other_segment[1]) + 1)
                seg_intersect = current_segment_date & set(other_segment_date)
                if len(seg_intersect) > k:
                    current_segment_date -= seg_intersect

            current_loc_changed_date.extend(list(current_segment_date))

        if current_loc_changed_date:
            current_loc_changed_date.sort()
            remove_overlap_date_dict[location] = current_loc_changed_date

    return find_segment(remove_overlap_date_dict, d)



def find_migration_by_segment(segments, k):
    """
    Filter migrations where home segment overlaps with destination segment.
    Returns [home_segment, destination_segment, home_id, destination_id]
    """
    if not segments:
        return []

    all_segments_date = []
    segment_loc_sort = []

    # Build flattened lists of segments and locations
    for loc, seg_list in segments.items():
        for segment in seg_list:
            all_segments_date.append(segment)
            segment_loc_sort.append(loc)

    if not all_segments_date:
        return []

    # Sort segments chronologically
    sorted_indices = np.argsort([seg[0] for seg in all_segments_date])
    all_segments_date = [all_segments_date[i] for i in sorted_indices]
    segment_loc_sort = [segment_loc_sort[i] for i in sorted_indices]

    migration_result = []

    for index, current_segment in enumerate(all_segments_date[:-1]):
        current_loc = segment_loc_sort[index]

        # Find segments with different locations
        different_loc_indices = [i for i, loc in enumerate(segment_loc_sort)
                                 if loc != current_loc and i > index]

        if different_loc_indices:
            next_segment_idx = different_loc_indices[0]
            next_segment = all_segments_date[next_segment_idx]

            # Check if gap between segments is acceptable
            if next_segment[0] - current_segment[1] >= -k + 1:
                migration_result.append([
                    current_segment,
                    next_segment,
                    current_loc,
                    segment_loc_sort[next_segment_idx]
                ])

    return migration_result


def filter_seg_appear_prop(row, filter_column, prop):
    """Filter segments based on appearance proportion"""
    all_record = row['all_record']
    to_filter_record = row[filter_column]

    result_dict = {}
    for location, segments in to_filter_record.items():
        loc_daily_record = set(all_record.get(location, []))
        new_loc_record = []

        for segment in segments:
            segment_days = set(range(int(segment[0]), int(segment[1]) + 1))
            appear_len = len(segment_days & loc_daily_record)

            if appear_len >= prop * len(segment_days):
                new_loc_record.append(segment)

        if new_loc_record:
            result_dict[location] = new_loc_record

    return result_dict

def create_migration_dict(x):
    """
    Transform migration information into a dictionary for plotting.
    """
    home_id = x[2]
    des_id = x[3]
    home_segment = x[0]
    des_segment = x[1]
    return {home_id: [home_segment], des_id: [des_segment]}


# Functions to infer migration day
def find_migration_day_segment(x):
    """
    Return migration day and minimum number of error days.
    """
    home = x['home']
    des = x['destination']
    home_seg = x['migration_segment'][home][0]
    des_seg = x['migration_segment'][des][0]
    home_end = home_seg[1]
    des_start = des_seg[0]

    # Find the day of minimun error day
    home_record_bwn = [d for d in x['all_record'][home]
                       if home_end <= d <= des_start]
    des_record_bwn = [d for d in x['all_record'][des]
                      if home_end <= d <= des_start]
    poss_m_day = range(int(home_end), int(des_start)+1)
    wrong_day_before = [[des_day
                         for des_day in des_record_bwn
                         if des_day < current_day]
                         for current_day in poss_m_day]
    num_wrong_day_before = [len(des_day_list)
                            for des_day_list in wrong_day_before]
    wrong_day_after = [[home_day
                        for home_day in home_record_bwn
                        if home_day > current_day]
                        for current_day in poss_m_day]
    num_wrong_day_after = [len(home_day_list)
                           for home_day_list in wrong_day_after]
    num_error_day = np.array(num_wrong_day_before) + np.array(num_wrong_day_after)
    min_error_idx = np.where(num_error_day == num_error_day.min())[0][-1]

    # If there are several days that the num_error_day is the minimun,
    # take the last one.
    migration_day = poss_m_day[min_error_idx]
    min_error_day = min(num_error_day)

    return [int(migration_day), int(min_error_day)]


# Functions for short term migration
def filter_migration_segment_len(x, hmin=0, hmax=float("inf"), dmin=0,
                                 dmax=float("inf")):
    """
    Filter migration segment by home segment length and destination length.
    x: ['migration_list']
    hmin: min_home_segment_len
    hmax: max_home_segment_len
    dmin: min_des_segment_len
    dmax: max_des_segment_len
    """
    home_segment = x[0]
    des_segment = x[1]
    home_len = home_segment[1]-home_segment[0]+1
    des_len = des_segment[1]-des_segment[0]+1
    if (hmin <= home_len <= hmax) and (dmin <= des_len <= dmax):
        return 1
    else:
        return 0


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

    # Generate month labels
    month_labels = [d.strftime('%Y-%m-%d') for d in month_start]
    month_positions = [(d - pd.to_datetime(start_date, format='%Y%m%d')).days for d in month_start]

    # Filter daily records
    daily_record = traj[
        (traj['user_id'] == user_id) &
        (traj['date_num'].between(start_day, end_day))
        ].copy()

    daily_record['date_count'] = 1
    appear_loc = sorted(daily_record['location'].unique())

    # Create heatmap data
    template_df = pd.DataFrame({
        'location': np.repeat(appear_loc, duration),
        'date_num': np.tile(np.arange(start_day, end_day + 1), len(appear_loc))
    })

    heatmap_df = template_df.merge(
        daily_record[['location', 'date_count', 'date_num']],
        on=['date_num', 'location'],
        how='left'
    ).fillna(0)

    heatmap_pivot = pd.pivot_table(
        heatmap_df,
        values='date_count',
        index='location',
        columns='date_num',
        fill_value=0
    )

    # Create plot
    height = len(appear_loc)
    fig_width = max(28.0 / 365 * duration, 10)  # Minimum width of 10
    fig, ax = plt.subplots(dpi=300, figsize=(fig_width, height))
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95)

    # Plot heatmap
    sns.heatmap(heatmap_pivot, cmap='YlOrRd', cbar=False, linewidths=0.5)

    # Add gridlines
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels, rotation=30, ha='right')

    # Set labels and ticks
    plt.yticks(np.arange(len(appear_loc)) + 0.5, appear_loc, rotation=0)

    plt.ylabel('Location', fontsize=12)
    plt.xlabel('Date', fontsize=12)

    location_y_order = dict(zip(appear_loc, range(len(appear_loc))))

    return fig, ax, location_y_order, appear_loc


def encode_locations(full_traj_df, evacuees_df, home_locations):
    """
    Encode locations in trajectory data based on evacuation patterns and home locations.
    """
    # Create a copy without the index column
    encoded_traj = full_traj_df.copy().reset_index(drop=True)

    # Create mapping dictionary for each user
    user_mappings = {}

    for user_id in encoded_traj['user_id'].unique():
        mapping = {}

        # Set home location as 1
        home_loc = home_locations.get(user_id)
        if home_loc is not None:
            mapping[home_loc] = 1

        # Get all destinations for this user from evacuees_df
        user_destinations = evacuees_df[evacuees_df['user_id'] == user_id]['destination'].tolist()

        # Assign values 2 onwards for unique destinations
        next_value = 2
        for dest in user_destinations:
            if dest not in mapping:  # Only assign if not already mapped
                mapping[dest] = next_value
                next_value += 1

        user_mappings[user_id] = mapping

    # Function to map locations
    def map_location(row):
        user_map = user_mappings.get(row['user_id'], {})
        return user_map.get(row['location'], 0)  # Default to 0 if not mapped

    # Apply mapping
    encoded_traj['location'] = encoded_traj.apply(map_location, axis=1)

    return encoded_traj[['user_id', 'date', 'location', 'date_num']]







