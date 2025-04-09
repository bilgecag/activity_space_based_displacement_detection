import pandas as pd
from tqdm.auto import tqdm


def create_dataframes_for_displacement_mining(df_pre_earthquake_stays, df_post_earthquake_stays, df_origin_areas_km,
                                              df_overlay_km, km):
    # Get the appropriate dataframes based on km
    origin_df = df_origin_areas_km.copy()
    overlay_df = df_overlay_km.copy()
    df_pre_earthquake_stays_copy = df_pre_earthquake_stays.copy()
    df_post_earthquake_stays_copy = df_post_earthquake_stays.copy()

    # Origin areas
    o = pd.merge(
        df_pre_earthquake_stays_copy[["customer_id", "stay_id"]],
        origin_df[["customer_id", "stay_id_aggregated", "night_relevance"]].explode('stay_id_aggregated').reset_index(
            drop=True)
        .rename(columns={'stay_id_aggregated': 'stay_id'}),
        on=["customer_id", "stay_id"],
        how="left"
    )
    # Rename the column with the km value
    o = o.rename(columns={'night_relevance': f'habitual_night_relevance_{km}'})

    # Overlay
    t = pd.merge(
        df_post_earthquake_stays_copy[["customer_id", "stay_id"]],
        overlay_df[["customer_id", "habitual_night_relevance", "destination_stay_ids"]].explode('destination_stay_ids')
        .reset_index(drop=True).rename(columns={'destination_stay_ids': 'stay_id'}),
        on=["customer_id", "stay_id"],
        how="left"
    )
    t["habitual_night_relevance"] = t["habitual_night_relevance"].fillna(0)
    t = t.groupby(["customer_id", "stay_id"])["habitual_night_relevance"].mean().reset_index().rename(
        columns={'habitual_night_relevance': f'habitual_night_relevance_{km}'})

    return pd.concat([o, t], axis=0).reset_index(drop=True)


def displacement_sequence_mining(df):
    def is_night_hour(hour):
        return hour >= 22 or (hour >= 0 and hour <= 6)

    def is_day_hour(hour):
        return hour >= 7 and hour <= 21

    def calculate_hours_in_range(row, date_start, date_end, hour_filter_func):
        stay_start = max(row['start_time'], date_start)
        stay_end = min(row['end_time'], date_end)

        if stay_start >= stay_end:
            return 0
        hours_range = pd.date_range(start=stay_start, end=stay_end, freq='H')
        filtered_hours = sum(hour_filter_func(hour.hour) for hour in hours_range)

        return filtered_hours

    def get_relevance_columns(df):
        # Find all columns that start with 'habitual_night_relevance_'
        relevance_cols = [col for col in df.columns if col.startswith('habitual_night_relevance_')]
        # Extract km values from column names
        km_values = [col.split('_')[-1] for col in relevance_cols]
        return relevance_cols, km_values

    def process_customer_date(group, date, relevance_cols, km_values):
        """Process stays for a specific customer and date"""
        date_start = pd.Timestamp(date)
        date_end = date_start + pd.Timedelta(days=1)

        # Filter stays that overlap with this date
        date_stays = group[
            (group['start_time'] < date_end) &
            (group['end_time'] > date_start)
            ].copy()

        if len(date_stays) == 0:
            return None

        # Calculate night hours for each stay
        date_stays['night_hours'] = date_stays.apply(
            lambda row: calculate_hours_in_range(row, date_start, date_end, is_night_hour),
            axis=1
        )

        # Calculate day hours for each stay
        date_stays['day_hours'] = date_stays.apply(
            lambda row: calculate_hours_in_range(row, date_start, date_end, is_day_hour),
            axis=1
        )

        # Filter stays with night hours
        night_stays = date_stays[date_stays['night_hours'] > 0]

        # Use the first customer_id from the date_stays (they should all be the same anyway)
        customer_id = date_stays['customer_id'].iloc[0]

        result = {
            'date': date.date(),
            'customer_id': customer_id
        }

        # If we have nighttime stays, calculate average of habitual relevance values
        if len(night_stays) > 0:
            for col, km in zip(relevance_cols, km_values):
                result[f'location_{km}'] = night_stays[col].mean()
        else:
            # If no nighttime stays, check daytime stays
            day_stays = date_stays[date_stays['day_hours'] > 0]

            if len(day_stays) > 0:
                # Use daytime stays but still look at their habitual_night_relevance values
                for col, km in zip(relevance_cols, km_values):
                    result[f'location_{km}'] = day_stays[col].mean()
            else:
                # No stays at all during this day
                return None

        return result

    # Main function logic starts here
    # Ensure datetime columns
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    # Get all unique dates in the dataset
    all_dates = pd.date_range(
        start=df['start_time'].min().date(),
        end=df['end_time'].max().date()
    )

    # Get the relevance columns and km values
    relevance_cols, km_values = get_relevance_columns(df)

    results = []

    # Group by customer_id
    from tqdm.auto import tqdm

    pbar = tqdm(total=df['customer_id'].nunique(),
                desc="Processing customers",
                position=0,
                leave=True)

    # Process each customer group
    for customer_id, customer_stays in df.groupby('customer_id'):
        date_results = [
            process_customer_date(customer_stays, date, relevance_cols, km_values)
            for date in all_dates
        ]
        results.extend([r for r in date_results if r is not None])
        pbar.update(1)

    pbar.close()

    # Create final DataFrame
    result_df = pd.DataFrame(results)

    # Sort by customer_id and date
    if not result_df.empty:
        result_df = result_df.sort_values(['customer_id', 'date'])

    return result_df


def all_dataframes_for_displacement_sequence_mining(
        all_stays_before,
        all_stays_after,
        activity_spaces_origin,
        activity_spaces_destination,
        overlay_results,
        radius
):
    """
    Prepare dataframes for displacement sequence mining.

    Parameters:
    -----------
    all_stays_before : GeoDataFrame
        Stays before earthquake
    all_stays_after : GeoDataFrame
        Stays after earthquake
    activity_spaces_origin : dict
        Dictionary with origin activity spaces
    activity_spaces_destination : dict
        Dictionary with destination activity spaces
    overlay_results : dict
        Dictionary with overlay results
    radius : int
        Radius used for activity spaces in meters

    Returns:
    --------
    dict
        Dictionary with dataframes ready for displacement mining
    """
    to_be_mined = {}

    # Add the basic stays data
    to_be_mined["stays_before"] = all_stays_before
    to_be_mined["stays_after"] = all_stays_after

    # Add origin, destination, and overlay data
    km_label = f"{radius // 1000}km"
    origin_key = f"df_origin_areas_{km_label}"
    destination_key = f"df_destination_areas_{km_label}"
    overlay_key = f"overlay_{km_label}"

    # Add origin spaces
    if origin_key in activity_spaces_origin:
        to_be_mined[f"origin_{km_label}"] = activity_spaces_origin[origin_key]
    else:
        print(f"Warning: {origin_key} not found in origin spaces")

    # Add destination spaces
    if destination_key in activity_spaces_destination:
        to_be_mined[f"destination_{km_label}"] = activity_spaces_destination[destination_key]
    else:
        print(f"Warning: {destination_key} not found in destination spaces")

    # Add overlay results
    if overlay_key in overlay_results:
        to_be_mined[f"overlay_{km_label}"] = overlay_results[overlay_key]
    else:
        print(f"Warning: {overlay_key} not found in overlay results")

    return to_be_mined


def detect_displacements(days_away, max_gap, displacement_results=None, output_dir=None, base_path=None):
    """
    Detect migration patterns from mobility data using specified parameters.

    Parameters:
    -----------
    days_away : int
        Number of consecutive days a person must stay away to be considered a migrant

    max_gap : int
        Maximum number of days with missing data that is allowed in a migration pattern

    displacement_results : DataFrame, optional
        Displacement results dataframe to analyze

    output_dir : str, optional
        Directory to save results

    base_path : str, optional
        Base directory containing the displacement CSV files
    """
    if base_path is None and displacement_results is None:
        base_path = "/Users/bilgecagaydogdu/Downloads/mobile_data/trips_and_stays/outgoing_2km_2hours/displacements"

    # Display parameter information to the user
    print(f"\nRunning migration detection with parameters:")
    print(f"  - days_away (i): {days_away} days")
    print(f"  - max_gap (j): {max_gap} days")
    print(f"  - Data source: location_2 (5km radius)")

    # Fixed parameters for this simplified version
    cutoff_value = 5
    location_type = 'location_2'

    if displacement_results is not None:
        # Use provided displacement results
        trajectory_data = displacement_results
        print("\nUsing provided displacement results.")
    else:
        # Construct the file path for the displacement data
        file_name = f"displacement_by_location_cut_of_{cutoff_value}_habitual_night_ratio_including_day_time_binary_cut-off_{location_type}.csv"
        file_path = os.path.join(base_path, file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: Could not find displacement data at {file_path}")
            print("Please check the path and make sure the data files are available.")
            return

        # Load the trajectory data
        print(f"Reading displacement data from {file_path}")
        trajectory_data = pd.read_csv(file_path)

    # Additional migration detection parameters
    small_seg_len = 2  # Minimum length of a segment to be considered
    seg_prop = 0.5  # Proportion threshold for segment validation
    min_overlap_part_len = 0  # Minimum overlap length
    max_gap_home_des = 7  # Maximum gap between home and destination

    print("\nDetecting migration patterns...")
    print("This may take some time depending on the size of your data...")

    # Create dummy object for trajectory_data if needed
    if not hasattr(trajectory_data, 'find_migrants'):
        # This is a stub for compatibility
        # In the real implementation, this would be imported from migration_detector
        class MigrationDetector:
            def __init__(self, data):
                self.data = data

            def find_migrants(self, **kwargs):
                # This is a placeholder implementation
                # In a real scenario, this would be implemented properly
                print("Using placeholder migration detector")
                # Just return the data with additional migration flags
                result = self.data.copy()
                result['is_migrant'] = True
                result['start_date'] = pd.to_datetime(result['date'].min())
                result