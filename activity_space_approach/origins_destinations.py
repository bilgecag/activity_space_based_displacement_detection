
import pandas as pd
from collections import Counter
from shapely.geometry import Point
from tqdm import tqdm  # For progress_apply

def classify_displacement_return(migration_df, earthquake_date='20230206'):
    migration_df['migration_date'] = pd.to_datetime(migration_df['migration_date'], format='%Y%m%d')
    earthquake_date = pd.to_datetime(earthquake_date, format='%Y%m%d')

    def classify_movement(row):
        if row['home'] == 1 and row['destination'] == 0:
            if row['migration_date'] < earthquake_date:
                return 'migration'
            else:
                return 'displacement'
        elif row['home'] == 0 and row['destination'] == 1:
            return 'return_pending_classification'

    migration_df['movement_type'] = migration_df.apply(classify_movement, axis=1)

    user_movement_history = {}
    result_df = migration_df.copy()

    for idx, row in result_df.sort_values(['user_id', 'migration_date']).iterrows():
        user_id = row['user_id']

        if user_id not in user_movement_history:
            user_movement_history[user_id] = []

        if row['movement_type'] != 'return_pending_classification':
            user_movement_history[user_id].append(row['movement_type'])
        else:
            if 'displacement' in user_movement_history[user_id]:
                result_df.loc[idx, 'movement_type'] = 'return_displacement'
            elif 'migration' in user_movement_history[user_id]:
                result_df.loc[idx, 'movement_type'] = 'return_migration'
            else:
                result_df.loc[idx, 'movement_type'] = 'return_migration'

    result_df['migration_date'] = result_df['migration_date'].dt.strftime('%Y%m%d')

    movement_dummies = pd.get_dummies(result_df['movement_type'], prefix='movement_type').astype(int)
    result_df = pd.concat([result_df.drop('movement_type', axis=1), movement_dummies], axis=1)

    return result_df


def match_home_locations_with_displacements(eq, cluster_voronoi, clusters, customer_list):
    """
    Match home locations with displacement locations for customers in the given list.

    Args:
        eq: DataFrame with [time, customer_id, site_id, segment]
        cluster_voronoi: DataFrame with [site_id, voronoi_geometry] and other columns
        customer_list: List of customer IDs to filter by

    Returns:
        DataFrame with [customer_id, origin_geometry, destination_geometry]
    """
    eq_filtered = eq[eq['customer_id'].isin(customer_list)].copy()
    eq_filtered = eq_filtered.merge(pd.read_csv(clusters, index_col=0), on="site_id", how="left")
    eq_filtered["cluster"] = eq_filtered["cluster"].astype(int)
    cluster_voronoi["cluster"] = cluster_voronoi["cluster"].astype(int)
    cutoff_time = pd.Timestamp('2023-02-06 00:04:00')

    before_eq = eq_filtered[eq_filtered['time'] < cutoff_time]
    after_eq = eq_filtered[eq_filtered['time'] >= cutoff_time]

    home_locations = []
    for customer, group in before_eq.groupby('customer_id'):
        if len(group) > 0:
            most_common = Counter(group['cluster']).most_common()
            if most_common:
                home_locations.append({
                    'customer_id': customer,
                    'home_cluster_id': most_common[0][0]
                })

    home_df = pd.DataFrame(home_locations)

    displacement_locations = []
    for customer, group in after_eq.groupby('customer_id'):
        if len(group) > 0 and customer in home_df['customer_id'].values:
            home_site = home_df.loc[home_df['customer_id'] == customer, 'home_cluster_id'].iloc[0]
            counter = Counter(group['cluster'])
            most_common = counter.most_common()

            displacement_site = None
            for site, count in most_common:
                if site != home_site:
                    displacement_site = site
                    break

            if displacement_site is None and most_common:
                displacement_site = most_common[0][0]

            displacement_locations.append({
                'customer_id': customer,
                'displacement_cluster_id': displacement_site
            })

    displacement_df = pd.DataFrame(displacement_locations)

    home_with_geo = home_df.merge(
        cluster_voronoi[['cluster', 'voronoi_geometry']],
        left_on='home_cluster_id',
        right_on='cluster',
        how='left'
    ).drop(columns=['cluster'])

    home_with_geo = home_with_geo.rename(columns={'voronoi_geometry': 'origin_geometry'})

    displacement_with_geo = displacement_df.merge(
        cluster_voronoi[['cluster', 'voronoi_geometry']],
        left_on='displacement_cluster_id',
        right_on='cluster',
        how='left'
    ).drop(columns=['cluster'])

    displacement_with_geo = displacement_with_geo.rename(columns={'voronoi_geometry': 'destination_geometry'})

    result = home_with_geo.merge(
        displacement_with_geo,
        on='customer_id',
        how='inner'
    )

    result = result.rename(columns={
        'home_cluster_id': 'origin_cluster_id',
        'displacement_cluster_id': 'destination_cluster_id'
    })

    result = result[
        ['customer_id', 'origin_cluster_id', 'destination_cluster_id', 'origin_geometry', 'destination_geometry']]

    return result

def match_stays_with_displacements(labeled_migrations, all_stays, match_type='origin'):
    """
    Match migration entries with stay data based on specified match type

    Parameters:
    labeled_migrations: DataFrame with migration/displacement data
    all_stays: DataFrame with stay/location data (all_stays_before or all_stays_after)
    match_type: 'origin' to match home dates, 'destination' to match destination dates

    Returns:
    DataFrame with matched migration and polygon data
    """
    labeled_migrations = labeled_migrations.copy()
    all_stays_copy = all_stays.copy()
    if match_type == 'origin':
        start_date_col = 'home_start_date'
        end_date_col = 'home_end_date'
    elif match_type == 'destination':
        start_date_col = 'destination_start_date'
        end_date_col = 'destination_end_date'
    else:
        raise ValueError("match_type must be 'origin' or 'destination'")

    labeled_migrations[start_date_col] = pd.to_datetime(labeled_migrations[start_date_col], format='%Y%m%d')
    labeled_migrations[end_date_col] = pd.to_datetime(labeled_migrations[end_date_col], format='%Y%m%d')

    labeled_migrations['user_id'] = labeled_migrations['user_id'].astype(int)

    matched_results = []

    for _, migration_row in labeled_migrations.iterrows():
        user_id = migration_row['user_id']
        period_start = migration_row[start_date_col]
        period_end = migration_row[end_date_col]

        # Filter stays for this user
        user_stays = all_stays_copy[all_stays_copy['customer_id'] == user_id]

        if len(user_stays) == 0:
            continue

        # Find stays that overlap with the period
        matching_stays = user_stays[
            (pd.to_datetime(user_stays['start_time']) <= period_end) &
            (pd.to_datetime(user_stays['end_time']) >= period_start)
            ]

        for _, stay_row in matching_stays.iterrows():
            # Create a new row combining migration and stay data
            combined_row = {**migration_row.to_dict(), **stay_row.to_dict()}
            matched_results.append(combined_row)

    if matched_results:
        result_df = pd.DataFrame(matched_results)
        return result_df
    else:
        columns = list(labeled_migrations.columns) + list(all_stays_copy.columns)
        return pd.DataFrame(columns=columns)


def calculate_weighted_midpoints(df, threshold, param='destination'):
    if param == 'destination':
        filtered_df = df[df['habitual_night_relevance'] < threshold].copy()
    else:
        filtered_df = df[df['night_relevance'] > threshold].copy()

    def get_weighted_midpoint(group):
        total_weight = group['night_duration'].sum()

        if total_weight == 0:
            return None

        centroids = group['geometry'].apply(lambda geom: geom.centroid)
        weights = group['night_duration']

        weighted_x = sum(centroids.apply(lambda c: c.x) * weights) / total_weight
        weighted_y = sum(centroids.apply(lambda c: c.y) * weights) / total_weight

        return Point(weighted_x, weighted_y)

    def process_customer(group):
        midpoint = get_weighted_midpoint(group)
        if midpoint is None:
            return None

        return pd.Series({
            f"{param}_midpoint": midpoint
        })

    result = filtered_df.groupby('customer_id').progress_apply(process_customer)
    result = result.dropna()
    result = result.reset_index()

    midpoint_column = f"{param}_midpoint"

    return result[['customer_id', midpoint_column]]
