import pandas as pd
from tqdm import tqdm
import dask.dataframe as dd
from dask import delayed
from activity_space_approach.activity_spaces import get_convex_hull
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.")
pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)

tqdm.pandas()

def calculate_pre_post_stay_locations(all_stays=None, filepath=None, earthquake_timestamp=pd.Timestamp('2023-02-06 04:00:00')):
    """
    Calculate pre-earthquake and post-earthquake customer stay locations.
    """
    if all_stays is None and filepath is not None:
        all_stays = pd.read_csv(filepath, index_col=0)
    elif all_stays is None and filepath is None:
        raise ValueError("Either all_stays or filepath must be provided")

    all_stays["end_time"] = pd.to_datetime(all_stays['end_time'])
    all_stays["start_time"] = pd.to_datetime(all_stays['start_time'])


    df_pre_earthquake_stays = all_stays[all_stays['start_time'] < earthquake_timestamp].reset_index(drop=True)
    df_pre_earthquake_stays = gpd.GeoDataFrame(df_pre_earthquake_stays, geometry='geometry', crs="EPSG:32636")
    df_pre_earthquake_stays['end_time'] = df_pre_earthquake_stays['end_time'].clip(upper=earthquake_timestamp)
    df_pre_earthquake_stays = df_pre_earthquake_stays.drop(columns="duration", errors='ignore')
    df_pre_earthquake_stays['duration'] = (df_pre_earthquake_stays['end_time'] - df_pre_earthquake_stays['start_time']).dt.total_seconds() / 3600.0
    df_pre_earthquake_stays = df_pre_earthquake_stays[df_pre_earthquake_stays["geometry"].notna()].reset_index(drop=True)
    df_pre_earthquake_stays = df_pre_earthquake_stays[["customer_id", "clusters", "geometry", "start_time", "end_time", "stay_id", "duration"]]

    df_post_earthquake_stays = all_stays[all_stays['end_time'] > earthquake_timestamp].reset_index(drop=True)
    df_post_earthquake_stays = gpd.GeoDataFrame(df_post_earthquake_stays, geometry='geometry', crs="EPSG:32636")
    df_post_earthquake_stays['start_time'] = df_post_earthquake_stays['start_time'].clip(lower=earthquake_timestamp)
    df_post_earthquake_stays = df_post_earthquake_stays.drop(columns="duration", errors='ignore')
    df_post_earthquake_stays['duration'] = (df_post_earthquake_stays['end_time'] - df_post_earthquake_stays['start_time']).dt.total_seconds() / 3600.0
    df_post_earthquake_stays = df_post_earthquake_stays[df_post_earthquake_stays["geometry"].notna()].reset_index(drop=True)
    df_post_earthquake_stays = df_post_earthquake_stays[["customer_id", "clusters", "geometry", "start_time", "end_time", "stay_id", "duration"]]

    return df_pre_earthquake_stays, df_post_earthquake_stays

def calculate_all_stays(gdf_all, final_df, distance_threshold=2000, duration_threshold=7200, use_dask=False,
                        npartitions=None):
    """
    Calculate all customer stays based on mobility diary.
    """
    all_stays, all_trips = create_customer_mobility_diary(
        gdf_all.reset_index(drop=True).reset_index(drop=True),
        distance_threshold=distance_threshold,
        duration_threshold=duration_threshold,
    )

    all_stays = all_stays[all_stays['duration'] != 0].reset_index()
    all_stays["end_time"] = pd.to_datetime(all_stays['end_time'])
    all_stays["start_time"] = pd.to_datetime(all_stays['start_time'])

    all_stays = all_stays.drop(columns=['index']).merge(
        final_df.drop_duplicates(subset='customer_id', keep='first')[['customer_id', 'segment']],
        on='customer_id'
    )

    all_stays["stay_id"] = all_stays.groupby('customer_id').cumcount()

    all_stays["clusters"] = all_stays["clusters"].astype(str)

    return all_stays, all_trips


def calculate_stay_polygons(all_stays, cluster_voronoi):
    """
    Calculate stay polygons from customer stay data.
    """
    all_stays['convex_hull'] = all_stays['clusters'].progress_apply(lambda x: get_convex_hull(x, cluster_voronoi))

    all_stays_gdf = gpd.GeoDataFrame(all_stays, geometry='convex_hull')

    valid_stays = all_stays_gdf[all_stays_gdf.is_valid].reset_index(drop=True)
    valid_stays["clusters"] = valid_stays["clusters"].astype(str)

    unique_stays = valid_stays[['customer_id', 'clusters']].drop_duplicates(
        subset=['customer_id', 'clusters'],
        keep='first'
    )

    unique_stays_with_geom = unique_stays.merge(
        valid_stays[['customer_id', 'clusters', 'convex_hull']],
        on=['customer_id', 'clusters']
    )

    gdf_stays = gpd.GeoDataFrame(unique_stays_with_geom, geometry='convex_hull')
    gdf_stays = gdf_stays.rename(columns={'convex_hull': 'geometry'})
    gdf_stays = gdf_stays.drop_duplicates(["clusters"])[["clusters", "geometry"]]

    valid_stays['convex_hull'] = valid_stays['convex_hull'].astype(str)
    valid_stays['geometry'] = gpd.GeoSeries.from_wkt(valid_stays['convex_hull'])
    valid_stays = valid_stays.set_geometry('geometry')
    valid_stays = valid_stays.drop(columns=['convex_hull'])
    valid_stays["end_geometry"] = valid_stays["end_geometry"].astype(str)
    valid_stays["start_geometry"] = valid_stays["start_geometry"].astype(str)

    for col in valid_stays.select_dtypes(include=['object']):
        if col != 'geometry':
            valid_stays[col] = valid_stays[col].astype(str)

    return gdf_stays, valid_stays




def create_customer_mobility_diary(df, distance_threshold=20000, duration_threshold=21600):
    """
    Creates a mobility diary for customers by identifying stays and trips based on location data.
    """

    def process_group(group):
        customer_id = group['customer_id'].iloc[0]  
        group = group.sort_values('time')

        stays = []
        trips = []
        current_stay = None
        cumulative_distance = 0
        stay_start_point = None

        for i in range(len(group)):
            row = group.iloc[i]

            if i == 0:
                current_stay = {
                    'customer_id': customer_id,
                    'start_time': row['time'],
                    'end_time': row['time'],
                    'start_geometry': row['geometry'],
                    'end_geometry': row['geometry'],
                    'clusters': [row['cluster']]
                }
                stay_start_point = row['geometry']
            else:
                prev_row = group.iloc[i - 1]
                distance_from_prev = row['geometry'].distance(prev_row['geometry'])
                distance_from_start = row['geometry'].distance(stay_start_point)
                current_duration = (row['time'] - current_stay['start_time']).total_seconds()

                if distance_from_prev <= distance_threshold and distance_from_start <= distance_threshold and current_duration >= duration_threshold:
                    # Continue current stay
                    if row['cluster'] not in current_stay['clusters']:
                        current_stay['clusters'].append(row['cluster'])
                    cumulative_distance += distance_from_prev
                    current_stay['end_time'] = row['time']
                    current_stay['cumulative_distance'] = cumulative_distance
                    current_stay['end_geometry'] = row['geometry']
                else:
                    if current_stay:
                        current_stay['duration'] = (current_stay['end_time'] - current_stay[
                            'start_time']).total_seconds() / 3600
                        stays.append(current_stay)

                    trips.append({
                        'customer_id': customer_id,
                        'start_time': prev_row['time'],
                        'end_time': row['time'],
                        'origin': prev_row['geometry'],
                        'destination': row['geometry'],
                        'distance': max(distance_from_prev, distance_from_start)
                    })

                    current_stay = {
                        'customer_id': customer_id,
                        'start_time': row['time'],
                        'end_time': row['time'],
                        'start_geometry': row['geometry'],
                        'end_geometry': row['geometry'],
                        'clusters': [row['cluster']]
                    }
                    stay_start_point = row['geometry']
                    cumulative_distance = 0

        if current_stay:
            current_stay['duration'] = (current_stay['end_time'] - current_stay['start_time']).total_seconds() / 3600
            current_stay['cumulative_distance'] = cumulative_distance
            stays.append(current_stay)

        stays_df = pd.DataFrame(stays)
        trips_df = pd.DataFrame(trips)

        if not trips_df.empty:
            trips_df['duration'] = (trips_df['end_time'] - trips_df['start_time']).dt.total_seconds() / 3600
            trips_df['speed'] = trips_df['distance'] / (trips_df['duration'] * 1000)  # km/h

        return stays_df, trips_df

    results = []
    for _, group in tqdm(df.groupby('customer_id'), desc="Processing customers"):
        try:
            results.append(process_group(group))
        except Exception as e:
            print(f"Error processing customer: {e}")

    all_stays = []
    all_trips = []
    for stay_df, trip_df in results:
        if not stay_df.empty:
            all_stays.append(stay_df)
        if not trip_df.empty:
            all_trips.append(trip_df)

    if all_stays:
        all_stays = gpd.GeoDataFrame(pd.concat(all_stays, ignore_index=True), geometry='end_geometry', crs=df.crs)
    else:
        all_stays = gpd.GeoDataFrame(
            columns=['customer_id', 'start_time', 'end_time', 'duration', 'cumulative_distance', 'clusters',
                     'geometry'], geometry='geometry', crs=df.crs)

    if all_trips:
        all_trips = gpd.GeoDataFrame(pd.concat(all_trips, ignore_index=True), geometry='destination', crs=df.crs)
    else:
        all_trips = gpd.GeoDataFrame(
            columns=['customer_id', 'start_time', 'end_time', 'duration', 'distance', 'speed', 'geometry'],
            geometry='geometry', crs=df.crs)

    print(f"Total stays identified: {len(all_stays)}")
    print(f"Total trips identified: {len(all_trips)}")

    return all_stays, all_trips

