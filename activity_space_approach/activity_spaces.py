from activity_space_approach.relevance_ratio import calculate_relevance
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
from tqdm import tqdm

# Initialize tqdm for pandas
tqdm.pandas()


def activity_spaces(df, radius, area_type):
    """
    Create activity spaces for either origin or destination.

    Parameters:
    -----------
    df : GeoDataFrame
        DataFrame containing stay locations
    radius : int
        Radius threshold in meters
    area_type : str
        Either "origin" or "destination"

    Returns:
    --------
    dict
        Dictionary of activity spaces with proper keys
    """
    # Validate area_type
    if area_type not in ["origin", "destination"]:
        raise ValueError("area_type must be either 'origin' or 'destination'")

    # Process DBSCAN and convex hull for the specified area_type
    results = process_dbscan_convex_hull(df, type_area=area_type, threshold=radius)

    # Prepare output dictionary with proper processing
    processed_results = {}

    km_label = f"{radius // 1000}km"
    key = f"df_{area_type}_areas_{km_label}"

    # Get the dataframe from the results
    if key in results:
        df_result = results[key]

        # Drop unnecessary columns
        df_result = df_result.drop(columns=["night_relevance_aggregated", "day_relevance_aggregated"],
                                   errors="ignore")

        # Apply relevance calculation
        df_result = calculate_relevance(df_result)

        # Store processed dataframe in dictionary with the same key
        processed_results[key] = df_result
    else:
        print(f"Warning: Key '{key}' not found in results. Available keys: {list(results.keys())}")

    return processed_results


def create_convex_hull(df, geometry_col='geometry', crs='EPSG:32636',
                       distance_threshold=20000, min_samples=1,
                       group_by='customer_id', cluster_col='clusters',
                       aggregate_cols=None, time_col=None):
    """
    Create a flexible convex hull for grouped data points.

    :param df: Input DataFrame or GeoDataFrame
    :param geometry_col: Name of the column containing geometry data
    :param crs: Coordinate reference system
    :param distance_threshold: Maximum distance between two samples for them to be considered as in the same neighborhood
    :param min_samples: Minimum number of samples in a neighborhood for a point to be considered as a core point
    :param group_by: Column name to group the data by (e.g., 'customer_id')
    :param cluster_col: Column name containing cluster information
    :param aggregate_cols: Dictionary of columns to aggregate and their respective aggregation functions
    :param time_col: Column name containing time information (if available)
    :return: GeoDataFrame with convex hulls and aggregated data
    """

    # Ensure we're working with a GeoDataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(df, geometry=geometry_col, crs=crs)
    else:
        gdf = df.copy()

    gdf = gdf.to_crs(crs)

    def get_agg_func(func):
        if callable(func):
            return func
        elif isinstance(func, str):
            return getattr(pd.Series, func)
        else:
            raise ValueError(f"Unsupported aggregation function: {func}")

    def get_representative_point(geom):
        if isinstance(geom, (Polygon, MultiPolygon)):
            return geom.centroid
        elif isinstance(geom, Point):
            return geom
        elif isinstance(geom, LineString):
            return geom.centroid
        else:
            raise ValueError(f"Unsupported geometry type: {type(geom)}")

    # Create a new column for representative points
    gdf['points'] = gdf[geometry_col].apply(get_representative_point)

    def process_group(group):
        coords = np.array(group["points"].apply(lambda point: (point.x, point.y)).tolist())

        distance_matrix = pdist(coords)
        distance_matrix = squareform(distance_matrix)
        clustering = DBSCAN(eps=distance_threshold, min_samples=min_samples, metric='precomputed').fit(distance_matrix)

        results = []

        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue

            cluster_mask = clustering.labels_ == cluster_id
            cluster_points = coords[cluster_mask]
            cluster_geometries = group[geometry_col][cluster_mask]

            # Create convex hull from original geometries
            convex_hull = unary_union(cluster_geometries).convex_hull

            result = {
                'cluster_id': cluster_id,
                'geometry': convex_hull,
                'num_points': len(cluster_points)
            }
            # Aggregate specified columns
            if aggregate_cols:
                for col, agg_func in aggregate_cols.items():
                    if col in group.columns:
                        agg_func = get_agg_func(agg_func)
                        result[f'{col}_aggregated'] = agg_func(group[col][cluster_mask])

            # Handle time information if available
            if time_col and time_col in group.columns:
                result['first_time'] = group[time_col][cluster_mask].min()
                result['last_time'] = group[time_col][cluster_mask].max()

            results.append(result)

        return pd.DataFrame(results)

    # Process each group
    result_df = gdf.groupby(group_by).progress_apply(process_group).reset_index()

    # Flatten the resulting DataFrame
    result_df = result_df.drop(columns='level_1').reset_index(drop=True)

    # Convert back to GeoDataFrame
    result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs=crs)

    return result_gdf


def get_convex_hull(clusters_str, cluster_voronoi):
    """
    Get convex hull for a list of clusters using their voronoi geometries
    """
    # Remove brackets and split by comma to get cluster IDs
    clusters = clusters_str.strip('[]').split(',')
    clusters = [np.int64(c.strip()) for c in clusters if c.strip()]

    # Get corresponding voronoi geometries
    geometries = cluster_voronoi[cluster_voronoi['cluster'].isin(clusters)]['geometry']

    # If no geometries found, return None
    if geometries.empty:
        return None

    # Combine all geometries and get convex hull
    return unary_union(geometries).convex_hull


def process_dbscan_convex_hull(df_input, df_secondary=None, type_area="origin", threshold=5000):
    """
    Process DBSCAN clustering and convex hull for activity spaces.

    Parameters:
    -----------
    df_input : GeoDataFrame
        Primary DataFrame to process
    df_secondary : GeoDataFrame, optional
        Secondary DataFrame (not used when processing a single type)
    type_area : str
        Either "origin" or "destination"
    threshold : int
        Radius threshold in meters (default: 5000)

    Returns:
    --------
    dict
        Dictionary of processed DataFrames
    """
    # Make a copy of the input DataFrame
    df_copy = df_input.copy()

    aggregate_cols = {'stay_id': list, 'duration': 'sum', 'day_duration': 'sum', 'night_duration': 'sum',
                      'night_relevance': 'mean', 'day_relevance': 'mean'}

    prefix = "df_destination_areas_" if type_area == "destination" else "df_origin_areas_"

    # Create a label for the radius
    km_label = f"{threshold // 1000}km"

    result_dfs = {}

    # Create the convex hull for the given threshold
    df = create_convex_hull(
        df_copy[
            ["customer_id", "geometry", "stay_id", 'duration', 'day_duration', 'night_duration', 'night_relevance',
             'day_relevance']],
        geometry_col='geometry',
        group_by='customer_id',
        cluster_col='clusters',
        distance_threshold=threshold,
        aggregate_cols=aggregate_cols
    ).rename(columns={"cluster_id": "origin_id"})

    result_dfs[f"{prefix}{km_label}"] = df

    return result_dfs


def apply_pre_post_overlay(activity_spaces_origin, activity_spaces_destination, radius):
    """
    Apply overlay analysis between origin and destination activity spaces.

    Parameters:
    -----------
    activity_spaces_origin : dict
        Dictionary of origin activity spaces
    activity_spaces_destination : dict
        Dictionary of destination activity spaces
    radius : int
        Radius threshold in meters

    Returns:
    --------
    dict
        Dictionary of overlay results
    """
    results = {}

    km_label = f"{radius // 1000}km"
    origin_key = f"df_origin_areas_{km_label}"
    destination_key = f"df_destination_areas_{km_label}"

    try:
        # Access origin and destination dataframes from their respective dictionaries
        df_origin = activity_spaces_origin[origin_key]
        df_destination = activity_spaces_destination[destination_key]

        # Define column subsets for overlay function
        origin_cols = ["customer_id", "origin_id", "geometry", "num_points", "stay_id_aggregated",
                       "night_relevance"]
        dest_cols = ["customer_id", "origin_id", "geometry", "num_points", "stay_id_aggregated"]

        # Perform overlay operation
        result = pre_post_habitual_areas_overlay(
            df_origin[origin_cols],
            df_destination[dest_cols]
        )

        # Store results using consistent key pattern
        results[f"overlay_{km_label}"] = result

    except KeyError as e:
        print(f"Key error: {e}")
        print(f"Origin keys: {list(activity_spaces_origin.keys())}")
        print(f"Destination keys: {list(activity_spaces_destination.keys())}")
        results[f"overlay_{km_label}"] = pd.DataFrame()

    return results


def overlay_activity_spaces(df_pre_earthquake_stays, df_post_earthquake_stays, radius):
    """
    Create activity spaces for origin and destination, then overlay them.

    Parameters:
    -----------
    df_pre_earthquake_stays : GeoDataFrame
        DataFrame containing pre-earthquake stays (origin)
    df_post_earthquake_stays : GeoDataFrame
        DataFrame containing post-earthquake stays (destination)
    radius : int
        Radius threshold in meters

    Returns:
    --------
    tuple
        (origin_spaces, destination_spaces, overlay_results)
    """
    # Step 1: Generate origin activity spaces
    activity_spaces_origin = activity_spaces(df_pre_earthquake_stays, radius, "origin")

    # Step 2: Generate destination activity spaces
    activity_spaces_destination = activity_spaces(df_post_earthquake_stays, radius, "destination")

    # Step 3: Apply the overlay function using the generated dataframes
    overlay_results = apply_pre_post_overlay(activity_spaces_origin, activity_spaces_destination, radius)

    # Return all three results as expected by the main function
    return activity_spaces_origin, activity_spaces_destination, overlay_results


def pre_post_habitual_areas_overlay(df_origin, df_destination):
    df_origin_copy = df_origin.copy()
    df_origin_copy = df_origin_copy[
        ["customer_id", "origin_id", "geometry", "num_points", "stay_id_aggregated", "night_relevance"]]
    df_destination_copy = df_destination.copy()
    df_destination_copy = df_destination_copy[
        ["customer_id", "origin_id", "geometry", "num_points", "stay_id_aggregated"]]

    df_pairs = df_origin_copy.merge(
        df_destination_copy,
        on='customer_id',
        suffixes=('_pre', '_post')
    )

    # Calculate areas using progress_apply
    df_pairs['intersection_area'] = df_pairs.progress_apply(
        lambda row: row['geometry_pre'].intersection(row['geometry_post']).area
        if row['geometry_pre'].intersects(row['geometry_post']) else 0,
        axis=1
    )

    # Drop rows where intersection_area is 0
    df_pairs = df_pairs[df_pairs['intersection_area'] > 0].reset_index(drop=True)

    df_pairs['origin_area'] = df_pairs.progress_apply(
        lambda row: row['geometry_pre'].area,
        axis=1
    )

    df_pairs['area_ratio'] = df_pairs['intersection_area'] / df_pairs['origin_area']
    df_pairs['habitual_night_relevance'] = df_pairs['night_relevance'] * df_pairs['area_ratio']

    # Select and rename columns for final output
    result_df = df_pairs[[
        'customer_id',
        'origin_id_pre',
        'origin_id_post',
        'intersection_area',
        'origin_area',
        'area_ratio',
        'night_relevance',
        'habitual_night_relevance',
        'stay_id_aggregated_pre',
        'stay_id_aggregated_post'
    ]].rename(columns={
        'origin_id_pre': 'origin_id',
        'origin_id_post': 'destination_id',
        'night_relevance': 'origin_night_relevance',
        'stay_id_aggregated_pre': 'origin_stay_ids',
        'stay_id_aggregated_post': 'destination_stay_ids'
    })

    return result_df