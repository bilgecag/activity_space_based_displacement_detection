
import pandas as pd
import numpy as np
#from shapely.ops import unary_union
from tqdm import tqdm
import dask.dataframe as dd
from typing import List, Union
from shapely.ops import unary_union
#from tqdm.auto import tqdm

import geopandas as gpd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.")
pd.set_option('display.max_columns', None)
from shapely.geometry import Point, Polygon,MultiPolygon
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)


def calculate_urban_rural_context_per_cluster(gdf_turkcell, gdf_merged, context_location):
    gdf_turkcell = gdf_turkcell.merge(
        pd.read_csv(context_location),
        how='left',
        left_on='matcher',
        right_on='site_id'
    )
    gdf_turkcell['context'] = gdf_turkcell['context'].fillna('Unknown')
    gdf_turkcell['area_turkcell'] = gdf_turkcell['geometry'].area
    gdf_turkcell = gdf_turkcell.rename(columns={'geometry': 'site_geometry'})
    gdf_turkcell = gdf_turkcell.set_geometry("site_geometry")
    # gdf_turkcell = gdf_turkcell.drop(columns=['site_id'])
    gdf_merged = gdf_merged[['cluster', 'voronoi_geometry']]
    gdf_merged = gdf_merged.set_geometry("voronoi_geometry")
    gdf_overlay = gpd.overlay(gdf_merged, gdf_turkcell, how='intersection')
    gdf_overlay['area_overlay'] = gdf_overlay['geometry'].area
    gdf_overlay['context'] = gdf_overlay['context'].fillna('Unknown')

    context_dummies = pd.get_dummies(gdf_overlay['context'])
    for context in context_dummies.columns:
        gdf_overlay[context] = context_dummies[context] * gdf_overlay['area_overlay']

    cluster_context = gdf_overlay.groupby('cluster')[context_dummies.columns].sum()

    cluster_total_area = gdf_overlay.groupby('cluster')['area_overlay'].sum()

    for context in context_dummies.columns:
        cluster_context[context] = (cluster_context[context] / cluster_total_area) * 100
    #print("Here are the seasonal cells for double check:")
    #gpd.GeoDataFrame(urban_rural_context_per_cluster.merge(cluster_voronoi), geometry='voronoi_geometry').plot(
    #    column='SEASONAL')
    return cluster_context.reset_index()


def add_border_context(
        gdf: gpd.GeoDataFrame,
        buffer_distance: int = 20000,
        countries: List[str] = None
) -> gpd.GeoDataFrame:
    """
    Add border context to a GeoDataFrame by identifying areas that intersect with buffered border zones
    of neighboring countries.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing geometries to analyze
    buffer_distance : int, optional
        Distance in meters to buffer the border (default: 20000)
    countries : List[str], optional
        List of countries to consider. If None, uses default list of Turkey's neighbors

    Returns:
    --------
    gpd.GeoDataFrame
        Original GeoDataFrame with additional boolean columns for each border
    """
    if countries is None:
        countries = ['Turkey', 'Greece', 'Bulgaria', 'Georgia', 'Armenia',
                     'Azerbaijan', 'Iran', 'Iraq', 'Syria']

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    turkey_neighbors = world.loc[world['name'].isin(countries)]

    turkey_neighbors = turkey_neighbors.set_crs('EPSG:4326').to_crs('EPSG:32636')
    if gdf.crs is None or gdf.crs != 'EPSG:32636':
        gdf = gdf.set_crs('EPSG:4326').to_crs('EPSG:32636')

    turkey_boundaries = turkey_neighbors.loc[turkey_neighbors['name'] == 'Turkey', 'geometry'].iloc[0]

    def compute_border(country_geometry):
        try:
            return turkey_boundaries.intersection(country_geometry)
        except Exception as e:
            print(f"Error computing intersection: {e}")
            return None

    def buffer_border(border_multiline):
        if border_multiline is None:
            return None
        try:
            return border_multiline.buffer(buffer_distance)
        except Exception as e:
            print(f"Error creating buffer: {e}")
            return None

    for country in countries:
        if country == 'Turkey':
            continue

        try:
            country_boundaries = turkey_neighbors.loc[turkey_neighbors['name'] == country, 'geometry']
            if len(country_boundaries) == 0:
                print(f"No boundaries found for {country}")
                continue

            country_geom = country_boundaries.iloc[0]

            # Create border buffer
            border = compute_border(country_geom)
            if border is None:
                print(f"Could not compute border for {country}")
                continue

            border = border.simplify(tolerance=0.001, preserve_topology=True)
            border_buffer = buffer_border(border)

            if border_buffer is None:
                print(f"Could not create buffer for {country}")
                continue

            border_buffer = border_buffer.simplify(tolerance=0.001, preserve_topology=True)

            # Create border dummy variable
            gdf[f"{country.lower()}_border"] = gdf.geometry.intersects(border_buffer).astype(int)

            # Print diagnostic information
            intersecting_count = gdf[f"{country.lower()}_border"].sum()
            print(f"{country}: {intersecting_count} intersecting cells")
        # gdf.plot(column='syria_border')

        except Exception as e:
            print(f"Error processing {country}: {e}")
            continue

    return gdf


def add_wealth_context(
        gdf: gpd.GeoDataFrame,
        wealth_csv_path: str,
        lon_col: str = 'longitude',
        lat_col: str = 'latitude',
        wealth_col: str = 'rwi'
) -> gpd.GeoDataFrame:
    """
    Add wealth context to a GeoDataFrame by joining with wealth index data.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame to add wealth context to
    wealth_csv_path : str
        Path to CSV file containing wealth index data
    lon_col : str, optional
        Name of longitude column in wealth CSV (default: 'longitude')
    lat_col : str, optional
        Name of latitude column in wealth CSV (default: 'latitude')
    wealth_col : str, optional
        Name of wealth index column in wealth CSV (default: 'rwi')

    Returns:
    --------
    gpd.GeoDataFrame
        Original GeoDataFrame with added mean wealth index column
    """
    try:
        # Read and prepare wealth data
        df_wealth = pd.read_csv(wealth_csv_path)
        df_wealth['geometry_wealth'] = df_wealth.apply(
            lambda row: Point(row[lon_col], row[lat_col]),
            axis=1
        )
        gdf_wealth = gpd.GeoDataFrame(df_wealth, geometry='geometry_wealth')
        print(gdf_wealth.head())
        # Ensure consistent CRS
        gdf_wealth = gdf_wealth.set_crs('EPSG:4326').to_crs('EPSG:32636')
        if gdf.crs is None or gdf.crs != 'EPSG:32636':
            gdf = gdf.set_crs('EPSG:4326').to_crs('EPSG:32636')

        # Perform spatial join
        gdf = gdf[gdf['voronoi_geometry'] != "nan"].reset_index(drop=True)
        gdf = gdf[gdf['voronoi_geometry'].isnull() == False].reset_index(drop=True)
        # print(gdf.head())
        gdf = gdf.set_geometry("voronoi_geometry")

        wealth_context = gpd.sjoin(gdf_wealth, gdf, how="inner", predicate="within")
        # wealth_context.plot(column='mean_rwi')
        # Calculate mean wealth index
        mean_wealth = wealth_context.groupby(wealth_context.index_right)[wealth_col].mean()
        gdf['mean_rwi'] = mean_wealth

    except Exception as e:
        print(f"Error in add_wealth_context: {e}")
        return gdf

    return gdf[['cluster', 'mean_rwi']]


def process_building_categories(shp_buildings_path, cluster_voronoi):
    """
    """
    gdf_buildings = gpd.read_file(shp_buildings_path)

    categories = {
        'residential': ['apartments', 'residential', 'house', 'detached', 'dormitory', 'bungalow', 'villa', 'block',
                        'chalet', 'container', 'manor', 'apartment_building', 'static_caravan', 'allotment_house',
                        'terrace_lounge', 'gecekondu', 'EV', 'yes;apartments', 'house;yes', 'residential;retail'],
        'commercial': ['office', 'commercial', 'hotel', 'retail', 'supermarket', 'caravanseray', 'restaurant', 'shop',
                       'conference_centre', 'marketplace', 'Dorado Restaurant', 'caravanserai', 'mall',
                       'SAFA TARIM SANAYİ SİTESİ', 'rest', 'bank', 'cafe',
                       'commercialhttps://datatracker.ietf.org/doc/draft-arslan-mimi-outer/',
                       'commercialnufacture', 'government_office', 'food', 'shop'],
        'industrial': ['yes;industrial', 'industrial', 'warehouse', 'factory', 'depot', 'Factory', 'data_center',
                       'manufacture'],
        'religious_sites': ['Cami', 'yes;mosque', 'mosque', 'church', 'synagogue', 'cathedral', 'chapel', 'monastery',
                            'shrine', 'cami', 'medrese', 'madrasa', 'cami̇', 'mosqe', 'religious', 'Kilise', 'minaret',
                            'temple'],
        'healthcare': ['hospital', 'clinic'],
        'education': ['university;yes', 'school', 'university', 'college', 'kindergarten', 'Okul', 'high school',
                      'education center', 'HES'],
        'transport': ['train_station', 'bus_station', 'airport_terminal', 'garage', 'parking', 'bridge',
                      'Suspension_Bridge', 'suspension bridge', 'taxi', 'substation', 'transportation'],
        'public_services': ['fire_station', 'prison', 'courthouse', 'townhall', 'post_office', 'government',
                            'community_center', 'courthouse', 'government_office', 'security_booth', 'military',
                            'pumping_station', 'ground_station', 'lighthouse', 'post_office', 'Community_Services'],
        'sports_recreation': ['sports_hall', 'sports_centre', 'stadium', 'spor center', 'park', 'playground',
                              'greenhouse', 'sera', 'pool', 'observation', 'canopy'],
        'cultural_historic': ['theatre', 'museum', 'library', 'art_centre', 'gallery', 'historic', 'gateway',
                              'monastery', 'garden', 'zoo', 'historic', 'outbuilding', 'Government_Opera_and_Theathe',
                              'amphitheatre', 'historic']
    }

    gdf_buildings = gdf_buildings.set_crs('EPSG:4326')
    gdf_buildings = gdf_buildings.to_crs('EPSG:32636')

    gdf_buildings["mid_point"] = gdf_buildings["geometry"].centroid
    gdf_buildings = gdf_buildings.set_geometry("mid_point")
    cluster_voronoi = cluster_voronoi[["voronoi_geometry", "cluster"]]
    cluster_voronoi = cluster_voronoi.set_geometry("voronoi_geometry")
    df_buildings = gpd.sjoin(gdf_buildings, cluster_voronoi, how='left', op='within')

    type_to_category = {subtype: category
                        for category, subtypes in categories.items()
                        for subtype in subtypes}

    df_buildings['category'] = df_buildings['building'].apply(
        lambda x: type_to_category.get(x, 'other')
    )

    df_buildings_dummies = pd.get_dummies(df_buildings['category'],
                                          prefix='category',
                                          dtype=float)
    df_buildings = pd.concat([df_buildings, df_buildings_dummies], axis=1)

    columns = ['cluster', 'building', 'category_commercial', 'category_cultural_historic',
               'category_education', 'category_healthcare', 'category_industrial',
               'category_other', 'category_public_services', 'category_religious_sites',
               'category_residential', 'category_sports_recreation', 'category_transport']

    df_buildings = df_buildings[columns]
    df_buildings['building'] = 1
    df_buildings = df_buildings[df_buildings["cluster"].isnull() == False]

    for column in columns[1:]:
        building_counts = df_buildings.groupby('cluster')[column].sum()
        building_counts = pd.DataFrame(building_counts).reset_index()
        cluster_voronoi = cluster_voronoi.merge(building_counts,
                                                on='cluster',
                                                how='left')
        cluster_voronoi[column] = cluster_voronoi[column].fillna(0)

    return cluster_voronoi


def process_damage_categories(damage_categories_dict, cluster_voronoi):
    """
    Process multiple damage category files and aggregate counts by cluster.

    Parameters:
    -----------
    damage_categories_dict : dict
        Dictionary with damage category names as keys and file paths as values
    cluster_voronoi : GeoDataFrame
        GeoDataFrame containing voronoi polygons with cluster information

    Returns:
    --------
    GeoDataFrame
        cluster_voronoi with added damage category counts
    """
    # Create a DataFrame to store all damage counts
    damage_counts = pd.DataFrame({'index': cluster_voronoi.index})
    damage_counts.set_index('index', inplace=True)

    # Process each damage category
    for category, file_path in damage_categories_dict.items():
        # Read and process damage category file
        gdf_damaged = gpd.read_file(file_path, geometry='geometry')
        gdf_damaged = gdf_damaged.set_crs('EPSG:4326')
        gdf_damaged = gdf_damaged.to_crs('EPSG:32636')

        # Spatial join with cluster_voronoi
        joined = gpd.sjoin(gdf_damaged, cluster_voronoi, how='inner', op='within')

        # Count points per polygon
        points_per_polygon = joined.groupby('index_right').size()

        # Add to damage_counts DataFrame
        column_name = f'{category}_buildings'
        damage_counts[column_name] = points_per_polygon

    # Fill NaN values with 0
    damage_counts = damage_counts.fillna(0)

    # Merge damage counts with cluster_voronoi
    result = cluster_voronoi.join(damage_counts)
    result.plot(column='collapsed_buildings')
    result = result[['cluster', 'collapsed_buildings', 'heavily_damaged_buildings', 'needs_demolished_buildings',
                     'slightly_damaged_buildings']]

    return result


def calculate_damange_index(df):
    damage_cols = ['collapsed_buildings', 'heavily_damaged_buildings',
                   'needs_demolished_buildings', 'slightly_damaged_buildings']

    damage_weights = {
        'collapsed_buildings': 1.0,
        'heavily_damaged_buildings': 0.7,
        'needs_demolished_buildings': 0.8,
        'slightly_damaged_buildings': 0.3
    }

    df_damage = df[damage_cols].apply(pd.to_numeric, errors='coerce')
    damage_sum = sum(df_damage[col] * damage_weights[col] for col in damage_cols)
    total_damage = df_damage.sum(axis=1)
    df['damage_index'] = np.where(total_damage > 0, damage_sum / total_damage, 0)

    return df

def calculate_urbanization_seasonality_indices(df):
    cols = ["CITY SEASONAL", "CITY SUBURBAN", "DENSE URBAN", "RURAL",
            "SEASONAL", "SUBURBAN", "URBAN", "Unknown"]
    df_copy = df[cols].apply(pd.to_numeric, errors='coerce')

    # Calculate total excluding Unknown
    total_valid = df_copy.drop('Unknown', axis=1).sum(axis=1)

    # Seasonality Index
    seasonal_sum = df_copy['SEASONAL'] + df_copy['CITY SEASONAL']
    seasonality_index = seasonal_sum / total_valid

    # Urbanization Index
    weights = {
        'DENSE URBAN': 1.0,
        'URBAN': 0.8,
        'CITY SEASONAL': 0.8,
        'SUBURBAN': 0.4,
        'CITY SUBURBAN': 0.4,
        'SEASONAL': 0.4,
        'RURAL': 0.0
    }

    weighted_sum = sum(df_copy[col] * weights[col] for col in weights.keys())
    urbanization_index = weighted_sum / total_valid

    df['urbanization_index'] = urbanization_index
    df['seasonality_index'] = seasonality_index

    return df


def calculate_stay_polygons_per_cluster(
   cluster_turkcell, 
   cluster_voronoi, 
   rural_urban_context_location, 
   wealth_file, 
   damage_categories, 
   gdf_stays
):
   """
   Calculate stay polygons per cluster with additional contextual information.
   
   Parameters:
   - cluster_turkcell: GeoDataFrame containing Turkcell cluster data
   - cluster_voronoi: GeoDataFrame containing Voronoi diagrams for clusters
   - rural_urban_context_location: Path to CSV file with rural/urban context information
   - wealth_file: Path to CSV file with wealth index data
   - damage_categories: Dictionary mapping damage types to GeoJSON file paths
   - gdf_stays: GeoDataFrame with stay polygon geometries
   
   Returns:
   - processed_stay_polygons: GeoDataFrame containing processed stay polygons with contextual information
   """
   # Calculate urban/rural context per cluster
   urban_rural_context_per_cluster = calculate_urban_rural_context_per_cluster(
       cluster_turkcell, 
       cluster_voronoi, 
       rural_urban_context_location
   )
   
   # Add border context to clusters
   cluster_border_context = add_border_context(cluster_voronoi.copy())
   
   # Add wealth context to clusters
   cluster_wealth_context = add_wealth_context(cluster_voronoi.copy(), wealth_file)
   
   # Process damage categories and merge with contextual data
   gdf_damaged = process_damage_categories(
       damage_categories, 
       cluster_voronoi.copy().set_geometry("voronoi_geometry")
   )
   
   # Merge all contextual information
   context_cluster = gdf_damaged.merge(
       cluster_border_context[['cluster', 'greece_border', 'bulgaria_border', 'georgia_border', 
                             'armenia_border', 'azerbaijan_border', 'iran_border', 'iraq_border', 'syria_border']],
       on='cluster'
   ).merge(
       cluster_wealth_context, on='cluster'
   ).merge(
       urban_rural_context_per_cluster, on='cluster'
   )
   
   # Define column categories
   dummy_cols = [col for col in context_cluster.columns if col.endswith('_border')]
   count_cols = ['collapsed_buildings', 'heavily_damaged_buildings', 
                'needs_demolished_buildings', 'slightly_damaged_buildings']
   percentage_cols = ['CITY SEASONAL', 'CITY SUBURBAN', 'DENSE URBAN', 
                     'RURAL', 'SEASONAL', 'SUBURBAN', 'URBAN', 'Unknown']
   
   # Process context cluster data
   context_cluster_indexed = context_cluster.set_index('cluster')
   context_cluster_indexed['voronoi_geometry'] = context_cluster_indexed['voronoi_geometry'].apply(
       lambda x: wkt.loads(x)
   )
   
   # Create GeoDataFrame with proper CRS
   context_cluster_indexed = gpd.GeoDataFrame(
       context_cluster_indexed, 
       geometry='voronoi_geometry'
   ).set_crs("EPSG:32636")
   
   # Calculate voronoi area
   context_cluster_indexed["voronoi_area"] = context_cluster_indexed.area
   context_cluster_indexed = context_cluster_indexed.reset_index()
   context_cluster_indexed = context_cluster_indexed.set_geometry("voronoi_geometry")
   
   # Process stay polygons
   gdf_stays = gdf_stays.rename(columns={"geometry": "stay_polygon_geometry"})
   gdf_stays = gpd.GeoDataFrame(gdf_stays, geometry="stay_polygon_geometry")
   gdf_stays["stay_polygon_area"] = gdf_stays.area
   gdf_stays = gdf_stays.set_crs("EPSG:32636", allow_override=True)
   
   # Create minimal context for overlay operation
   context_minimal = context_cluster_indexed[['cluster', 'voronoi_geometry']].copy()
   
   # Intersect stay polygons with voronoi polygons
   intersected_gdf = gpd.overlay(gdf_stays, context_minimal, how='intersection')
   intersected_gdf = intersected_gdf.rename(columns={"geometry": "intersected_geometry"})
   intersected_gdf = gpd.GeoDataFrame(intersected_gdf, geometry="intersected_geometry")
   
   # Calculate intersection area and filter out very small intersections
   intersected_gdf["intersected_area"] = intersected_gdf.area
   intersected_gdf = intersected_gdf[intersected_gdf['intersected_area'] >= 1e-06].reset_index(drop=True)
   
   # Merge with full context data
   intersected_gdf = context_cluster_indexed.merge(
       intersected_gdf, 
       how='right', 
       on="cluster"
   ).drop(columns=['index', 'Unnamed: 0'])
   
   # Process final stay polygons
   processed_stay_polygons = process_stay_polygons(intersected_gdf)
   
   return processed_stay_polygons