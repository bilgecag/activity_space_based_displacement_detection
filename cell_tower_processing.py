#from geopy.distance import geodesic
import pandas as pd
import geopandas as gpd
import numpy as np

def read_tower_data(tower_location,voronoi_file=None,crs="EPSG:4326"):
    if crs:
        target_crs=crs
    else:
        target_crs = "EPSG:4326"

    if not voronoi_file:
        voronoi_file="/Volumes/Extreme Pro/Cell_Tower_Locations/turkcell_voronoi/voronoi.shp"

    tower = pd.read_csv(
        tower_location,
        sep="|",
        header=0, encoding='ISO-8859-1')
    tower = tower.drop(['Unnamed: 0', 'Unnamed: 4'], axis=1)
    tower = tower.iloc[1:, :]
    tower = tower.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    tower = tower.rename(columns=lambda x: x.strip())
    tower = tower.rename(columns={'matcher': 'site_id'})
    tower['site_id'] = tower['site_id'].astype(int)

    tower_sorted_district = tower.sort_values(by=['city', 'district'])
    tower_sorted_district = tower_sorted_district.drop_duplicates(subset=['city', 'district'])

    tower_sorted_district['city_district_id'] = tower_sorted_district.groupby(['city', 'district']).ngroup() + 1

    tower_sorted_city = tower.sort_values(by=['city'])
    tower_sorted_city = tower_sorted_city.drop_duplicates(subset=['city'])
    tower_sorted_city['city_id'] = tower_sorted_city.groupby(['city']).ngroup() + 1


    tower_merged = tower.merge(tower_sorted_district[['city', 'district', 'city_district_id']], on=['city', 'district'])
    tower_merged = tower_merged.merge(tower_sorted_city[['city', 'city_id']], on=['city'])
    tower_merged["city_district"] = tower_merged['city'].astype(str) + '_' + tower_merged['district'].astype(str)

    voronoi_gdf=gpd.read_file(voronoi_file, geometry='geometry').rename(columns={"matcher":"site_id"})[["site_id","geometry"]]
    voronoi_gdf.crs = "EPSG:5636"
    voronoi_gdf = voronoi_gdf.to_crs(target_crs)

    tower_merged=tower_merged.merge(voronoi_gdf,on="site_id",how="right")
    tower_merged=gpd.GeoDataFrame(tower_merged,geometry="geometry")
    tower_merged['centroid'] = tower_merged['geometry'].centroid
    tower_merged['lat'] = tower_merged['centroid'].y
    tower_merged['lng'] = tower_merged['centroid'].x
    tower_merged=tower_merged[tower_merged['geometry'].isnull()==False].reset_index(drop=True)
    tower_merged=tower_merged.rename(columns={"geometry": "voronoi_geometry","centroid":"geometry"})
    #print('There are {} cell towers in the datas   et'.format(df.site_id.nunique()))
    return tower_merged

def get_cartesian(lat=None,lon=None):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371 # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x,y, #z


def find_sites_in_buffer(gdf, circle):
    intersection = gpd.overlay(gdf, circle, how='intersection')
    sites = intersection['matcher'].unique().tolist()
    return sites


def find_nearby_towers(df, buffer_distance=30000):
    tower_ids = []
    neighbor_ids = []
    distances = []
    for idx, tower in df.iterrows():
        buffer = tower.geometry.buffer(buffer_distance)
        potential_neighbors = df[df.geometry.intersects(buffer) & (df.index != idx)]
        
        if len(potential_neighbors) > 0:
            distances_to_neighbors = potential_neighbors.geometry.distance(tower.geometry)
            min_distance = distances_to_neighbors.min()
            closest_neighbors = potential_neighbors[distances_to_neighbors == min_distance]
            for neighbor_idx, neighbor in closest_neighbors.iterrows():
                tower_ids.append(tower['site_id'])
                neighbor_ids.append(neighbor['site_id'])
                distances.append(min_distance)
    result = pd.DataFrame({
        'tower_id': tower_ids,
        'nearest_neighbor_id': neighbor_ids,
        'distance_meters': distances
    })
    return result
