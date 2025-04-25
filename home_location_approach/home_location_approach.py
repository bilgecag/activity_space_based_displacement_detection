import sys
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plts
from shapely.geometry import Point
import pandas as pd
import warnings
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.")
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import timedelta


def home_location_individual(traj, start_night='22:00', end_night='07:00'):
    """
    Compute the home location of a single individual given their trajectory data.
    """
    night_visits = traj.set_index(pd.DatetimeIndex(traj['time'])).between_time(start_night, end_night)
    
    if len(night_visits) > 0:
        location_counts = night_visits.groupby(['lat_caller', 'lng_caller']).size().sort_values(ascending=False)
        home_lat, home_lng = location_counts.index[0]
    else:
        location_counts = traj.groupby(['lat_caller', 'lng_caller']).size().sort_values(ascending=False)
        home_lat, home_lng = location_counts.index[0]
        
    return (home_lat, home_lng)


def home_location(df, start_night='22:00', end_night='07:00', show_progress=True):
    """
    Compute the home location of each individual in the DataFrame.
    """
    if 'customer_id' not in df.columns:
        home_coords = home_location_individual(df, start_night, end_night)
        return pd.DataFrame([home_coords], columns=['lat_caller', 'lng_caller'])
    
    if show_progress:
        tqdm.pandas()
        homes = df.groupby('customer_id').progress_apply(
            lambda x: home_location_individual(x, start_night, end_night))
    else:
        homes = df.groupby('customer_id').apply(
            lambda x: home_location_individual(x, start_night, end_night))
    
    result = pd.DataFrame(homes.tolist(), index=homes.index)
    result.columns = ['lat', 'lng']
    result = result.reset_index()
    
    return result


def calculate_daily_home_locations(df, start_night='22:00', end_night='07:00', show_progress=True):
    """
    Calculate home locations separately for each day for each user.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    df['date'] = df['time'].dt.date
    
    grouped = df.groupby(['customer_id', 'date'])
    
    results = []
    
    iterator = tqdm(grouped, desc="Processing user-days") if show_progress else grouped
    
    for (customer_id, date), group in iterator:
        try:
            if group['lat_caller'].isna().all() or group['lng_caller'].isna().all():
                continue
                
            home_lat, home_lng = home_location_individual(group, start_night, end_night)
            
            results.append({
                'customer_id': customer_id,
                'date': date,
                'home_lat': home_lat,
                'home_lng': home_lng
            })
        except Exception as e:
            if show_progress:
                pass
    
    return pd.DataFrame(results)

def calculate_daily_home_locations_in_batches(df, batch_size=1000, start_night='22:00', end_night='07:00'):
    """
    Calculate home locations by processing customers in batches to reduce memory usage.
    """
    unique_customers = df['customer_id'].unique()
    total_customers = len(unique_customers)
    
    all_results = []
    
    for i in range(0, total_customers, batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(total_customers+batch_size-1)//batch_size} - Customers {i} to {min(i+batch_size, total_customers)}")
        
        batch_customers = unique_customers[i:i+batch_size]
        
        batch_df = df[df['customer_id'].isin(batch_customers)]
        
        batch_results = calculate_daily_home_locations(batch_df, start_night, end_night)
        
        all_results.append(batch_results)
        
        del batch_df
        
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['customer_id', 'date', 'home_lat', 'home_lng'])
