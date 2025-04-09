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
    
    Parameters
    ----------
    traj : DataFrame
        The trajectory of the individual with time, lat_caller, lng_caller columns
    
    start_night : str, optional
        The starting time of the night (format HH:MM). Default is '22:00'.
        
    end_night : str, optional
        The ending time for the night (format HH:MM). Default is '07:00'.
    
    Returns
    -------
    tuple
        The (latitude, longitude) coordinates of the individual's home location 
    """
    # Filter for nighttime visits
    night_visits = traj.set_index(pd.DatetimeIndex(traj['time'])).between_time(start_night, end_night)
    
    if len(night_visits) > 0:
        # Get most frequent location during nighttime
        location_counts = night_visits.groupby(['lat_caller', 'lng_caller']).size().sort_values(ascending=False)
        home_lat, home_lng = location_counts.index[0]
    else:
        # If no nighttime data, use most frequent location overall
        location_counts = traj.groupby(['lat_caller', 'lng_caller']).size().sort_values(ascending=False)
        home_lat, home_lng = location_counts.index[0]
        
    return (home_lat, home_lng)


def home_location(df, start_night='22:00', end_night='07:00', show_progress=True):
    """
    Compute the home location of each individual in the DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with columns 'customer_id', 'time', 'lat_caller', 'lng_caller'
    
    start_night : str, optional
        The starting time of the night (format HH:MM). Default is '22:00'.
        
    end_night : str, optional
        The ending time for the night (format HH:MM). Default is '07:00'.
    
    show_progress : boolean, optional
        If True, show a progress bar. Default is True.
    
    Returns
    -------
    pandas DataFrame
        DataFrame with columns 'customer_id', 'lat', 'lng' containing home locations
    """
    # If there's no uid column (single user data)
    if 'customer_id' not in df.columns:
        home_coords = home_location_individual(df, start_night, end_night)
        return pd.DataFrame([home_coords], columns=['lat_caller', 'lng_caller'])
    
    # Process multiple users
    if show_progress:
        tqdm.pandas()
        homes = df.groupby('customer_id').progress_apply(
            lambda x: home_location_individual(x, start_night, end_night))
    else:
        homes = df.groupby('customer_id').apply(
            lambda x: home_location_individual(x, start_night, end_night))
    
    # Convert results to DataFrame
    result = pd.DataFrame(homes.tolist(), index=homes.index)
    result.columns = ['lat', 'lng']
    result = result.reset_index()
    
    return result


def calculate_daily_home_locations(df, start_night='22:00', end_night='07:00', show_progress=True):
    """
    Calculate home locations separately for each day for each user.
    This function uses the home_location_individual function for each day-user combination.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with columns 'customer_id', 'time', 'lat_caller', 'lng_caller'
    
    start_night : str, optional
        The starting time of the night (format HH:MM). Default is '22:00'.
        
    end_night : str, optional
        The ending time for the night (format HH:MM). Default is '07:00'.
    
    show_progress : boolean, optional
        If True, show a progress bar. Default is True.
        
    Returns
    -------
    pandas DataFrame
        DataFrame with columns 'customer_id', 'date', 'home_lat', 'home_lng'
    """
    # Ensure time column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    # Extract date from time
    df['date'] = df['time'].dt.date
    
    # Group by customer_id and date
    grouped = df.groupby(['customer_id', 'date'])
    
    results = []
    
    # Process each group
    iterator = tqdm(grouped, desc="Processing user-days") if show_progress else grouped
    
    for (customer_id, date), group in iterator:
        try:
            # Skip groups with no location data
            if group['lat_caller'].isna().all() or group['lng_caller'].isna().all():
                continue
                
            # Use the home_location_individual function to find home for this day
            home_lat, home_lng = home_location_individual(group, start_night, end_night)
            
            results.append({
                'customer_id': customer_id,
                'date': date,
                'home_lat': home_lat,
                'home_lng': home_lng
            })
        except Exception as e:
            if show_progress:
                pass#print(f"Error processing customer {customer_id} on {date}: {e}")
    
    # Create DataFrame from results
    return pd.DataFrame(results)

def calculate_daily_home_locations_in_batches(df, batch_size=1000, start_night='22:00', end_night='07:00'):
    """
    Calculate home locations by processing customers in batches to reduce memory usage.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with columns 'customer_id', 'time', 'lat_caller', 'lng_caller'
    
    batch_size : int, optional
        Number of customers to process in each batch. Default is 1000.
    
    start_night : str, optional
        The starting time of the night (format HH:MM). Default is '22:00'.
        
    end_night : str, optional
        The ending time for the night (format HH:MM). Default is '07:00'.
        
    Returns
    -------
    pandas DataFrame
        DataFrame with columns 'customer_id', 'date', 'home_lat', 'home_lng'
    """
    # Get unique customer IDs
    unique_customers = df['customer_id'].unique()
    total_customers = len(unique_customers)
    
    # Initialize results list
    all_results = []
    
    # Process in batches
    for i in range(0, total_customers, batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(total_customers+batch_size-1)//batch_size} - Customers {i} to {min(i+batch_size, total_customers)}")
        
        # Get customer IDs for this batch
        batch_customers = unique_customers[i:i+batch_size]
        
        # Filter data for this batch
        batch_df = df[df['customer_id'].isin(batch_customers)]
        
        # Calculate home locations for this batch
        batch_results = calculate_daily_home_locations(batch_df, start_night, end_night)
        
        # Append to results
        all_results.append(batch_results)
        
        # Clear memory
        del batch_df
        
    # Combine all results
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['customer_id', 'date', 'home_lat', 'home_lng'])
