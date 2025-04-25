import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from cell_tower_processing import read_tower_data
def read_fine_grained(dir_finegrained, datatype):
    if datatype=='XDR':
        df = pd.read_csv(dir_finegrained,
                    sep="|", skiprows=2,
                    header=None, encoding='ISO-8859-1')
        df = df.drop([0, 5], axis=1)
        df.columns = ['time', 'customer_id', 'segment', 'site_id']
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H')
    elif datatype == 'CDR':
        df = pd.read_csv(dir_finegrained)
        df = df.rename(columns={"CUSTOMER_ID": "customer_id", "DAY": "date", "HOUR": "hour",
                                "CALLE_SITE_ID": "site_id_callee", \
                                "CALER_SITE_ID": "site_id_caller", "CALLEE_SEGMENT": "segment_callee",\
                                "CALER_SEGMENT": "segment_caller",\
                                "CALEE_SEGMENT":"segment_callee",\
                                "CALLER_SEGMENT": "segment_caller"})
        #df['time'] = pd.to_datetime(df['date'], infer_datetime_format=True) + pd.to_timedelta(df['hour'], unit='h')
        df['time'] = pd.to_datetime(df['date'], format="%d/%m/%Y") + pd.to_timedelta(df['hour'], unit='h')
    return df



def filter_customers(cust_df, signal_count, unique_days_threshold, night_signal_threshold, customer_list=None, include_customers=True):
    filtered_df = cust_df[
        (cust_df['signal_count'] >= signal_count) &
        (cust_df['unique_days_count'] >= unique_days_threshold) &
        (cust_df['signal_at_night'] > night_signal_threshold)
    ]

    if customer_list is not None:
        if include_customers:
            filtered_df = filtered_df[filtered_df['customer_id'].isin(customer_list)]
        else:
            filtered_df = filtered_df[~filtered_df['customer_id'].isin(customer_list)]

    return filtered_df


def customer_signals_analysis(df,site_id):

    df['site_count'] = df.groupby('customer_id')[site_id].transform('nunique')
    df['day_count'] = df.groupby('customer_id')['day'].transform('nunique')

    customers_analysis = df.groupby('customer_id').agg({
        'time': 'count',
        #'23_dummy': 'sum',
        #'is_weekend': 'sum',
        'night_dummy': 'sum',
        'day_count': 'first',
        'site_count': 'first'
    }).rename(columns={
        'time': 'signal_count',
        #'23_dummy': 'signal_at_23',
        #'is_weekend': 'signal_on_weekend',
        'night_dummy': 'signal_at_night',
        'day_count': 'unique_days_count',
        'site_count': 'unique_sites_count'
    })

    return customers_analysis.reset_index()

def calculate_date_values(df):
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['week'] = df['time'].dt.isocalendar().week
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['dayofweek'] = df['time'].dt.dayofweek
    df['night_dummy'] = df['hour'].apply(lambda x: 1 if x >= 19 or x < 7 else 0)
    return df


def process_cdr_data(file_path_outgoing, file_ids=['7','7_2','7_3','7_4','7_5']):
    final_df = pd.DataFrame()
    
    for i in file_ids:
        df1 = read_fine_grained(file_path_outgoing.format(i), 'CDR')
        
        df1["incoming"] = 0
        
        df1 = calculate_date_values(df1)
        
        cust_df1 = customer_signals_analysis(df1, 'site_id_caller')
        
        filtered_cust1 = filter_customers(
            cust_df1, 
            signal_count=3, 
            unique_days_threshold=0, 
            night_signal_threshold=0,
            customer_list=None, 
            include_customers=True
        )['customer_id'].unique().tolist()
        
        df1 = df1[df1['customer_id'].isin(filtered_cust1)].reset_index(drop=True)
        
        final_df = pd.concat([final_df, df1], ignore_index=True)
    
    final_df = final_df[final_df['incoming'] == 0]
    
    return final_df

def filter_affected_customers(
    clusters_file, 
    earthquake_cities, 
    tower_location, 
    voronoi_file, 
    final_df, 
    clusters_shapefile
):
    
    df_clusters = pd.read_csv(clusters_file, index_col=0)
    
    
    df_tower = read_tower_data(
        tower_location,
        voronoi_file=voronoi_file,
        crs="EPSG:4326"
    ).drop_duplicates(subset='site_id').reset_index(drop=True)
    
    df_tower = df_tower[['city_district_id', 'city_id', 'site_id', 'city_district']].merge(df_clusters)
    
    
    last_observed_locations = pd.DataFrame.from_dict(
        get_last_locations(
            final_df[
                (final_df['time'] < pd.Timestamp('2023-02-06 04:00:00')) & 
                (final_df['time'] > pd.Timestamp('2023-01-06 04:00:00'))
            ].reset_index(drop=True)
        ), 
        orient='index'
    ).reset_index().rename(columns={'index': 'customer_id'}).merge(df_tower)
    
    # Filter customers who were in earthquake affected cities
    affected_customers = last_observed_locations[
        last_observed_locations['city_id'].isin(earthquake_cities)
    ]['customer_id'].unique().tolist()
    
    # Filter CDR data to include only affected customers
    final_df = final_df[final_df['customer_id'].isin(affected_customers)].reset_index(drop=True)
    final_df['segment'] = final_df['segment_caller'].where(final_df['incoming'] == 0, final_df['segment_callee'])
    final_df['site_id'] = final_df['site_id_caller'].where(final_df['incoming'] == 0, final_df['site_id_callee'])
    dataframe_all = final_df.merge(
        df_clusters, 
        left_on="site_id", 
        right_on="site_id"
    ).merge(
        gpd.read_file(clusters_shapefile, geometry='geometry'),
        on="cluster"
    )[
        ["cluster", "geometry", "time", "customer_id", "incoming", "segment_caller", "segment_callee"]
    ].reset_index(drop=True)
    gdf_all = gpd.GeoDataFrame(dataframe_all, geometry="geometry").set_crs("EPSG:4326").to_crs("EPSG:32636")
    
    return gdf_all

def get_last_locations(data):
    data['site_id'] = data['site_id_caller'].where(data['incoming'] == 0, data['site_id_callee'])
    data = data.dropna(subset=['site_id'])
    data_sorted = data.sort_values('time', ascending=False)
    last_locations = data_sorted.drop_duplicates(subset='customer_id', keep='first')
    result = last_locations[['customer_id', 'site_id', 'time']]
    result.set_index('customer_id', inplace=True)
    return result.to_dict(orient='index')
