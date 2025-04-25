import pandas as pd
from tqdm.auto import tqdm  
def calculate_day_night_duration(df):
    df = df.copy()
    df['start_date'] = df['start_time'].dt.date
    df['end_date'] = df['end_time'].dt.date

    df['morning_bound_start'] = pd.to_datetime(df['start_date'].astype(str) + ' 07:00:00')
    df['evening_bound_start'] = pd.to_datetime(df['start_date'].astype(str) + ' 22:00:00')
    df['morning_bound_end'] = pd.to_datetime(df['end_date'].astype(str) + ' 07:00:00')
    df['evening_bound_end'] = pd.to_datetime(df['end_date'].astype(str) + ' 22:00:00')

    df['spans_multiple_days'] = df['start_date'] != df['end_date']
    df['starts_before_morning'] = df['start_time'].dt.hour < 7
    df['starts_after_evening'] = df['start_time'].dt.hour >= 22
    df['ends_before_morning'] = df['end_time'].dt.hour < 7
    df['ends_after_evening'] = df['end_time'].dt.hour >= 22

    def process_time_chunks(row):
        if not row['spans_multiple_days']:
            if row['starts_before_morning']:
                night_mins = (row['morning_bound_start'] - row['start_time']).total_seconds() / 3600
                if row['ends_before_morning']:
                    day_mins = 0
                    night_mins = (row['end_time'] - row['start_time']).total_seconds() / 3600
                elif row['ends_after_evening']:
                    day_mins = 15  # 7AM to 22PM
                    night_mins += (row['end_time'] - row['evening_bound_start']).total_seconds() / 3600
                else:
                    day_mins = (row['end_time'] - row['morning_bound_start']).total_seconds() / 3600
            elif row['starts_after_evening']:
                night_mins = (row['end_time'] - row['start_time']).total_seconds() / 3600
                day_mins = 0
            else:
                if row['ends_after_evening']:
                    day_mins = (row['evening_bound_start'] - row['start_time']).total_seconds() / 3600
                    night_mins = (row['end_time'] - row['evening_bound_start']).total_seconds() / 3600
                else:
                    day_mins = (row['end_time'] - row['start_time']).total_seconds() / 3600
                    night_mins = 0
        else:
            full_days = (row['end_date'] - row['start_date']).days - 1
            day_mins = max(0, full_days * 15) 
            night_mins = max(0, full_days * 9)  

            if row['starts_before_morning']:
                night_mins += (row['morning_bound_start'] - row['start_time']).total_seconds() / 3600
                day_mins += 15  
                night_mins += 2  
            elif row['starts_after_evening']:
                night_mins += (pd.Timestamp(row['start_date'] + pd.Timedelta(days=1)) - row[
                    'start_time']).total_seconds() / 3600
            else:
                day_mins += (row['evening_bound_start'] - row['start_time']).total_seconds() / 3600
                night_mins += 2 
            if row['ends_before_morning']:
                night_mins += (row['end_time'] - pd.Timestamp(row['end_date'])).total_seconds() / 3600
            elif row['ends_after_evening']:
                day_mins += 15  # Full day portion
                night_mins += (row['end_time'] - row['evening_bound_end']).total_seconds() / 3600
            else:
                night_mins += 7  # Morning portion
                day_mins += (row['end_time'] - row['morning_bound_end']).total_seconds() / 3600

        return pd.Series({'day_duration': day_mins, 'night_duration': night_mins})

    result = df.groupby(df.index // 1000).progress_apply(
        lambda chunk: chunk.apply(process_time_chunks, axis=1)
    ).reset_index(level=0, drop=True)

    return result

def calculate_relevance(df):
    customer_totals = df.groupby('customer_id').agg({
        'duration_aggregated': 'sum',
        'day_duration_aggregated': 'sum',
        'night_duration_aggregated': 'sum'
    }).reset_index()
    df = df.merge(customer_totals,
                  on='customer_id',
                  suffixes=('', '_total'))
    df['total_relevance'] = (df['duration_aggregated'] / df['duration_aggregated_total']) * 100
    df['day_relevance'] = (df['day_duration_aggregated'] / df['day_duration_aggregated_total']) * 100
    df['night_relevance'] = (df['night_duration_aggregated'] / df['night_duration_aggregated_total']) * 100
    df = df.drop(['duration_aggregated_total',
                  'day_duration_aggregated_total',
                  'night_duration_aggregated_total'], axis=1)

    return df

def calculate_night_relevance_pre_disaster(df_pre_earthquake_stays, df_post_earthquake_stays):
    """
    Process stay durations by calculating day/night divisions and aggregating durations for the same locations.
    """
    day_and_night_division_before = calculate_day_night_duration(df_pre_earthquake_stays)
    day_and_night_division_after = calculate_day_night_duration(df_post_earthquake_stays)

    df_pre_earthquake_stays = pd.concat([day_and_night_division_before, df_pre_earthquake_stays], axis=1)
    df_post_earthquake_stays = pd.concat([day_and_night_division_after, df_post_earthquake_stays], axis=1)

    df_pre_earthquake_stays = df_pre_earthquake_stays.merge(
        df_pre_earthquake_stays[["customer_id", "clusters", "duration", "day_duration", "night_duration"]]
        .groupby(["customer_id", "clusters"]).sum().reset_index()
        .rename(columns={
            "duration": "duration_aggregated",
            "day_duration": "day_duration_aggregated",
            "night_duration": "night_duration_aggregated"
        }),
        how="left",
        on=["customer_id", "clusters"]
    )

    df_post_earthquake_stays = df_post_earthquake_stays.merge(
        df_post_earthquake_stays[["customer_id", "clusters", "duration", "day_duration", "night_duration"]]
        .groupby(["customer_id", "clusters"]).sum().reset_index()
        .rename(columns={
            "duration": "duration_aggregated",
            "day_duration": "day_duration_aggregated",
            "night_duration": "night_duration_aggregated"
        }),
        how="left",
        on=["customer_id", "clusters"]
    )

    df_pre_earthquake_stays = calculate_relevance(df_pre_earthquake_stays)
    df_post_earthquake_stays = calculate_relevance(df_post_earthquake_stays)

    return df_pre_earthquake_stays, df_post_earthquake_stays