import pandas as pd
from datetime import datetime
import numpy as np

def read_csv(path):
    df = pd.read_csv(path)
    return df

def load_users(path):
    df = pd.read_csv(path)
    users = df['Id'].unique()
    return users

def load_sleep_data(connection):
    minute_sleep = pd.read_sql_query("SELECT * FROM minute_sleep;", connection)
    minute_sleep['date'] = pd.to_datetime(minute_sleep['date'], format='%m/%d/%Y %I:%M:%S %p')
    minute_sleep['Id'] = minute_sleep['Id'].astype(int)

    # Assign "sleep day" correctly (-12 hours)
    minute_sleep['sleep_day'] = (minute_sleep['date'] - pd.Timedelta(hours=12)).dt.date
    return minute_sleep

def load_heartrate_data(connection):
    df_heartrate = pd.read_sql_query("SELECT * FROM heart_rate;", connection)
    df_heartrate["Time"] = pd.to_datetime(df_heartrate["Time"])
    df_heartrate['Date'] = df_heartrate['Time'].dt.date
    df_heartrate['Id'] = df_heartrate['Id'].astype(int)
    return df_heartrate

def load_activity_data(connection):
    df_activity = pd.read_sql_query("SELECT * FROM hourly_intensity;", connection)
    df_activity['ActivityHour'] = pd.to_datetime(df_activity['ActivityHour'])
    return df_activity

def load_daily_activity(connection):
    df_daily_activity = pd.read_sql_query("SELECT * FROM daily_activity;", connection)
    return df_daily_activity

def classify_user(df, person_id):
    person_count = len(df[df["Id"] == person_id])

    user_class = (
        'Light user' if person_count <= 10 
        else 'Moderate user' if 11 <= person_count <= 15 
        else 'Heavy user'
    )
    return person_count, user_class

def process_sleep_sessions(df_person, nap_threshold=3):
    """Process sleep sessions for a single user"""
    # Group by logId
    sessions = df_person.groupby('logId').agg(
        start_time=('date', 'min'),
        end_time=('date', 'max'),
        duration_minutes=('date', 'count'),
        sleep_day=('sleep_day', 'first')
    ).reset_index()

    # Sort and merge sessions <2h gap
    sessions = sessions.sort_values('start_time').reset_index(drop=True)
    merged_sessions = []
    current_session = sessions.iloc[0].copy()

    for i in range(1, len(sessions)):
        next_session = sessions.iloc[i]
        gap = (next_session['start_time'] - current_session['end_time']).total_seconds() / 3600
        if gap < 2:
            current_session['end_time'] = next_session['end_time']
            current_session['duration_minutes'] += next_session['duration_minutes']
        else:
            merged_sessions.append(current_session)
            current_session = next_session.copy()
    merged_sessions.append(current_session)

    merged_df = pd.DataFrame(merged_sessions)
    merged_df['SleepHours'] = merged_df['duration_minutes'] / 60
    merged_df['sleep_day'] = merged_df['start_time'].dt.date
    merged_df['sleep_day_dt'] = pd.to_datetime(merged_df['sleep_day'])

    # Split main sleep vs naps
    main_sleep = merged_df[merged_df['SleepHours'] >= nap_threshold].sort_values('sleep_day_dt')
    naps = merged_df[merged_df['SleepHours'] < nap_threshold]

    # Line with NaNs for non-consecutive days
    sleep_hours_line = main_sleep['SleepHours'].copy()
    sleep_day_diff = main_sleep['sleep_day_dt'].diff().dt.days
    sleep_hours_line[sleep_day_diff > 1] = np.nan

    return main_sleep, naps, sleep_hours_line, merged_df

def compute_total_distance(df):
    df = df.groupby('Id', as_index=False)['TotalDistance'].sum()
    df = df.sort_values('TotalDistance', ascending=False)
    df['Id'] = df['Id'].astype(str)

    return df


def convert_date(date):
    return datetime.strptime(date, '%m/%d/%Y')


def burnt_calories(df, user_id, start_date, end_date):
    start_date = convert_date(start_date)
    end_date = convert_date(end_date)

    user_data = df[df['Id'] == user_id][['ActivityDate', 'Calories']]

    # add converted date column
    user_data['ConvertedDate'] = [convert_date(d) for d in user_data['ActivityDate']]

    # filter for range of dates
    user_data = user_data[(user_data['ConvertedDate'] >= start_date) & 
                          (user_data['ConvertedDate'] <= end_date)]

    return user_data


def day_of_week(df):
    dates = df['ActivityDate'].tolist()
    days = []

    for date in dates:
        converted_date = convert_date(date)
        days.append(converted_date.strftime('%A'))

    return days


def get_workout_per_day(df):
    weekday = day_of_week(df)
    df['Weekday'] = weekday 
    day_frequency = df['Weekday'].value_counts()
    workout_per_day = day_frequency.to_frame().reset_index()
    
    return workout_per_day


def get_distance_by_activity_level(df):
    df = df.groupby('Id')[['TotalDistance', 
                           'VeryActiveDistance', 
                           'ModeratelyActiveDistance', 
                           'LightActiveDistance', 
                           'SedentaryActiveDistance']].sum().sort_values('TotalDistance')
    df = df.drop(columns='TotalDistance')

    return df  
