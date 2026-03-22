import pandas as pd
from datetime import datetime
import numpy as np
import requests
import urllib.parse


def read_csv(path):
    df = pd.read_csv(path)
    return df

def load_users(path):
    df = pd.read_csv(path)
    users = df['Id'].unique()
    return users

def load_and_prepare(path):
    df = pd.read_csv(path)
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'], format='%m/%d/%Y')
    df['Weekday'] = pd.Categorical(
        df['ActivityDate'].dt.day_name(),
        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
        ordered=True
        )
    
    return df

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
    df_daily_activity["ActivityDate"] = pd.to_datetime(df_daily_activity["ActivityDate"])
    return df_daily_activity

def load_weight_data(connection):
    df_weight = pd.read_sql_query("SELECT * FROM weight_log;", connection)
    df_weight["Date"] = pd.to_datetime(df_weight["Date"])
    return df_weight

def load_calories_data(connection):
    df_calories = pd.read_sql_query('SELECT * FROM hourly_calories;', connection)
    df_calories['ActivityHour'] = pd.to_datetime(df_calories['ActivityHour'])
    return df_calories

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


def get_total_distance(df):
    df = (df.groupby('Id', as_index=False)['TotalDistance']
          .sum()
          .sort_values('TotalDistance', ascending=False)
          .assign(Id=lambda x: x['Id'].astype(str))
    )
    return df


def get_workout_per_day(df):
    df = (df['Weekday']
          .value_counts()
          .sort_index()
          .reset_index()
    )
    return df


def assign_date_of_activity(row):
    if row['SleepEnd'].hour < 12:
        return row['SleepEnd'].date()
    else:
        # sleep session is likely an afternoon/evening nap -> related to tommorrow's activity
        return (row['SleepEnd'] + pd.Timedelta(days=1)).date()


def get_sleep_duration_per_session(connection):
    '''for regression between sleep duration & activity'''
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM minute_sleep')

    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=['Id', 'date', 'value', 'logId'])

    # fix data types
    df['Id'] = df['Id'].astype(int).astype(str)
    df['logId'] = df['logId'].astype(int).astype(str)
    df['date'] = pd.to_datetime(df['date'])
    
    # sleep sessions
    sleep_sessions = df.groupby(['Id', 'logId']).agg(
        SleepEnd=('date', 'max'),
        SleepDuration=('value', 'count')
    ).reset_index()

    sleep_sessions['ActivityDate'] = sleep_sessions.apply(assign_date_of_activity, axis=1)

    sleep_df = sleep_sessions.groupby(['Id', 'ActivityDate'])['SleepDuration'].sum().reset_index()

    return sleep_df


def get_activity_data(connection):
    '''for regression between sleep & activity'''
    cursor = connection.cursor()
    active_minutes = '''
    SELECT
        Id,
        ActivityDate,
        (VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes) as TotalActiveMinutes,
        SedentaryMinutes
    FROM daily_activity
    GROUP BY Id, ActivityDate
    '''
    cursor.execute(active_minutes)
    activity_data = cursor.fetchall()
    df = pd.DataFrame(activity_data, columns = [x[0] for x in cursor.description])
    
    df['Id'] = df['Id'].astype(int).astype(str)
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate']).dt.date
    
    return df


def merge_sleep_and_activity_data(connection):
    '''use for regression_sleep_activity()'''
    sleep = get_sleep_duration_per_session(connection)
    activity = get_activity_data(connection)

    df = pd.merge(
        sleep,
        activity,
        on=['Id', 'ActivityDate'],
        how='inner'
    )

    return df

    
def download_weather_data(API_KEY):
    UnitGroup = 'us'
    StartDate = '2016-03-12'
    EndDate = '2016-04-12'
    location = "Chicago,IL,USA"
    ContentType = "csv"

    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{location}/{StartDate}/{EndDate}?unitGroup={UnitGroup}&key={API_KEY}&contentType={ContentType}&include=days"
    )

    try:
        weather_df = pd.read_csv(url)
        weather_df = weather_df.drop(columns=['feelslikemax',
       'feelslikemin', 'feelslike', 'dew', 'humidity','precipprob',
       'precipcover', 'snow', 'snowdepth', 'windgust',
       'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
       'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'sunrise',
       'sunset', 'moonphase', 'conditions', 'description', 'icon', 'stations'])
        weather_df.to_csv("data/chicago_weather.csv", index=False)
        print("Weather data saved to chicago_weather.csv!")
        return weather_df
    except Exception as e:
        print(f"Failed to download weather data: {e}")
        return None
    
def merge_weather_and_steps_data(df_weather, df_daily_activity, user_id):
    df_user = df_daily_activity[df_daily_activity['Id'] == user_id].copy()
    df_user['ActivityDate'] = pd.to_datetime(df_user['ActivityDate'])
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_merged = pd.merge(df_user, df_weather, left_on='ActivityDate', right_on='datetime', how='left')
    return df_merged


def assign_blocks(df, date_column):
    df['Hour'] = pd.to_datetime(df[date_column]).dt.hour
    df.loc[df['Hour'] < 4, 'Block'] = '0-4'
    df.loc[(df['Hour'] >= 4) & (df['Hour'] < 8), 'Block'] = '4-8'
    df.loc[(df['Hour'] >= 8) & (df['Hour'] < 12), 'Block'] = '8-12'
    df.loc[(df['Hour'] >= 12) & (df['Hour'] < 16), 'Block'] = '12-16'
    df.loc[(df['Hour'] >= 16) & (df['Hour'] < 20), 'Block'] = '16-20'
    df.loc[(df['Hour'] >= 20) & (df['Hour'] < 24), 'Block'] = '20-24'

    df['Block'] = pd.Categorical(
        df['Block'], 
        categories=['0-4', '4-8', '8-12', '12-16', '16-20', '20-24'], 
        ordered=True)

    return df
  
  
def get_steps_per_block(connection):
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM hourly_steps')
    hourly_steps = cursor.fetchall()

    df = pd.DataFrame(hourly_steps,
                      columns = [x[0] for x in cursor.description])
   
    df = assign_blocks(df, 'ActivityHour')
    df = df.groupby('Block', as_index=False).agg(
        StepTotal=('StepTotal', 'sum'),
        Count=('StepTotal', 'size')
    )
    df['AverageSteps'] = df['StepTotal'] / df['Count']

    return df


def get_calories_per_block(connection):
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM hourly_calories')
    hourly_calories = cursor.fetchall()
    df = pd.DataFrame(hourly_calories,
                      columns = [x[0] for x in cursor.description])
   
    df = assign_blocks(df, 'ActivityHour')
    df = df.groupby('Block', as_index=False).agg(
        Calories=('Calories', 'sum'),
        Count=('Calories', 'size')
    )
    df['AverageCalories'] = df['Calories'] / df['Count']

    return df


def get_sleep_per_block(connection):
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM minute_sleep')
    minute_sleep = cursor.fetchall()

    df = pd.DataFrame(minute_sleep, columns=[x[0] for x in cursor.description])

    df['Id'] = df['Id'].astype(int).astype(str)
    df['Day'] = pd.to_datetime(df['date'], format='%m/%d/%Y %I:%M:%S %p').dt.date

    df = assign_blocks(df, 'date')

    df['User_and_Date'] = df['Id'].astype(str) + ' ' + df['Day'].astype(str)

    df = df.groupby('Block', as_index=False).agg(
        TotalMins=('value', 'count'),
        SessionsPerBlock=('User_and_Date', 'nunique')
    )
    df['AverageSleep'] = df['TotalMins'] / df['SessionsPerBlock']

    return df
