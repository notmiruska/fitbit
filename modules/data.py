from datetime import datetime
import pandas as pd


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
