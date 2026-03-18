import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import plotly.express as px
from data import *

def plot_total_distance(df):
    total_distance_per_user = df.groupby('Id')['TotalDistance'].sum()

    total_distance_per_user.plot(kind='bar')

    plt.xlabel("User ID")
    plt.ylabel("Total Distance")
    plt.title("Total Distance per User")

    plt.show()

def plot_calories_for_user(df, user_id, start_date=None, end_date=None):
    
    # Convert to datetime
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    
    # Filter by user
    user_data = df[df['Id'] == user_id]
    
    # Filter by date range (if provided)
    if start_date:
        start_date = pd.to_datetime(start_date)
        user_data = user_data[user_data['ActivityDate'] >= start_date]
        
    if end_date:
        end_date = pd.to_datetime(end_date)
        user_data = user_data[user_data['ActivityDate'] <= end_date]
    
    # Sort by date
    user_data = user_data.sort_values('ActivityDate')
    
    # Plot
    plt.figure()
    plt.plot(user_data['ActivityDate'], user_data['Calories'])
    
    plt.xlabel("Date")
    plt.ylabel("Calories Burnt")
    plt.title(f"Calories Burnt Over Time (User {user_id})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_workout_frequency_by_weekday(df):
    
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    df['Weekday'] = df['ActivityDate'].dt.day_name()
    workout_counts = df['Weekday'].value_counts()
    
    ordered_days = ["Monday", "Tuesday", "Wednesday", 
                    "Thursday", "Friday", "Saturday", "Sunday"]
    
    workout_counts = workout_counts.reindex(ordered_days)
    
    plt.figure()
    workout_counts.plot(kind='bar')
    
    plt.xlabel("Day of the Week")
    plt.ylabel("Workout Frequency")
    plt.title("Workout Frequency per Weekday (All Users)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_regression_for_user(df, model, user_id):
    
    user_data = df[df['Id'] == user_id]
    
    # Scatter plot
    plt.figure()
    plt.scatter(user_data['TotalSteps'], user_data['Calories'])
    
    # Create smooth line
    steps_range = np.linspace(
        user_data['TotalSteps'].min(),
        user_data['TotalSteps'].max(),
        100
    )
    
    pred_df = pd.DataFrame({
        'TotalSteps': steps_range,
        'Id': user_id
    })
    
    predictions = model.predict(pred_df)
    
    plt.plot(steps_range, predictions)
    
    plt.xlabel("Total Steps")
    plt.ylabel("Calories")
    plt.title(f"Calories vs Steps (User {user_id})")
    plt.tight_layout()
    plt.show()

def plot_sleep_timeline(main_sleep, naps, sleep_hours_line, merged_df, person_id):

    fig = go.Figure()

    # Main sleep line
    fig.add_trace(go.Scatter(
        x=main_sleep['sleep_day_dt'],
        y=sleep_hours_line,
        mode='lines+markers',
        name='Main sleep (>=3h)',
        line=dict(color='orange', width=3),
        marker=dict(size=8)
    ))

    # Naps as stars
    fig.add_trace(go.Scatter(
        x=naps['sleep_day_dt'],
        y=naps['SleepHours'],
        mode='markers',
        name='Nap (<3h)',
        marker=dict(color='red', size=12, symbol='star')
    ))

    fig.update_layout(
        title=f"Sleep Sessions Timeline for User {person_id}",
        xaxis_title="Date",
        yaxis_title="Hours of sleep per session",
        template="plotly_white",
        xaxis=dict(
            range=[
                merged_df['sleep_day_dt'].min() - pd.Timedelta(days=1),
                merged_df['sleep_day_dt'].max() + pd.Timedelta(days=1)
            ]
        )
    )

    return fig

def last_24_hours_plot(df, id):

    heartrate_user = df[df['Id'] == id]

    last_time = heartrate_user['Time'].max()
    start_time = last_time - pd.Timedelta(hours=24)

    df_last24 = heartrate_user[heartrate_user['Time'] >= start_time]
    df_last24 = df_last24.sort_values('Time')

    df_last24['diff'] = df_last24['Time'].diff()
    df_last24['plot_hr'] = df_last24['Value']
    df_last24.loc[df_last24['diff'] > pd.Timedelta(minutes=10), 'plot_hr'] = None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_last24['Time'],
        y=df_last24['plot_hr'],
        mode='lines',
        name='Heartrate',
        line=dict(width=2)
    ))

    fig.update_layout(
        title='Heartrate last 24 hours (line breaks on gaps > 10 min)',
        xaxis_title='Time',
        yaxis_title='Heartrate (bpm)',
        template='plotly_white'
    )

    return fig


def plot_stats_heartrate(df, id):

    heartrate_user = df[df['Id'] == id].copy()
    heartrate_user['Date'] = heartrate_user['Time'].dt.date

    daily_stats = heartrate_user.groupby('Date')['Value'].agg(['mean', 'min', 'max']).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_stats['Date'],
        y=daily_stats['mean'],
        mode='lines+markers',
        name='Mean'
    ))

    fig.add_trace(go.Scatter(
        x=daily_stats['Date'],
        y=daily_stats['max'],
        mode='lines+markers',
        name='Max'
    ))

    fig.add_trace(go.Scatter(
        x=daily_stats['Date'],
        y=daily_stats['min'],
        mode='lines+markers',
        name='Min'
    ))

    fig.update_layout(
        title='Daily Heart Rate Summary',
        xaxis_title='Date',
        yaxis_title='Heartrate (bpm)',
        template='plotly_white'
    )

    return fig

def plot_total_distance(df):
    sns.barplot(df, x='TotalDistance', y='Id', orient='h', color='tab:blue')
    plt.title('Total Distance Per User')
    plt.xlabel('Total Distance')
    plt.ylabel('User ID')
    plt.yticks(fontsize=7)
    plt.grid(axis='x')
    plt.tight_layout()
    
    plt.show()


def plot_burnt_calories(df, user_id, start_date, end_date):
    user_data = burnt_calories(df, user_id, start_date, end_date)

    plt.plot(user_data['ConvertedDate'], user_data['Calories'])
    plt.title(f'Calories burnt by user {user_id} per day')
    plt.xlabel('Date')
    plt.ylabel('Burnt calories')
    plt.xticks(fontsize=7, rotation=30)
    plt.tight_layout()

    plt.show()


def plot_workout_per_day(df):
    sns.barplot(df, x='Weekday', y='count', color='tab:blue', 
                order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title('Workout frequency per day')
    plt.xlabel('Day')
    plt.ylabel('Frequency')
    plt.show()


def plot_calories(df, user_id):
    # note: check this for correctness
    user_data = df[df['Id'] == user_id]
    
    sns.regplot(data=user_data,
                x = 'TotalSteps',
                y='Calories',
                scatter_kws={'alpha':0.5},
                line_kws={'color':'tab:pink'})
    plt.title(f'Relationship between calories and steps taken for user {user_id}')
    plt.xlabel('Total Amount of Steps')
    plt.ylabel('Burnt Calories')
    plt.show()


def plot_distance_by_activity_level(df):
    df.plot.barh(stacked=True, color=['tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray'])
    plt.title('Total Distance Per User Grouped by Activity Level')
    plt.xlabel('Distance')
    plt.ylabel('User ID')
    plt.yticks(fontsize=7)
    plt.legend(['Very Active', 'Moderately Active', 'Light Active', 'Sedentary Active'])
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show() 


def plot_total_intensity_hourly(df, user_id):
    
    # Sort by time (important for line plots)
    df = df.sort_values(by='ActivityHour')

    # Plot ALL datapoints
    fig = px.line(
        df,
        x='ActivityHour',
        y='TotalIntensity',
        title=f'Total Intensity Over Time (User {user_id})',
        labels={
            'ActivityHour': 'Time',
            'TotalIntensity': 'Total Intensity'
        }
    )
    
    return fig

