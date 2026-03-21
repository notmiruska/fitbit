import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import seaborn as sns
import plotly.express as px
import math
from data import *



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
        yaxis=dict(
            title="Hours of sleep per session",
            tickformat=".1f",
        ),
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
    '''input: output of get_total_distance()'''

    dynamic_height = max(400, len(df) * 25)
    fig = px.bar(
        df,
        x='TotalDistance',
        y='Id',
        orientation='h',
        height=dynamic_height,
        title='Total Distance Per User', 
    )

    fig.update_layout(
        yaxis_title='User ID',
        xaxis_title='Total Distance',
        yaxis={'type': 'category'},
    )

    fig.update_yaxes(categoryorder='total ascending')
    
    return fig


def plot_workout_per_day(df):
    '''input: output of workout_per_day()'''

    fig = px.bar(
        df,
        x='Weekday',
        y='count',
        title='Workout Frequency per Day'
    )

    return fig


def plot_regression_steps_calories(df):
    fig = px.scatter(
        df,
        x='TotalSteps',
        y='Calories',
        trendline='ols',
        color='Id',
        title='Relationship Between Amount of Steps Taken and Calories Burnt'
    )

    return fig


def plot_regression_sleep_activity(df):
    fig = px.scatter(
        df,
        x='SleepDuration',
        y='TotalActiveMinutes',
        trendline='ols',
        title='Relationship Between Sleep Duration and Total Active Minutes'
    )

    return fig


def plot_regression_sleep_sedentary(df):
    fig = px.scatter(
        df,
        x='SleepDuration',
        y='SedentaryMinutes',
        trendline='ols',
        title='Relationship Between Sleep Duration and Sedentary Minutes'
    )

    return fig


def plot_steps_per_block(df):
    fig = px.bar(
        df,
        x='Block',
        y='AverageSteps',
        title='Average Steps Taken Per Time Block'
    )

    return fig


def plot_calories_per_block(df):
    fig = px.bar(
        df,
        x='Block',
        y='AverageCalories',
        title='Average Calories Burnt Per Time Block'
    )
    
    return fig


def plot_sleep_per_block(df):
    fig = px.bar(
        df,
        x='Block',
        y='AverageSleep',
        title='Average Sleep Duration Per Time Block'
    )

    return fig


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

def plot_user_class(n_activities, user_class):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=n_activities,
        number={
            'suffix': " activities",
            'font': {'size': 28}
        },
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "User class based on number of activities",
            'font': {'size': 28}
        },
        
        gauge={
            'axis': {'range': [0, 32]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, 10], 'color': "#68a8df"},
                {'range': [10, 15], 'color': "#4968b0"},
                {'range': [15, 32], 'color': "#343cac"}
            ],
        }
    ))

    fig.add_annotation(
        x=0.5,
        y=0.25,
        text=user_class,
        showarrow=False,
        font=dict(size=36)
    )
    return fig

def plot_activity_vs_weather(df_merged, user_id):
    
    fig = go.Figure()

    # Left axis (Steps)
    fig.add_trace(go.Bar(
        x=df_merged['ActivityDate'],
        y=df_merged['TotalSteps'],  
        name='Steps',
        yaxis='y1',
        marker_color= "#4b70b5",
        opacity=0.7
    ))
    
    # Right axis (Temp)
    fig.add_trace(go.Scatter(
        x=df_merged['ActivityDate'],
        y=df_merged['temp'],
        name='Avg Temp',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color="#130f4b", width=2)
    ))
    

    fig.update_layout(
        title=f"Total steps vs Average temperature for User {user_id}",
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='Steps',
            side='left',
            showgrid=False
        ),
        yaxis2=dict(
            title='Average Temperature (°F)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(x=0.1, y=1.1, orientation='h'),
        template='plotly_white'
    )
    return fig

def barplot_steps_vs_precip(df_merged, user_id):

    df_merged['Had_Precip'] = df_merged['precip'].apply(lambda x: 'Yes' if x > 0 else 'No')
    avg_steps = df_merged.groupby('Had_Precip')['TotalSteps'].mean().reset_index()
    avg_steps['TotalSteps'] = np.floor(avg_steps['TotalSteps'])

    fig = px.bar(
        avg_steps,
        x='Had_Precip',
        y='TotalSteps',
        color='Had_Precip',
        color_discrete_map={'Yes':"#4968b0",'No':"#4968b0"},
        text='TotalSteps',
        title=f"Average Steps on days with or withour precipitation for User {user_id}"
    )
    
    fig.update_layout(
        yaxis_title="Average Total Steps",
        xaxis_title="Precipitation (Yes/No)",
        showlegend=False,
        template="plotly_white"
    )
    return fig


