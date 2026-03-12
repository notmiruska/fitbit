import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

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
