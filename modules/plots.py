import seaborn as sns
import matplotlib.pyplot as plt
from modules.data import *


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

