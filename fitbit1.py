import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.formula.api as smf


def count_unique_users(df):
    return df['Id'].nunique()


def compute_total_distance(df):
    df = df.groupby('Id', as_index=False)['TotalDistance'].sum()
    df = df.sort_values('TotalDistance', ascending=False)
    return df


def visualize_total_distance(df):
    df = compute_total_distance(df)
    df['Id'] = df['Id'].astype(str)

    sns.barplot(df, x='TotalDistance', y='Id', orient='h', color='tab:blue')
    plt.title('Total Distance Per User')
    plt.xlabel('Total Distance')
    plt.ylabel('User ID')
    plt.yticks(fontsize=7)
    plt.grid(axis='x')
    plt.tight_layout()
    
    plt.show()


def convert_date(date):
    return datetime.strptime(date, '%m/%d/%Y')


def burnt_calories(user_id, start_date, end_date):
    start_date = convert_date(start_date)
    end_date = convert_date(end_date)

    data = pd.read_csv('daily_activity.csv')
    user_data = data[data['Id'] == user_id][['ActivityDate', 'Calories']]

    # add converted date column
    user_data['ConvertedDate'] = [convert_date(d) for d in user_data['ActivityDate']]

    # filter for range of dates
    user_data = user_data[(user_data['ConvertedDate'] >= start_date) & 
                          (user_data['ConvertedDate'] <= end_date)]

    return user_data


def visualize_burnt_calories(user_id, start_date, end_date):
    user_data = burnt_calories(user_id, start_date, end_date)

    plt.plot(user_data['ConvertedDate'], user_data['Calories'])
    plt.title(f'Calories burnt by user {user_id} per day')
    plt.xlabel('Date')
    plt.ylabel('Burnt calories')
    plt.xticks(fontsize=7, rotation=30)
    plt.tight_layout()

    plt.show()


def day_of_week(df):
    dates = df['ActivityDate'].tolist()
    days = []

    for date in dates:
        converted_date = convert_date(date)
        days.append(converted_date.strftime('%A'))

    return days


def add_day_of_week_to(df):
    weekday = day_of_week(df)
    df['Weekday'] = weekday 
    return df


def workout_per_day(df):
    df = add_day_of_week_to(df)
    day_frequency = df['Weekday'].value_counts()
    weekday_count = day_frequency.to_frame().reset_index()

    sns.barplot(weekday_count, x='Weekday', y='count', color='tab:blue', 
                order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title('Workout frequency per day')
    plt.xlabel('Day')
    plt.ylabel('Frequency')
    plt.show()


def steps_calories_regression(df):
    model = smf.ols('Calories ~ TotalSteps + C(Id)', data=df)
    results = model.fit()

    return results.summary()


def visualize_calories(df, user_id):
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
    


# EXTRA:

def distance_by_activity_level(df):
    # note: check totals 
    df = df.groupby('Id')[['TotalDistance', 
                           'VeryActiveDistance', 
                           'ModeratelyActiveDistance', 
                           'LightActiveDistance', 
                           'SedentaryActiveDistance']].sum().sort_values('TotalDistance')
    df = df.drop(columns='TotalDistance')
    
    df.plot.barh(stacked=True, color=['tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray'])
    plt.title('Total Distance Per User Grouped by Activity Level')
    plt.xlabel('Distance')
    plt.ylabel('User ID')
    plt.yticks(fontsize=7)
    plt.legend(['Very Active', 'Moderately Active', 'Light Active', 'Sedentary Active'])
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()


def main():
    data = pd.read_csv('daily_activity.csv')

    # basic inspection of the data
    print('Number of unique users: ', count_unique_users(data), '\n')
    print('Total distance per user:\n', compute_total_distance(data), '\n')
    visualize_total_distance(data)
    visualize_burnt_calories(1503960366, '3/25/2016', '4/09/2016')
    workout_per_day(data)

    # relationship between calories and steps taken
    print(steps_calories_regression(data))
    visualize_calories(data, 1503960366)

    # extra
    distance_by_activity_level(data)


if __name__ == '__main__':
    main()