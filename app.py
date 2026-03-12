from modules.data import *
from modules.stats import *
from modules.plots import *

data = pd.read_csv('./data/daily_activity.csv')

print('Number of unique users: ', data['Id'].nunique(), '\n')

total_distance = compute_total_distance(data)
print('Total distance per user:\n', total_distance, '\n')
plot_total_distance(total_distance)

plot_burnt_calories(data, 1503960366, '3/25/2016', '4/09/2016')

workout_per_day = get_workout_per_day(data)
plot_workout_per_day(workout_per_day)

# relationship between calories and steps taken
print(steps_calories_regression(data))
plot_calories(data, 1503960366)

# extra
distance_by_activity_level = get_distance_by_activity_level(data)
plot_distance_by_activity_level(distance_by_activity_level)
