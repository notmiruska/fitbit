# Fitbit Project

In this project, we analyse user data collected by Fitbit, a wearable fitness tracker that monitors user information such as activity, sleep and health (read more about Fitbit [here](https://en.wikipedia.org/wiki/Fitbit) and [here](https://www.fitbit.com/sg/about)). A dashboard has also been created for displaying the insights of this research.

# Project Structure

## Overview
```
fitbit-project/
├── data/
│   ├── daily_activity.csv
│   └── fitbit_database.db
├── modules/
│   ├── data.py
│   ├── stats.py
│   └── plots.py
└── app.py
```

## Data

Two sources of data are used in our analyses: 
### `daily_activity.csv`
Includes information about Fitbit users' daily activity, including but not limited to: steps taken, levels of activity, distance walked.
### `fitbit_database.db`
A database with the following tables:
* `daily_activity`: corresponds to `daily_activity.csv`
* `heart_rate`: heart rate of users with 5-second frequency
* `hourly_calories`: amount of calories burnt each hour
* `hourly_intensity`: exercise intensity per hour, given in both total & average amounts
* `hourly_steps`: amount of steps taken per hour
* `minute_sleep`: per-minute sleep data
* `weight_log`: weight, fat, BMI of each user

## Modules

### `data.py` 
Used for accessing relevant parts of the data for later analysis and plotting.


### `plots.py`
Includes plotting functions, with dashboarding solutions in mind.


### `stats.py`
File for the statistical analysis of fitbit data.


### Dashboard: `app.py`
Creates a dashboard using Streamlit. To display the dashboard, run:
```bash
streamlit run app.py
```

