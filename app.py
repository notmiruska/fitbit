import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.patches as mpatches
import sqlite3
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from modules.plots import *
from modules.data import *
from modules.stats import *

#CACHED FUNCTIONS

@st.cache_data(show_spinner=False)
def load_data():
    data = load_and_prepare('data/daily_activity.csv')
    
    connection = sqlite3.connect("data/fitbit_database.db")

    daily_activity = get_total_distance(data)
    workout_per_day = get_workout_per_day(data)
    sleep_and_activity = merge_sleep_and_activity_data(connection)
    steps_blocks = get_steps_per_block(connection)
    calories_blocks = get_calories_per_block(connection)
    sleep_blocks = get_sleep_per_block(connection)

    connection.close()

    return data, daily_activity, workout_per_day, sleep_and_activity, steps_blocks, calories_blocks, sleep_blocks


@st.cache_resource
def get_connection(db_path):
    return sqlite3.connect(db_path, check_same_thread=False)

@st.cache_data
def cached_load_users(path):
    return load_users(path)

@st.cache_data
def cached_load_sleep_data(_connection):
    return load_sleep_data(_connection)

@st.cache_data
def cached_load_heartrate_data(_connection):
    return load_heartrate_data(_connection)

@st.cache_data
def cached_load_activity_data(_connection):
    return load_activity_data(_connection)

@st.cache_data
def cached_load_daily_activity(_connection):
    return load_daily_activity(_connection)

@st.cache_data
def cached_load_weight_data(_connection):
    return load_weight_data(_connection)

@st.cache_data
def cached_load_calories_data(_connection):
    return load_calories_data(_connection)

def display_general_stats(data, users, df_daily_activity, daily_activity, workout_per_day, sleep_and_activity, steps_blocks, calories_blocks, sleep_blocks):
    st.title("Fitbit Dashboard")
    st.header("General Statistics")
    
    # initialize tabs
    tabs = st.tabs(['Overview', 'Activity', 'Sleep'])

    # Overview
    with tabs[0]:
        cols = st.columns(4)
        cols[0].metric("Number of Users", len(users))
        cols[1].metric("Average Distance Walked", round(daily_activity['TotalDistance'].mean(), 2))
        cols[2].metric("Average Steps per Day", int(np.floor(df_daily_activity['TotalSteps'].mean())))

        cols = st.columns(2)
        with cols[0]:
            st.plotly_chart(active_minutes_piechart(df_daily_activity))
        with cols[1]:
            # slider: display top 5 users by default, can slide up to 35
            number_of_users_to_show = st.slider(
                "Select number of users to display",
                min_value=5,
                max_value=10,
                value=min(5, 10)
            )
            data_to_show = daily_activity.head(number_of_users_to_show)
            st.plotly_chart(
                plot_total_distance(data_to_show), 
                use_container_width=True)
            

        cols = st.columns(3)
        with cols[0]:
            
            st.plotly_chart(plot_workout_per_day(workout_per_day), use_container_width=True)
            
        with cols[1]:
            st.plotly_chart(plot_regression_steps_calories(data), use_container_width=True)

        with cols[2]:
            st.plotly_chart(plot_steps_per_block(steps_blocks), use_container_width=True)
    
    # Activity Metrics
    with tabs[1]:
        st.plotly_chart(plot_calories_per_block(calories_blocks))

    # Sleep Metrics
    with tabs[2]:
        cols = st.columns(2)
        with cols[0]:
            st.plotly_chart(plot_regression_sleep_sedentary(sleep_and_activity))
        with cols[1]:
            st.plotly_chart(plot_sleep_per_block(sleep_blocks))


# MAIN DASHBOARD
def main():
    st.set_page_config(layout="wide")
        # Load data
    API_KEY = "FX2TTAYYJZ4YSXPEWUHS536LQ"
    connection = get_connection("data/fitbit_database.db")
    users = cached_load_users("data/daily_activity.csv")
    minute_sleep = cached_load_sleep_data(connection)
    df_heartrate = cached_load_heartrate_data(connection)
    df_activity = cached_load_activity_data(connection)
    df_daily_activity = cached_load_daily_activity(connection)
    df_weather = pd.read_csv("data/chicago_weather.csv")
    print(df_weather.head())
    df_weight = cached_load_weight_data(connection)
    df_calories = cached_load_calories_data(connection)
    data, daily_activity, workout_per_day, sleep_and_activity, steps_blocks, calories_blocks, sleep_blocks = load_data()


    # Sidebar: select a user with a placeholder default
    user_options = [""] + list(users)
    selected_user = st.sidebar.selectbox("Select a user", options=user_options, index=0)

    if selected_user == "":
        display_general_stats(data, users, df_daily_activity, daily_activity, workout_per_day, sleep_and_activity, steps_blocks, calories_blocks, sleep_blocks)


    if selected_user != "":
        person_id = selected_user

        st.title("Fitbit User Dashboard")
        st.subheader(f"User {person_id}")

        df_sleep_person = minute_sleep[minute_sleep["Id"] == person_id]
        df_hr_person = df_heartrate[df_heartrate["Id"] == person_id]
        df_activity_person = df_activity[df_activity["Id"] == person_id]
        df_weight_person = df_weight[df_weight["Id"] == person_id]
        df_calories_person = df_calories[df_calories['Id'] == person_id]

        #TABS FOR USERS
        tab1, tab2, tab3, tab4 = st.tabs(["General Stats", "Sleep", "Heartrate", "Intensity"])

        #General TAB
        with tab1:
            if not df_weight_person["WeightPounds"].empty:
            
                latest = df_weight_person.sort_values("Date").iloc[-1]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Weight (lbs)", f"{latest['WeightPounds']:.1f}")
                with col2:
                    st.metric("BMI", f"{latest['BMI']:.1f}")
                st.caption(f"Measured on: {latest['Date'].strftime('%Y-%m-%d')}")
                st.divider()
        
            n_activities, user_class = classify_user(df_daily_activity, person_id)
            fig_activity = plot_user_class(n_activities, user_class)
            st.plotly_chart(fig_activity, use_container_width=True)

            if not df_weather is None:
                df_merged = merge_weather_and_steps_data(df_weather, df_daily_activity, person_id)
                cols = st.columns(2)
                with cols[0]:
                    fig_steps_temp = plot_activity_vs_weather(df_merged, person_id)
                    st.plotly_chart(fig_steps_temp, use_container_width=True)
                with cols[1]:
                    fig_steps_precip = barplot_steps_vs_precip(df_merged, person_id)
                    st.plotly_chart(fig_steps_precip, use_container_width=True)
    

        # SLEEP TAB
        with tab2:

            if df_sleep_person.empty:
                st.warning("No sleep data available.")
            else:
                main_sleep, naps, sleep_hours_line, merged_df = process_sleep_sessions(df_sleep_person, nap_threshold=3)

                col1, = st.columns(1)

                with col1:
                    avg_hr = round(sleep_hours_line.mean(), 1)
                    st.metric("Mean sleep hours per night", f"{avg_hr} hours")

                st.divider()

                col1, col2 = st.columns(2)
                with col1:

                    fig_sleep = plot_sleep_timeline(main_sleep, naps, sleep_hours_line, merged_df, person_id)
                    st.plotly_chart(fig_sleep, use_container_width=True)

        # HEARTRATE TAB
        with tab3:

            col1, col2 = st.columns(2)

            with col1:
                if not df_hr_person.empty:
                    avg_hr = round(df_hr_person["Value"].mean(), 1)
                    st.metric("Average Heartrate", f"{avg_hr} bpm")

            with col2:
                if not df_hr_person.empty:
                    max_hr = df_hr_person["Value"].max()
                    st.metric("Max Heartrate", f"{max_hr} bpm")

            st.divider()

            if df_hr_person.empty:
                st.warning("No heartrate data available.")
            else:

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Last 24 hours")
                    fig_hr_24 = last_24_hours_plot(df_hr_person, person_id)
                    st.plotly_chart(fig_hr_24, width="stretch")

                with col2:
                    st.subheader("Daily statistics")
                    fig_hr_stats = plot_stats_heartrate(df_hr_person, person_id)
                    st.plotly_chart(fig_hr_stats, width="stretch")

        # ACTIVITY TAB
        with tab4:
            if df_activity_person.empty:
                st.warning("No activity data available.")
            else:
                # Get min/max available dates
                min_date = df_activity_person["ActivityHour"].min().date()
                max_date = df_activity_person["ActivityHour"].max().date()

                # Checkbox to filter by specific day
                filter_by_day = st.checkbox("Filter by specific day", value=False)

                if filter_by_day:
                    # Show calendar picker
                    selected_date = st.date_input(
                        "Select a day to display",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                    # Filter data for that day
                    df_to_plot = df_activity_person[df_activity_person["ActivityHour"].dt.date == selected_date]
                    df_calories_to_plot = df_calories_person[df_calories_person['ActivityHour'].dt.date == selected_date]

                    if df_to_plot.empty:
                        st.warning("No activity data for this day.")
                    else:
                        fig = plot_total_intensity_hourly(df_to_plot, person_id)
                        st.plotly_chart(fig, use_container_width=True)
                        st.plotly_chart(plot_calories_for_user(df_calories_to_plot), use_container_width=True)

                else:
                    # Show all days by default
                    fig = plot_total_intensity_hourly(df_activity_person, person_id)
                    st.plotly_chart(fig, use_container_width=True)
                    st.plotly_chart(plot_calories_for_user(df_calories_person))
    
if __name__ == "__main__":
    main()