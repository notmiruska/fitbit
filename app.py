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
from modules.data import *
from modules.plots import *
from modules.stats import *

@st.cache_data(show_spinner=False)
def load_data():
    users = load_users("data/daily_activity.csv")
    
    connection = sqlite3.connect("data/fitbit_database.db")

    minute_sleep = load_sleep_data(connection)
    df_heartrate = load_heartrate_data(connection)

    connection.close()

    return users, minute_sleep, df_heartrate


def select_user(users):
    user_options = [""] + list(users)  # "" will be the default placeholder
    selected_user = st.sidebar.selectbox("Select a user", options=user_options, index=0)

    return selected_user


def display_general_statistics(users):
    st.title("Fitbit Dashboard")
    st.header("General Statistics")
    st.markdown("This page displays general insights across users from Fitbit data."
                "For user-specific statistics,  select a user ID from the sidebar!")
    
    st.metric("Number of Fitbit Users", len(users))
    

def display_sleep_tab(df_sleep_person, person_id):
    if df_sleep_person.empty:
        st.warning("No sleep data available.")
        return
    else:
        main_sleep, naps, sleep_hours_line, merged_df = process_sleep_sessions(
            df_sleep_person, nap_threshold=3
        )

        avg_hr = round(sleep_hours_line.mean(), 1)
        st.metric("Mean sleep hours per night", f"{avg_hr} hours")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            fig_sleep = plot_sleep_timeline(main_sleep, naps, sleep_hours_line, merged_df, person_id)
            st.plotly_chart(fig_sleep, use_container_width=True)
    

def display_heartrate_tab(df_hr_person, person_id):
    if df_hr_person.empty:
        st.warning("No heartrate data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        avg_hr = round(df_hr_person["Value"].mean(), 1)
        st.metric("Average Heartrate", f"{avg_hr} bpm")
    with col2:
        max_hr = df_hr_person["Value"].max()
        st.metric("Max Heartrate", f"{max_hr} bpm")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Last 24 hours")
        fig_hr_24 = last_24_hours_plot(df_hr_person, person_id)
        st.plotly_chart(fig_hr_24, width="stretch")

    with col2:
        st.subheader("Daily statistics")
        fig_hr_stats = plot_stats_heartrate(df_hr_person, person_id)
        st.plotly_chart(fig_hr_stats, width="stretch")


def main():
    st.set_page_config(layout="wide")

    # ---------------------
    # Load data
    # ---------------------
    with st.spinner("Loading Fitbit Dashboard..."):
        users, minute_sleep, df_heartrate = load_data()

    # ---------------------
    # Sidebar: Select user
    # ---------------------
    selected_user = select_user(users)

    # ---------------------
    # General statistics
    # ---------------------
    if selected_user == "":
        display_general_statistics(users)

    # ---------------------
    # User-specific statistics
    # ---------------------
    if selected_user != "":
        person_id = selected_user

        st.title("Fitbit User Dashboard")
        st.subheader(f"User {person_id}")

        # Filter data once
        df_sleep_person = minute_sleep[minute_sleep["Id"] == person_id]
        df_hr_person = df_heartrate[df_heartrate["Id"] == person_id]

        # Initialize tabs
        tab1, tab2 = st.tabs(["Sleep", "Heartrate"])

        # Sleep tab
        with tab1:
            display_sleep_tab(df_sleep_person, person_id)
        
        # Heartrate tab
        with tab2:
            display_heartrate_tab(df_hr_person, person_id)


if __name__ == '__main__':
    main()
