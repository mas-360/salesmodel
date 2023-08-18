# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 08:34:45 2023

@author: 27823
"""

import streamlit as st
import pandas as pd
from fbprophet import Prophet  # Import Prophet from the fbprophet library

def main():
    st.title("Sales Model")
    st.sidebar.header("User Input")

    # Let user upload a CSV file
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = process_uploaded_file(uploaded_file)
        predictions = generate_forecast(df)
        display_forecast(predictions)

def process_uploaded_file(uploaded_file):
    # Read the uploaded CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    return df

def generate_forecast(df):
    # Prepare the DataFrame for Prophet
    df = df.rename(columns={"date_column": "ds", "sales_column": "y"})  # Adjust column names
    df["ds"] = pd.to_datetime(df["ds"])  # Convert the "ds" column to datetime
    
    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df)
    
    # Create a DataFrame with future dates for forecasting
    future = model.make_future_dataframe(periods=30)  # Forecast for the next 30 days
    
    # Generate forecasts
    forecast = model.predict(future)
    return forecast

def display_forecast(forecast):
    st.write("## Below are our sales predictions:")
    window = st.slider("Forecast window (days)")
    st.line_chart(forecast[["ds", "yhat"]].set_index("ds").tail(window))  # Display the forecasted values

if __name__ == "__main__":
    main()

