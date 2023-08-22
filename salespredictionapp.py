# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 08:34:45 2023

@author: 27823
"""

import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="Sales Prediction Model", page_icon="bar-chart-line" ,layout="centered")

def main():
    img_logo = Image.open("images/Capture.PNG")

    st.header("Below are our sales predictions:")
    st.sidebar.image(img_logo)
    st.sidebar.header("User Input")
    

    # Let user upload a CSV file
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    # Add input for forecast days (limited to a maximum of 31 days)
    forecast_days = st.sidebar.slider("Forecast for how many days?", 1, 31, 30)

    # Add radio button for forecast method
    forecast_method = st.sidebar.radio("Select forecast method:", ("ARIMA", "Holt-Winters"))

    if uploaded_file is not None:
        df = process_uploaded_file(uploaded_file)
        if forecast_method == "ARIMA":
            predictions = generate_arima_forecast(df, forecast_days)
        else:
            predictions = generate_holt_winters_forecast(df, forecast_days)
        display_forecast(df, predictions, forecast_days, forecast_method)

#Info
with st.expander(
    "**Time series forecasting model explainer:**", expanded=False
):
    st.write(""" Apply the right model according to your data:  
             **ARIMA** focuses on modeling the relationships between past observations in time series data and can handle various data patterns that might not be predictable and consistent.  
             **Holt-Winters** emphasizes capturing trend and seasonality in time series data and is particularly useful for data with predictable and consistent seasonal patterns.
             """)
    
def process_uploaded_file(uploaded_file):
    # Read the uploaded CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    return df

def generate_arima_forecast(df, forecast_days):
    # Prepare the DataFrame
    df['date_column'] = pd.to_datetime(df['date_column'])
    df.set_index('date_column', inplace=True)
    
    # Resample the data to the desired frequency (e.g., daily)
    resampled_df = df['sales_column'].resample('D').sum()
    
    # Perform time series decomposition (trend, seasonal, residual)
    decomposition = sm.tsa.seasonal_decompose(resampled_df, model='additive')
    
    # Extract the trend component and perform forecasting
    trend = decomposition.trend.dropna()
    model = sm.tsa.ARIMA(trend, order=(5, 1, 0))  # Example ARIMA order (p, d, q)
    fitted_model = model.fit()
    
    # Forecast for the specified number of days
    forecast = fitted_model.forecast(steps=forecast_days)
    
    return forecast

def generate_holt_winters_forecast(df, forecast_days):
    # Prepare the DataFrame
    df['date_column'] = pd.to_datetime(df['date_column'])
    df.set_index('date_column', inplace=True)
    
    # Resample the data to the desired frequency (e.g., daily)
    resampled_df = df['sales_column'].resample('D').sum()
    
    # Initialize Holt-Winters model
    model = sm.tsa.ExponentialSmoothing(resampled_df, seasonal='add', seasonal_periods=7)
    
    # Fit the model
    fitted_model = model.fit()
    
    # Forecast for the specified number of days
    forecast = fitted_model.forecast(steps=forecast_days)
    
    return forecast

def display_forecast(df, forecast, forecast_days, forecast_method):
    #st.write("## Below are our sales predictions:")
    
    # Get the last date in the original sales data
    last_date = df.index[-1]
    
    # Create forecast date range
    forecast_dates = pd.date_range(start=last_date, periods=forecast_days, freq='D')
    
    # Combine sales data and forecast into a single DataFrame
    combined_df = pd.DataFrame({'Date': df.index.tolist() + forecast_dates.tolist(),
                                'Sales': df['sales_column'].tolist() + forecast.tolist(),
                                'Type': ['Actual'] * len(df) + ['Forecast'] * len(forecast_dates)})
    
    # Create a Plotly figure
    fig = px.line(combined_df, x='Date', y='Sales', color='Type', line_dash='Type')
    fig.update_layout(title='Sales Forecast with range slider', xaxis_title='Date', yaxis_title='Sales')
    fig.update_xaxes(rangeslider_visible=True)
    # Print sales prediction information
    st.write(f"Sales prediction for {forecast_days} days using {forecast_method} method is {forecast.sum():.2f}")
    
    # Show the Plotly figure using st.plotly_chart
    st.write("##")
    st.plotly_chart(fig, use_container_width=True)

#Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
local_css("style/style.css") 
       
st.markdown('---')

# --- CONTACT FORM ---
st.header("Unlock the potential of your data today!")
st.write("Take the first step towards getting your very own data app by completing the simple form below.")

contact_form = """
<form action="https://formsubmit.co/masinsight360@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="email" name="email" placeholder="Your email" required>
    <button type="submit">Send</button>
</form>
"""
# Use columns to organize the content
col1, col2 = st.columns(2)

with col1:
    st.markdown(contact_form, unsafe_allow_html=True)

with col2:
    pass
  


if __name__ == "__main__":
    main()
























