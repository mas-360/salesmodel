# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 08:34:45 2023

@author: 27823
"""

import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
from streamlit_extras.buy_me_a_coffee import button

st.set_page_config(page_title="Sales Prediction Model", page_icon="📊" ,layout="centered")

# Function to load Lottie animation from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#---LOAD ASSETS---
video_file1 = open("videos/salespredictv2.mp4", "rb")
video_bytes1 = video_file1.read()

# Load Lottie animation
lottie_coding = load_lottieurl("https://lottie.host/093aa35f-2e91-4872-82d7-260e4480c984/bl0ai0rKts.json")
st_lottie(lottie_coding, height=100, key="coding")
st.write("Refine business strategy, optimize resources, or set goals in a few clicks!")
#first_tab, second_tab = st.tabs(["📊 Sales Prediction"])
##with first_tab:
    # Print the recommendation
    #recommendation_style = "background-color: #e9c46a; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0px;"
    #title_style = "font-size: 20px; margin-bottom: 10px;"
    #content_style = "font-size: 15px;"
    #st.markdown(
        #f"<div style='{recommendation_style}'>"
        #f"<h2 style='{title_style}'>Apply the right model according to your data and Industry:</h2>"
        #f"<p style='{content_style}'><strong>ARIMA</strong> focuses on modeling the relationships between past observations in time series data and can handle various data patterns that might not be predictable and consistent. <strong>Suitable Industry:</strong> Finance and Economics, Healthcare, Manufacturing, and Marketing."
        #f"<p style='{content_style}'><strong>Holt-Winters</strong> emphasizes capturing trend and seasonality in time series data and is particularly useful for data with predictable and consistent seasonal patterns. <strong>Suitable Industry:</strong> Retail, Hospitality & Tourism, Agriculture and Energy."
        #"</div>",
        #unsafe_allow_html=True)
    #st.video(video_bytes1)

    
#with second_tab:    
def main():
    # Let user upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    # Add input for forecast days (limited to a maximum of 365 days)
    forecast_days = st.slider("Forecast for how many days?", 1, 365, 30)
    
    # Add radio button for forecast method
    forecast_method = st.radio("Select forecast method:", ("Holt-Winters", "ARIMA",))

    # Only display the seasonal period select box if ARIMA is selected
    if forecast_method == "ARIMA":
        m = st.selectbox("Select seasonal period:", (4, 7, 12), help="4 = Quarterly data with a yearly pattern, 7 = Daily Data with a weekly pattern, 12 = Monthly data with a yearly pattern")
    else:
        m = None

    st.subheader("Below are your sales predictions:")
    
    if uploaded_file is not None:
        df = process_uploaded_file(uploaded_file)
    else:
        fixed_file_path = "randomsalesdata.csv"
        df = pd.read_csv(fixed_file_path)
        forecast_method = "Holt-Winters"

    if forecast_method == "ARIMA":
        predictions = generate_arima_forecast(df, forecast_days, m)
    else:
        predictions = generate_holt_winters_forecast(df, forecast_days)
    display_forecast(df, predictions, forecast_days, forecast_method)
     
                
def process_uploaded_file(uploaded_file):
    # Read the uploaded CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    return df

def generate_arima_forecast(df, forecast_days, m):
    # Prepare the DataFrame
    df['date_column'] = pd.to_datetime(df['date_column'])
    df.set_index('date_column', inplace=True)

    # Resample the data to daily frequency
    resampled_df = df['sales_column'].resample('D').sum()

    # Fit a simple seasonal ARIMA (SARIMAX) model
    # For simplicity, we use order=(1,1,1) and seasonal_order=(1,1,1,m)
    model = sm.tsa.statespace.SARIMAX(
        resampled_df,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, m),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)

    # Forecast
    forecast = fitted_model.get_forecast(steps=forecast_days).predicted_mean

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
    #st.write("## Below are your sales predictions:")
    
    # Get the last date in the original sales data
    last_date = df.index[-1]
    
    # Create forecast date range
    forecast_dates = pd.date_range(start=last_date, periods=forecast_days, freq='D')
    
    # Combine sales data and forecast into a single DataFrame
    combined_df = pd.DataFrame({'Date': df.index.tolist() + forecast_dates.tolist(),
                                'Sales': df['sales_column'].tolist() + forecast.tolist(),
                                'Type': ['Actual'] * len(df) + ['Forecast'] * len(forecast_dates)})
    
    # Specify color mapping for "Actual" and "Forecast"
    color_mapping = {
        "Actual": "#E76F51",  # or any other color you prefer
        "Forecast": "#E9C46A"  # or any other color you prefer
    }
    
    # Create a Plotly figure
    fig = px.line(combined_df, x='Date', y='Sales', color='Type', line_dash='Type', color_discrete_map=color_mapping)
    fig.update_layout(title='Sales Forecast with range slider', xaxis_title='Date', yaxis_title='Sales')
    fig.update_xaxes(rangeslider_visible=True)
    
    # Print sales prediction information
    st.write(f"Sales prediction for {forecast_days} days using {forecast_method} method is {forecast.sum():,.2f}")
    
    # Show the Plotly figure using st.plotly_chart
    st.write("##")
    st.plotly_chart(fig, use_container_width=True)


       
if __name__ == "__main__":
    main()

st.markdown("---")

st.markdown("**Disclaimer:** It is important to note that all sales predictions are inherently uncertain and should not be relied upon as a guarantee of future performance. Actual sales may vary significantly from the predicted amount, depending on a number of factors, including changes in the market, competitive activity, and economic conditions.")
# Adding a footer
footer = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
            display: flex;
            justify-content: space-around;  # equally space the divs
        }
        .footer div {
            flex: 1;  # each div will take up an equal amount of space
            border: 1px solid #ccc;  # just to visualize the divs, can be removed
            padding: 10px;
        }
    </style>
    <div class="footer">
        <div>© Made by Anthea Sago</div>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)






























