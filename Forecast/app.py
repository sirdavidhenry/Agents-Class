import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from dotenv import load_dotenv

# Load API key securely (for any external APIs, such as Groq if needed in the future)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit Page Configuration
st.set_page_config(page_title="Revenue Forecasting with Prophet", page_icon="📊", layout="wide")

# Streamlit UI Elements
st.title("Revenue Forecasting Using Prophet")
st.subheader("Upload an Excel file with 'Date' and 'Revenue' columns")

# File Upload
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    # Load the uploaded Excel file into a pandas DataFrame
    df = pd.read_excel(uploaded_file)
    
    # Check if necessary columns are present
    if 'Date' not in df.columns or 'Revenue' not in df.columns:
        st.error("The Excel file must contain 'Date' and 'Revenue' columns.")
    else:
        # Preprocess the data
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' column is in datetime format
        df = df[['Date', 'Revenue']]  # Ensure only necessary columns are included

        # Rename columns for Prophet
        df.rename(columns={'Date': 'ds', 'Revenue': 'y'}, inplace=True)

        # Forecasting with Prophet
        model = Prophet()
        model.fit(df)

        # Create future dataframe for prediction
        future = model.make_future_dataframe(df, periods=12, freq='M')  # Forecasting for 12 months
        forecast = model.predict(future)

        # Plot the forecast
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Display the forecast data
        st.subheader("Forecast Data")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # Option to download the forecasted data as Excel
        forecast_file = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast_file.to_excel("forecasted_data.xlsx", index=False)
        with open("forecasted_data.xlsx", "rb") as f:
            st.download_button("Download Forecast Data", f, file_name="forecasted_data.xlsx")
