import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
import os
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# **🎨 Streamlit UI Styling**
st.set_page_config(page_title="AI Forecasting with Prophet", page_icon="📊", layout="wide")

# **Upload Excel File**
st.title("📈 AI-Driven Revenue Forecasting")
st.subheader("Upload your Excel file with 'Date' and 'Revenue' columns")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load data from Excel file
    data = pd.read_excel(uploaded_file)

    # Ensure the dataset contains required columns
    if 'Date' not in data.columns or 'Revenue' not in data.columns:
        st.error("The dataset must contain 'Date' and 'Revenue' columns.")
        st.stop()

    # Convert Date column to datetime type
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

    # Rename columns for Prophet compatibility
    data = data.rename(columns={'Date': 'ds', 'Revenue': 'y'})

    # **Fit Prophet Model**
    model = Prophet()
    model.fit(data)

    # Make future dataframe for predictions
    future = model.make_future_dataframe(data, periods=365)  # Forecast for the next year
    forecast = model.predict(future)

    # **Plot the Forecast**
    st.subheader("🔮 Forecasted Revenue")
    fig = model.plot(forecast)
    st.pyplot(fig)

    # **Plot the Forecast Components**
    st.subheader("📊 Forecast Components (Trend, Weekly, Yearly)")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # **Display Forecast Data**
    st.subheader("📊 Forecast Data")
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    st.write(forecast_df.tail())

    # **AI Forecast Summary (Optional - Use Groq or another service for this)**
    # If you're integrating the Groq AI commentary as in the initial code, you can use that here as well
    # Replace below with your actual AI processing logic

    st.subheader("🤖 AI Commentary (Optional)")
    # If you want to include AI-generated commentary, here you would insert code similar to the one in the original snippet.
    # ai_commentary = generate_ai_commentary(forecast_df)  # Replace with your function
    # st.write(ai_commentary)
