import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables securely
load_dotenv()

# Load the Groq API Key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if the API key is missing
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Define a function to handle the file upload
def load_data(file):
    data = pd.read_excel(file)
    if "Date" not in data.columns or "Revenue" not in data.columns:
        st.error("The Excel file must contain 'Date' and 'Revenue' columns.")
        st.stop()
    return data

# Set up Streamlit UI
st.set_page_config(page_title="Revenue Forecasting with Prophet", page_icon="📊", layout="wide")

# Header and file upload prompt
st.title("🚀 AI-Powered Revenue Forecasting")
st.markdown("Upload your Excel file containing 'Date' and 'Revenue' columns to get a forecast.")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    # Load the data
    df = load_data(uploaded_file)
    st.subheader("📊 Data Overview")
    st.write(df.head())

    # Ensure the 'Date' column is in datetime format and rename columns
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Coerce errors to NaT (Not a Time)
    df = df.rename(columns={'Date': 'ds', 'Revenue': 'y'})  # Prophet expects these columns

    # Drop rows with invalid or missing dates or revenue
    df = df.dropna(subset=['ds', 'y'])

    # Check for empty or insufficient data
    if df.shape[0] < 2:
        st.error("Not enough data points to perform forecasting. Ensure you have multiple data points.")
        st.stop()

    # Verify that the 'ds' column is in datetime format and 'y' column is numeric
    if not pd.api.types.is_datetime64_any_dtype(df['ds']):
        st.error("The 'Date' column should be in datetime format.")
        st.stop()

    if not pd.api.types.is_numeric_dtype(df['y']):
        st.error("The 'Revenue' column should be numeric.")
        st.stop()

    # Fit the Prophet model
    model = Prophet()

    try:
        # Create future dataframe for predictions (365 days ahead)
        future = model.make_future_dataframe(df, periods=365)  # Ensure the correct parameter passing
        
        # Ensure future dates are of type datetime
        future['ds'] = pd.to_datetime(future['ds'])
        
        # Generate the forecast
        forecast = model.predict(future)

        # Display forecasted data
        st.subheader("📅 Forecasting Results")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Plot forecast
        st.subheader("📈 Forecast Plot")
        fig = model.plot(forecast)
        st.pyplot(fig)

        # Plot components (trend, holidays, etc.)
        st.subheader("🧩 Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # Prepare data for Groq API call
        data_for_ai = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')

        # Generate AI-based Commentary using Groq
        st.subheader("🤖 AI-Generated Commentary")
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        You are a revenue forecasting expert. Here is the forecast data for the next year:

        - Date and predicted revenue
        - Lower and upper bounds of the predictions

        Provide a comprehensive analysis that includes:
        - Key insights from the data.
        - Areas of concern and key drivers for revenue growth/decline.
        - Actionable recommendations based on the forecast.

        Here is the data for the forecast: {data_for_ai}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a revenue forecasting expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",  # Make sure to choose the correct model based on your account
        )

        ai_commentary = response.choices[0].message.content

        # Display AI Commentary
        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
        st.subheader("📖 AI-Generated Commentary")
        st.write(ai_commentary)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while generating the forecast: {e}")
