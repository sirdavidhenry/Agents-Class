import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI Setup
st.set_page_config(page_title="AI Forecasting with Prophet", page_icon="📈", layout="wide")
st.title("📈 AI-Driven Revenue Forecasting")
st.subheader("Upload an Excel file with 'Date' and 'Revenue' columns")

# File Upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### Preview of Uploaded Data", df.head())

    # Data Preprocessing
    df.columns = [col.lower() for col in df.columns]
    if "date" not in df.columns or "revenue" not in df.columns:
        st.error("Excel file must contain 'Date' and 'Revenue' columns.")
        st.stop()
    
    df = df.rename(columns={"date": "ds", "revenue": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    
    # Prophet Model Training
    model = Prophet()
    model.fit(df)
    
    # Future Prediction
    future = model.make_future_dataframe(periods=30)  # Predict next 30 days
    forecast = model.predict(future)
    
    # Plot Forecast
    st.write("### Forecasted Revenue")
    fig, ax = plt.subplots(figsize=(10, 5))
    model.plot(forecast, ax=ax)
    st.pyplot(fig)
    
    # Generate AI Commentary
    st.subheader("🤖 AI-Generated Forecast Analysis")
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    You are a financial analyst. Analyze the revenue forecast and provide:
    - Key trends observed.
    - Potential risks and opportunities.
    - Actionable insights for business growth.
    Here is the forecast data:
    {forecast.to_json()}
    """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert financial analyst."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )
    
    ai_analysis = response.choices[0].message.content
    st.write(ai_analysis)
