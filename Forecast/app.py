import pandas as pd
from prophet import Prophet
import streamlit as st

# Streamlit file uploader for Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load the data
    df = pd.read_excel(uploaded_file)

    # Ensure 'Date' and 'Revenue' columns are present
    if 'Date' not in df.columns or 'Revenue' not in df.columns:
        st.error("Excel file must contain 'Date' and 'Revenue' columns.")
    else:
        # Prepare the dataframe for Prophet
        df['ds'] = pd.to_datetime(df['Date'])  # 'Date' column should be in datetime format
        df['y'] = df['Revenue']  # 'Revenue' column should be your target variable
        
        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(df)

        # Create a future dataframe for predictions
        future = model.make_future_dataframe(df, periods=12, freq='M')

        # Make predictions
        forecast = model.predict(future)

        # Plot the forecast
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        st.write("Forecast Plot:")
        fig = model.plot(forecast)
        st.pyplot(fig)
 
