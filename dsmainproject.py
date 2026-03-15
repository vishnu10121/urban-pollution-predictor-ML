import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("final_pollution_model.pkl")

st.title("Urban Air Pollution Predictor")

st.write("Enter environmental values to predict PM2.5 level")

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
wind_speed = st.number_input("Wind Speed")
pressure = st.number_input("Pressure")
visibility = st.number_input("Visibility")
hour = st.number_input("Hour of Day")
month = st.number_input("Month")

if st.button("Predict Pollution"):
    
    input_data = np.array([[temperature, humidity, wind_speed, pressure, visibility, hour, month]])
    
    prediction = model.predict(input_data)

    st.success(f"Predicted PM2.5 Level: {prediction[0]:.2f}")
