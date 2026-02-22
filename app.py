import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Weather ML Dashboard", layout="wide")

st.title("🌦 Weather Prediction & Risk Dashboard")

# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_models():
    temp_model = joblib.load("final_temperature_model.pkl")
    rain_model = joblib.load("final_rain_model.pkl")
    return temp_model, rain_model

temp_model, rain_model = load_models()

# ===============================
# Sidebar Navigation
# ===============================
option = st.sidebar.selectbox(
    "Select Task",
    [
        "Temperature Prediction",
        "Rain Prediction",
    ]
)

# ===============================
# Common Input Fields
# ===============================
def user_inputs():
    maxtempC = st.number_input("Max Temperature (°C)", value=35.0)
    mintempC = st.number_input("Min Temperature (°C)", value=25.0)
    sunHour = st.number_input("Sun Hours", value=8.0)
    uvIndex = st.number_input("UV Index", value=7.0)
    moon_illumination = st.number_input("Moon Illumination", value=50.0)
    DewPointC = st.number_input("Dew Point (°C)", value=22.0)
    FeelsLikeC = st.number_input("Feels Like (°C)", value=36.0)
    WindChillC = st.number_input("Wind Chill (°C)", value=30.0)
    WindGustKmph = st.number_input("Wind Gust (Kmph)", value=15.0)
    cloudcover = st.number_input("Cloud Cover (%)", value=40.0)
    humidity = st.number_input("Humidity (%)", value=60.0)
    pressure = st.number_input("Pressure (mb)", value=1012.0)
    visibility = st.number_input("Visibility (km)", value=10.0)
    windspeedKmph = st.number_input("Wind Speed (Kmph)", value=12.0)

    data = pd.DataFrame({
        'maxtempC': [maxtempC],
        'mintempC': [mintempC],
        'sunHour': [sunHour],
        'uvIndex': [uvIndex],
        'moon_illumination': [moon_illumination],
        'DewPointC': [DewPointC],
        'FeelsLikeC': [FeelsLikeC],
        'WindChillC': [WindChillC],
        'WindGustKmph': [WindGustKmph],
        'cloudcover': [cloudcover],
        'humidity': [humidity],
        'pressure': [pressure],
        'visibility': [visibility],
        'windspeedKmph': [windspeedKmph]
    })

    return data

# ===============================
# Temperature Prediction
# ===============================
if option == "Temperature Prediction":
    st.subheader("🌡 Predict Temperature")

    input_df = user_inputs()

    if st.button("Predict Temperature"):
        prediction = temp_model.predict(input_df)[0]
        st.success(f"Predicted Temperature: {prediction:.2f} °C")

# ===============================
# Rain Prediction
# ===============================
if option == "Rain Prediction":
    st.subheader("🌧 Predict Rain (Yes/No)")

    input_df = user_inputs()

    if st.button("Predict Rain"):
        prediction = rain_model.predict(input_df)[0]

        if prediction == 1:
            st.error("Rain Expected 🌧")
        else:
            st.success("No Rain Expected ☀")