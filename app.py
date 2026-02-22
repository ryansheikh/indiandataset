import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==================================
# Page Config
# ==================================
st.set_page_config(page_title="Weather ML Dashboard", layout="wide")
st.title("🌦 Weather Prediction & Risk Dashboard")

# ==================================
# Get Current Directory (IMPORTANT for deployment)
# ==================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_MODEL_PATH = os.path.join(BASE_DIR, "final_temperature_model.pkl")
RAIN_MODEL_PATH = os.path.join(BASE_DIR, "final_rain_model.pkl")

# ==================================
# Load Models (Cached)
# ==================================
@st.cache_resource
def load_models():
    try:
        temp_model = joblib.load(TEMP_MODEL_PATH)
        rain_model = joblib.load(RAIN_MODEL_PATH)
        return temp_model, rain_model
    except Exception as e:
        st.error("Model loading failed. Please check .pkl files.")
        st.error(str(e))
        return None, None

temp_model, rain_model = load_models()

# Stop app if models not loaded
if temp_model is None or rain_model is None:
    st.stop()

# ==================================
# Sidebar Navigation
# ==================================
option = st.sidebar.selectbox(
    "Select Task",
    [
        "Temperature Prediction",
        "Rain Prediction",
    ]
)

# ==================================
# User Input Function
# ==================================
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

# ==================================
# Temperature Prediction
# ==================================
if option == "Temperature Prediction":
    st.subheader("🌡 Predict Temperature")
    input_df = user_inputs()

    if st.button("Predict Temperature"):
        try:
            prediction = temp_model.predict(input_df)[0]
            st.success(f"Predicted Temperature: {prediction:.2f} °C")
        except Exception as e:
            st.error("Prediction failed")
            st.error(str(e))

# ==================================
# Rain Prediction
# ==================================
if option == "Rain Prediction":
    st.subheader("🌧 Predict Rain (Yes/No)")
    input_df = user_inputs()

    if st.button("Predict Rain"):
        try:
            prediction = rain_model.predict(input_df)[0]

            if prediction == 1:
                st.error("Rain Expected 🌧")
            else:
                st.success("No Rain Expected ☀")
        except Exception as e:
            st.error("Prediction failed")
            st.error(str(e))
