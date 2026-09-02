import os
import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Download model from Google Drive if not present locally
MODEL_PATH = "aep_smart_energy_model.pkl"
FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"  # Put your real File ID here inside quotes

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()

# 2. Streamlit Web App Interface
st.title("Smart Energy Predictor")
st.write("Predict hourly energy consumption using trained machine learning model.")

# User Inputs for Features
hour = st.slider("Hour of Day", 0, 23, 12)
dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
quarter = st.selectbox("Quarter of Year", [1, 2, 3, 4])
month = st.slider("Month", 1, 12, 6)
year = st.number_input("Year", min_value=2004, max_value=2030, value=2024)
lag_1 = st.number_input("Lag 1 Load (MW)", value=15000.0)

# Predict Button
if st.button("Predict Consumption"):
    features = np.array([[hour, dayofweek, quarter, month, year, lag_1]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Energy Load: {prediction:.2f} MW")
