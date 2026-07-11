import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(page_title="California House Price Predictor", layout="centered")

# Load the saved model components safely
@st.cache_resource
def load_model_objects():
    with open('polynomial_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('poly_features.pkl', 'rb') as f:
        poly = pickle.load(f)
    return model, scaler, poly

try:
    model, scaler, poly = load_model_objects()
except FileNotFoundError:
    st.error("Error: Model files (.pkl) not found in the current directory. Please make sure they are in the same folder as app.py.")
    st.stop()

# App UI Header
st.title("California House Price Prediction App")
st.markdown("Enter the region's metrics below to estimate the median house value.")
st.markdown("---")

# Input fields mapping exactly to your dataset columns
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (in tens of thousands, e.g., 3.5 = ₹35,000)", min_value=0.0, max_value=15.0, value=3.5, step=0.1)
    house_age = st.number_input("Median House Age", min_value=1.0, max_value=100.0, value=28.0, step=1.0)
    ave_rooms = st.number_input("Average Rooms per Household", min_value=1.0, max_value=50.0, value=5.0, step=0.1)
    ave_bedrms = st.number_input("Average Bedrooms per Household", min_value=1.0, max_value=20.0, value=1.0, step=0.1)

with col2:
    population = st.number_input("Block Population", min_value=1.0, max_value=100000.0, value=1400.0, step=10.0)
    ave_occup = st.number_input("Average House Occupancy", min_value=1.0, max_value=100.0, value=3.0, step=0.1)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.5, step=0.01)
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-119.5, step=0.01)

st.markdown("---")

# Predict button
if st.button(" Predict House Value", type="primary"):
    # 1. Structure the features into a DataFrame with the correct column names
    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
    input_df = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]], columns=feature_names)

    # 2. Apply identical transformations
    input_scaled = scaler.transform(input_df)
    input_poly = poly.transform(input_scaled)

    # 3. Predict value
    prediction = model.predict(input_poly)[0]

    # Format target output (The target column MedHouseVal represents $100,000s)
    actual_price = prediction * 100000

    if actual_price < 0:
        st.warning(" The model predicted an extreme negative valuation based on these out-of-bounds parameters. Try entering more realistic regional coordinates or metrics.")
    else:
        st.success(f"### Estimated Median House Value: **₹{actual_price:,.2f}**")
