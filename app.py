#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="Energy Consumption Prediction", page_icon="⚡", layout="centered")

# Load trained model
model_path = "linear_regressor_model.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("🔋 Energy Consumption Prediction App")
st.write("""
    Welcome to the Energy Consumption Prediction App! 
    Provide the following inputs to predict the total energy consumption (kWh).
""")

# User inputs
st.subheader("Input Parameters")

temp = st.number_input("Temperature", min_value= 15, max_value=50,value=750.0, step=5.0)
humidity = st.number_input("Humidity", min_value=15, max_value=200, value=1800.0, step=50.0)
monthly_cbs = st.number_input("Monthly CBS", min_value=14, max_value=26, value=18, step=1)
total = st.number_input("Total Direct Activities (TOTAL)", min_value=1800.0, max_value=7500.0, value=7000.0, step=100.0)
total_direct_activities = st.number_input("Total Direct Activities", min_value=11000.0, max_value=28000.0, value=25000.0, step=500.0)
lag_1 = st.number_input("Lag 1 Value (Lag_1)", min_value=128000.0, max_value=204000.0, value=180000.0, step=1000.0)


# Prediction button
# Button to trigger prediction
if st.button("🔮 Predict Energy Consumption"):
    data = np.array([[temp, humidity, monthly_cbs, total, total_direct_activities, lag_1]])  # Numeric input array
   



    # Predict using model
    prediction = model.predict(data)

    # Display result
    st.success(f"✨ **Predicted Energy Consumption:** {prediction[0]:.2f} kWh")

# Footer
st.markdown("""
    ---
    ⚡ **Note:** This app predicts energy consumption based on input parameters.
    For inquiries, contact us at **energy@prediction.com**.
""")

# Custom CSS for better styling
st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50; 
            color: white; 
            font-size: 16px; 
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stMarkdown {
            font-size: 16px;
            line-height: 1.6;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)
