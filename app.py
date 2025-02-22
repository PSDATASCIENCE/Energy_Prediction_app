#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pickle
import numpy as np

# Set a custom title and favicon for your app
st.set_page_config(page_title="Energy Consumption Prediction", page_icon="⚡", layout="centered")

# Load the trained model
model_path = "linear_regressor_model.pkl"
  # Ensure the correct model path
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("🔋 Energy Consumption Prediction App")
st.write("""
    Welcome to the Energy Consumption Prediction App! 
    Provide the following inputs to predict the total energy consumption (kWh).
    """)

# User inputs with improved labels
st.subheader("Input Parameters")
temp = st.number_input("Temperature", min_value=10.0, max_value=50.0, value=25.0, step=0.5)
humidity = st.number_input("Humidity", min_value=10.0, max_value=100.0, value=50.0, step=1.0)
monthly_cbs = st.number_input("Monthly CBS", min_value=10, max_value=25, value=18, step=1)
total = st.number_input("Total Direct Activities (TOTAL)", min_value=6000.0, max_value=8000.0, value=7000.0, step=100.0)
total_direct_activities = st.number_input("Total Direct Activities", min_value=100000.0, max_value=500000.0, value=300000.0, step=5000.0)
lag_1 = st.number_input("Lag 1 Value (Lag_1)", min_value=150000.0, max_value=200000.0, value=180000.0, step=1000.0)


# Prediction
data = np.array([['temp','humidity', 'Monthly CBS','TOTAL','Total Direct Activities','Lag_1']])

# Button to trigger prediction
if st.button("🔮 Predict Energy Consumption"):
    prediction = model.predict(data)
    st.success(f"✨ **Predicted Energy Consumption:** {prediction[0]:.2f} kWh")

# Footer with contact info or additional details
st.markdown("""
    ---
    ⚡ **Note:** This app predicts energy consumption based on input parameters.
    For inquiries, contact us at **energy@prediction.com**.
""")

# Additional styling with CSS to enhance the visual experience
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
