import streamlit as st
import pickle
import numpy as np

# Load the trained PCA model and Linear Regression model
with open("pca_model.pkl", "rb") as pca_file:
    pca = pickle.load(pca_file)  # Load the PCA transformer

with open("linear_regressor_model.pkl", "rb") as model_file:
    lin_reg_pca = pickle.load(model_file)  # Load the trained model

# Streamlit UI
st.title("Energy Consumption Prediction with PCA")

# User inputs
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
monthly_cbs = st.number_input("Monthly CBS", min_value=0, max_value=50, step=1)
total = st.number_input("TOTAL", min_value=0.0, max_value=100000.0, step=1.0)
direct_activities = st.number_input("Total Direct Activities", min_value=0.0, max_value=10000.0, step=1.0)
lag_1 = st.number_input("Lag_1", min_value=0.0, max_value=10000.0, step=1.0)

# Convert user input into a NumPy array
user_input = np.array([[temp, humidity, monthly_cbs, total, direct_activities, lag_1]])

# 🛑 FIX: Apply PCA transformation before making predictions
user_input_pca = pca.transform(user_input)  # Transform user input using PCA

# Prediction button
if st.button("🔮 Predict Energy Consumption"):
    prediction = lin_reg_pca.predict(user_input_pca)[0]  # Predict using PCA-transformed input

    st.subheader("⚡ Predicted Energy Consumption (kWh)")
    st.success(f"{prediction:.2f} kWh")
