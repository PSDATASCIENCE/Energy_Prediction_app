import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("linear_regressor_model.pkl", "rb") as file:
    lin_reg = pickle.load(file)

# Streamlit UI
st.title("Energy Consumption Prediction")

# User inputs
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
monthly_cbs = st.number_input("Monthly CBS", min_value=0, max_value=50, step=1)
total = st.number_input("TOTAL", min_value=0.0, max_value=100000.0, step=1.0)
direct_activities = st.number_input("Total Direct Activities", min_value=0.0, max_value=10000.0, step=1.0)
lag_1 = st.number_input("Lag_1", min_value=0.0, max_value=10000.0, step=1.0)

# Convert user input into a NumPy array
user_input = np.array([[temp, humidity, monthly_cbs, total, direct_activities, lag_1]])

# Prediction button
if st.button("Predict Energy Consumption"):
    prediction = lin_reg.predict(user_input)[0]
    
    st.subheader("Predicted Energy Consumption (kWh)")
    st.write(f"{prediction:.2f} kWh")
