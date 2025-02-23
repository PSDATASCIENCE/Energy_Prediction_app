import streamlit as st
import pickle
import numpy as np

# Load the trained model and PCA (if used)
with open("linear_regressor_model.pkl", "rb") as file:
    lin_reg = pickle.load(file)

with open("pca_model.pkl", "rb") as file:
    pca = pickle.load(file)  # Load PCA model if used

# Streamlit UI
st.title("🔋 Energy Consumption Prediction App")

st.markdown("### Enter the required input values to predict energy consumption.")

# User inputs
temp = st.number_input("🌡 Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1, value=25.0)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, value=50.0)
monthly_cbs = st.number_input("🏭 Monthly CBS", min_value=0, max_value=50, step=1, value=20)
total = st.number_input("⚡ TOTAL Energy (kWh)", min_value=0.0, max_value=100000.0, step=1.0, value=5000.0)
direct_activities = st.number_input("🛠 Total Direct Activities", min_value=0.0, max_value=10000.0, step=1.0, value=2000.0)
lag_1 = st.number_input("📊 Lag_1 (Previous Day Energy)", min_value=0.0, max_value=10000.0, step=1.0, value=4500.0)

# Convert user input into NumPy array
user_input = np.array([[temp, humidity, monthly_cbs, total, direct_activities, lag_1]])

# Apply PCA transformation if necessary
user_input_pca = pca.transform(user_input)  # Only use if PCA was applied in training

# Prediction button
if st.button("🔮 Predict Energy Consumption"):
    prediction = lin_reg.predict(user_input_pca)[0]
    
    st.subheader("⚡ Predicted Energy Consumption (kWh)")
    st.success(f"{prediction:.2f} kWh")
