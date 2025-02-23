import streamlit as st
import pickle
import numpy as np

# Load the trained PCA model and Linear Regression model
with open("pca_model.pkl", "rb") as pca_file:
    pca = pickle.load(pca_file)  # Load the trained PCA transformer

with open("linear_regressor_model.pkl", "rb") as model_file:
    lin_reg_pca = pickle.load(model_file)  # Load the trained PCA-based model

# Streamlit UI
st.title("Energy Consumption Prediction with PCA")

# User inputs
# User Inputs
temp = st.number_input("Temperature (°C)", min_value=10.0, max_value=50.0, step=5.0)
humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, step=10.0)
monthly_cbs = st.number_input("Monthly CBS", min_value=5, max_value=50, step=1)
total = st.number_input("TOTAL", min_value=0.0, max_value=100000.0, step=100.0)
direct_activities = st.number_input("Total Direct Activities", min_value=0.0, max_value=1000000.0, step=100.0)
lag_1 = st.number_input("Lag_1", min_value=0.0, max_value=10000000.0, step=10000.0)


# Convert user input into a NumPy array
user_input = np.array([[temp, humidity, monthly_cbs, total, direct_activities, lag_1]])

# Debugging: Check input shape before PCA transformation
st.write(f"📌 **Original Input Shape:** {user_input.shape}")  # Should be (1, 6)
st.write(f"🔢 **PCA Components Expected:** {pca.n_components_}")  # Should be 2

# 🛑 Check PCA transformation before prediction
try:
    user_input_pca = pca.transform(user_input)  # Transform user input using PCA

    # Debugging: Check output shape after PCA transformation
    st.write(f"✅ **PCA Output Shape:** {user_input_pca.shape}")  # Should be (1, 2)

    if user_input_pca.shape[1] != lin_reg_pca.n_features_in_:
        st.error(
            f"🚨 Feature mismatch! Model expects {lin_reg_pca.n_features_in_} features, "
            f"but PCA returned {user_input_pca.shape[1]}."
        )
    else:
        # Prediction button
        if st.button("🔮 Predict Energy Consumption"):
            prediction = lin_reg_pca.predict(user_input_pca)[0]  # Predict using PCA-transformed input
            st.subheader("⚡ Predicted Energy Consumption (kWh)")
            st.success(f"{prediction:.2f} kWh")

except Exception as e:
    st.error(f"❌ Error: {e}")
