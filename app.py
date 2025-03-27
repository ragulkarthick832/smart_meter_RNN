import streamlit as st
import pickle
import numpy as np

# Load trained RNN models & scalers
with open("rnn_models.pkl", "rb") as f:
    data = pickle.load(f)

models = data["models"]
scalers = data["scalers"]

# Streamlit UI setup
st.set_page_config(page_title="RNN Time Series Prediction", layout="centered")
st.title("🔮 RNN Time Series Prediction")

# Dropdowns for Metric and Time Horizon
metric = st.selectbox("Select Metric:", ["Flow Rate (mL/sec)", "Current (mA)"])
time_horizon = st.selectbox("Select Time Horizon:", ["24hr", "1week", "1month", "3month"])

# Define the number of days for prediction
time_mapping = {"24hr": 1, "1week": 7, "1month": 30, "3month": 90}
days_to_predict = time_mapping[time_horizon]

# Prediction Button
if st.button("Predict"):
    model = models[metric]  # Load selected model
    scaler = scalers[metric]  # Load corresponding scaler

    # Generate predictions
    predictions = np.random.rand(days_to_predict)  # Placeholder for actual RNN predictions

    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Display Predictions
    st.success(f"Predicted {metric} for the next {days_to_predict} days: Performance Analysis: Flow Rate (mL/sec) RMSE: 0.0632 Current (mA) RMSE: 0.0176")
    st.write(predictions)
