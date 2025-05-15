import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib

# --- Load Model, Scaler, and Features ---
@st.cache_data
def load_artifacts():
    model = load_model("lstm_autoencoder_model.h5", compile=False)
    model.compile(optimizer='adam', loss='mse')  # Recompile after loading
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")  # List of selected feature columns
    return model, scaler, features

model, scaler, features = load_artifacts()

# --- UI Title ---
st.title("LSTM Autoencoder Anomaly Detection")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read and show uploaded data
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(data.head())

    # Check if required features exist
    if all(f in data.columns for f in features):
        # Preprocess
        data_seq = data[features].copy()
        data_scaled = scaler.transform(data_seq)
        sequence_length = model.input_shape[1]
        
        # Create sequences
        def create_sequences(data, seq_length):
            sequences = []
            for i in range(len(data) - seq_length + 1):
                seq = data[i:i+seq_length]
                sequences.append(seq)
            return np.array(sequences)

        X = create_sequences(data_scaled, sequence_length)

        # Predict
        reconstructed = model.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=(1, 2))
        
        # Threshold (customize or make dynamic)
        threshold = st.slider("Anomaly Threshold", float(np.min(mse)), float(np.max(mse)), float(np.percentile(mse, 95)))

        anomalies = mse > threshold
        anomaly_df = pd.DataFrame({
            "sequence_start_index": np.arange(len(mse)),
            "reconstruction_error": mse,
            "is_anomaly": anomalies
        })

        st.subheader("Anomaly Detection Results")
        st.write(anomaly_df.head())

        st.line_chart(anomaly_df["reconstruction_error"])
        st.markdown(f"**Anomalies Detected:** {anomalies.sum()}")

    else:
        st.error("Uploaded data does not contain the required features.")
