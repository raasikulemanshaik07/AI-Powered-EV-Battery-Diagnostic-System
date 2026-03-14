import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import json
import os

# Set page configuration
st.set_page_config(page_title="Battery ML Dashboard", layout="wide")

st.title("🔋 Battery Health ML Diagnostic Dashboard")
st.markdown("Real-time monitoring and prediction dashboard based on Random Forest & LSTM models.")

@st.cache_resource
def load_models():
    # Load Models
    label_encoder = joblib.load("rf_label_encoder.pkl")
    rf_model = joblib.load("random_forest_fault_model.pkl")
    lstm_risk_model = load_model("lstm_risk_model.h5", compile=False)
    ae_model = load_model("lstm_autoencoder.h5", compile=False)
    ae_scaler = joblib.load("ae_scaler.pkl")
    anomaly_threshold = joblib.load("anomaly_threshold.pkl")
    return label_encoder, rf_model, lstm_risk_model, ae_model, ae_scaler, anomaly_threshold

try:
    with st.spinner("Loading ML Models..."):
        le, rf_model, lstm_risk_model, ae_model, ae_scaler, anomaly_threshold = load_models()
    st.sidebar.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar for Input
st.sidebar.header("Real-Time Sensor Data")
v = st.sidebar.slider("Voltage (V)", 2.0, 4.5, 3.1, 0.01)
i = st.sidebar.slider("Current (A)", -10.0, 10.0, 3.5, 0.1)
t = st.sidebar.slider("Temperature (°C)", 10.0, 100.0, 65.0, 0.5)
c_rate = st.sidebar.slider("C-Rate", 0.0, 5.0, 2.0, 0.1)
soc = st.sidebar.slider("State of Charge (SOC)", 0.0, 1.0, 0.5, 0.01)

if st.sidebar.button("Run Diagnostics", type="primary"):
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    # Random Forest Inference
    rf_input = np.array([[v, i, t, c_rate, soc]])
    rf_pred_encoded = rf_model.predict(rf_input)[0]
    rf_pred_label = le.inverse_transform([rf_pred_encoded])[0]
    
    # LSTM Sequence Data (using last available simulated data)
    try:
        lstm_df = pd.read_csv("../LSTM_Ready.csv")
        last_seq_id = lstm_df["sequence_id"].max()
        seq = lstm_df[lstm_df["sequence_id"] == last_seq_id].sort_values("time_step")
        
        lstm_features = ["voltage_measured", "current_measured", "temperature_measured", "c_rate", "soc", "soh"]
        lstm_input = seq[lstm_features].values.reshape(1, 100, 6)
        
        # Risk Score
        risk_score = float(lstm_risk_model.predict(lstm_input, verbose=0)[0][0])
        
        # Anomaly Detection
        ae_input = ae_scaler.transform(
            lstm_input.reshape(-1, len(lstm_features))
        ).reshape(1, 100, len(lstm_features))
        
        ae_recon = ae_model.predict(ae_input, verbose=0)
        recon_error = float(np.mean((ae_input - ae_recon) ** 2))
        is_anomaly = recon_error > float(anomaly_threshold)
        
    except Exception as e:
        st.error(f"Error computing LSTM sequences: {e}")
        st.stop()
        
    # Final Decision Logic
    RISK_THRESHOLD = 0.6

    status = "NORMAL"
    action = "No action required"
    color = "green"

    if rf_pred_label != "NORMAL":
        status = "CRITICAL"
        action = "Stop vehicle and inspect battery immediately"
        color = "red"
    elif is_anomaly:
        status = "ANOMALY WARNING"
        action = "Abnormal battery behavior detected. Inspect sensors and battery system"
        color = "orange"
    elif risk_score >= RISK_THRESHOLD:
        status = "EARLY WARNING"
        action = "Reduce load and inspect battery cooling system"
        color = "#FFD700" # yellow

    st.markdown(f"### System Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    st.info(f"**Recommended Action:** {action}")
    
    with col1:
        st.subheader("🌲 Random Forest")
        st.metric(label="Detected Fault", value=rf_pred_label)
        
    with col2:
        st.subheader("📈 LSTM Risk Assessment")
        st.metric(label="Future Risk Score", value=f"{risk_score:.2f}", delta="Critical" if risk_score >= RISK_THRESHOLD else "Safe", delta_color="inverse")
        st.caption(f"Risk Threshold: {RISK_THRESHOLD:.2f}")

    with col3:
        st.subheader("🕵️ LSTM Autoencoder")
        st.metric(label="Reconstruction Error", value=f"{recon_error:.4f}", delta="Anomaly" if is_anomaly else "Normal", delta_color="inverse")
        st.caption(f"Anomaly Threshold: {float(anomaly_threshold):.4f}")

    # Display full JSON
    with st.expander("View Raw JSON Output"):
        dashboard_output = {
            "status": status,
            "fault_detection": {
                "model": "Random Forest",
                "detected_fault": rf_pred_label
            },
            "risk_assessment": {
                "model": "LSTM",
                "risk_score": risk_score,
                "risk_threshold": RISK_THRESHOLD
            },
            "anomaly_detection": {
                "model": "LSTM Autoencoder",
                "anomaly_score": recon_error,
                "anomaly_threshold": float(anomaly_threshold),
                "is_anomaly": bool(is_anomaly)
            },
            "recommended_action": action,
            "timestamp": datetime.now().isoformat()
        }
        st.json(dashboard_output)
else:
    st.info("👈 Adjust the sensor data in the sidebar and click **Run Diagnostics** to start.")
