import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

print("Environment OK")
lstm_risk_model = load_model("lstm_risk_model.h5", compile=False)



# =========================================================
# 1. LOAD MODELS
# =========================================================

label_encoder = joblib.load("rf_label_encoder.pkl")
rf_model = joblib.load("random_forest_fault_model.pkl")



# Autoencoder (Anomaly Detection)
ae_model = load_model("lstm_autoencoder.h5", compile=False)
ae_scaler = joblib.load("ae_scaler.pkl")
ANOMALY_THRESHOLD = joblib.load("anomaly_threshold.pkl")

# =========================================================
# 2. RANDOM FOREST INPUT (Snapshot / Current State)
# =========================================================

# NOTE:
# In real deployment, these will come from sensors.
# For now, we manually simulate stressed conditions.

v = 3.1        # Voltage (V)
i = 3.5        # Current (A)
t = 65.0       # Temperature (°C)
c_rate = 2.0   # C-rate
soc = 0.5      # State of Charge

rf_input = np.array([[v, i, t, c_rate, soc]])

rf_pred_encoded = rf_model.predict(rf_input)[0]
rf_pred_label = label_encoder.inverse_transform([rf_pred_encoded])[0]

# =========================================================
# 3. LOAD LSTM SEQUENCE DATA
# =========================================================

lstm_df = pd.read_csv("../LSTM_Ready.csv")

last_seq_id = lstm_df["sequence_id"].max()
seq = lstm_df[lstm_df["sequence_id"] == last_seq_id]
seq = seq.sort_values("time_step")

lstm_features = [
    "voltage_measured",
    "current_measured",
    "temperature_measured",
    "c_rate",
    "soc",
    "soh"
]

# Safety check
missing = set(lstm_features) - set(seq.columns)
assert not missing, f"Missing LSTM columns: {missing}"

lstm_input = seq[lstm_features].values.reshape(1, 100, 6)

# =========================================================
# 4. LSTM RISK PREDICTION (EARLY WARNING)
# =========================================================

risk_score = float(lstm_risk_model.predict(lstm_input)[0][0])
risk_score = round(risk_score, 2)

# =========================================================
# 5. LSTM AUTOENCODER (ANOMALY DETECTION)
# =========================================================

# Scale input using AE scaler
ae_input = ae_scaler.transform(
    lstm_input.reshape(-1, len(lstm_features))
).reshape(1, 100, len(lstm_features))

# Reconstruction
ae_recon = ae_model.predict(ae_input)

# Reconstruction error
recon_error = np.mean((ae_input - ae_recon) ** 2)
recon_error = round(float(recon_error), 6)

is_anomaly = recon_error > ANOMALY_THRESHOLD

# =========================================================
# 6. FINAL DECISION LOGIC (FUSION)
# =========================================================

RISK_THRESHOLD = 0.6

if rf_pred_label != "NORMAL":
    status = "CRITICAL"
    action = "Stop vehicle and inspect battery immediately"

elif is_anomaly:
    status = "ANOMALY WARNING"
    action = "Abnormal battery behavior detected. Inspect sensors and battery system"

elif rf_pred_label == "NORMAL" and risk_score >= RISK_THRESHOLD:
    status = "EARLY WARNING"
    action = "Reduce load and inspect battery cooling system"

else:
    status = "NORMAL"
    action = "No action required"

# =========================================================
# 7. OUTPUT
# =========================================================

print("\n================ FINAL DIAGNOSTICS =================")
print("Status:", status)
print("Detected Fault (RF):", rf_pred_label)
print("LSTM Risk Score:", risk_score)
print("Anomaly Score:", recon_error)
print("Anomaly Threshold:", round(float(ANOMALY_THRESHOLD), 6))
print("Suggested Action:")
print(action)
print("===================================================")
