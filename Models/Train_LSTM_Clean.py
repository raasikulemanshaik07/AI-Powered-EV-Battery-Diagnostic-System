import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ------------------------------
# Load LSTM-ready dataset
# ------------------------------
data = pd.read_csv("LSTM_Ready.csv")
print("\n=== DATASET COLUMNS ===")
print(list(data.columns))
print("=======================\n")


# ------------------------------
# Correct feature names
# ------------------------------
features = [
    "voltage_measured",
    "current_measured",
    "temperature_measured",
    "c_rate",
    "soc",
    "soh"
]

TIME_STEPS = 100

# ------------------------------
# Build sequences
# ------------------------------
sequences = []
for seq_id in data["sequence_id"].unique():
    seq = data[data["sequence_id"] == seq_id]
    seq = seq.sort_values("time_step")
    sequences.append(seq[features].values)

X = np.array(sequences)
print("X shape:", X.shape)

# ------------------------------
# Scale features
# ------------------------------
n_samples, n_steps, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)

# ------------------------------
# Create risk score target
# ------------------------------
y_risk = []

for seq in X:
    temp = seq[:, features.index("temperature_measured")]
    volt = seq[:, features.index("voltage_measured")]

    temp_slope = np.mean(np.diff(temp))
    volt_drop = np.mean(np.diff(volt))

    r = 0.0
    if temp_slope > 0:
        r += min(temp_slope / 5, 0.6)
    if volt_drop < 0:
        r += min(abs(volt_drop) / 2, 0.4)

    y_risk.append(min(r, 1.0))

y_risk = np.array(y_risk)

# ------------------------------
# Train / test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_risk, test_size=0.2, random_state=42
)

# ------------------------------
# LSTM model
# ------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, n_features)),
    Dropout(0.2),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ------------------------------
# Train
# ------------------------------
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ------------------------------
# Predict & alert
# ------------------------------
risk_score = model.predict(X_test[-1].reshape(1, TIME_STEPS, n_features))[0][0]
risk_score = round(float(risk_score), 2)

last_seq = X[-1]
temp_trend = np.mean(np.diff(last_seq[:, features.index("temperature_measured")]))

print("\nStatus: EARLY WARNING")
print(f"Overall Risk Score: {risk_score}")

if temp_trend > 0.3:
    print("• Abnormal battery heating trend detected")

print("Suggested Action:")
print("Reduce load and inspect battery cooling system")

model.save("models/lstm_risk_model.h5")
print("LSTM model saved as .h5")
