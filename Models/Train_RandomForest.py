import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------------------------------
# Load labeled dataset
# ---------------------------------
df = pd.read_csv("RF_Ready.csv")

# ---------------------------------
# Snapshot features (LATEST timestep)
# ---------------------------------
features = [
    "Voltage_measured_t99",
    "Current_measured_t99",
    "Temperature_measured_t99",
    "C_rate_t99",
    "SoC_t99"
]

X = df[features]
y = df["Fault_Label"]

# ---------------------------------
# Convert labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ---------------------------------
# Train-test split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

# ---------------------------------
# Train Random Forest
# ---------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

# ---------------------------------
# Predictions
# ---------------------------------
y_pred = rf_model.predict(X_test)

# ---------------------------------
# Evaluation
# ---------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------
# Feature importance
# ---------------------------------
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance:")
for idx in indices:
    print(features[idx], ":", importances[idx])

# ---------------------------------
# Save model
# ---------------------------------
joblib.dump(rf_model, "models/random_forest_fault_model.pkl")
joblib.dump(label_encoder, "models/rf_label_encoder.pkl")

print("Random Forest model saved successfully")
