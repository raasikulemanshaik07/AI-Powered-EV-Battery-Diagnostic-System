# AI-Based Preventive Diagnostics for EV Batteries using Hybrid Machine Learning

## Overview
This project implements a hybrid machine learning pipeline to analyze battery datasets and predict battery health conditions. The system detects faults, estimates future risk, and identifies anomalies using multiple machine learning models integrated into a unified diagnostic framework.

## Objective
The objective of this project is to develop a predictive battery diagnostics system capable of:
- Detecting battery faults
- Predicting degradation risks
- Identifying abnormal patterns in battery data

The system enables intelligent monitoring and early warning for battery health management.

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- JSON (for structured output)

## Machine Learning Models

### Random Forest Classifier
Used for battery fault classification based on dataset features.

**Purpose**
Detect known battery conditions such as:
- NORMAL
- OVERHEAT
- HIGH_LOAD
- LOW_SOC

### LSTM Risk Prediction Model
A time-series deep learning model that analyzes sequential battery data to estimate future risk.

**Architecture**
- LSTM Layer (64 units)
- Dropout Layer (0.2)
- LSTM Layer (32 units)
- Dense Layer (Sigmoid activation)

## System Output

The diagnostic results are displayed through a **real-time monitoring dashboard** instead of raw JSON outputs.  
The dashboard presents battery health information in an easy-to-understand format for users.

### Dashboard Displays

- **Battery Status** (Normal / Early Warning / Critical)
- **Detected Fault** from Random Forest model
- **Risk Score** predicted by the LSTM model
- **Anomaly Detection Status** from the LSTM Autoencoder
- **Recommended Action** for battery safety

### Example Dashboard Information

Battery Status: CRITICAL  
Detected Fault: OVERHEAT  
Risk Score: 0.01  
Anomaly Score: 0.000783  
Recommended Action: Stop vehicle and inspect battery immediately

The dashboard provides a centralized interface for monitoring battery health and supports intelligent decision-making for preventive diagnostics.
The project integrates multiple AI models into a single diagnostic pipeline.
