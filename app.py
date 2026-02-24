import streamlit as st
import pandas as pd
from datetime import timedelta
import numpy as np

from src.config import *
from src.data_loader import load_data
from src.feature_engineering import create_features
from src.model import train_models

st.title("Fuel Price Forecasting System")

# Load & Prepare Data
df_raw = load_data(DATA_PATH)
df_features = create_features(df_raw)

# Select Metro City
states = sorted(df_features["state"].unique())
selected_state = st.selectbox("Select Metro City", states)

df_state = df_features[df_features["state"] == selected_state].copy()
df_state = df_state.sort_values("date")

# Historical Chart
st.subheader("Historical Prices")
st.line_chart(
    df_state.set_index("date")[["petrol_price", "diesel_price"]]
)

# Forecast Horizon
forecast_months = st.slider(
    "Select Forecast Horizon (Months)",
    min_value=1,
    max_value=24,
    value=6
)

# Feature Columns
feature_cols = [
    col for col in df_state.columns
    if col not in ["date", "state", "city", "petrol_price", "diesel_price"]
]

X = df_state[feature_cols]
y_petrol = df_state["petrol_price"]
y_diesel = df_state["diesel_price"]

# Train Models
models_petrol, eval_petrol = train_models(X, y_petrol, RANDOM_STATE)
models_diesel, eval_diesel = train_models(X, y_diesel, RANDOM_STATE)

model_petrol = models_petrol["random_forest"]
model_diesel = models_diesel["random_forest"]

# ==============================
# 🔥 Recursive Multi-Step Forecast
# ==============================

future_rows = []
current_row = df_state.iloc[-1:].copy()

for _ in range(forecast_months):

    feature_input = current_row[feature_cols]

    petrol_pred = model_petrol.predict(feature_input)[0]
    diesel_pred = model_diesel.predict(feature_input)[0]

    future_date = current_row["date"].iloc[0] + pd.DateOffset(months=1)

    future_rows.append({
        "forecast_date": future_date,
        "predicted_petrol_price": round(petrol_pred, 2),
        "predicted_diesel_price": round(diesel_pred, 2)
    })

    # Create next step row
    new_row = current_row.copy()

    new_row["date"] = future_date
    new_row["petrol_price"] = petrol_pred
    new_row["diesel_price"] = diesel_pred

    # Update time features
    new_row["month"] = future_date.month
    new_row["year"] = future_date.year

    # Update lag features
    new_row["petrol_lag_1"] = petrol_pred
    new_row["diesel_lag_1"] = diesel_pred

    # Update rolling mean (approximate)
    new_row["petrol_rolling_mean_3"] = (
        (current_row["petrol_rolling_mean_3"].iloc[0] * 2 + petrol_pred) / 3
    )

    new_row["diesel_rolling_mean_3"] = (
        (current_row["diesel_rolling_mean_3"].iloc[0] * 2 + diesel_pred) / 3
    )

    # Update percentage change
    new_row["petrol_pct_change"] = (
        (petrol_pred - current_row["petrol_price"].iloc[0]) /
        current_row["petrol_price"].iloc[0]
    )

    new_row["diesel_pct_change"] = (
        (diesel_pred - current_row["diesel_price"].iloc[0]) /
        current_row["diesel_price"].iloc[0]
    )

    current_row = new_row

df_forecast = pd.DataFrame(future_rows)

# Forecast Output
st.subheader("Forecast Results")
st.dataframe(df_forecast)

st.subheader("Forecast Chart")
st.line_chart(
    df_forecast.set_index("forecast_date")
)

# Model Evaluation
st.subheader("Model Evaluation (Random Forest)")

st.write("Petrol Metrics:", eval_petrol["random_forest"])
st.write("Diesel Metrics:", eval_diesel["random_forest"])