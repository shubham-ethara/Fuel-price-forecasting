import pandas as pd
from datetime import timedelta


def generate_forecast(model, last_row, forecast_months):
    future_predictions = []
    current_row = last_row.copy()

    for i in range(forecast_months):
        pred = model.predict(current_row.values.reshape(1, -1))[0]
        future_predictions.append(pred)

    return future_predictions