import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series features grouped by state.
    """

    df = df.copy()

    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    df["petrol_lag_1"] = df.groupby("state")["petrol_price"].shift(1)
    df["diesel_lag_1"] = df.groupby("state")["diesel_price"].shift(1)

    df["petrol_rolling_mean_3"] = (
        df.groupby("state")["petrol_price"]
        .rolling(3).mean()
        .reset_index(0, drop=True)
    )

    df["diesel_rolling_mean_3"] = (
        df.groupby("state")["diesel_price"]
        .rolling(3).mean()
        .reset_index(0, drop=True)
    )

    df["petrol_pct_change"] = df.groupby("state")["petrol_price"].pct_change()
    df["diesel_pct_change"] = df.groupby("state")["diesel_price"].pct_change()

    df = df.dropna()

    return df