# Load processed datasets
from pathlib import Path

import numpy as np
import pandas as pd
import holidays


BASE_DIR = Path(__file__).resolve().parents[2]

LOAD_PATH = BASE_DIR / "data" / "processed" / "greece_hourly_load.csv"
WEATHER_PATH = BASE_DIR / "data" / "processed" / "greece_hourly_weather.csv"


load_df = pd.read_csv(
    LOAD_PATH,
    parse_dates=["timestamp"]
)

weather_df = pd.read_csv(
    WEATHER_PATH,
    parse_dates=["timestamp"]
)

df = load_df.merge(
    weather_df,
    on="timestamp",
    how="left"
)

# Sanity check
print(df.shape)
print(df.isna().sum())

# 1. Calendar features
df["local_timestamp"] = (
    df["timestamp"]
    .dt.tz_convert("Europe/Athens")
)

df["hour"] = df["local_timestamp"].dt.hour
df["day_of_week"] = df["local_timestamp"].dt.dayofweek
df["month"] = df["local_timestamp"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Add Greek holidays
greek_holidays = holidays.Greece(
    years=range(
        df["local_timestamp"].dt.year.min(),
        df["local_timestamp"].dt.year.max() + 1
    )
)

df["is_holiday"] = (
    df["local_timestamp"]
    .dt.date
    .isin(greek_holidays)
    .astype(int)
)

df["is_non_working_day"] = (
    (df["is_weekend"] == 1) |
    (df["is_holiday"] == 1)
).astype(int)

print(df["is_holiday"].value_counts())
print(df["is_non_working_day"].value_counts())

print(
    df.loc[
        df["is_holiday"] == 1,
        ["local_timestamp", "load_mw", "is_weekend", "is_holiday"]
    ].head(30)
)

# 2. Cyclical encodings (hour 0 and 23 should be close in the feature space)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

# 3. Legal day-ahead lag features
df["lag_24"] = df["load_mw"].shift(24)
df["lag_48"] = df["load_mw"].shift(48)
df["lag_168"] = df["load_mw"].shift(168)

# 4. Rolling window features
# Removed for now

# 5. Rows incomplete by lags are dropped
feature_cols = [
    "lag_24",
    "lag_48",
    "lag_168"
]

df = df.dropna(subset=feature_cols).copy()

# Final inspection
print(df.shape)

print(
    df[
        [
            "timestamp",
            "load_mw",
            "lag_24",
            "lag_48",
            "lag_168",
        ]
    ].head()
)

print(df.isna().sum().sort_values(ascending=False).head(10))
print(df["is_holiday"].value_counts())
print(df["is_non_working_day"].value_counts())

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "wattahead_features.csv"
)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Feature dataset saved to: {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")