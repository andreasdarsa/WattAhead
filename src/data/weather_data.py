import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


CITIES = {
    "athens": (37.98, 23.73),
    "thessaloniki": (40.64, 22.94),
    "patras": (38.25, 21.73),
    "larissa": (39.64, 22.42),
}


def fetch_weather(city, latitude, longitude):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2015-01-01",
        "end_date": "2020-09-30",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
        ],
        "timezone": "UTC",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()["hourly"]

    df = pd.DataFrame(data)

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    return df

# Fetch data for all four cities
weather_frames = []

for city, (lat, lon) in CITIES.items():
    df = fetch_weather(city, lat, lon)

    df = df.rename(
        columns={
            "time": "timestamp",
            "temperature_2m": f"{city}_temperature",
            "relative_humidity_2m": f"{city}_humidity",
            "wind_speed_10m": f"{city}_wind_speed",
        }
    )

    weather_frames.append(df)

# Merge into one df
weather_df = weather_frames[0]

for df in weather_frames[1:]:
    weather_df = weather_df.merge(
        df,
        on="timestamp",
        how="outer"
    )

weather_df.info()
weather_df.head()
weather_df.isna().sum()

weather_df["timestamp"].min(), weather_df["timestamp"].max()

# Create aggregate features for temperature (not city-specific)

temperature_cols = [
    f"{city}_temperature"
    for city in CITIES
]

weather_df["temperature_mean"] = (
    weather_df[temperature_cols].mean(axis=1)
)

weather_df["temperature_min"] = (
    weather_df[temperature_cols].min(axis=1)
)

weather_df["temperature_max"] = (
    weather_df[temperature_cols].max(axis=1)
)

# Same for humidity and wind
humidity_cols = [
    f"{city}_humidity"
    for city in CITIES
]

wind_cols = [
    f"{city}_wind_speed"
    for city in CITIES
]

weather_df["humidity_mean"] = (
    weather_df[humidity_cols].mean(axis=1)
)

weather_df["wind_speed_mean"] = (
    weather_df[wind_cols].mean(axis=1)
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "greece_hourly_weather.csv"
)

weather_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Weather data saved to {OUTPUT_PATH}")

