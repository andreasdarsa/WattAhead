from pathlib import Path
import json

import pandas as pd
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "wattahead"
    / "models"
    / "wattahead_xgboost.json"
)

METADATA_PATH = (
    BASE_DIR
    / "wattahead"
    / "models"
    / "wattahead_xgboost_metadata.json"
)

DATA_PATH = (
    BASE_DIR
    / "wattahead"
    / "data"
    / "processed"
    / "wattahead_features.csv"
)


# Load metadata
with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)

FEATURES = metadata["features"]


# Load trained model
model = XGBRegressor()
model.load_model(MODEL_PATH)


# Load data
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"],
)

df["local_timestamp"] = (
    df["timestamp"]
    .dt.tz_convert("Europe/Athens")
)


# Take one example observation
sample = df[
    df["local_timestamp"] < pd.Timestamp(
        "2020-10-01 00:00:00",
        tz="Europe/Athens",
    )
].iloc[[-1]]

X_sample = sample[FEATURES]

prediction = model.predict(X_sample)[0]


print("Timestamp:")
print(sample["local_timestamp"].iloc[0])

print("\nActual load:")
print(f"{sample['load_mw'].iloc[0]:.2f} MW")

print("\nPredicted load:")
print(f"{prediction:.2f} MW")