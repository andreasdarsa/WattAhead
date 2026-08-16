from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load our feature dataset
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "wattahead_features.csv"
)

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"]
)

df["local_timestamp"] = (
    df["timestamp"]
    .dt.tz_convert("Europe/Athens")
)

# 2. Choose v1 features
FEATURES = [
    # Cyclical calendar
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",

    # Calendar
    "is_weekend",
    "is_holiday",
    "is_non_working_day",

    # Historical demand
    "lag_24",
    "lag_48",
    "lag_168",

    # Weather
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "humidity_mean",
    "wind_speed_mean",
]

TARGET = "load_mw"

# 3. Chronological split
train_df = df[
    df["local_timestamp"].dt.year <= 2018
].copy()

val_df = df[
    df["local_timestamp"].dt.year == 2019
].copy()

test_df = df[
    df["local_timestamp"].dt.year == 2020
].copy()

for name, split in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    print(
        name,
        split.shape,
        split["local_timestamp"].min(),
        split["local_timestamp"].max(),
    )

# 4. Exclude imputed targets
train_df = train_df[~train_df["was_imputed"]].copy()
val_df = val_df[~val_df["was_imputed"]].copy()
test_df = test_df[~test_df["was_imputed"]].copy()

# 5. Build X/y
X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

assert X_train.isna().sum().sum() == 0
assert X_val.isna().sum().sum() == 0
assert X_test.isna().sum().sum() == 0

# 6. Build our pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])

model.fit(X_train, y_train)

# 7. Evaluation helper
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }

# Predict validation
val_pred = model.predict(X_val)

val_metrics = evaluate(
    y_val,
    val_pred,
)

print("\nValidation:")
for metric, value in val_metrics.items():
    print(f"{metric}: {value:.3f}")

