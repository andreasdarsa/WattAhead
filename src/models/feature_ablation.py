from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "wattahead_features.csv"
)

LAG_FEATURES = [
    "lag_24",
    "lag_48",
    "lag_168",
]

CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "is_non_working_day",
]

WEATHER_FEATURES = [
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "humidity_mean",
    "wind_speed_mean",
]

FEATURE_SETS = {
    "Lags only": LAG_FEATURES,
    "Lags + Calendar": LAG_FEATURES + CALENDAR_FEATURES,
    "Lags + Weather": LAG_FEATURES + WEATHER_FEATURES,
    "All Features": (
        LAG_FEATURES
        + CALENDAR_FEATURES
        + WEATHER_FEATURES
    ),
}

TARGET = "load_mw"


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


# Load dataset
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"],
)

df["local_timestamp"] = (
    df["timestamp"]
    .dt.tz_convert("Europe/Athens")
)


# Chronological split
train_df = df[
    df["local_timestamp"].dt.year <= 2018
].copy()

val_df = df[
    df["local_timestamp"].dt.year == 2019
].copy()

test_df = df[
    df["local_timestamp"].dt.year == 2020
].copy()


# Do not train or evaluate against reconstructed target values
train_df = train_df[~train_df["was_imputed"]].copy()
val_df = val_df[~val_df["was_imputed"]].copy()
test_df = test_df[~test_df["was_imputed"]].copy()


# Inspect splits
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


y_train = train_df[TARGET]
y_val = val_df[TARGET]


# Evaluation helper is also used for our persistence baselines
baselines = {
    "Previous Day": "lag_24",
    "Two Days Ago": "lag_48",
    "Previous Week": "lag_168",
}

print("\nValidation Baselines")

for name, column in baselines.items():
    metrics = evaluate(
        y_val,
        val_df[column],
    )

    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")


results = []

for name, features in FEATURE_SETS.items():

    X_train = train_df[features]
    X_val = val_df[features]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    val_pred = model.predict(X_val)

    metrics = evaluate(
        y_val,
        val_pred,
    )

    results.append({
        "Feature Set": name,
        **metrics,
    })


results_df = pd.DataFrame(results)

print("\nFeature Ablation — 2019 Validation")

print(
    results_df
    .sort_values("MAE")
    .to_string(index=False)
)

