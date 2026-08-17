"""
Final model for load forecasting using XGBoost.
The selection on model and its hyperparameters was done based on the validation set performance.
The final model is trained on the train + validation set and evaluated on the test set.
"""

from pathlib import Path
import json

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

FEATURES = [
    # Raw calendar
    "hour",
    "day_of_week",
    "month",

    # Cyclical calendar
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",

    # Calendar flags
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

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"],
)

df["local_timestamp"] = (
    df["timestamp"]
    .dt.tz_convert("Europe/Athens")
)

test_start = pd.Timestamp(
    "2020-01-01 00:00:00",
    tz="Europe/Athens",
)

test_end = pd.Timestamp(
    "2020-10-01 00:00:00",
    tz="Europe/Athens",
)

train_df = df[
    df["local_timestamp"] < test_start
].copy()

test_df = df[
    (df["local_timestamp"] >= test_start)
    & (df["local_timestamp"] < test_end)
].copy()

train_df = train_df[
    ~train_df["was_imputed"]
].copy()

test_df = test_df[
    ~test_df["was_imputed"]
].copy()

print(
    "Train:",
    train_df.shape,
    train_df["local_timestamp"].min(),
    "->",
    train_df["local_timestamp"].max(),
)

print(
    "Test:",
    test_df.shape,
    test_df["local_timestamp"].min(),
    "->",
    test_df["local_timestamp"].max(),
)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

assert X_train.isna().sum().sum() == 0
assert X_test.isna().sum().sum() == 0

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

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    MODEL_DIR
    / "wattahead_xgboost.json"
)

model.save_model(MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")

test_pred = model.predict(X_test)

test_metrics = evaluate(
    y_test,
    test_pred,
)

print("\nXGBoost — Final 2020 Test")

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.3f}")

baseline_metrics = evaluate(
    y_test,
    test_df["lag_24"],
)

print("\nPrevious Day — 2020 Test")

for metric, value in baseline_metrics.items():
    print(f"{metric}: {value:.3f}")

mae_improvement = (
    baseline_metrics["MAE"]
    - test_metrics["MAE"]
) / baseline_metrics["MAE"] * 100

print(
    f"\nMAE improvement over previous-day baseline: "
    f"{mae_improvement:.2f}%"
)

metadata = {
    "model": "XGBoost",
    "target": TARGET,
    "features": FEATURES,
    "training_period": {
        "start": str(train_df["local_timestamp"].min()),
        "end": str(train_df["local_timestamp"].max()),
    },
    "test_period": {
        "start": str(test_df["local_timestamp"].min()),
        "end": str(test_df["local_timestamp"].max()),
    },
    "hyperparameters": {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
    },
    "test_metrics": {
        key: float(value)
        for key, value in test_metrics.items()
    },
    "baseline_metrics": {
        key: float(value)
        for key, value in baseline_metrics.items()
    },
    "mae_improvement_over_baseline": float(mae_improvement),
}

METADATA_PATH = (
    MODEL_DIR
    / "wattahead_xgboost_metadata.json"
)

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
    )

print(f"Metadata saved to: {METADATA_PATH}")

test_results = test_df[
    [
        "local_timestamp",
        "load_mw",
    ]
].copy()

test_results["prediction"] = test_pred

test_results["absolute_error"] = (
    test_results["load_mw"]
    - test_results["prediction"]
).abs()

test_results["percentage_error"] = (
    test_results["absolute_error"]
    / test_results["load_mw"]
    * 100
)

test_results["month"] = (
    test_results["local_timestamp"].dt.month
)

monthly_errors = (
    test_results
    .groupby("month")
    .agg(
        MAE=("absolute_error", "mean"),
        MAPE=("percentage_error", "mean"),
        observations=("load_mw", "size"),
    )
)

print("\nMonthly Test Performance")
print(monthly_errors)

test_results["error"] = (
    test_results["prediction"]
    - test_results["load_mw"]
)

monthly_errors = (
    test_results
    .groupby("month")
    .agg(
        MAE=("absolute_error", "mean"),
        MAPE=("percentage_error", "mean"),
        Bias=("error", "mean"),
        observations=("load_mw", "size"),
    )
)

test_results["error"] = (
    test_results["prediction"]
    - test_results["load_mw"]
)

monthly_errors = (
    test_results
    .groupby("month")
    .agg(
        MAE=("absolute_error", "mean"),
        MAPE=("percentage_error", "mean"),
        Bias=("error", "mean"),
        observations=("load_mw", "size"),
    )
)

print("\nMonthly Test Performance")
print(monthly_errors)

# A closer look at Spring 2020 (March, April, May) when the COVID-19 lockdowns were in effect
lockdown_mask = (
    test_results["local_timestamp"].dt.month.isin([3, 4, 5])
)

normal_period = test_results[~lockdown_mask]
lockdown_period = test_results[lockdown_mask]

print("\nMarch-May 2020")
print(
    evaluate(
        lockdown_period["load_mw"],
        lockdown_period["prediction"],
    )
)

print("\nOutside March-May 2020")
print(
    evaluate(
        normal_period["load_mw"],
        normal_period["prediction"],
    )
)

# Visualize monthly MAE
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.bar(
    monthly_errors.index,
    monthly_errors["MAE"],
)

plt.xlabel("Month")
plt.ylabel("MAE (MW)")
plt.title("Monthly Forecast Error — 2020 Test Set")
plt.xticks(range(1, 10))
plt.tight_layout()

output_path = (
    BASE_DIR
    / "reports"
    / "figures"
    / "monthly_test_mae.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

# Visualise predicted vs actual demand between Feb and June 2020
plot_df = test_results[
    (test_results["local_timestamp"] >= "2020-02-01")
    & (test_results["local_timestamp"] < "2020-07-01")
].copy()

plt.figure(figsize=(14, 6))

plt.plot(
    plot_df["local_timestamp"],
    plot_df["load_mw"],
    label="Actual",
    linewidth=1,
)

plt.plot(
    plot_df["local_timestamp"],
    plot_df["prediction"],
    label="XGBoost prediction",
    linewidth=1,
    alpha=0.8,
)

plt.title("Actual vs Predicted Electricity Demand — Feb–Jun 2020")
plt.xlabel("Date")
plt.ylabel("Electricity Load (MW)")
plt.legend()
plt.tight_layout()

output_path = (
    BASE_DIR
    / "reports"
    / "figures"
    / "actual_vs_predicted_2020.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

# Visualise predicted vs actual demand in April 2020 (first full month of COVID restrictions)
april_df = test_results[
    (test_results["local_timestamp"] >= "2020-04-01")
    & (test_results["local_timestamp"] < "2020-05-01")
].copy()

plt.figure(figsize=(14, 6))

plt.plot(
    april_df["local_timestamp"],
    april_df["load_mw"],
    label="Actual",
    linewidth=1.5,
)

plt.plot(
    april_df["local_timestamp"],
    april_df["prediction"],
    label="XGBoost prediction",
    linewidth=1.5,
    alpha=0.8,
)

plt.title("Actual vs Predicted Electricity Demand — April 2020")
plt.xlabel("Date")
plt.ylabel("Electricity Load (MW)")
plt.legend()
plt.tight_layout()

output_path = (
    BASE_DIR
    / "reports"
    / "figures"
    / "actual_vs_predicted_april_2020.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

# Visualise a week's sample from April 2020
week_df = test_results[
    (test_results["local_timestamp"] >= "2020-04-06")
    & (test_results["local_timestamp"] < "2020-04-13")
].copy()

plt.figure(figsize=(14, 6))

plt.plot(
    week_df["local_timestamp"],
    week_df["load_mw"],
    label="Actual",
    linewidth=1.5,
)

plt.plot(
    week_df["local_timestamp"],
    week_df["prediction"],
    label="XGBoost prediction",
    linewidth=1.5,
    alpha=0.8,
)

plt.title(
    "Actual vs Predicted Electricity Demand — April 6–12, 2020"
)
plt.xlabel("Date")
plt.ylabel("Electricity Load (MW)")
plt.legend()
plt.tight_layout()

output_path = (
    BASE_DIR
    / "reports"
    / "figures"
    / "actual_vs_predicted_6-12_april_2020.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()