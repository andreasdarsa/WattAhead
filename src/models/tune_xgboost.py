from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "wattahead_features.csv"
)

FEATURES = [
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
    "lag_24",
    "lag_48",
    "lag_168",
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


# Prepare features and targets
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


PARAM_GRID = [
    {
        "max_depth": 4,
        "min_child_weight": 1,
        "learning_rate": 0.03,
    },
    {
        "max_depth": 4,
        "min_child_weight": 3,
        "learning_rate": 0.03,
    },
    {
        "max_depth": 6,
        "min_child_weight": 1,
        "learning_rate": 0.03,
    },
    {
        "max_depth": 6,
        "min_child_weight": 3,
        "learning_rate": 0.03,
    },
    {
        "max_depth": 8,
        "min_child_weight": 3,
        "learning_rate": 0.03,
    },
    {
        "max_depth": 6,
        "min_child_weight": 3,
        "learning_rate": 0.05,
    },
]

results = []

for params in PARAM_GRID:
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        learning_rate=params["learning_rate"],
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
        **params,
        **metrics,
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values("MAE")

print("\nXGBoost Hyperparameter Tuning — 2019 Validation")
print(results_df.to_string(index=False))

best = results_df.iloc[0]

print("\nBest configuration:")
print(best)