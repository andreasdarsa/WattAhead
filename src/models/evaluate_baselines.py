from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

# 1. Load our feature dataset
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "wattahead_features.csv"
)

FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"]
)

df["local_timestamp"] = (
    df["timestamp"]
    .dt.tz_convert("Europe/Athens")
)

print(df["timestamp"].dtype)
print(df["local_timestamp"].dtype)

print(df.shape)
print(df["timestamp"].min())
print(df["timestamp"].max())

# 2. Define a chronological test set (first experiment will have 2020 as our test sample)
# Convert to datetime if not already
df["local_timestamp"] = pd.to_datetime(df["local_timestamp"])

test_df = df[
    df["local_timestamp"].dt.year == 2020
].copy()

print(test_df.shape)
print(test_df["local_timestamp"].min())
print(test_df["local_timestamp"].max())

# 3. Exclude imputed targets
test_df = test_df[
    ~test_df["was_imputed"]
].copy()

# 4. Define baselines
baselines = {
    "Previous Day": "lag_24",
    "Two Days Ago": "lag_48",
    "Previous Week": "lag_168",
}


# Previous Day
# ŷ(t) = y(t - 24)

# Two Days Ago
# ŷ(t) = y(t - 48)

# Previous Week
# ŷ(t) = y(t - 168)

# 5. Evaluate baselines
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    mape = (
        mean_absolute_percentage_error(
            y_true,
            y_pred,
        )
        * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }

results = []

for name, column in baselines.items():
    metrics = evaluate(
        test_df["load_mw"],
        test_df[column],
    )

    metrics["Model"] = name
    results.append(metrics)


results_df = pd.DataFrame(results)

results_df = results_df[
    ["Model", "MAE", "RMSE", "MAPE"]
]

print(results_df)

# 6. Visualization of actual demand against lag24
import matplotlib.pyplot as plt


sample = test_df[
    (test_df["local_timestamp"] >= "2020-02-03") &
    (test_df["local_timestamp"] < "2020-02-10")
]

plt.figure(figsize=(14, 6))

plt.plot(
    sample["local_timestamp"],
    sample["load_mw"],
    label="Actual",
)

plt.plot(
    sample["local_timestamp"],
    sample["lag_24"],
    label="Previous-day baseline",
    linestyle="--",
)

plt.title("Actual vs Previous-Day Baseline")
plt.xlabel("Time")
plt.ylabel("Electricity Load (MW)")
plt.legend()
plt.tight_layout()

output_path = FIGURES_DIR / "baseline_previous_day_week.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(f"Figure saved to: {output_path}")
