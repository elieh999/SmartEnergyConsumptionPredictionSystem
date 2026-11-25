"""Create clear report graphs from the saved test predictions.

The Linear Regression and LSTM models do not have exactly the same number of
test rows because the LSTM removes incomplete 24 hour windows. This script
matches both models by the end time of their input window before comparing
them, so every comparison uses the same 5,050 observations.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
WEEK_HOURS = 168


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = actual - predicted
    return {
        "mae_kw": float(np.mean(np.abs(errors))),
        "rmse_kw": float(np.sqrt(np.mean(errors**2))),
        "r_squared": float(1 - np.sum(errors**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def make_prediction_graph(
    dates: list[datetime], actual: np.ndarray, predicted: np.ndarray, model_name: str, filename: str
) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(dates, actual, color="#1f1f1f", linewidth=1.8, label="Actual electricity consumption")
    axis.plot(dates, predicted, color="#a12d2d", linewidth=1.5, label=f"{model_name} prediction")
    axis.set_title(f"Actual Electricity Consumption and {model_name} Predictions")
    axis.set_xlabel("Test date and time")
    axis.set_ylabel("Global active power (kW)")
    axis.xaxis.set_major_locator(mdates.DayLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.grid(axis="y", color="#d7d7d7", linewidth=0.7)
    axis.legend(frameon=False)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / filename, dpi=220)
    plt.close(figure)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    linear_rows = read_csv_rows(RESULTS_DIR / "linear_regression_test_predictions.csv")
    lstm_rows = read_csv_rows(RESULTS_DIR / "lstm_test_predictions.csv")
    linear_by_time = {row["datetime"]: row for row in linear_rows}

    common_rows = []
    for lstm_row in lstm_rows:
        linear_row = linear_by_time.get(lstm_row["input_end_datetime"])
        if linear_row is None:
            continue
        common_rows.append((linear_row, lstm_row))

    if not common_rows:
        raise ValueError("No matching test predictions were found.")

    dates = [datetime.fromisoformat(lstm["target_datetime"]) for _, lstm in common_rows]
    actual = np.array([float(lstm["actual_power_next_hour_kw"]) for _, lstm in common_rows])
    linear_prediction = np.array([float(linear["predicted_power_next_hour_kw"]) for linear, _ in common_rows])
    lstm_prediction = np.array([float(lstm["predicted_power_next_hour_kw"]) for _, lstm in common_rows])

    if not np.allclose(
        actual,
        np.array([float(linear["actual_power_next_hour_kw"]) for linear, _ in common_rows]),
        atol=1e-5,
    ):
        raise ValueError("The two files do not agree on the actual target values.")

    linear_metrics = regression_metrics(actual, linear_prediction)
    lstm_metrics = regression_metrics(actual, lstm_prediction)
    comparison = {
        "comparison_rows": len(common_rows),
        "comparison_note": "Both models are compared only on matching LSTM test windows.",
        "multiple_linear_regression": linear_metrics,
        "lstm": lstm_metrics,
    }
    with (RESULTS_DIR / "model_comparison_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    week_dates = dates[:WEEK_HOURS]
    make_prediction_graph(
        week_dates,
        actual[:WEEK_HOURS],
        linear_prediction[:WEEK_HOURS],
        "Multiple Linear Regression",
        "linear_regression_actual_vs_predicted.png",
    )
    make_prediction_graph(
        week_dates,
        actual[:WEEK_HOURS],
        lstm_prediction[:WEEK_HOURS],
        "LSTM",
        "lstm_actual_vs_predicted.png",
    )

    with (RESULTS_DIR / "lstm_training_history.json").open("r", encoding="utf-8") as file:
        history = json.load(file)["history"]
    epochs = np.arange(1, len(history["loss"]) + 1)
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.plot(epochs, history["loss"], marker="o", color="#1f1f1f", label="Training loss")
    axis.plot(epochs, history["val_loss"], marker="s", color="#a12d2d", label="Validation loss")
    axis.set_title("LSTM Training and Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error loss")
    axis.set_xticks(epochs)
    axis.grid(axis="y", color="#d7d7d7", linewidth=0.7)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "lstm_training_validation_loss.png", dpi=220)
    plt.close(figure)

    models = ["Multiple Linear\nRegression", "LSTM"]
    positions = np.arange(len(models))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    mae_bars = axis.bar(
        positions - width / 2,
        [linear_metrics["mae_kw"], lstm_metrics["mae_kw"]],
        width,
        color="#4c4c4c",
        label="MAE",
    )
    rmse_bars = axis.bar(
        positions + width / 2,
        [linear_metrics["rmse_kw"], lstm_metrics["rmse_kw"]],
        width,
        color="#a12d2d",
        label="RMSE",
    )
    axis.set_title("Test Error Comparison on the Same 5,050 Hours")
    axis.set_ylabel("Error in kW")
    axis.set_xticks(positions, models)
    axis.grid(axis="y", color="#d7d7d7", linewidth=0.7)
    axis.legend(frameon=False)
    for bars in (mae_bars, rmse_bars):
        axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "model_error_comparison.png", dpi=220)
    plt.close(figure)

    print(f"Created 4 graphs in {FIGURES_DIR}")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
