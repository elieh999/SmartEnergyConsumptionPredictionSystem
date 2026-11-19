"""Train and evaluate the Multiple Linear Regression baseline model.

The model is trained only on the chronological training split. Validation and
test files are never used during training. Results are written to the results
folder so they can be used in the final report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
TARGET_COLUMN = "power_next_hour"
DATETIME_COLUMN = "datetime"


def load_split(name: str) -> pd.DataFrame:
    """Load one already scaled chronological data split."""
    return pd.read_csv(DATA_DIR / f"{name}_scaled.csv")


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numerical input features and the next hour power target."""
    feature_columns = [
        column for column in frame.columns if column not in {DATETIME_COLUMN, TARGET_COLUMN}
    ]
    return frame[feature_columns], frame[TARGET_COLUMN]


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics in the original kW target scale."""
    actual_values = actual.to_numpy(dtype=float)
    errors = actual_values - predicted
    squared_error = float(np.sum(errors**2))
    total_variation = float(np.sum((actual_values - actual_values.mean()) ** 2))
    return {
        "mae_kw": float(np.mean(np.abs(errors))),
        "rmse_kw": float(np.sqrt(np.mean(errors**2))),
        "r_squared": float(1 - (squared_error / total_variation)),
    }


def add_intercept(features: pd.DataFrame) -> np.ndarray:
    """Add the constant column needed to learn the model intercept."""
    values = features.to_numpy(dtype=float)
    return np.column_stack([np.ones(len(values)), values])


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    x_train, y_train = split_features_target(train)
    x_validation, y_validation = split_features_target(validation)
    x_test, y_test = split_features_target(test)

    # Ordinary least squares finds the intercept and coefficients that minimise
    # the squared prediction error on the training data.
    training_matrix = add_intercept(x_train)
    parameters, _, _, _ = np.linalg.lstsq(training_matrix, y_train.to_numpy(dtype=float), rcond=None)
    intercept = float(parameters[0])
    coefficients = parameters[1:]

    validation_predictions = add_intercept(x_validation) @ parameters
    test_predictions = add_intercept(x_test) @ parameters

    results = {
        "model": "Multiple Linear Regression",
        "target": "power_next_hour in kW",
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "feature_count": int(x_train.shape[1]),
        "validation_metrics": metrics(y_validation, validation_predictions),
        "test_metrics": metrics(y_test, test_predictions),
    }

    (RESULTS_DIR / "linear_regression_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    pd.DataFrame(
        {
            "datetime": test[DATETIME_COLUMN],
            "actual_power_next_hour_kw": y_test,
            "predicted_power_next_hour_kw": test_predictions,
            "absolute_error_kw": (y_test - test_predictions).abs(),
        }
    ).to_csv(RESULTS_DIR / "linear_regression_test_predictions.csv", index=False)

    coefficient_table = pd.DataFrame(
        {
            "feature": x_train.columns,
            "coefficient": coefficients,
        }
    ).sort_values("coefficient", key=lambda values: values.abs(), ascending=False)
    coefficient_table.to_csv(RESULTS_DIR / "linear_regression_coefficients.csv", index=False)

    saved_model = {
        "model": "Multiple Linear Regression using ordinary least squares",
        "target": TARGET_COLUMN,
        "intercept": intercept,
        "feature_columns": list(x_train.columns),
        "coefficients": {column: float(value) for column, value in zip(x_train.columns, coefficients)},
    }
    (MODELS_DIR / "linear_regression_model.json").write_text(
        json.dumps(saved_model, indent=2), encoding="utf-8"
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
