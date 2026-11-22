"""Create chronological 24 hour sequences for the LSTM model.

Every sample contains 24 completed hourly observations ending at time t. Its
target is global active power at t + 1 hour. Input feature scaling uses only
the training split statistics already created during preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
SEQUENCE_LENGTH = 24
TARGET_COLUMN = "power_next_hour"

# The sequence itself already contains the last 24 power readings, so the
# separate power_lag columns are deliberately not repeated as LSTM inputs.
LSTM_FEATURES = [
    "global_active_power_current",
    "global_reactive_power_mean",
    "voltage_mean",
    "global_intensity_mean",
    "sub_metering_1_sum",
    "sub_metering_2_sum",
    "sub_metering_3_sum",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]


def load_boundaries() -> tuple[set[str], set[str], set[str]]:
    """Return the end timestamps assigned to each chronological split."""
    splits = []
    for name in ("train", "validation", "test"):
        frame = pd.read_csv(DATA_DIR / f"{name}_unscaled.csv")
        splits.append(set(frame["datetime"]))
    return tuple(splits)  # type: ignore[return-value]


def make_sequences() -> dict[str, object]:
    """Write LSTM sequence arrays without mixing future observations into input."""
    REPORTS_DIR.mkdir(exist_ok=True)
    all_data = pd.read_csv(DATA_DIR / "hourly_model_data.csv")
    all_data["datetime"] = pd.to_datetime(all_data["datetime"])

    scaler = json.loads((DATA_DIR / "feature_scaler.json").read_text(encoding="utf-8"))
    means = pd.Series(scaler["training_means"])[LSTM_FEATURES]
    scales = pd.Series(scaler["training_scales"])[LSTM_FEATURES]

    scaled_features = ((all_data[LSTM_FEATURES] - means) / scales).to_numpy(dtype=np.float32)
    targets = all_data[TARGET_COLUMN].to_numpy(dtype=np.float32)
    datetimes = all_data["datetime"]

    train_times, validation_times, test_times = load_boundaries()
    split_inputs: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    split_targets: dict[str, list[float]] = {"train": [], "validation": [], "test": []}
    split_end_times: dict[str, list[str]] = {"train": [], "validation": [], "test": []}

    skipped_non_consecutive = 0
    for end_index in range(SEQUENCE_LENGTH - 1, len(all_data)):
        sequence_times = datetimes.iloc[end_index - SEQUENCE_LENGTH + 1 : end_index + 1]
        expected_hours = pd.date_range(sequence_times.iloc[0], periods=SEQUENCE_LENGTH, freq="h")
        if not sequence_times.reset_index(drop=True).equals(pd.Series(expected_hours)):
            skipped_non_consecutive += 1
            continue

        end_time_text = str(datetimes.iloc[end_index])
        if end_time_text in train_times:
            split_name = "train"
        elif end_time_text in validation_times:
            split_name = "validation"
        elif end_time_text in test_times:
            split_name = "test"
        else:
            raise ValueError(f"Unassigned timestamp: {end_time_text}")

        split_inputs[split_name].append(scaled_features[end_index - SEQUENCE_LENGTH + 1 : end_index + 1])
        split_targets[split_name].append(float(targets[end_index]))
        split_end_times[split_name].append(end_time_text)

    summary: dict[str, object] = {
        "model": "LSTM",
        "sequence_length_hours": SEQUENCE_LENGTH,
        "input_feature_count": len(LSTM_FEATURES),
        "input_features": LSTM_FEATURES,
        "target": "power_next_hour in kW",
        "scaling": "z score scaling using training data mean and standard deviation only",
        "chronological_split": "The end time of each sequence determines its train validation or test split. Earlier observations may be used only as historical input.",
        "skipped_non_consecutive_windows": skipped_non_consecutive,
        "splits": {},
    }

    for split_name in ("train", "validation", "test"):
        x_values = np.asarray(split_inputs[split_name], dtype=np.float32)
        y_values = np.asarray(split_targets[split_name], dtype=np.float32)
        end_times = np.asarray(split_end_times[split_name])
        np.savez_compressed(
            DATA_DIR / f"lstm_{split_name}_sequences.npz",
            x=x_values,
            y=y_values,
            end_datetime=end_times,
        )
        summary["splits"][split_name] = {
            "samples": int(len(x_values)),
            "input_shape": list(x_values.shape),
            "first_input_end": str(end_times[0]),
            "last_input_end": str(end_times[-1]),
        }

    (REPORTS_DIR / "lstm_sequence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(make_sequences(), indent=2))


if __name__ == "__main__":
    main()
