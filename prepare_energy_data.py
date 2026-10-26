"""Prepare the UCI household electricity dataset for next-hour forecasting.

The script preserves the raw UCI data, creates hourly observations, and writes
chronological train, validation, and test splits.  It does not train a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

MEASUREMENT_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return 70% train, 15% validation, and 15% test data in time order."""
    n_rows = len(frame)
    train_end = int(n_rows * 0.70)
    validation_end = int(n_rows * 0.85)
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the UCI household electricity dataset for next-hour forecasting."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_FILE,
        help=f"raw household_power_consumption.txt file (default: {RAW_FILE})",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"directory for generated CSV and scaler files (default: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help=f"directory for the preprocessing summary (default: {REPORTS_DIR})",
    )
    return parser


def prepare_dataset(raw_file: Path, processed_dir: Path, reports_dir: Path) -> dict[str, object]:
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_file}. Download and extract the UCI dataset first."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_file, sep=";", na_values="?", low_memory=False)
    original_rows = len(raw)

    raw["Datetime"] = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    raw[MEASUREMENT_COLUMNS] = raw[MEASUREMENT_COLUMNS].apply(pd.to_numeric, errors="coerce")

    rows_with_missing = int(raw[["Datetime", *MEASUREMENT_COLUMNS]].isna().any(axis=1).sum())
    clean = raw.dropna(subset=["Datetime", *MEASUREMENT_COLUMNS]).copy()
    clean = clean.set_index("Datetime").sort_index()

    hourly = clean[MEASUREMENT_COLUMNS].resample("h").agg(
        {
            "Global_active_power": "mean",
            "Global_reactive_power": "mean",
            "Voltage": "mean",
            "Global_intensity": "mean",
            "Sub_metering_1": "sum",
            "Sub_metering_2": "sum",
            "Sub_metering_3": "sum",
        }
    )
    hourly.columns = [
        "global_active_power_current",
        "global_reactive_power_mean",
        "voltage_mean",
        "global_intensity_mean",
        "sub_metering_1_sum",
        "sub_metering_2_sum",
        "sub_metering_3_sum",
    ]

    hourly["hour"] = hourly.index.hour
    hourly["day_of_week"] = hourly.index.dayofweek
    hourly["month"] = hourly.index.month
    hourly["is_weekend"] = (hourly.index.dayofweek >= 5).astype(int)
    hourly["power_lag_1"] = hourly["global_active_power_current"].shift(1)
    hourly["power_lag_2"] = hourly["global_active_power_current"].shift(2)
    hourly["power_lag_24"] = hourly["global_active_power_current"].shift(24)
    hourly["power_next_hour"] = hourly["global_active_power_current"].shift(-1)

    model_data = hourly.dropna().reset_index()
    model_data = model_data.rename(columns={"Datetime": "datetime"})

    if len(model_data) < 4:
        raise ValueError(
            "The dataset produced fewer than four model-ready hourly rows. "
            "Provide a longer continuous time range before creating train, validation, and test splits."
        )

    feature_columns = [
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
        "power_lag_1",
        "power_lag_2",
        "power_lag_24",
    ]
    target_column = "power_next_hour"
    train, validation, test = chronological_split(model_data)

    # Mean normalization / standardization is fitted on training data only.
    means = train[feature_columns].mean()
    scales = train[feature_columns].std(ddof=0).replace(0, 1)

    def scale_split(split: pd.DataFrame) -> pd.DataFrame:
        scaled = split.copy()
        scaled[feature_columns] = (scaled[feature_columns] - means) / scales
        return scaled

    model_data.to_csv(processed_dir / "hourly_model_data.csv", index=False)
    train.to_csv(processed_dir / "train_unscaled.csv", index=False)
    validation.to_csv(processed_dir / "validation_unscaled.csv", index=False)
    test.to_csv(processed_dir / "test_unscaled.csv", index=False)
    scale_split(train).to_csv(processed_dir / "train_scaled.csv", index=False)
    scale_split(validation).to_csv(processed_dir / "validation_scaled.csv", index=False)
    scale_split(test).to_csv(processed_dir / "test_scaled.csv", index=False)

    scaler_details = {
        "method": "z-score standardization: (x - training_mean) / training_standard_deviation",
        "features": feature_columns,
        "training_means": means.to_dict(),
        "training_scales": scales.to_dict(),
    }
    (processed_dir / "feature_scaler.json").write_text(
        json.dumps(scaler_details, indent=2), encoding="utf-8"
    )

    summary = {
        "source": "UCI Individual Household Electric Power Consumption dataset",
        "raw_rows": int(original_rows),
        "rows_with_missing_measurements_removed": rows_with_missing,
        "clean_minute_rows": int(len(clean)),
        "hourly_rows_before_lag_target_cleanup": int(len(hourly)),
        "model_ready_hourly_rows": int(len(model_data)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "target": "power_next_hour: Global Active Power one hour after the input hour",
        "feature_columns": feature_columns,
        "split_method": "chronological 70% train, 15% validation, 15% test; no shuffling",
    }
    (reports_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_dataset(args.input, args.processed_dir, args.reports_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
