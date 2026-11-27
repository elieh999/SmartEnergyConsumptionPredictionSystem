"""Open a small desktop dashboard for the finished energy prediction project.

The dashboard reads the saved model predictions. It does not retrain a model or
modify project results. Use the Previous and Next buttons to inspect individual
test hours and compare the two model predictions with the real consumption.
"""

from __future__ import annotations

import csv
import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


@dataclass
class PredictionRow:
    """One common test hour with both model predictions."""

    target_time: str
    actual: float
    linear_prediction: float
    lstm_prediction: float


def load_rows() -> list[PredictionRow]:
    """Match Linear Regression and LSTM predictions by their input end time."""
    linear_path = RESULTS_DIR / "linear_regression_test_predictions.csv"
    lstm_path = RESULTS_DIR / "lstm_test_predictions.csv"

    with linear_path.open("r", encoding="utf-8", newline="") as file:
        linear_by_time = {row["datetime"]: row for row in csv.DictReader(file)}
    with lstm_path.open("r", encoding="utf-8", newline="") as file:
        lstm_rows = list(csv.DictReader(file))

    matched = []
    for lstm in lstm_rows:
        linear = linear_by_time.get(lstm["input_end_datetime"])
        if linear is None:
            continue
        matched.append(
            PredictionRow(
                target_time=lstm["target_datetime"].replace("T", " "),
                actual=float(lstm["actual_power_next_hour_kw"]),
                linear_prediction=float(linear["predicted_power_next_hour_kw"]),
                lstm_prediction=float(lstm["predicted_power_next_hour_kw"]),
            )
        )
    if not matched:
        raise ValueError("No matching model predictions were found.")
    return matched


def load_metrics() -> dict[str, dict[str, float]]:
    """Load final metrics already calculated on the same test hours."""
    with (RESULTS_DIR / "model_comparison_metrics.json").open("r", encoding="utf-8") as file:
        results = json.load(file)
    return {
        "linear": results["multiple_linear_regression"],
        "lstm": results["lstm"],
    }


class EnergyDashboard(tk.Tk):
    """Interactive viewer for the saved project predictions."""

    def __init__(self, rows: list[PredictionRow], metrics: dict[str, dict[str, float]]) -> None:
        super().__init__()
        self.rows = rows
        self.metrics = metrics
        self.index = 0
        self.title("Smart Energy Consumption Prediction System")
        self.minsize(960, 700)
        self.configure(background="#f4f4f4")
        self._configure_style()
        self._create_widgets()
        self._update_screen()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#202020", background="#f4f4f4")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#555555", background="#f4f4f4")
        style.configure("MetricName.TLabel", font=("Segoe UI", 10), foreground="#555555", background="#ffffff")
        style.configure("MetricValue.TLabel", font=("Segoe UI", 18, "bold"), foreground="#202020", background="#ffffff")
        style.configure("Info.TLabel", font=("Segoe UI", 11), foreground="#202020", background="#f4f4f4")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))
        style.configure("Accent.TButton", foreground="#ffffff", background="#9f2b2b")
        style.map("Accent.TButton", background=[("active", "#7e2020")])

    def _create_widgets(self) -> None:
        content = ttk.Frame(self, padding=22)
        content.pack(fill="both", expand=True)

        ttk.Label(content, text="Smart Energy Consumption Prediction System", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            content,
            text="Saved test results: predict household electricity consumption one hour ahead.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 16))

        model_frame = ttk.Frame(content)
        model_frame.pack(fill="x")
        self._metric_card(model_frame, "Linear Regression MAE", "linear_mae").pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._metric_card(model_frame, "LSTM MAE", "lstm_mae").pack(side="left", fill="x", expand=True, padx=8)
        self._metric_card(model_frame, "LSTM R squared", "lstm_r2").pack(side="left", fill="x", expand=True, padx=(8, 0))

        controls = ttk.Frame(content)
        controls.pack(fill="x", pady=(22, 10))
        ttk.Button(controls, text="Previous hour", command=self._previous).pack(side="left")
        ttk.Button(controls, text="Next hour", style="Accent.TButton", command=self._next).pack(side="left", padx=(8, 18))
        ttk.Label(controls, text="Test hour:", style="Info.TLabel").pack(side="left")
        self.hour_var = tk.IntVar(value=1)
        self.hour_box = ttk.Spinbox(
            controls,
            from_=1,
            to=len(self.rows),
            width=8,
            textvariable=self.hour_var,
            command=self._jump_to_hour,
        )
        self.hour_box.pack(side="left", padx=(8, 8))
        self.hour_box.bind("<Return>", self._jump_to_hour)
        self.time_label = ttk.Label(controls, style="Info.TLabel")
        self.time_label.pack(side="left")

        prediction_frame = ttk.Frame(content)
        prediction_frame.pack(fill="x", pady=(0, 16))
        self._metric_card(prediction_frame, "Actual consumption", "actual").pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._metric_card(prediction_frame, "Linear Regression prediction", "linear_prediction").pack(side="left", fill="x", expand=True, padx=8)
        self._metric_card(prediction_frame, "LSTM prediction", "lstm_prediction").pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.summary_label = ttk.Label(content, style="Info.TLabel", wraplength=880)
        self.summary_label.pack(anchor="w", pady=(0, 10))

        self.canvas = tk.Canvas(content, height=330, background="#ffffff", highlightthickness=1, highlightbackground="#cfcfcf")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw_chart())

        legend = ttk.Frame(content)
        legend.pack(anchor="w", pady=(10, 0))
        for colour, text in (("#222222", "Actual"), ("#9f2b2b", "Linear Regression"), ("#27734c", "LSTM")):
            tk.Label(legend, background=colour, width=2).pack(side="left", padx=(0, 5))
            ttk.Label(legend, text=text, style="Subtitle.TLabel").pack(side="left", padx=(0, 16))

    def _metric_card(self, parent: ttk.Frame, label: str, key: str) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        ttk.Label(card, text=label, style="MetricName.TLabel").pack(anchor="w")
        value = ttk.Label(card, style="MetricValue.TLabel")
        value.pack(anchor="w", pady=(5, 0))
        setattr(self, f"{key}_label", value)
        return card

    def _previous(self) -> None:
        self.index = max(0, self.index - 1)
        self._update_screen()

    def _next(self) -> None:
        self.index = min(len(self.rows) - 1, self.index + 1)
        self._update_screen()

    def _jump_to_hour(self, _event: object | None = None) -> None:
        try:
            self.index = min(len(self.rows) - 1, max(0, int(self.hour_var.get()) - 1))
        except tk.TclError:
            self.hour_var.set(self.index + 1)
        self._update_screen()

    def _update_screen(self) -> None:
        row = self.rows[self.index]
        linear_error = abs(row.actual - row.linear_prediction)
        lstm_error = abs(row.actual - row.lstm_prediction)
        self.linear_mae_label.config(text=f"{self.metrics['linear']['mae_kw']:.3f} kW")
        self.lstm_mae_label.config(text=f"{self.metrics['lstm']['mae_kw']:.3f} kW")
        self.lstm_r2_label.config(text=f"{self.metrics['lstm']['r_squared']:.3f}")
        self.actual_label.config(text=f"{row.actual:.3f} kW")
        self.linear_prediction_label.config(text=f"{row.linear_prediction:.3f} kW")
        self.lstm_prediction_label.config(text=f"{row.lstm_prediction:.3f} kW")
        self.hour_var.set(self.index + 1)
        self.time_label.config(text=f"Prediction time: {row.target_time}")
        better_model = "LSTM" if lstm_error <= linear_error else "Multiple Linear Regression"
        self.summary_label.config(
            text=(
                f"For this hour, Linear Regression error is {linear_error:.3f} kW and LSTM error is "
                f"{lstm_error:.3f} kW. The closer prediction is {better_model}."
            )
        )
        self._draw_chart()

    def _draw_chart(self) -> None:
        if not hasattr(self, "canvas"):
            return
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 700)
        height = max(canvas.winfo_height(), 250)
        left, right, top, bottom = 62, 24, 24, 42
        start = max(0, self.index - 24)
        end = min(len(self.rows), self.index + 25)
        subset = self.rows[start:end]
        values = [value for row in subset for value in (row.actual, row.linear_prediction, row.lstm_prediction)]
        minimum = min(values)
        maximum = max(values)
        padding = max(0.1, (maximum - minimum) * 0.1)
        minimum -= padding
        maximum += padding
        chart_width = width - left - right
        chart_height = height - top - bottom

        for fraction in range(5):
            y_value = minimum + (maximum - minimum) * fraction / 4
            y = top + chart_height - chart_height * fraction / 4
            canvas.create_line(left, y, width - right, y, fill="#dedede")
            canvas.create_text(left - 8, y, text=f"{y_value:.1f}", anchor="e", fill="#555555", font=("Segoe UI", 9))
        canvas.create_line(left, top, left, height - bottom, fill="#555555")
        canvas.create_line(left, height - bottom, width - right, height - bottom, fill="#555555")
        canvas.create_text(17, top + chart_height / 2, text="kW", angle=90, fill="#555555", font=("Segoe UI", 10, "bold"))

        def coordinates(series: list[float]) -> list[float]:
            points = []
            denominator = max(1, len(series) - 1)
            for point_index, value in enumerate(series):
                x = left + chart_width * point_index / denominator
                y = top + chart_height * (maximum - value) / (maximum - minimum)
                points.extend((x, y))
            return points

        series = (
            ("#222222", [row.actual for row in subset]),
            ("#9f2b2b", [row.linear_prediction for row in subset]),
            ("#27734c", [row.lstm_prediction for row in subset]),
        )
        for colour, values_for_line in series:
            canvas.create_line(*coordinates(values_for_line), fill=colour, width=2, smooth=True)

        chosen = self.index - start
        x = left + chart_width * chosen / max(1, len(subset) - 1)
        canvas.create_line(x, top, x, height - bottom, fill="#777777", dash=(4, 3))
        canvas.create_text(left, height - 18, text="24 hours before", anchor="w", fill="#555555", font=("Segoe UI", 9))
        canvas.create_text(x, height - 18, text="Selected hour", fill="#555555", font=("Segoe UI", 9))
        canvas.create_text(width - right, height - 18, text="24 hours after", anchor="e", fill="#555555", font=("Segoe UI", 9))


def main() -> None:
    try:
        app = EnergyDashboard(load_rows(), load_metrics())
    except (FileNotFoundError, ValueError, KeyError) as error:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Dashboard cannot start", f"Could not load the saved project results.\n\n{error}")
        root.destroy()
        return
    app.mainloop()


if __name__ == "__main__":
    main()
