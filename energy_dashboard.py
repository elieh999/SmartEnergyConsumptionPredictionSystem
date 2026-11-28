"""Modern desktop dashboard for the Smart Energy Consumption Prediction System.

It is a read only viewer for the final saved project results. Run this file
directly during development, or open SmartEnergyDashboard.exe after building it.
"""

from __future__ import annotations

import csv
import json
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


BACKGROUND = "#0b1220"
PANEL = "#121c2d"
CARD = "#17243a"
CARD_HOVER = "#1d2d46"
TEXT = "#f5f7fb"
MUTED = "#a7b3c7"
GRID = "#33445e"
ACCENT = "#57d6a4"
ORANGE = "#ffb454"
BLUE = "#70a7ff"


def resource_root() -> Path:
    """Use bundled data files when running as an executable."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


RESULTS_DIR = resource_root() / "results"


@dataclass
class PredictionRow:
    target_time: str
    actual: float
    linear_prediction: float
    lstm_prediction: float


def load_rows() -> list[PredictionRow]:
    """Match both model results on the same test hours."""
    with (RESULTS_DIR / "linear_regression_test_predictions.csv").open("r", encoding="utf-8", newline="") as file:
        linear_by_time = {row["datetime"]: row for row in csv.DictReader(file)}
    with (RESULTS_DIR / "lstm_test_predictions.csv").open("r", encoding="utf-8", newline="") as file:
        lstm_rows = list(csv.DictReader(file))

    matched = []
    for lstm in lstm_rows:
        linear = linear_by_time.get(lstm["input_end_datetime"])
        if linear is not None:
            matched.append(PredictionRow(lstm["target_datetime"].replace("T", " "), float(lstm["actual_power_next_hour_kw"]), float(linear["predicted_power_next_hour_kw"]), float(lstm["predicted_power_next_hour_kw"])))
    if not matched:
        raise ValueError("No matching predictions were found in the results folder.")
    return matched


def load_metrics() -> dict[str, dict[str, float]]:
    with (RESULTS_DIR / "model_comparison_metrics.json").open("r", encoding="utf-8") as file:
        comparison = json.load(file)
    return {"linear": comparison["multiple_linear_regression"], "lstm": comparison["lstm"]}


class EnergyDashboard(tk.Tk):
    """Interactive desktop app that presents the completed project."""

    def __init__(self, rows: list[PredictionRow], metrics: dict[str, dict[str, float]]) -> None:
        super().__init__()
        self.rows, self.metrics, self.index, self.current_page = rows, metrics, 0, "Overview"
        self.figure_photo: ImageTk.PhotoImage | None = None
        self.title("Smart Energy Dashboard")
        self.geometry("1280x800")
        self.minsize(1040, 690)
        self.configure(background=BACKGROUND)
        self._build_shell()
        self.show_page("Overview")

    def _build_shell(self) -> None:
        self.sidebar = tk.Frame(self, background=PANEL, width=235)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.main = tk.Frame(self, background=BACKGROUND)
        self.main.pack(side="left", fill="both", expand=True)

        brand = tk.Frame(self.sidebar, background=PANEL)
        brand.pack(fill="x", padx=22, pady=(28, 32))
        tk.Label(brand, text="⚡", font=("Segoe UI Emoji", 28), foreground=ACCENT, background=PANEL).pack(side="left")
        text = tk.Frame(brand, background=PANEL)
        text.pack(side="left", padx=(10, 0))
        tk.Label(text, text="ENERGY AI", font=("Segoe UI", 16, "bold"), foreground=TEXT, background=PANEL).pack(anchor="w")
        tk.Label(text, text="Forecasting dashboard", font=("Segoe UI", 9), foreground=MUTED, background=PANEL).pack(anchor="w")

        self.nav_buttons: dict[str, tk.Button] = {}
        for page, icon in (("Overview", "◉"), ("Model comparison", "◫"), ("Project details", "⌘"), ("Final graphs", "▧")):
            button = tk.Button(self.sidebar, text=f"  {icon}   {page}", command=lambda item=page: self.show_page(item), anchor="w", relief="flat", borderwidth=0, font=("Segoe UI", 11, "bold"), foreground=MUTED, background=PANEL, activeforeground=TEXT, activebackground=CARD_HOVER, cursor="hand2", padx=21, pady=13)
            button.pack(fill="x", padx=10, pady=3)
            self.nav_buttons[page] = button

        bottom = tk.Frame(self.sidebar, background=PANEL)
        bottom.pack(side="bottom", fill="x", padx=22, pady=24)
        tk.Label(bottom, text="Final project results", font=("Segoe UI", 10, "bold"), foreground=TEXT, background=PANEL).pack(anchor="w")
        tk.Label(bottom, text="UCI household electricity data\nOne hour ahead forecasting", justify="left", font=("Segoe UI", 9), foreground=MUTED, background=PANEL).pack(anchor="w", pady=(4, 0))

    def show_page(self, page: str) -> None:
        self.current_page = page
        for name, button in self.nav_buttons.items():
            active = name == page
            button.configure(background=CARD if active else PANEL, foreground=TEXT if active else MUTED)
        for child in self.main.winfo_children():
            child.destroy()
        if page == "Overview": self._overview_page()
        elif page == "Model comparison": self._comparison_page()
        elif page == "Project details": self._project_page()
        else: self._graphs_page()

    def _page_header(self, title: str, subtitle: str) -> None:
        header = tk.Frame(self.main, background=BACKGROUND)
        header.pack(fill="x", padx=34, pady=(28, 16))
        tk.Label(header, text=title, font=("Segoe UI", 24, "bold"), foreground=TEXT, background=BACKGROUND).pack(anchor="w")
        tk.Label(header, text=subtitle, font=("Segoe UI", 10), foreground=MUTED, background=BACKGROUND).pack(anchor="w", pady=(4, 0))

    def _metric_card(self, parent: tk.Frame, title: str, value: str, note: str, colour: str) -> tuple[tk.Frame, tk.Label]:
        card = tk.Frame(parent, background=CARD, highlightthickness=1, highlightbackground="#263956")
        tk.Frame(card, height=4, background=colour).pack(fill="x")
        inner = tk.Frame(card, background=CARD)
        inner.pack(fill="both", expand=True, padx=17, pady=15)
        tk.Label(inner, text=title, font=("Segoe UI", 10), foreground=MUTED, background=CARD).pack(anchor="w")
        value_label = tk.Label(inner, text=value, font=("Segoe UI", 22, "bold"), foreground=TEXT, background=CARD)
        value_label.pack(anchor="w", pady=(5, 1))
        tk.Label(inner, text=note, font=("Segoe UI", 9), foreground=colour, background=CARD).pack(anchor="w")
        return card, value_label

    def _overview_page(self) -> None:
        self._page_header("Project overview", "Smart Energy Consumption Prediction System  •  final test results")
        improvement = (1 - self.metrics["lstm"]["mae_kw"] / self.metrics["linear"]["mae_kw"]) * 100
        cards = tk.Frame(self.main, background=BACKGROUND)
        cards.pack(fill="x", padx=34)
        cards.columnconfigure((0, 1, 2), weight=1, uniform="cards")
        for column, (title, value, note, colour) in enumerate((("LSTM MAE", f"{self.metrics['lstm']['mae_kw']:.3f} kW", "Lower is better", ACCENT), ("LSTM R squared", f"{self.metrics['lstm']['r_squared']:.3f}", "Higher is better", BLUE), ("Improvement in MAE", f"{improvement:.1f}%", "Compared with Linear Regression", ORANGE))):
            card, _ = self._metric_card(cards, title, value, note, colour)
            card.grid(row=0, column=column, sticky="ew", padx=(0, 8) if column == 0 else (8, 0) if column == 2 else 8)

        toolbar = tk.Frame(self.main, background=BACKGROUND)
        toolbar.pack(fill="x", padx=34, pady=(24, 11))
        tk.Label(toolbar, text="Test hour explorer", font=("Segoe UI", 15, "bold"), foreground=TEXT, background=BACKGROUND).pack(side="left")
        tk.Label(toolbar, text="Inspect the real result and both predictions", font=("Segoe UI", 10), foreground=MUTED, background=BACKGROUND).pack(side="left", padx=12)
        tk.Button(toolbar, text="◀  Previous", command=self.previous, font=("Segoe UI", 10, "bold"), foreground=TEXT, background=CARD, activebackground=CARD_HOVER, activeforeground=TEXT, relief="flat", padx=12, pady=7, cursor="hand2").pack(side="right")
        tk.Button(toolbar, text="Next  ▶", command=self.next, font=("Segoe UI", 10, "bold"), foreground=BACKGROUND, background=ACCENT, activebackground="#76e7ba", relief="flat", padx=12, pady=7, cursor="hand2").pack(side="right", padx=(0, 8))

        self.hour_info = tk.Label(self.main, font=("Segoe UI", 10), foreground=MUTED, background=BACKGROUND)
        self.hour_info.pack(anchor="w", padx=34)
        selected = tk.Frame(self.main, background=BACKGROUND)
        selected.pack(fill="x", padx=34, pady=(10, 12))
        selected.columnconfigure((0, 1, 2), weight=1, uniform="selected")
        card, self.actual_text = self._metric_card(selected, "Actual consumption", "", "Measured electricity use", TEXT); card.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        card, self.linear_text = self._metric_card(selected, "Linear Regression", "", "Predicted consumption", ORANGE); card.grid(row=0, column=1, sticky="ew", padx=8)
        card, self.lstm_text = self._metric_card(selected, "LSTM", "", "Predicted consumption", ACCENT); card.grid(row=0, column=2, sticky="ew", padx=(8, 0))

        chart_frame = tk.Frame(self.main, background=CARD, highlightthickness=1, highlightbackground="#263956")
        chart_frame.pack(fill="both", expand=True, padx=34, pady=(2, 28))
        self.insight_text = tk.Label(chart_frame, font=("Segoe UI", 10), foreground=MUTED, background=CARD, anchor="w")
        self.insight_text.pack(fill="x", padx=18, pady=(14, 0))
        self.canvas = tk.Canvas(chart_frame, background=CARD, highlightthickness=0, height=270)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(4, 10))
        self.canvas.bind("<Configure>", lambda _event: self.draw_chart())
        self.update_overview()

    def previous(self) -> None:
        self.index = max(0, self.index - 1); self.update_overview()

    def next(self) -> None:
        self.index = min(len(self.rows) - 1, self.index + 1); self.update_overview()

    def update_overview(self) -> None:
        if self.current_page != "Overview" or not hasattr(self, "canvas"): return
        row = self.rows[self.index]
        linear_error, lstm_error = abs(row.actual - row.linear_prediction), abs(row.actual - row.lstm_prediction)
        self.hour_info.configure(text=f"Test hour {self.index + 1:,} of {len(self.rows):,}   •   prediction time: {row.target_time}")
        self.actual_text.configure(text=f"{row.actual:.3f} kW"); self.linear_text.configure(text=f"{row.linear_prediction:.3f} kW"); self.lstm_text.configure(text=f"{row.lstm_prediction:.3f} kW")
        winner = "LSTM" if lstm_error <= linear_error else "Linear Regression"
        self.insight_text.configure(text=f"For this test hour: Linear Regression error = {linear_error:.3f} kW   •   LSTM error = {lstm_error:.3f} kW   •   Closer model: {winner}")
        self.draw_chart()

    def draw_chart(self) -> None:
        if self.current_page != "Overview" or not hasattr(self, "canvas"): return
        canvas = self.canvas; canvas.delete("all")
        width, height = max(760, canvas.winfo_width()), max(260, canvas.winfo_height())
        left, right, top, bottom = 64, 24, 22, 34
        start, end = max(0, self.index - 24), min(len(self.rows), self.index + 25)
        subset = self.rows[start:end]
        numbers = [number for row in subset for number in (row.actual, row.linear_prediction, row.lstm_prediction)]
        lower, upper = min(numbers), max(numbers); margin = max(0.10, (upper - lower) * 0.12); lower, upper = lower - margin, upper + margin
        chart_width, chart_height = width - left - right, height - top - bottom
        for tick in range(5):
            value = lower + (upper - lower) * tick / 4; y = top + chart_height - chart_height * tick / 4
            canvas.create_line(left, y, width - right, y, fill=GRID); canvas.create_text(left - 10, y, text=f"{value:.1f}", anchor="e", fill=MUTED, font=("Segoe UI", 9))
        canvas.create_line(left, top, left, height - bottom, fill=MUTED); canvas.create_line(left, height - bottom, width - right, height - bottom, fill=MUTED)
        canvas.create_text(18, top + chart_height / 2, text="kW", angle=90, fill=MUTED, font=("Segoe UI", 9, "bold"))
        def points(values: list[float]) -> list[float]:
            return [coordinate for number, value in enumerate(values) for coordinate in (left + chart_width * number / max(1, len(values) - 1), top + chart_height * (upper - value) / (upper - lower))]
        for colour, series in ((TEXT, [row.actual for row in subset]), (ORANGE, [row.linear_prediction for row in subset]), (ACCENT, [row.lstm_prediction for row in subset])):
            canvas.create_line(*points(series), fill=colour, width=2, smooth=True)
        chosen = left + chart_width * (self.index - start) / max(1, len(subset) - 1)
        canvas.create_line(chosen, top, chosen, height - bottom, fill=BLUE, dash=(4, 3)); canvas.create_text(chosen, height - 12, text="selected", fill=BLUE, font=("Segoe UI", 9))

    def _comparison_page(self) -> None:
        self._page_header("Model comparison", "Both models were evaluated on the same 5,050 test hours")
        table = tk.Frame(self.main, background=CARD, highlightthickness=1, highlightbackground="#263956"); table.pack(fill="x", padx=34, pady=(3, 22))
        headings = ("Model", "MAE", "RMSE", "R squared")
        rows = (("Multiple Linear Regression", self.metrics["linear"]["mae_kw"], self.metrics["linear"]["rmse_kw"], self.metrics["linear"]["r_squared"]), ("LSTM", self.metrics["lstm"]["mae_kw"], self.metrics["lstm"]["rmse_kw"], self.metrics["lstm"]["r_squared"]))
        for column, heading in enumerate(headings):
            tk.Label(table, text=heading, font=("Segoe UI", 10, "bold"), foreground=MUTED, background=CARD, anchor="w").grid(row=0, column=column, sticky="ew", padx=20, pady=(18, 10)); table.columnconfigure(column, weight=2 if column == 0 else 1)
        for row_number, row in enumerate(rows, start=1):
            colour = ACCENT if row[0] == "LSTM" else ORANGE
            tk.Label(table, text=row[0], font=("Segoe UI", 13, "bold"), foreground=TEXT, background=CARD, anchor="w").grid(row=row_number, column=0, sticky="ew", padx=20, pady=16)
            for column, number in enumerate(row[1:], start=1):
                tk.Label(table, text=f"{number:.3f}{' kW' if column < 3 else ''}", font=("Segoe UI", 13, "bold"), foreground=colour, background=CARD, anchor="w").grid(row=row_number, column=column, sticky="ew", padx=20, pady=16)
        insight = tk.Frame(self.main, background=PANEL, highlightthickness=1, highlightbackground="#263956"); insight.pack(fill="x", padx=34)
        mae_drop = (1 - self.metrics["lstm"]["mae_kw"] / self.metrics["linear"]["mae_kw"]) * 100; rmse_drop = (1 - self.metrics["lstm"]["rmse_kw"] / self.metrics["linear"]["rmse_kw"]) * 100
        tk.Label(insight, text="What this means", font=("Segoe UI", 16, "bold"), foreground=TEXT, background=PANEL).pack(anchor="w", padx=22, pady=(19, 6))
        tk.Label(insight, text=f"The LSTM reduces average error by {mae_drop:.1f}% and RMSE by {rmse_drop:.1f}% compared with Multiple Linear Regression. It also explains more of the variation in household electricity use.", wraplength=850, justify="left", font=("Segoe UI", 11), foreground=MUTED, background=PANEL).pack(anchor="w", padx=22, pady=(0, 19))

    def _project_page(self) -> None:
        self._page_header("Project details", "From raw household measurements to a next hour electricity forecast")
        sections = (("1", "Dataset", "UCI Individual Household Electric Power Consumption dataset. 2,075,259 minute readings from one household, collected from December 2006 to November 2010."), ("2", "Preprocessing", "Question marks were treated as missing values and 25,979 incomplete readings were removed. Minute readings were grouped into hourly observations and normalized using only training data."), ("3", "Features", "Hour, day of week, month, weekend status, electrical measurements, sub metering totals, and historical power readings were prepared. The LSTM reads 24 completed hours with 11 features per hour."), ("4", "Models", "Multiple Linear Regression is the baseline. The final deep learning model is one LSTM layer with 32 units followed by one output neuron."), ("5", "Evaluation", "Data was split in chronological order: 70% training, 15% validation, and 15% test. MAE, RMSE, and R squared were used to compare the models."))
        content = tk.Frame(self.main, background=BACKGROUND); content.pack(fill="both", expand=True, padx=34, pady=(0, 22))
        for number, title, description in sections:
            item = tk.Frame(content, background=CARD, highlightthickness=1, highlightbackground="#263956"); item.pack(fill="x", pady=5)
            tk.Label(item, text=number, font=("Segoe UI", 15, "bold"), foreground=BACKGROUND, background=ACCENT, width=3, pady=7).pack(side="left", padx=16, pady=12)
            text = tk.Frame(item, background=CARD); text.pack(side="left", fill="x", expand=True, pady=12)
            tk.Label(text, text=title, font=("Segoe UI", 13, "bold"), foreground=TEXT, background=CARD).pack(anchor="w")
            tk.Label(text, text=description, wraplength=810, justify="left", font=("Segoe UI", 10), foreground=MUTED, background=CARD).pack(anchor="w", pady=(3, 0))

    def _graphs_page(self) -> None:
        self._page_header("Final graphs", "The same report ready figures included in the project results folder")
        controls = tk.Frame(self.main, background=BACKGROUND); controls.pack(fill="x", padx=34, pady=(0, 12))
        tk.Label(controls, text="Choose a graph:", font=("Segoe UI", 10, "bold"), foreground=TEXT, background=BACKGROUND).pack(side="left")
        self.graph_names = {"LSTM actual versus predicted": "lstm_actual_vs_predicted.png", "Linear Regression actual versus predicted": "linear_regression_actual_vs_predicted.png", "LSTM training and validation loss": "lstm_training_validation_loss.png", "Model error comparison": "model_error_comparison.png"}
        self.graph_choice = tk.StringVar(value="LSTM actual versus predicted")
        box = ttk.Combobox(controls, textvariable=self.graph_choice, values=list(self.graph_names), state="readonly", width=40); box.pack(side="left", padx=10); box.bind("<<ComboboxSelected>>", lambda _event: self.show_graph())
        self.graph_label = tk.Label(self.main, background=CARD); self.graph_label.pack(fill="both", expand=True, padx=34, pady=(0, 28)); self.graph_label.bind("<Configure>", lambda _event: self.show_graph())
        self.show_graph()

    def show_graph(self) -> None:
        if self.current_page != "Final graphs" or not hasattr(self, "graph_label"): return
        try:
            image = Image.open(RESULTS_DIR / "figures" / self.graph_names[self.graph_choice.get()])
            image.thumbnail((max(700, self.graph_label.winfo_width() - 24), max(420, self.graph_label.winfo_height() - 24)), Image.Resampling.LANCZOS)
            self.figure_photo = ImageTk.PhotoImage(image); self.graph_label.configure(image=self.figure_photo, text="")
        except FileNotFoundError:
            self.graph_label.configure(image="", text="Graph file was not found.", foreground=TEXT, font=("Segoe UI", 14, "bold"))


def main() -> None:
    try:
        EnergyDashboard(load_rows(), load_metrics()).mainloop()
    except (FileNotFoundError, KeyError, ValueError) as error:
        root = tk.Tk(); root.withdraw(); messagebox.showerror("Smart Energy Dashboard", f"Could not load the saved project results.\n\n{error}"); root.destroy()


if __name__ == "__main__":
    main()
