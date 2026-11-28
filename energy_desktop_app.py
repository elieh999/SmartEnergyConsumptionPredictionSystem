"""Native Qt desktop application for the finished energy prediction project."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


NAVY = "#17324d"
BLUE = "#1f5d8f"
PALE_BLUE = "#edf4fa"
TEXT = "#1e2a35"
MUTED = "#667789"
LINEAR = "#a65b20"
LSTM = "#147a58"


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


RESULTS_DIR = resource_root() / "results"


@dataclass
class PredictionRow:
    timestamp: str
    actual: float
    linear: float
    lstm: float


def load_rows() -> list[PredictionRow]:
    with (RESULTS_DIR / "linear_regression_test_predictions.csv").open("r", encoding="utf-8", newline="") as file:
        linear = {row["datetime"]: row for row in csv.DictReader(file)}
    with (RESULTS_DIR / "lstm_test_predictions.csv").open("r", encoding="utf-8", newline="") as file:
        lstm = list(csv.DictReader(file))
    rows = [
        PredictionRow(
            row["target_datetime"].replace("T", " "),
            float(row["actual_power_next_hour_kw"]),
            float(linear[row["input_end_datetime"]]["predicted_power_next_hour_kw"]),
            float(row["predicted_power_next_hour_kw"]),
        )
        for row in lstm
        if row["input_end_datetime"] in linear
    ]
    if not rows:
        raise ValueError("No matching model predictions were found.")
    return rows


def load_metrics() -> dict[str, dict[str, float]]:
    with (RESULTS_DIR / "model_comparison_metrics.json").open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {"linear": data["multiple_linear_regression"], "lstm": data["lstm"]}


class TrendChart(QWidget):
    """Small custom chart for current test hour and nearby observations."""

    def __init__(self, app: "EnergyApplication") -> None:
        super().__init__()
        self.app = app
        self.setMinimumHeight(290)

    def paintEvent(self, _event: object) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        rows = self.app.rows
        selected = self.app.index
        start, end = max(0, selected - 24), min(len(rows), selected + 25)
        subset = rows[start:end]
        values = [value for row in subset for value in (row.actual, row.linear, row.lstm)]
        low, high = min(values), max(values)
        padding = max(0.10, (high - low) * 0.10)
        low, high = low - padding, high + padding
        left, top, right, bottom = 60, 25, 20, 35
        width, height = self.width() - left - right, self.height() - top - bottom

        painter.setFont(QFont("Segoe UI", 8))
        for tick in range(5):
            y = top + height - height * tick / 4
            value = low + (high - low) * tick / 4
            painter.setPen(QPen(QColor("#d9e0e6"), 1))
            painter.drawLine(left, int(y), left + width, int(y))
            painter.setPen(QColor(MUTED))
            painter.drawText(4, int(y) + 4, f"{value:.1f}")
        painter.setPen(QPen(QColor("#7b8794"), 1))
        painter.drawLine(left, top, left, top + height)
        painter.drawLine(left, top + height, left + width, top + height)
        painter.setPen(QColor(MUTED))
        painter.drawText(4, 16, "kW")

        def path(numbers: list[float]) -> QPainterPath:
            result = QPainterPath()
            for point, value in enumerate(numbers):
                x = left + width * point / max(1, len(numbers) - 1)
                y = top + height * (high - value) / (high - low)
                if point == 0:
                    result.moveTo(QPointF(x, y))
                else:
                    result.lineTo(QPointF(x, y))
            return result

        for colour, numbers in (("#263238", [row.actual for row in subset]), (LINEAR, [row.linear for row in subset]), (LSTM, [row.lstm for row in subset])):
            painter.setPen(QPen(QColor(colour), 2.2))
            painter.drawPath(path(numbers))
        x = left + width * (selected - start) / max(1, len(subset) - 1)
        painter.setPen(QPen(QColor(BLUE), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(x), top, int(x), top + height)
        painter.setPen(QColor(BLUE))
        painter.drawText(int(x) - 22, self.height() - 12, "selected")
        painter.end()


class EnergyApplication(QMainWindow):
    def __init__(self, rows: list[PredictionRow], metrics: dict[str, dict[str, float]]) -> None:
        super().__init__()
        self.rows, self.metrics, self.index = rows, metrics, 0
        self.graph_label: QLabel | None = None
        self.setWindowTitle("Smart Energy Consumption Prediction System")
        self.resize(1180, 760)
        self.setMinimumSize(980, 660)
        self._build()
        self._update_explorer()

    def _build(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._overview(), "Overview")
        tabs.addTab(self._comparison(), "Model comparison")
        tabs.addTab(self._explorer(), "Test hour explorer")
        tabs.addTab(self._project_details(), "Project details")
        tabs.addTab(self._graphs(), "Final graphs")
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(24, 18, 24, 24)
        heading = QLabel("Smart Energy Consumption Prediction System")
        heading.setObjectName("mainTitle")
        subheading = QLabel("One hour ahead household electricity forecasting  |  Final project application")
        subheading.setObjectName("subTitle")
        layout.addWidget(heading)
        layout.addWidget(subheading)
        layout.addWidget(tabs, 1)
        self.setCentralWidget(wrapper)

    def _metric_box(self, label: str, value: str, note: str) -> QGroupBox:
        box = QGroupBox(label)
        layout = QVBoxLayout(box)
        number = QLabel(value)
        number.setObjectName("metricValue")
        layout.addWidget(number)
        layout.addWidget(QLabel(note))
        return box

    def _overview(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel("Final result summary")
        intro.setObjectName("sectionTitle")
        layout.addWidget(intro)
        layout.addWidget(QLabel("The LSTM model performed better than the Multiple Linear Regression baseline on the same 5,050 test hours."))
        metrics = QGridLayout()
        improvement = (1 - self.metrics["lstm"]["mae_kw"] / self.metrics["linear"]["mae_kw"]) * 100
        metrics.addWidget(self._metric_box("LSTM MAE", f"{self.metrics['lstm']['mae_kw']:.3f} kW", "Average absolute prediction error"), 0, 0)
        metrics.addWidget(self._metric_box("LSTM RMSE", f"{self.metrics['lstm']['rmse_kw']:.3f} kW", "Penalises larger errors"), 0, 1)
        metrics.addWidget(self._metric_box("LSTM R squared", f"{self.metrics['lstm']['r_squared']:.3f}", "Explained variation"), 0, 2)
        metrics.addWidget(self._metric_box("MAE reduction", f"{improvement:.1f}%", "Compared with Linear Regression"), 0, 3)
        layout.addLayout(metrics)
        takeaway = QGroupBox("Conclusion")
        takeaway_layout = QVBoxLayout(takeaway)
        takeaway_layout.addWidget(QLabel("The LSTM uses the completed readings from the previous 24 hours as a sequence. This helps it recognise repeated daily consumption patterns and produces lower error than the baseline model."))
        layout.addWidget(takeaway)
        layout.addStretch()
        return page

    def _comparison(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Final comparison on identical test observations", objectName="sectionTitle"))
        table = QTableWidget(2, 4)
        table.setHorizontalHeaderLabels(["Model", "MAE (kW)", "RMSE (kW)", "R squared"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setAlternatingRowColors(True)
        values = (("Multiple Linear Regression", self.metrics["linear"]), ("LSTM", self.metrics["lstm"]))
        for row, (name, numbers) in enumerate(values):
            for column, text in enumerate((name, f"{numbers['mae_kw']:.3f}", f"{numbers['rmse_kw']:.3f}", f"{numbers['r_squared']:.3f}")):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if column else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, column, item)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        note = QGroupBox("Interpretation")
        note_layout = QVBoxLayout(note)
        note_layout.addWidget(QLabel("Lower MAE and RMSE mean predictions are closer to the actual electricity consumption. A higher R squared means the model explains more variation in the target. The LSTM is better on all three measurements."))
        layout.addWidget(note)
        layout.addStretch()
        return page

    def _explorer(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Choose test hour:"))
        self.hour_spin = QSpinBox()
        self.hour_spin.setRange(1, len(self.rows))
        self.hour_spin.valueChanged.connect(self._go_to_hour)
        controls.addWidget(self.hour_spin)
        previous = QPushButton("Previous")
        previous.clicked.connect(lambda: self.hour_spin.setValue(max(1, self.hour_spin.value() - 1)))
        controls.addWidget(previous)
        following = QPushButton("Next")
        following.clicked.connect(lambda: self.hour_spin.setValue(min(len(self.rows), self.hour_spin.value() + 1)))
        controls.addWidget(following)
        controls.addStretch()
        self.time_label = QLabel()
        controls.addWidget(self.time_label)
        outer.addLayout(controls)
        values = QGridLayout()
        self.actual_label = self._metric_box("Actual consumption", "", "Measured value")
        self.linear_label = self._metric_box("Linear Regression prediction", "", "Saved test prediction")
        self.lstm_label = self._metric_box("LSTM prediction", "", "Saved test prediction")
        values.addWidget(self.actual_label, 0, 0)
        values.addWidget(self.linear_label, 0, 1)
        values.addWidget(self.lstm_label, 0, 2)
        outer.addLayout(values)
        self.chart = TrendChart(self)
        outer.addWidget(self.chart, 1)
        legend = QLabel("— Actual consumption       — Linear Regression prediction       — LSTM prediction")
        legend.setObjectName("legend")
        outer.addWidget(legend)
        return page

    @staticmethod
    def _set_metric_value(box: QGroupBox, text: str) -> None:
        box.findChild(QLabel, "metricValue").setText(text)

    def _go_to_hour(self, value: int) -> None:
        self.index = value - 1
        self._update_explorer()

    def _update_explorer(self) -> None:
        if not hasattr(self, "hour_spin"):
            return
        row = self.rows[self.index]
        self.hour_spin.blockSignals(True)
        self.hour_spin.setValue(self.index + 1)
        self.hour_spin.blockSignals(False)
        self.time_label.setText(f"Prediction time: {row.timestamp}")
        self._set_metric_value(self.actual_label, f"{row.actual:.3f} kW")
        self._set_metric_value(self.linear_label, f"{row.linear:.3f} kW")
        self._set_metric_value(self.lstm_label, f"{row.lstm:.3f} kW")
        self.chart.update()

    def _project_details(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Project setup", objectName="sectionTitle"))
        form = QFormLayout()
        details = (
            ("Project task", "Predict Global Active Power one hour in the future"),
            ("Dataset", "UCI Individual Household Electric Power Consumption"),
            ("Original data", "2,075,259 minute readings, December 2006 to November 2010"),
            ("Cleaning", "25,979 incomplete records removed"),
            ("Prepared data", "33,972 hourly observations"),
            ("Data split", "70% training, 15% validation, 15% test in chronological order"),
            ("Baseline model", "Multiple Linear Regression with 14 input features"),
            ("Deep learning model", "One LSTM layer with 32 units and one output neuron"),
            ("LSTM input", "24 completed hours with 11 features for each hour"),
            ("Training", "13 epochs, batch size 128, learning rate 0.001"),
        )
        for label, value in details:
            form.addRow(QLabel(f"<b>{label}</b>"), QLabel(value))
        layout.addLayout(form)
        layout.addStretch()
        return page

    def _graphs(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Graph:"))
        self.graphs = {
            "LSTM actual versus predicted": "lstm_actual_vs_predicted.png",
            "Linear Regression actual versus predicted": "linear_regression_actual_vs_predicted.png",
            "LSTM training and validation loss": "lstm_training_validation_loss.png",
            "Model error comparison": "model_error_comparison.png",
        }
        self.graph_select = QComboBox()
        self.graph_select.addItems(self.graphs)
        self.graph_select.currentTextChanged.connect(self._show_graph)
        controls.addWidget(self.graph_select)
        controls.addStretch()
        layout.addLayout(controls)
        self.graph_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.graph_label.setStyleSheet("background: white; border: 1px solid #ccd6df;")
        layout.addWidget(self.graph_label, 1)
        self._show_graph(self.graph_select.currentText())
        return page

    def _show_graph(self, name: str) -> None:
        if self.graph_label is None:
            return
        pixmap = QPixmap(str(RESULTS_DIR / "figures" / self.graphs[name]))
        self.graph_label.setPixmap(pixmap.scaled(900, 520, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


def main() -> None:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    application.setStyleSheet(f"""
        QWidget {{ color: {TEXT}; background: #f7f9fb; font-family: 'Segoe UI'; font-size: 11px; }}
        QLabel#mainTitle {{ color: {NAVY}; font-size: 22px; font-weight: 700; }}
        QLabel#subTitle {{ color: {MUTED}; margin-bottom: 8px; }}
        QLabel#sectionTitle {{ color: {NAVY}; font-size: 16px; font-weight: 700; margin-top: 8px; }}
        QLabel#metricValue {{ color: {NAVY}; font-size: 22px; font-weight: 700; }}
        QLabel#legend {{ color: {MUTED}; }}
        QTabWidget::pane {{ border: 1px solid #ccd6df; background: white; top: -1px; }}
        QTabBar::tab {{ background: #e8eef3; color: {TEXT}; padding: 10px 16px; border: 1px solid #ccd6df; border-bottom: none; }}
        QTabBar::tab:selected {{ background: white; color: {BLUE}; font-weight: 700; }}
        QGroupBox {{ background: white; border: 1px solid #ccd6df; border-radius: 3px; margin-top: 12px; padding: 15px; font-weight: 700; color: {NAVY}; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; }}
        QPushButton {{ background: {NAVY}; color: white; border: none; border-radius: 3px; padding: 8px 16px; font-weight: 600; }}
        QPushButton:hover {{ background: {BLUE}; }}
        QSpinBox, QComboBox {{ background: white; border: 1px solid #aebdca; padding: 6px; min-width: 90px; }}
        QTableWidget {{ background: white; alternate-background-color: #f2f6f9; gridline-color: #d9e1e8; }}
        QHeaderView::section {{ background: {NAVY}; color: white; padding: 9px; border: none; font-weight: 700; }}
    """)
    window = EnergyApplication(load_rows(), load_metrics())
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
