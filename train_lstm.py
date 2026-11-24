"""Train a 32 unit LSTM for next hour household power prediction.

The network has one LSTM layer with 32 units and one output neuron. It learns
from 24 completed hourly observations and is monitored on the later validation
split. The test split is deliberately not loaded in this training script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from tinygrad import Context, Tensor, TinyJit, nn
from tinygrad.nn.optim import Adam


# Tinygrad compiles a small CPU helper when exporting model tensors on Windows.
# Use the available GCC installation instead of assuming that clang is installed.
GCC_PATH = Path(r"C:\msys64\ucrt64\bin\gcc.exe")
if GCC_PATH.exists():
    os.environ.setdefault("CC", str(GCC_PATH))


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
SEED = 42
EPOCHS = 13
BATCH_SIZE = 128
LEARNING_RATE = 0.001


class LSTMForecaster:
    """One LSTM layer followed by one output neuron."""

    def __init__(self, input_features: int, hidden_units: int = 32) -> None:
        self.lstm = nn.LSTMCell(input_features, hidden_units)
        self.output = nn.Linear(hidden_units, 1)

    def __call__(self, x: Tensor) -> Tensor:
        hidden_state = None
        for time_step in range(x.shape[1]):
            hidden_state = self.lstm(x[:, time_step, :], hidden_state)
        return self.output(hidden_state[0])


def load_sequences(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load sequences created by prepare_lstm_sequences.py."""
    archive = np.load(DATA_DIR / f"lstm_{split_name}_sequences.npz")
    return archive["x"], archive["y"]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return metrics in the original global active power kW scale."""
    errors = actual.reshape(-1) - predicted.reshape(-1)
    squared_error = float(np.sum(errors**2))
    total_variation = float(np.sum((actual.reshape(-1) - actual.mean()) ** 2))
    return {
        "mae_kw": float(np.mean(np.abs(errors))),
        "rmse_kw": float(np.sqrt(np.mean(errors**2))),
        "r_squared": float(1 - squared_error / total_variation),
    }


def predict_in_batches(model: LSTMForecaster, x_values: np.ndarray) -> np.ndarray:
    """Predict without holding the entire validation split in memory."""
    predictions = []
    for start in range(0, len(x_values), BATCH_SIZE):
        batch = Tensor(x_values[start : start + BATCH_SIZE])
        predictions.append(model(batch).detach().numpy().reshape(-1))
    return np.concatenate(predictions)


def main() -> None:
    Tensor.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    x_train, y_train = load_sequences("train")
    x_validation, y_validation = load_sequences("validation")

    model = LSTMForecaster(input_features=x_train.shape[2], hidden_units=32)
    optimizer = Adam(nn.state.get_parameters(model), lr=LEARNING_RATE)
    history = {"loss": [], "val_loss": []}
    best_validation_loss = float("inf")
    best_state = None
    best_validation_prediction = None
    patience = 4
    unchanged_epochs = 0

    with Context(TRAINING=1):
        @TinyJit
        def train_batch(x_batch: Tensor, y_batch: Tensor) -> Tensor:
            prediction = model(x_batch)
            loss = ((prediction - y_batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss

        for epoch in range(EPOCHS):
            batch_losses = []
            # All training batches have the same shape, which allows TinyJit to
            # compile the LSTM update once and reuse it for later batches.
            last_full_batch_start = len(x_train) - BATCH_SIZE + 1
            for start in range(0, last_full_batch_start, BATCH_SIZE):
                x_batch = Tensor(x_train[start : start + BATCH_SIZE])
                y_batch = Tensor(y_train[start : start + BATCH_SIZE].reshape(-1, 1))
                loss = train_batch(x_batch, y_batch)
                batch_losses.append(float(loss.item()))

            validation_prediction = predict_in_batches(model, x_validation)
            validation_loss = float(np.mean((validation_prediction - y_validation) ** 2))
            training_loss = float(np.mean(batch_losses))
            history["loss"].append(training_loss)
            history["val_loss"].append(validation_loss)
            print(f"Epoch {epoch + 1}/{EPOCHS}  loss={training_loss:.6f}  val_loss={validation_loss:.6f}")

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = {name: value.detach().clone() for name, value in nn.state.get_state_dict(model).items()}
                best_validation_prediction = validation_prediction.copy()
                unchanged_epochs = 0
            else:
                unchanged_epochs += 1
                if unchanged_epochs >= patience:
                    print("Stopped early because validation loss did not improve for 4 epochs.")
                    break

        if best_state is None or best_validation_prediction is None:
            raise RuntimeError("LSTM training did not produce a model state.")

        # The metrics below use the validation predictions from the best epoch.

    validation_results = regression_metrics(y_validation, best_validation_prediction)

    (RESULTS_DIR / "lstm_training_history.json").write_text(
        json.dumps(
            {
                "epochs_requested": EPOCHS,
                "epochs_completed": len(history["loss"]),
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "architecture": "One LSTM layer with 32 units followed by one output neuron",
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "lstm_validation_metrics.json").write_text(
        json.dumps(
            {
                "model": "LSTM",
                "split": "validation",
                "rows": int(len(y_validation)),
                "metrics": validation_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"validation_metrics": validation_results, "epochs_completed": len(history["loss"])}, indent=2))


if __name__ == "__main__":
    main()
