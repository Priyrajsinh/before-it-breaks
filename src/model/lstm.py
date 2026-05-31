"""2-layer LSTM (RULPredictor) for Remaining Useful Life regression.

Rules C38 (implements BaseMLModel ABC), C47 (state_dict save), C49 (window
shape [batch, seq_len=30, n_features=17], batch-first).
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from src.exceptions import ModelNotFoundError
from src.model.base import BaseMLModel


class RULPredictor(BaseMLModel, nn.Module):
    """2-layer LSTM for Remaining Useful Life regression (rules C38, C47, C49)."""

    def __init__(self, config: dict) -> None:
        """Build the LSTM stack from the ``model`` section of ``config``."""
        nn.Module.__init__(self)
        m = config["model"]
        self.lstm = nn.LSTM(
            input_size=m["input_size"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            batch_first=True,
            dropout=m["dropout"] if m["num_layers"] > 1 else 0.0,
        )
        self.dropout = nn.Dropout(m["dropout"])
        self.fc = nn.Linear(m["hidden_size"], 1)
        self._cfg = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len=30, n_features=17] -> returns [batch]."""
        lstm_out, _ = self.lstm(x)
        last = self.dropout(lstm_out[:, -1, :])
        return self.fc(last).squeeze(-1)

    # --- BaseMLModel API ---

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RULPredictor":
        """Not used: the training entry point lives in src/model/train.py."""
        raise NotImplementedError("Training entry point lives in src/model/train.py")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass on numpy input -> numpy RUL vector."""
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            return self.forward(t).cpu().numpy()

    def save(self, dir_: Path) -> None:
        """Save state_dict + config — rule C47 (never torch.save(model, ...))."""
        dir_.mkdir(parents=True, exist_ok=True)
        # Serializing a state_dict (tensors only) is not a deserialization risk.
        torch.save(self.state_dict(), dir_ / "lstm_rul.pt")  # nosec B614
        with open(dir_ / "lstm_config.pkl", "wb") as fh:
            joblib.dump(self._cfg, fh)

    @classmethod
    def load(cls, dir_: Path) -> "RULPredictor":
        """Reconstruct from saved config + state_dict; raise if weights missing."""
        cfg_path = dir_ / "lstm_config.pkl"
        weight_path = dir_ / "lstm_rul.pt"
        if not weight_path.exists():
            raise ModelNotFoundError(f"missing {weight_path}")
        with open(cfg_path, "rb") as fh:
            cfg = joblib.load(fh)
        model = cls(cfg)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
