"""Abstract base class enforcing the fit/predict/save/load contract (rule C38)."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseMLModel(ABC):
    """Abstract base for every model in the project (rule C38).

    For regression models (RULPredictor): `predict(X)` returns the RUL vector;
    `predict_proba` is not meaningful and raises NotImplementedError.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseMLModel":
        """Fit the model to (X, y). Returns self."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return RUL predictions for each window in X."""

    @abstractmethod
    def save(self, dir_: Path) -> None:
        """Persist the model to a directory (state_dict for torch models — rule C47)."""

    @classmethod
    @abstractmethod
    def load(cls, dir_: Path) -> "BaseMLModel":
        """Load the model from a directory."""
