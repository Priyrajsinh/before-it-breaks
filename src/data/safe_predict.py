"""NaN/inf/shape guard wrapping every model prediction call (rule C36)."""

from typing import Callable

import numpy as np

from src.exceptions import PredictionError
from src.logger import get_logger

logger = get_logger(__name__)


def safe_predict(
    model_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """NaN/inf/shape guard around any model prediction call (rule C36)."""
    if x.shape != expected_shape:
        raise PredictionError(
            f"shape mismatch: got {x.shape}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(x)):
        raise PredictionError("input contains NaN or inf")
    logger.info(
        "safe_predict input stats",
        extra={"shape": x.shape, "mean": float(x.mean()), "std": float(x.std())},
    )
    y = model_fn(x)
    if not np.all(np.isfinite(y)):
        raise PredictionError("model output contains NaN or inf")
    return y
