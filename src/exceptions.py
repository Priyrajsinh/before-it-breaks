"""Project-wide exception hierarchy."""


class ProjectBaseError(Exception):
    """Base error for the before-it-breaks project."""


class ConfigError(ProjectBaseError):
    """Raised when config/config.yaml is invalid or missing required keys."""


class DataLoadError(ProjectBaseError):
    """Raised when CMAPSS files cannot be loaded or fail schema validation."""


class ChecksumError(DataLoadError):
    """Raised when a CMAPSS file's SHA-256 does not match its sidecar."""


class ModelNotFoundError(ProjectBaseError):
    """Raised when models/lstm_rul.pt or models/scaler.pkl is missing at inference."""


class PredictionError(ProjectBaseError):
    """Raised by safe_predict() on NaN/inf/wrong-shape input or non-finite output."""


class DriftAlarmError(ProjectBaseError):
    """Raised when sensor PSI > threshold at predict-time (rule C42)."""
