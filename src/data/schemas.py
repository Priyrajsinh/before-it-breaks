"""Pydantic v2 request/response schemas for the FastAPI prediction service."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """A single engine inference request — 30-cycle sensor window."""

    engine_id: int = Field(..., ge=1)
    sensor_window: list[list[float]]

    @field_validator("sensor_window")
    @classmethod
    def _shape_check(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate sensor_window is exactly 30 rows × 17 features."""
        if len(v) != 30:
            raise ValueError(f"sensor_window must have 30 rows, got {len(v)}")
        if any(len(row) != 17 for row in v):
            raise ValueError("each row of sensor_window must have 17 features")
        return v


class PredictResponse(BaseModel):
    """Prediction result for a single engine."""

    engine_id: int
    predicted_rul: float
    health_status: Literal["Healthy", "Warning", "Critical"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    nl_summary: str = ""
    drift_flags: dict[str, bool] = {}


class ExplainResponse(BaseModel):
    """SHAP explanation for a single engine prediction."""

    engine_id: int
    predicted_rul: float
    shap_values: dict[str, float]
    top_features: list[str]
    nl_summary: str = ""


class HealthResponse(BaseModel):
    """API health check response."""

    status: Literal["healthy", "degraded", "down"]
    model_loaded: bool
    uptime_seconds: float
    memory_mb: float
    version: str
    total_predictions: int
