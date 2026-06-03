"""FastAPI application — RUL prediction service (rules C36, C39, C42, C45, C48)."""

import json
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import load_config
from src.data.dataset import CMAPSSDataset
from src.data.schemas import (
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.exceptions import PredictionError
from src.explainability.shap_explainer import RULExplainer
from src.logger import get_logger
from src.model.predict import ModelServer, _health_status, _nl_summary

logger = get_logger(__name__)
cfg = load_config("config/config.yaml")
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Load model artefacts at startup; release on shutdown."""
    app.state.server = ModelServer(cfg)
    yield


app = FastAPI(
    title="Before It Breaks — Predictive Maintenance API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg["api"]["cors_origins"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=cfg["api"]["trusted_hosts"],
)
app.state.limiter = limiter
_rl_handler: Any = _rate_limit_exceeded_handler
app.add_exception_handler(RateLimitExceeded, _rl_handler)


@app.middleware("http")
async def content_length_guard(request: Request, call_next: Any) -> Any:
    """Reject requests whose Content-Length exceeds the configured cap."""
    max_bytes = cfg["api"]["max_payload_mb"] * 1024 * 1024
    cl = request.headers.get("content-length")
    if cl and int(cl) > max_bytes:
        return JSONResponse({"detail": "payload too large"}, status_code=413)
    return await call_next(request)


@app.exception_handler(PredictionError)
async def prediction_error_handler(_: Request, exc: PredictionError) -> JSONResponse:
    """Map PredictionError to HTTP 422 (rule C39)."""
    return JSONResponse({"detail": str(exc)}, status_code=422)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse)
def health(request: Request) -> dict[str, Any]:
    """Return server health, model status, memory usage, and prediction count."""
    srv: ModelServer = request.app.state.server
    proc = psutil.Process()
    return {
        "status": "healthy" if srv.model is not None else "down",
        "model_loaded": srv.model is not None,
        "uptime_seconds": srv.uptime_seconds(),
        "memory_mb": proc.memory_info().rss / (1024 * 1024),
        "version": "0.1.0",
        "total_predictions": srv.total_predictions,
    }


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


@app.post("/api/v1/predict", response_model=PredictResponse)
@limiter.limit(cfg["api"]["rate_limit_predict"])
def predict(request: Request, payload: PredictRequest) -> dict[str, Any]:
    """Run a single-engine RUL prediction with drift detection."""
    srv: ModelServer = request.app.state.server
    return srv.predict(payload.engine_id, payload.sensor_window)


@app.post("/api/v1/predict_batch", response_model=list[PredictResponse])
@limiter.limit(cfg["api"]["rate_limit_predict"])
def predict_batch(
    request: Request, payload: list[PredictRequest]
) -> list[dict[str, Any]]:
    """Run RUL predictions for a batch of engines (max 50)."""
    if len(payload) > cfg["api"]["max_batch_size"]:
        raise HTTPException(
            status_code=422,
            detail=(f"batch size {len(payload)} > {cfg['api']['max_batch_size']}"),
        )
    srv: ModelServer = request.app.state.server
    return [srv.predict(p.engine_id, p.sensor_window) for p in payload]


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


@app.post("/api/v1/explain/{engine_id}", response_model=ExplainResponse)
def explain(
    request: Request, engine_id: int, payload: PredictRequest
) -> dict[str, Any]:
    """Return SHAP-based feature attributions for an engine window."""
    srv: ModelServer = request.app.state.server
    if srv._explainer is None:
        train_df = pd.read_parquet(cfg["data"]["processed_train"])
        ds = CMAPSSDataset(train_df, cfg["data"]["sequence_length"])
        n = cfg["shap"]["n_background_samples"]
        rng = np.random.default_rng(cfg["data"]["seed"])
        idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
        bg = torch.stack([ds[int(i)][0] for i in idx])
        srv._explainer = RULExplainer(srv.model, bg, cfg)
    explainer: RULExplainer = srv._explainer
    scaled = srv._scale_window(payload.sensor_window)
    window = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    exp = explainer.explain_engine(window)
    rul = exp["predicted_rul"]
    thresholds = cfg["evaluation"]["health_status_thresholds"]
    status_label = _health_status(rul, thresholds)
    top = exp["top_features"][0] if exp["top_features"] else None
    return {
        "engine_id": engine_id,
        "predicted_rul": rul,
        "shap_values": exp["shap_values"],
        "top_features": exp["top_features"],
        "nl_summary": _nl_summary(engine_id, rul, status_label, top),
    }


# ---------------------------------------------------------------------------
# Engines + model info
# ---------------------------------------------------------------------------


@app.get("/api/v1/engines", response_model=list[int])
def engines() -> list[int]:
    """Return sorted list of test engine IDs from the processed test set."""
    df = pd.read_parquet(cfg["data"]["processed_test"])
    return sorted(int(e) for e in df["engine_id"].unique())


@app.get("/api/v1/model_info")
def model_info() -> dict[str, Any]:
    """Return the contents of reports/results.json (RMSE, MAE, NASA Score)."""
    with open(cfg["paths"]["results_json"]) as fh:
        return json.load(fh)
