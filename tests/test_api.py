"""Integration tests for the FastAPI prediction API (Day 5)."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


class _FakeServer:
    """Minimal stand-in for ModelServer — no real artefacts required."""

    model: object = object()
    total_predictions: int = 0
    _explainer: object = None

    def predict(self, engine_id: int, sensor_window: list) -> dict:
        """Return a canned prediction result."""
        self.total_predictions += 1
        return {
            "engine_id": engine_id,
            "predicted_rul": 42.0,
            "health_status": "Warning",
            "confidence": 0.66,
            "nl_summary": "Test summary.",
            "drift_flags": {},
        }

    def uptime_seconds(self) -> float:
        """Return a fixed uptime value."""
        return 1.0


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncClient:
    """Async HTTP client with FakeServer injected via monkeypatch.

    ASGITransport does not trigger ASGI lifespan, so we set app.state.server
    directly rather than relying on the lifespan context manager.
    """
    from src.api import app as app_module

    fake = _FakeServer()
    monkeypatch.setattr(app_module, "ModelServer", lambda cfg: fake)
    app.state.server = fake  # bypass lifespan for tests
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://localhost") as c:
        yield c


async def test_health_returns_200(client: AsyncClient) -> None:
    """GET /health returns 200 with model_loaded=True."""
    r = await client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


async def test_predict_valid(client: AsyncClient) -> None:
    """POST /predict with valid 30x17 window returns 200 with required fields."""
    body = {"engine_id": 1, "sensor_window": [[0.0] * 17] * 30}
    r = await client.post("/api/v1/predict", json=body)
    assert r.status_code == 200
    j = r.json()
    assert "predicted_rul" in j
    assert "health_status" in j
    assert "nl_summary" in j


async def test_predict_wrong_shape_422(client: AsyncClient) -> None:
    """POST /predict with 29-row window is rejected as 422 by Pydantic."""
    body = {"engine_id": 1, "sensor_window": [[0.0] * 17] * 29}
    r = await client.post("/api/v1/predict", json=body)
    assert r.status_code == 422


async def test_predict_batch_too_large_422(client: AsyncClient) -> None:
    """POST /predict_batch with >50 items returns 422."""
    body = [{"engine_id": i, "sensor_window": [[0.0] * 17] * 30} for i in range(60)]
    r = await client.post("/api/v1/predict_batch", json=body)
    assert r.status_code == 422


async def test_metrics_endpoint(client: AsyncClient) -> None:
    """GET /metrics exposes Prometheus text including predictions_served_total."""
    r = await client.get("/metrics")
    assert r.status_code == 200
    assert "predictions_served_total" in r.text
