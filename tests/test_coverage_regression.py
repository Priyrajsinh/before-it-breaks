"""Coverage regression gate: the trained model must not silently degrade.

Rules B + C37 — the canonical CMAPSS test RMSE is wired as a hard pytest gate.
If a future change regresses the model past 30 cycles, CI fails. Do NOT lower
the threshold to make this pass — investigate why the model got worse.
"""

import json
from pathlib import Path

import pytest

RESULTS = Path("reports/results.json")
RMSE_THRESHOLD = 30.0  # rule C37 — hard gate


@pytest.fixture(scope="module")
def results() -> dict:
    """Load the evaluation results emitted by `make evaluate`."""
    if not RESULTS.exists():
        pytest.skip(
            "reports/results.json not yet generated (run /train then /evaluate)"
        )
    with open(RESULTS, encoding="utf-8") as fh:
        return json.load(fh)


def test_test_rmse_under_threshold(results: dict) -> None:
    """Rule C37: test RMSE must stay below 30 cycles. Hard CI gate."""
    rmse = results["rmse"]
    assert rmse <= RMSE_THRESHOLD, (
        f"test RMSE {rmse:.2f} > threshold {RMSE_THRESHOLD}. "
        "Do NOT lower the threshold — investigate why the model regressed."
    )


def test_nasa_score_finite(results: dict) -> None:
    """NASA Score must be finite and positive (asymmetric scoring sum)."""
    score = results["nasa_score"]
    assert 0 < score < float("inf")


def test_per_engine_predictions_complete(results: dict) -> None:
    """Every test engine should have a prediction (no silent skips)."""
    # 100 test engines; allow a small margin for short engines without a window.
    assert len(results["per_engine_predictions"]) >= 95
