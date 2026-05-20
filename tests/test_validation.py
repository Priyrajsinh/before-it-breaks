import pandas as pd
import pandera
import pytest


def _make_raw_row() -> dict:
    """Return one valid raw-schema row dict."""
    row: dict = {
        "engine_id": 1,
        "cycle": 1,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 100.0,
    }
    for i in range(1, 22):
        row[f"sensor_{i}"] = 0.0
    return row


def _make_processed_row() -> dict:
    """Return one valid processed-schema row dict."""
    row: dict = {
        "engine_id": 1,
        "cycle": 1,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 100.0,
        "rul": 50.0,
    }
    for s in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]:
        row[f"sensor_{s}"] = 0.0
    return row


def test_raw_schema_accepts_valid_df():
    """RAW_CMAPSS_SCHEMA validates a well-formed DataFrame."""
    from src.data.validation import validate_raw_cmapss

    df = pd.DataFrame([_make_raw_row()])
    result = validate_raw_cmapss(df)
    assert len(result) == 1


def test_raw_schema_rejects_zero_cycle():
    """RAW_CMAPSS_SCHEMA raises SchemaError when cycle=0."""
    from src.data.validation import validate_raw_cmapss

    row = _make_raw_row()
    row["cycle"] = 0
    df = pd.DataFrame([row])
    with pytest.raises(pandera.errors.SchemaError):
        validate_raw_cmapss(df)


def test_processed_schema_accepts_valid_df():
    """PROCESSED_CMAPSS_SCHEMA validates a well-formed DataFrame."""
    from src.data.validation import validate_processed_cmapss

    df = pd.DataFrame([_make_processed_row()])
    result = validate_processed_cmapss(df)
    assert len(result) == 1


def test_processed_schema_rejects_rul_over_cap():
    """PROCESSED_CMAPSS_SCHEMA raises SchemaError when rul > 125."""
    from src.data.validation import validate_processed_cmapss

    row = _make_processed_row()
    row["rul"] = 200.0
    df = pd.DataFrame([row])
    with pytest.raises(pandera.errors.SchemaError):
        validate_processed_cmapss(df)


def test_processed_schema_rejects_negative_rul():
    """PROCESSED_CMAPSS_SCHEMA raises SchemaError when rul < 0."""
    from src.data.validation import validate_processed_cmapss

    row = _make_processed_row()
    row["rul"] = -1.0
    df = pd.DataFrame([row])
    with pytest.raises(pandera.errors.SchemaError):
        validate_processed_cmapss(df)
