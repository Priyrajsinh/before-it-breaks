"""Pandera schemas for raw and processed NASA CMAPSS DataFrames (rule C34)."""

import pandas as pd
import pandera as pa
from pandera.pandas import Column, DataFrameSchema

RAW_CMAPSS_SCHEMA = DataFrameSchema(
    {
        "engine_id": Column(int, checks=pa.Check(lambda s: s >= 1)),
        "cycle": Column(int, checks=pa.Check(lambda s: s >= 1)),
        "setting_1": Column(float),
        "setting_2": Column(float),
        "setting_3": Column(float),
        **{f"sensor_{i}": Column(float) for i in range(1, 22)},
    },
    strict=True,
)

KEPT_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]

PROCESSED_CMAPSS_SCHEMA = DataFrameSchema(
    {
        "engine_id": Column(int, checks=pa.Check(lambda s: s >= 1)),
        "cycle": Column(int, checks=pa.Check(lambda s: s >= 1)),
        "setting_1": Column(float),
        "setting_2": Column(float),
        "setting_3": Column(float),
        **{f"sensor_{i}": Column(float) for i in KEPT_SENSORS},
        "rul": Column(float, checks=pa.Check(lambda s: (s >= 0) & (s <= 125))),
    },
    strict=True,
)


def validate_raw_cmapss(df: pd.DataFrame) -> pd.DataFrame:
    """Run RAW_CMAPSS_SCHEMA. Raise SchemaError on first violation. Rule C34."""
    return RAW_CMAPSS_SCHEMA.validate(df)


def validate_processed_cmapss(df: pd.DataFrame) -> pd.DataFrame:
    """Run PROCESSED_CMAPSS_SCHEMA. Raise SchemaError on first violation. Rule C34."""
    return PROCESSED_CMAPSS_SCHEMA.validate(df)
