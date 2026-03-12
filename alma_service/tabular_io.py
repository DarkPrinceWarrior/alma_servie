from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl


def read_table(
    path: str | Path,
    *,
    dtypes: dict[str, Any] | None = None,
    parse_dates: list[str] | tuple[str, ...] | None = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    src = Path(path)
    suffix = src.suffix.lower()

    if suffix == ".parquet":
        df = pl.read_parquet(src).to_pandas()
    elif suffix == ".csv":
        schema_overrides = None
        if dtypes:
            schema_overrides = {
                column: pl.String
                for column, dtype in dtypes.items()
                if dtype is str
            } or None
        df = pl.read_csv(
            src,
            try_parse_dates=True,
            infer_schema_length=1000,
            ignore_errors=False,
            schema_overrides=schema_overrides,
        ).to_pandas()
    else:
        raise ValueError(f"Unsupported table format: {src}")

    if dtypes:
        for column, dtype in dtypes.items():
            if column not in df.columns:
                continue
            if dtype is str:
                df[column] = df[column].astype(str)
            else:
                df[column] = df[column].astype(dtype)

    if parse_dates:
        for column in parse_dates:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")

    return df


def write_dataset_tables(
    df: pd.DataFrame,
    *,
    parquet_path: str | Path,
) -> None:
    parquet_dst = Path(parquet_path)
    pl.from_pandas(df).write_parquet(parquet_dst, compression="zstd")


def read_excel_sheet(
    path: str | Path,
    *,
    sheet_name: str | None = None,
    sheet_id: int | None = None,
    has_header: bool = False,
    infer_schema_length: int | None = 100,
    raise_if_empty: bool = False,
) -> pd.DataFrame:
    src = Path(path)
    frame = pl.read_excel(
        src,
        sheet_name=sheet_name,
        sheet_id=sheet_id,
        engine="calamine",
        has_header=has_header,
        infer_schema_length=infer_schema_length,
        raise_if_empty=raise_if_empty,
    )
    if isinstance(frame, dict):
        if not frame:
            return pd.DataFrame()
        frame = next(iter(frame.values()))
    return frame.to_pandas()


def read_excel_workbook(
    path: str | Path,
    *,
    has_header: bool = False,
    infer_schema_length: int | None = 100,
) -> dict[str, pd.DataFrame]:
    src = Path(path)
    workbook = pl.read_excel(
        src,
        sheet_id=0,
        engine="calamine",
        has_header=has_header,
        infer_schema_length=infer_schema_length,
        raise_if_empty=False,
    )
    if isinstance(workbook, dict):
        return {name: frame.to_pandas() for name, frame in workbook.items()}
    return {"Sheet1": workbook.to_pandas()}
