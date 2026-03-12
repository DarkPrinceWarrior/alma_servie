from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency in older envs
    pl = None


def has_polars() -> bool:
    return pl is not None


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
        if pl is not None:
            df = pl.read_parquet(src).to_pandas()
        else:
            df = pd.read_parquet(src)
    elif suffix == ".csv":
        if pl is not None:
            try:
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
            except Exception:
                df = pd.read_csv(src, low_memory=low_memory)
        else:
            df = pd.read_csv(src, low_memory=low_memory)
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
    csv_path: str | Path | None = None,
) -> None:
    parquet_dst = Path(parquet_path)
    if pl is not None:
        pl.from_pandas(df).write_parquet(parquet_dst, compression="zstd")
    else:  # pragma: no cover - exercised only in fallback envs
        df.to_parquet(parquet_dst, index=False, compression="zstd")

    if csv_path is not None:
        df.to_csv(csv_path, index=False)


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
    if pl is not None:
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

    read_kwargs: dict[str, Any] = {"header": 0 if has_header else None}
    if sheet_name is not None:
        read_kwargs["sheet_name"] = sheet_name
    elif sheet_id is not None:
        read_kwargs["sheet_name"] = sheet_id
    return pd.read_excel(src, **read_kwargs)


def read_excel_workbook(
    path: str | Path,
    *,
    has_header: bool = False,
    infer_schema_length: int | None = 100,
) -> dict[str, pd.DataFrame]:
    src = Path(path)
    if pl is not None:
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

    sheets = pd.read_excel(src, sheet_name=None, header=0 if has_header else None)
    return {str(name): frame for name, frame in sheets.items()}
