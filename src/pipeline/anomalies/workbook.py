from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover - safekeeping for API-less environments
    requests = None  # type: ignore[assignment]


class WorkbookSource:
    """Thin facade that provides `sheet_names` and `parse` no matter where data originates."""

    def __init__(self, sheet_names: Iterable[str], parser: Callable[[str, Dict[str, Any]], pd.DataFrame]):
        self._sheet_names: Tuple[str, ...] = tuple(sheet_names)
        self._parser = parser

    @property
    def sheet_names(self) -> Sequence[str]:
        return list(self._sheet_names)

    def parse(self, sheet_name: str, **kwargs: Any) -> pd.DataFrame:
        if sheet_name not in self._sheet_names:
            available = ", ".join(self._sheet_names)
            raise KeyError(f"Sheet '{sheet_name}' is not available. Known sheets: [{available}]")
        frame = self._parser(sheet_name, kwargs)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Sheet '{sheet_name}' parser returned {type(frame)!r}, expected pandas.DataFrame.")
        return frame

    @classmethod
    def from_excel(cls, path: Path | str) -> "WorkbookSource":
        # Try using calamine engine if available for performance
        try:
            import python_calamine  # noqa
            engine = "calamine"
        except ImportError:
            engine = None

        excel_file = pd.ExcelFile(path, engine=engine)

        def _parse(sheet: str, kwargs: Dict[str, Any]) -> pd.DataFrame:
            return excel_file.parse(sheet, **kwargs)

        source = cls(excel_file.sheet_names, _parse)
        source._excel_handle = excel_file  # keep reference alive
        return source

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame]) -> "WorkbookSource":
        stored: Dict[str, pd.DataFrame] = {
            name: df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df) for name, df in frames.items()
        }

        def _parse(sheet: str, kwargs: Dict[str, Any]) -> pd.DataFrame:
            # pandas.ExcelFile.parse accepts a `header` arg that drops top rows; API callers can prepare frames directly.
            kwargs.pop("header", None)
            return stored[sheet].copy()

        return cls(stored.keys(), _parse)

    @classmethod
    def from_api(cls, api_cfg: Mapping[str, Any]) -> "WorkbookSource":
        if requests is None:
            raise ImportError("requests is required for API workbook mode. Install it via `pip install requests`.")
        url = api_cfg.get("url")
        if not url:
            raise ValueError("API configuration must provide a `url` field.")
        method = str(api_cfg.get("method", "GET")).upper()
        headers = api_cfg.get("headers") or {}
        params = api_cfg.get("params") or {}
        data = api_cfg.get("data")
        json_payload = api_cfg.get("json")
        timeout = float(api_cfg.get("timeout", 30.0))
        verify = api_cfg.get("verify", True)

        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            data=data,
            json=json_payload,
            timeout=timeout,
            verify=verify,
        )
        response.raise_for_status()
        payload = response.json()

        sheets_key = api_cfg.get("sheets_key", "sheets")
        name_key = api_cfg.get("name_key", "name")
        records_key = api_cfg.get("records_key", "records")
        payload_path = api_cfg.get("payload_path")

        if payload_path:
            if isinstance(payload_path, str):
                payload = payload[payload_path]
            else:
                for key in payload_path:
                    payload = payload[key]

        frames = _payload_to_frames(payload, sheets_key=sheets_key, name_key=name_key, records_key=records_key)
        if not frames:
            raise ValueError("API response did not contain any sheets to parse.")

        return cls.from_frames(frames)


def _payload_to_frames(
    payload: Any,
    *,
    sheets_key: str,
    name_key: str,
    records_key: str,
) -> Dict[str, pd.DataFrame]:
    if isinstance(payload, Mapping):
        if sheets_key in payload and _is_sequence_like(payload[sheets_key]):
            return _frames_from_sheet_list(payload[sheets_key], name_key=name_key, records_key=records_key)
        if all(_is_sequence_like(value) for value in payload.values()):
            return {str(name): pd.DataFrame(value) for name, value in payload.items()}
        if all(isinstance(value, Mapping) for value in payload.values()):
            return {str(name): pd.DataFrame(value) for name, value in payload.items()}
    if _is_sequence_like(payload):
        return _frames_from_sheet_list(payload, name_key=name_key, records_key=records_key)
    raise ValueError(
        "Unsupported API payload structure. Expected mapping of sheet name to rows or a list of sheets with "
        f"keys '{name_key}'/'{records_key}'."
    )


def _frames_from_sheet_list(
    sheets: Iterable[Any],
    *,
    name_key: str,
    records_key: str,
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for sheet in sheets:
        if not isinstance(sheet, Mapping):
            raise ValueError("Each sheet description must be a mapping with name/records keys.")
        name = sheet.get(name_key)
        if not name:
            raise ValueError(f"Sheet description missing '{name_key}'. Got: {sheet}")
        records = sheet.get(records_key, [])
        frames[str(name)] = pd.DataFrame(records)
    return frames


def _is_sequence_like(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


@dataclass
class WorkbookSpec:
    source: WorkbookSource
    description: str


def resolve_workbook_source(
    config: Mapping[str, Any],
    *,
    workbook_override: Optional[Path] = None,
) -> WorkbookSpec:
    anomalies_cfg = config.get("anomalies", {}) or {}
    input_cfg = anomalies_cfg.get("input", {}) or {}
    mode = (input_cfg.get("mode") or "excel").lower()

    if workbook_override is not None:
        if not workbook_override.exists():
            raise FileNotFoundError(f"Override workbook not found: {workbook_override}")
        source = WorkbookSource.from_excel(workbook_override)
        return WorkbookSpec(source=source, description=f"excel:{workbook_override}")

    if mode == "excel":
        workbook_path_raw = anomalies_cfg.get("source_workbook")
        if not workbook_path_raw:
            raise ValueError("Excel mode selected but `anomalies.source_workbook` is not set in config.")
        workbook_path = Path(workbook_path_raw)
        if not workbook_path.exists():
            raise FileNotFoundError(f"Workbook with well data not found: {workbook_path}")
        source = WorkbookSource.from_excel(workbook_path)
        return WorkbookSpec(source=source, description=f"excel:{workbook_path}")

    if mode == "api":
        api_cfg = input_cfg.get("api") or {}
        source = WorkbookSource.from_api(api_cfg)
        api_label = api_cfg.get("url", "api")
        return WorkbookSpec(source=source, description=f"api:{api_label}")

    raise ValueError(f"Unsupported anomalies.input.mode '{mode}'. Expected 'excel' or 'api'.")


__all__ = [
    "WorkbookSource",
    "WorkbookSpec",
    "resolve_workbook_source",
]
