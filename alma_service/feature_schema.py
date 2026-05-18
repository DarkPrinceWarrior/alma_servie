from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from alma_service.engineered_features import PreparedWellData


DEFAULT_GLOBAL_SCHEMA_PATH = Path("configs/alma_global_feature_schema.json")


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    version: str
    raw_channels: tuple[str, ...]
    excluded_channels: tuple[str, ...] = ()

    @property
    def raw_channel_set(self) -> set[str]:
        return set(self.raw_channels)

    def to_detail(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "raw_channels": list(self.raw_channels),
            "raw_channel_count": len(self.raw_channels),
            "excluded_channels": list(self.excluded_channels),
        }


def load_feature_schema(path: str | Path = DEFAULT_GLOBAL_SCHEMA_PATH) -> FeatureSchema:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_channels = tuple(str(item) for item in payload["raw_channels"])
    if len(raw_channels) != len(set(raw_channels)):
        raise ValueError(f"Feature schema contains duplicate raw channels: {path}")
    return FeatureSchema(
        name=str(payload.get("name", Path(path).stem)),
        version=str(payload.get("version", "unknown")),
        raw_channels=raw_channels,
        excluded_channels=tuple(str(item) for item in payload.get("excluded_channels", ())),
    )


def feature_root(feature_name: str) -> str:
    return str(feature_name).split("::", 1)[0]


def restrict_prepared_to_schema(
    prepared: PreparedWellData,
    schema: FeatureSchema,
    *,
    strict: bool = True,
) -> tuple[PreparedWellData | None, dict[str, Any]]:
    raw_set = schema.raw_channel_set
    raw_columns = list(prepared.raw_columns)
    feature_columns = list(prepared.feature_columns)
    missing_raw = [channel for channel in schema.raw_channels if channel not in raw_columns]

    if strict and missing_raw:
        return None, {
            "well_id": prepared.well_id,
            "reason": "missing_required_raw_channels",
            "missing_raw_channels": missing_raw,
            "available_raw_channels": raw_columns,
        }

    raw_indices = [idx for idx, channel in enumerate(raw_columns) if channel in raw_set]
    feature_indices = [
        idx
        for idx, channel in enumerate(feature_columns)
        if feature_root(channel) in raw_set
    ]

    if not raw_indices or not feature_indices:
        return None, {
            "well_id": prepared.well_id,
            "reason": "no_schema_features_after_filter",
            "missing_raw_channels": missing_raw,
            "available_raw_channels": raw_columns,
        }

    kept_raw_columns = [raw_columns[idx] for idx in raw_indices]
    kept_feature_columns = [feature_columns[idx] for idx in feature_indices]
    detail = dict(prepared.detail)
    detail["fixed_feature_schema"] = {
        **schema.to_detail(),
        "strict": bool(strict),
        "missing_raw_channels": missing_raw,
        "raw_channels_before": len(raw_columns),
        "raw_channels_after": len(kept_raw_columns),
        "feature_count_before": len(feature_columns),
        "feature_count_after": len(kept_feature_columns),
    }
    detail["raw_channels"] = int(len(kept_raw_columns))
    detail["feature_count"] = int(len(kept_feature_columns))

    filtered = replace(
        prepared,
        raw_columns=kept_raw_columns,
        feature_columns=kept_feature_columns,
        raw_matrix=np.asarray(prepared.raw_matrix[:, raw_indices], dtype=np.float32),
        feature_matrix=np.asarray(prepared.feature_matrix[:, feature_indices], dtype=np.float32),
        detail=detail,
    )
    return filtered, {
        "well_id": prepared.well_id,
        "reason": "ok",
        "missing_raw_channels": missing_raw,
        "raw_channels_after": len(kept_raw_columns),
        "feature_count_after": len(kept_feature_columns),
    }


def restrict_prepared_mapping_to_schema(
    prepared_wells: dict[str, PreparedWellData],
    schema: FeatureSchema,
    *,
    strict: bool = True,
) -> tuple[dict[str, PreparedWellData], dict[str, dict[str, Any]]]:
    filtered: dict[str, PreparedWellData] = {}
    audit: dict[str, dict[str, Any]] = {}
    for key, prepared in prepared_wells.items():
        item, detail = restrict_prepared_to_schema(prepared, schema, strict=strict)
        audit[key] = detail
        if item is not None:
            filtered[key] = item
    return filtered, audit
