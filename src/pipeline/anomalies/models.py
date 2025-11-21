from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ReferenceInterval:
    well: str
    start: pd.Timestamp
    end: pd.Timestamp
    label: str  # "anomaly" or "normal"
    cause: str
    notes: List[str]


@dataclass
class DetectionSettings:
    window_minutes: int = 60
    shift_minutes: int = 60
    min_samples: int = 30
    delta_factor: float = 0.6
    threshold_clip_quantile: float = 0.9
    delta_quantile: float = 0.1
    slope_margin: float = 0.03
    slope_quantile: float = 0.15
    min_duration_minutes: int = 60
    gap_minutes: int = 10


@dataclass
class InterpretationThresholds:
    strong_increase: float = 0.1
    mild_increase: float = 0.03
    mild_drop: float = -0.03
    strong_drop: float = -0.1


@dataclass
class WellTimeseries:
    base: pd.DataFrame
    pressure_fast: Optional[pd.DataFrame]


@dataclass
class FrequencyBaseline:
    bin_width: float
    metrics: Dict[str, Dict[int, Dict[str, float]]]
    summary: pd.DataFrame


@dataclass
class ResidualDetectionSettings:
    enabled: bool = True
    ewma_lambda: float = 0.2
    ewma_l_multiplier: float = 3.0
    spike_threshold: float = 4.0
    t2_alpha: float = 0.01
    min_t2_points: int = 30


@dataclass
class ResidualDetectionModel:
    metrics: List[str]
    covariance: np.ndarray
    inv_covariance: np.ndarray
    t2_threshold: float
    settings: ResidualDetectionSettings


__all__ = [
    "DetectionSettings",
    "FrequencyBaseline",
    "InterpretationThresholds",
    "ReferenceInterval",
    "ResidualDetectionModel",
    "ResidualDetectionSettings",
    "WellTimeseries",
]
