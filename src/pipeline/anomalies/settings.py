from __future__ import annotations

from typing import Dict, Tuple

from .models import DetectionSettings, InterpretationThresholds, ResidualDetectionSettings


def load_detection_settings(config: Dict) -> Tuple[DetectionSettings, InterpretationThresholds]:
    detection_cfg = config.get("anomalies", {}).get("detection", {}) or {}
    interpretation_cfg = config.get("anomalies", {}).get("interpretation", {}) or {}
    settings = DetectionSettings(
        window_minutes=int(detection_cfg.get("window_minutes", 60)),
        shift_minutes=int(detection_cfg.get("shift_minutes", 60)),
        min_samples=int(detection_cfg.get("min_samples", 30)),
        delta_factor=float(detection_cfg.get("delta_factor", 0.6)),
        delta_quantile=float(detection_cfg.get("delta_quantile", 0.1)),
        slope_margin=float(detection_cfg.get("slope_margin", 0.03)),
        slope_quantile=float(detection_cfg.get("slope_quantile", 0.15)),
        min_duration_minutes=int(detection_cfg.get("min_duration_minutes", 60)),
        gap_minutes=int(detection_cfg.get("gap_minutes", 10)),
    )
    interpretation = InterpretationThresholds(
        strong_increase=float(interpretation_cfg.get("strong_increase", 0.1)),
        mild_increase=float(interpretation_cfg.get("mild_increase", 0.03)),
        mild_drop=float(interpretation_cfg.get("mild_drop", -0.03)),
        strong_drop=float(interpretation_cfg.get("strong_drop", -0.1)),
    )
    return settings, interpretation


def load_residual_settings(config: Dict) -> ResidualDetectionSettings:
    residual_cfg = config.get("anomalies", {}).get("detection_residual", {}) or {}
    return ResidualDetectionSettings(
        enabled=bool(residual_cfg.get("enabled", True)),
        ewma_lambda=float(residual_cfg.get("ewma_lambda", 0.2)),
        ewma_l_multiplier=float(residual_cfg.get("ewma_l_multiplier", 3.0)),
        spike_threshold=float(residual_cfg.get("spike_threshold", 4.0)),
        t2_alpha=float(residual_cfg.get("t2_alpha", 0.01)),
        min_t2_points=int(residual_cfg.get("min_t2_points", 30)),
    )


__all__ = [
    "load_detection_settings",
    "load_residual_settings",
]
