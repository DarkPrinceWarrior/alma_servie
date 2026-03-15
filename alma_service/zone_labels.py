"""
4-zone data curation for anomaly detection pipelines.

Assigns every timestep to one of four zones:
  0 = clean_normal   — safe for encoder training and memory bank
  1 = pre_anomaly_buffer — excluded, precursor region before labelled anomaly
  2 = anomaly         — the labelled anomaly interval itself
  3 = post_anomaly_recovery — excluded, recovery region after anomaly

Buffer widths are configurable per anomaly family.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd

from alma_service.onset_detection import infer_step_seconds


class Zone(IntEnum):
    CLEAN_NORMAL = 0
    PRE_ANOMALY_BUFFER = 1
    ANOMALY = 2
    POST_ANOMALY_RECOVERY = 3


# Per-family buffer configuration.
# Values in minutes; converted to point-counts using inferred step size.
# Minimum buffer in *points* is always max(config_points, patch_size).
ZONE_BUFFER_CONFIG: dict[str, dict[str, float]] = {
    "negermet": {"pre_buffer_minutes": 30.0,  "post_buffer_minutes": 60.0},
    "pritok":   {"pre_buffer_minutes": 60.0,  "post_buffer_minutes": 180.0},
    "salt":     {"pre_buffer_minutes": 60.0,  "post_buffer_minutes": 180.0},
}

_DEFAULT_BUFFER = {"pre_buffer_minutes": 60.0, "post_buffer_minutes": 120.0}


def _minutes_to_points(minutes: float, step_seconds: float) -> int:
    """Convert a duration in minutes to a number of timestep points."""
    return max(int(np.ceil((minutes * 60.0) / max(step_seconds, 1.0))), 1)


def label_zones(
    timestamps: np.ndarray,
    anomaly_intervals: pd.DataFrame,
    patch_size: int,
    anomaly_key: str = "",
) -> np.ndarray:
    """Assign a :class:`Zone` label to every point in the timeseries.

    Parameters
    ----------
    timestamps:
        Sorted array of datetime64 timestamps for the well.
    anomaly_intervals:
        DataFrame with columns ``start_date``, ``end_date`` (pd.Timestamp).
        Only rows relevant to *this* well should be passed.
    patch_size:
        Long-patch size.  The minimum buffer width in points is
        ``max(config_points, patch_size)`` to ensure that no patch
        overlapping the anomaly boundary contaminates the normal pool.
    anomaly_key:
        One of ``"negermet"``, ``"pritok"``, ``"salt"`` (or empty for
        default buffer widths).

    Returns
    -------
    np.ndarray of int8 with shape ``(len(timestamps),)``
    """
    n = len(timestamps)
    labels = np.full(n, Zone.CLEAN_NORMAL, dtype=np.int8)

    if anomaly_intervals is None or anomaly_intervals.empty:
        return labels

    step_seconds = infer_step_seconds(timestamps)
    buf_cfg = ZONE_BUFFER_CONFIG.get(anomaly_key, _DEFAULT_BUFFER)
    pre_points = max(
        _minutes_to_points(buf_cfg["pre_buffer_minutes"], step_seconds),
        patch_size,
    )
    post_points = max(
        _minutes_to_points(buf_cfg["post_buffer_minutes"], step_seconds),
        patch_size,
    )

    ts = pd.DatetimeIndex(timestamps)

    for _, row in anomaly_intervals.iterrows():
        start = pd.Timestamp(row["start_date"])
        end = pd.Timestamp(row["end_date"])
        if pd.isna(start) or pd.isna(end):
            continue

        # Core anomaly mask
        anom_mask = (ts >= start) & (ts <= end)
        anom_indices = np.flatnonzero(anom_mask)
        if len(anom_indices) == 0:
            continue

        first_anom = int(anom_indices[0])
        last_anom = int(anom_indices[-1])

        # Mark the anomaly zone
        labels[first_anom : last_anom + 1] = Zone.ANOMALY

        # Pre-anomaly buffer
        pre_start = max(first_anom - pre_points, 0)
        for i in range(pre_start, first_anom):
            if labels[i] == Zone.CLEAN_NORMAL:
                labels[i] = Zone.PRE_ANOMALY_BUFFER

        # Post-anomaly recovery
        post_end = min(last_anom + 1 + post_points, n)
        for i in range(last_anom + 1, post_end):
            if labels[i] == Zone.CLEAN_NORMAL:
                labels[i] = Zone.POST_ANOMALY_RECOVERY

    return labels


def make_clean_normal_mask(zone_labels: np.ndarray) -> np.ndarray:
    """Boolean mask: ``True`` only where zone is ``CLEAN_NORMAL``."""
    return np.asarray(zone_labels, dtype=np.int8) == Zone.CLEAN_NORMAL


def make_training_exclusion_mask(zone_labels: np.ndarray) -> np.ndarray:
    """Boolean mask: ``True`` where a point should be **excluded** from
    normal training pool (i.e. pre-buffer, anomaly, or post-recovery)."""
    return np.asarray(zone_labels, dtype=np.int8) != Zone.CLEAN_NORMAL


def make_onset_allowed_from_zones(
    zone_labels: np.ndarray,
    reference_end_idx: int,
) -> np.ndarray:
    """Boolean mask for onset detection: ``False`` inside reference window
    and inside pre/post buffers.  ``True`` only in anomaly or clean_normal
    segments *after* the reference window."""
    z = np.asarray(zone_labels, dtype=np.int8)
    n = len(z)
    mask = np.ones(n, dtype=bool)
    # Suppress reference window
    mask[:reference_end_idx] = False
    # Suppress buffer zones even after reference window
    mask[z == Zone.PRE_ANOMALY_BUFFER] = False
    mask[z == Zone.POST_ANOMALY_RECOVERY] = False
    return mask
