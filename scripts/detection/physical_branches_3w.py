"""Per-class physical anomaly scoring for 3W event classes.

These branches complement the PaAno representation-based score for events whose
signature is not well captured by patch-level autoencoding:

- Class 4 FLOW_INSTABILITY: sustained slug-like oscillations in downhole
  pressure. Captured by the ratio of fast-rolling std (5m) to slow-rolling std
  (30m) on `P-TPT`, z-scored against the per-well reference window.
- Class 6 QUICK_RESTRICTION_IN_PCK: sharp rise of pressure drop across the
  production choke. Captured by `(P-MON-CKP - P-JUS-CKP)` deviation from its
  per-well rolling baseline.

All scores are causal (no future leak) and NaN-safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _causal_zscore(x: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    ref = x[ref_mask]
    ref = ref[np.isfinite(ref)]
    if len(ref) < 8:
        ref = x[np.isfinite(x)][:max(8, len(x) // 10)]
    if len(ref) < 2:
        return np.zeros_like(x, dtype=np.float32)
    mu = float(np.mean(ref))
    sd = float(np.std(ref))
    if sd < 1e-8:
        sd = 1e-8
    z = (x - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float32)
    if not finite.any():
        return out
    order = np.argsort(np.argsort(np.where(finite, x, x[finite].min() - 1.0)))
    out = (order.astype(np.float32) / max(1, len(x) - 1)).astype(np.float32)
    return out


def class4_oscillation_score(df: pd.DataFrame, ref_mask: np.ndarray) -> np.ndarray:
    """Slug-oscillation regime score for FLOW_INSTABILITY (class 4)."""
    fast = df.get("P-TPT_roll5m_std")
    slow = df.get("P-TPT_roll30m_std")
    if fast is None or slow is None:
        return np.zeros(len(df), dtype=np.float32)
    fast_v = np.asarray(fast.values, dtype=np.float64)
    slow_v = np.asarray(slow.values, dtype=np.float64)
    slow_v = np.where(slow_v < 1e-6, 1e-6, slow_v)
    ratio = fast_v / slow_v
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.0)
    z = _causal_zscore(ratio, ref_mask)
    z = np.clip(z, 0.0, None)
    return _rank_normalize(z)


def class6_choke_step_score(df: pd.DataFrame, ref_mask: np.ndarray) -> np.ndarray:
    """Choke pressure-drop step score for QUICK_RESTRICTION_IN_PCK (class 6)."""
    p_up = df.get("P-MON-CKP")
    p_down = df.get("P-JUS-CKP")
    p_up_baseline = df.get("P-MON-CKP_roll30m_mean")
    if p_up is None or p_down is None or p_up_baseline is None:
        return np.zeros(len(df), dtype=np.float32)
    up = np.asarray(p_up.values, dtype=np.float64)
    down = np.asarray(p_down.values, dtype=np.float64)
    base = np.asarray(p_up_baseline.values, dtype=np.float64)
    dp = up - down
    dp = np.nan_to_num(dp, nan=0.0)
    dev = up - base
    dev = np.nan_to_num(dev, nan=0.0)
    z_dp = _causal_zscore(dp, ref_mask)
    z_dev = _causal_zscore(dev, ref_mask)
    composite = np.maximum(z_dp, z_dev)
    composite = np.clip(composite, 0.0, None)
    return _rank_normalize(composite)


CLASS_SCORERS = {
    4: class4_oscillation_score,
    6: class6_choke_step_score,
}


def compute_physical_score(df: pd.DataFrame, event_class: int, ref_mask: np.ndarray) -> np.ndarray | None:
    fn = CLASS_SCORERS.get(int(event_class))
    if fn is None:
        return None
    return fn(df, ref_mask)
