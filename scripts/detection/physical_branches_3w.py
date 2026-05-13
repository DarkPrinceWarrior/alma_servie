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


def class3_slug_score(df: pd.DataFrame, ref_mask: np.ndarray) -> np.ndarray:
    """Slug-regime score for SEVERE_SLUGGING (class 3).

    Approach: detect quasi-periodic oscillations in downhole pressure with
    typical slug period 5-30 min. Uses two complementary indicators:
    (a) ratio of fast-rolling std (5m) to slow-rolling std (30m) — same as
        class 4 but on P-PDG (downhole gauge, closer to slug source);
    (b) z-score of P-TPT_roll5m_std (fast variance growth), since slug events
        produce sustained high-frequency pressure oscillation.

    Both indicators are z-scored against the per-well reference window and
    fused via max-pool, then rank-normalized.
    """
    p_pdg_fast = df.get("P-PDG_roll5m_std")
    p_pdg_slow = df.get("P-PDG_roll30m_std")
    p_tpt_fast = df.get("P-TPT_roll5m_std")
    p_tpt_slow = df.get("P-TPT_roll30m_std")
    if p_tpt_fast is None or p_tpt_slow is None:
        return np.zeros(len(df), dtype=np.float32)

    tpt_fast = np.asarray(p_tpt_fast.values, dtype=np.float64)
    tpt_slow = np.asarray(p_tpt_slow.values, dtype=np.float64)
    tpt_slow = np.where(tpt_slow < 1e-6, 1e-6, tpt_slow)
    tpt_ratio = np.nan_to_num(tpt_fast / tpt_slow, nan=1.0, posinf=10.0, neginf=0.0)
    z_tpt = np.clip(_causal_zscore(tpt_ratio, ref_mask), 0.0, None)

    if p_pdg_fast is not None and p_pdg_slow is not None:
        pdg_fast = np.asarray(p_pdg_fast.values, dtype=np.float64)
        pdg_slow = np.asarray(p_pdg_slow.values, dtype=np.float64)
        pdg_slow = np.where(pdg_slow < 1e-6, 1e-6, pdg_slow)
        pdg_ratio = np.nan_to_num(pdg_fast / pdg_slow, nan=1.0, posinf=10.0, neginf=0.0)
        z_pdg = np.clip(_causal_zscore(pdg_ratio, ref_mask), 0.0, None)
        composite = np.maximum(z_tpt, z_pdg)
    else:
        composite = z_tpt

    # Additional indicator: absolute fast variance (slug bursts are not just
    # ratios but also high amplitude).
    z_abs = np.clip(_causal_zscore(tpt_fast, ref_mask), 0.0, None)
    composite = np.maximum(composite, 0.5 * z_abs)

    return _rank_normalize(composite)


CLASS_SCORERS = {
    3: class3_slug_score,
    4: class4_oscillation_score,
    6: class6_choke_step_score,
}


def compute_physical_score(df: pd.DataFrame, event_class: int, ref_mask: np.ndarray) -> np.ndarray | None:
    fn = CLASS_SCORERS.get(int(event_class))
    if fn is None:
        return None
    return fn(df, ref_mask)
