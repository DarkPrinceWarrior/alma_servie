"""Synthetic anomaly injection transforms for PaAno transfer fine-tune.

Used by `scripts/evaluation/transfer_3w_to_alma.py` during ALMA-side fine-tune
to expose the encoder to anomaly-like perturbations on otherwise-normal target
windows. The encoder still trains via the standard PaAno patch-reconstruction
objective; injection only modifies a fraction of input rows so the encoder
learns to keep clean reconstructions for normal patterns and produce
high reconstruction residual for injected ones.

All transforms operate on a 2D matrix `pool` of shape (rows, channels) and
return a new modified copy. Channels are interpreted as already-standardized
features. The functions are pure numpy and stochastic via the supplied RNG.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InjectionConfig:
    rate: float = 0.3                   # fraction of (row, channel) blocks affected
    spike_prob: float = 0.25
    scale_shift_prob: float = 0.25
    collective_prob: float = 0.25
    jitter_prob: float = 0.25
    spike_magnitude: float = 6.0        # multiples of channel std (already standardized -> ~6 sigma)
    scale_shift_magnitude: float = 3.0  # additive shift in sigma units
    scale_shift_min_len: int = 16
    scale_shift_max_len: int = 96
    collective_min_len: int = 32
    collective_max_len: int = 128
    jitter_sigma: float = 0.4

    def normalize(self) -> InjectionConfig:
        probs = np.array(
            [self.spike_prob, self.scale_shift_prob, self.collective_prob, self.jitter_prob],
            dtype=np.float64,
        )
        total = float(probs.sum())
        if total <= 0:
            return self
        norm = probs / total
        return InjectionConfig(
            rate=self.rate,
            spike_prob=float(norm[0]),
            scale_shift_prob=float(norm[1]),
            collective_prob=float(norm[2]),
            jitter_prob=float(norm[3]),
            spike_magnitude=self.spike_magnitude,
            scale_shift_magnitude=self.scale_shift_magnitude,
            scale_shift_min_len=self.scale_shift_min_len,
            scale_shift_max_len=self.scale_shift_max_len,
            collective_min_len=self.collective_min_len,
            collective_max_len=self.collective_max_len,
            jitter_sigma=self.jitter_sigma,
        )


def _inject_spike(window: np.ndarray, rng: np.random.Generator, cfg: InjectionConfig) -> np.ndarray:
    out = window.copy()
    n_rows, n_chan = out.shape
    if n_rows == 0 or n_chan == 0:
        return out
    n_spikes = max(1, int(n_rows * 0.02))
    rows = rng.integers(0, n_rows, size=n_spikes)
    chans = rng.integers(0, n_chan, size=n_spikes)
    signs = rng.choice([-1.0, 1.0], size=n_spikes)
    out[rows, chans] = out[rows, chans] + signs * cfg.spike_magnitude
    return out


def _inject_scale_shift(window: np.ndarray, rng: np.random.Generator, cfg: InjectionConfig) -> np.ndarray:
    out = window.copy()
    n_rows, n_chan = out.shape
    if n_rows < cfg.scale_shift_min_len + 2 or n_chan == 0:
        return out
    seg_len = int(rng.integers(cfg.scale_shift_min_len, min(cfg.scale_shift_max_len, max(n_rows // 2, cfg.scale_shift_min_len + 1)) + 1))
    seg_start = int(rng.integers(0, n_rows - seg_len + 1))
    chan_count = max(1, int(n_chan * rng.uniform(0.1, 0.4)))
    chan_count = min(chan_count, n_chan)
    chans = rng.choice(n_chan, size=chan_count, replace=False)
    shift = rng.choice([-1.0, 1.0]) * cfg.scale_shift_magnitude
    out[seg_start:seg_start + seg_len, chans] = out[seg_start:seg_start + seg_len, chans] + shift
    return out


def _inject_collective(window: np.ndarray, rng: np.random.Generator, cfg: InjectionConfig) -> np.ndarray:
    out = window.copy()
    n_rows, n_chan = out.shape
    if n_rows < cfg.collective_min_len + 2 or n_chan == 0:
        return out
    seg_len = int(rng.integers(cfg.collective_min_len, min(cfg.collective_max_len, max(n_rows // 2, cfg.collective_min_len + 1)) + 1))
    seg_start = int(rng.integers(0, n_rows - seg_len + 1))
    # Reverse the segment (pattern flip) — preserves marginal distribution but breaks temporal structure.
    out[seg_start:seg_start + seg_len, :] = out[seg_start:seg_start + seg_len, :][::-1]
    return out


def _inject_jitter(window: np.ndarray, rng: np.random.Generator, cfg: InjectionConfig) -> np.ndarray:
    noise = rng.normal(0.0, cfg.jitter_sigma, size=window.shape).astype(window.dtype)
    return window + noise


_INJECTORS = [
    ("spike", _inject_spike),
    ("scale_shift", _inject_scale_shift),
    ("collective", _inject_collective),
    ("jitter", _inject_jitter),
]


def random_inject(window: np.ndarray, rng: np.random.Generator, cfg: InjectionConfig) -> tuple[np.ndarray, str]:
    """Apply one randomly-chosen injection transform to the window."""
    cfg_n = cfg.normalize()
    probs = np.array(
        [cfg_n.spike_prob, cfg_n.scale_shift_prob, cfg_n.collective_prob, cfg_n.jitter_prob],
        dtype=np.float64,
    )
    idx = int(rng.choice(len(_INJECTORS), p=probs))
    name, fn = _INJECTORS[idx]
    return fn(window, rng, cfg), name


def inject_pool(
    pool: np.ndarray,
    cfg: InjectionConfig,
    rng: np.random.Generator | None = None,
    window_len: int = 256,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply injection to a fraction `cfg.rate` of non-overlapping windows.

    Returns a new pool with the same shape and a counter dict of which
    transforms were applied.
    """
    if rng is None:
        rng = np.random.default_rng()
    if pool.size == 0 or cfg.rate <= 0:
        return pool, {"spike": 0, "scale_shift": 0, "collective": 0, "jitter": 0, "windows": 0}

    out = pool.copy()
    n = len(out)
    stride = max(1, int(window_len))
    starts = list(range(0, n - stride + 1, stride))
    n_inject = int(round(len(starts) * float(cfg.rate)))
    counter = {"spike": 0, "scale_shift": 0, "collective": 0, "jitter": 0, "windows": len(starts)}
    if n_inject <= 0:
        return out, counter

    chosen = rng.choice(len(starts), size=n_inject, replace=False)
    for idx in chosen:
        s = starts[int(idx)]
        window = out[s:s + stride]
        new_window, name = random_inject(window, rng, cfg)
        out[s:s + stride] = new_window
        counter[name] = counter.get(name, 0) + 1
    return out, counter
