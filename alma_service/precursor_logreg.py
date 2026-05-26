from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PRESSURE_COLUMN_ALIAS = "давление на приеме"
DEFAULT_EMA_ALPHA = 0.08
DEFAULT_ROLLING_WINDOW_MIN = 120.0
FEATURE_NAMES = (
    "smoothed_score",
    "delta_ema_score",
    "rolling_sigma_score",
    "rolling_sigma_pressure",
)


@dataclass(frozen=True)
class PrecursorLogregModel:
    feature_names: tuple[str, ...]
    coef: tuple[float, ...]
    intercept: float
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    ema_alpha: float
    rolling_window_minutes: float
    train_pos: int
    train_neg: int
    n_wells: int
    oof_auc: float | None = None
    cv_metrics: dict[str, float] = field(default_factory=dict)

    def to_json_payload(self) -> dict:
        payload = asdict(self)
        payload["coef"] = list(self.coef)
        payload["scaler_mean"] = list(self.scaler_mean)
        payload["scaler_scale"] = list(self.scaler_scale)
        payload["feature_names"] = list(self.feature_names)
        return payload


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_std(values: np.ndarray, window_points: int) -> np.ndarray:
    if window_points <= 1 or len(values) == 0:
        return np.zeros_like(values, dtype=np.float64)
    return (
        pd.Series(values, dtype="float64")
        .rolling(window_points, min_periods=max(window_points // 2, 2))
        .std()
        .fillna(0.0)
        .to_numpy()
    )


def _step_minutes_from_timestamps(timestamps: np.ndarray) -> float:
    ts = pd.to_datetime(timestamps)
    if len(ts) < 2:
        return 5.0
    diffs = (ts[1:] - ts[:-1]).total_seconds() / 60.0
    diffs = np.asarray(diffs)
    diffs = diffs[(diffs > 0) & (diffs < 24 * 60)]
    return float(np.median(diffs)) if len(diffs) else 5.0


def resolve_pressure_index(raw_columns: Iterable[str]) -> int | None:
    for idx, name in enumerate(raw_columns):
        if PRESSURE_COLUMN_ALIAS in str(name).strip().lower():
            return idx
    return None


def extract_features(
    timestamps: np.ndarray,
    score: np.ndarray,
    raw_columns: Iterable[str] | None = None,
    raw_matrix: np.ndarray | None = None,
    *,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    rolling_window_minutes: float = DEFAULT_ROLLING_WINDOW_MIN,
) -> np.ndarray:
    score = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0)
    smoothed = _ema(score, ema_alpha)
    delta = np.zeros_like(smoothed)
    if len(smoothed) > 1:
        delta[1:] = smoothed[1:] - smoothed[:-1]

    step_min = _step_minutes_from_timestamps(np.asarray(timestamps))
    window_points = max(int(round(rolling_window_minutes / max(step_min, 1e-6))), 2)
    rolling_sigma_score = _rolling_std(score, window_points)

    pressure = np.zeros_like(score)
    if raw_columns is not None and raw_matrix is not None:
        idx = resolve_pressure_index(raw_columns)
        if idx is not None and idx < raw_matrix.shape[1]:
            pressure = np.nan_to_num(np.asarray(raw_matrix[:, idx], dtype=np.float64), nan=0.0)
    rolling_sigma_pressure = _rolling_std(pressure, window_points)

    return np.stack(
        [smoothed, delta, rolling_sigma_score, rolling_sigma_pressure],
        axis=1,
    )


def _logistic(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def score_proba(model: PrecursorLogregModel, features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.zeros(0, dtype=np.float64)
    scale = np.asarray(model.scaler_scale, dtype=np.float64)
    scale = np.where(scale == 0, 1.0, scale)
    z = (features - np.asarray(model.scaler_mean, dtype=np.float64)) / scale
    logits = z @ np.asarray(model.coef, dtype=np.float64) + float(model.intercept)
    return _logistic(logits)


def _train_with_sklearn(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    model = LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")
    model.fit(Xz, y)
    return (
        model.coef_.ravel().astype(np.float64),
        float(model.intercept_[0]),
        scaler.mean_.astype(np.float64),
        scaler.scale_.astype(np.float64),
    )


def _safe_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n1, n0 = len(pos), len(neg)
    all_s = np.concatenate([pos, neg])
    ranks = pd.Series(all_s).rank().to_numpy()
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_lowo(
    per_well: dict[str, dict[str, np.ndarray]],
    *,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    rolling_window_minutes: float = DEFAULT_ROLLING_WINDOW_MIN,
) -> PrecursorLogregModel:
    """per_well[well_id] = {"features": ndarray(N,4), "target": ndarray(N,), "valid": ndarray(N,)}."""
    if not per_well:
        raise ValueError("train_lowo: empty per_well mapping")
    X_all = np.concatenate(
        [d["features"][d["valid"]] for d in per_well.values()], axis=0
    )
    y_all = np.concatenate(
        [d["target"][d["valid"]] for d in per_well.values()], axis=0
    )
    n_pos = int((y_all == 1).sum())
    n_neg = int((y_all == 0).sum())
    if n_pos < 5 or n_neg < 5 or len(set(y_all)) < 2:
        raise ValueError(
            f"train_lowo: insufficient class balance (pos={n_pos}, neg={n_neg})"
        )

    coef, intercept, mean, scale = _train_with_sklearn(X_all, y_all)

    oof_pos: list[np.ndarray] = []
    oof_neg: list[np.ndarray] = []
    if len(per_well) >= 2:
        for hold in list(per_well.keys()):
            train_keys = [k for k in per_well.keys() if k != hold]
            X_tr = np.concatenate(
                [per_well[k]["features"][per_well[k]["valid"]] for k in train_keys],
                axis=0,
            )
            y_tr = np.concatenate(
                [per_well[k]["target"][per_well[k]["valid"]] for k in train_keys],
                axis=0,
            )
            if len(set(y_tr)) < 2 or (y_tr == 1).sum() < 3:
                continue
            try:
                f_coef, f_int, f_mean, f_scale = _train_with_sklearn(X_tr, y_tr)
            except Exception:
                continue
            fold_model = PrecursorLogregModel(
                feature_names=FEATURE_NAMES,
                coef=tuple(f_coef.tolist()),
                intercept=f_int,
                scaler_mean=tuple(f_mean.tolist()),
                scaler_scale=tuple(f_scale.tolist()),
                ema_alpha=ema_alpha,
                rolling_window_minutes=rolling_window_minutes,
                train_pos=int((y_tr == 1).sum()),
                train_neg=int((y_tr == 0).sum()),
                n_wells=len(train_keys),
            )
            X_te = per_well[hold]["features"][per_well[hold]["valid"]]
            y_te = per_well[hold]["target"][per_well[hold]["valid"]]
            if len(y_te) == 0:
                continue
            proba_te = score_proba(fold_model, X_te)
            oof_pos.append(proba_te[y_te == 1])
            oof_neg.append(proba_te[y_te == 0])

    auc = None
    if oof_pos and oof_neg:
        pos = np.concatenate(oof_pos)
        neg = np.concatenate(oof_neg)
        if len(pos) and len(neg):
            auc = _safe_auc(pos, neg)

    return PrecursorLogregModel(
        feature_names=FEATURE_NAMES,
        coef=tuple(coef.tolist()),
        intercept=intercept,
        scaler_mean=tuple(mean.tolist()),
        scaler_scale=tuple(scale.tolist()),
        ema_alpha=ema_alpha,
        rolling_window_minutes=rolling_window_minutes,
        train_pos=n_pos,
        train_neg=n_neg,
        n_wells=len(per_well),
        oof_auc=auc,
        cv_metrics={"oof_auc": auc} if auc is not None else {},
    )


def save_model(model: PrecursorLogregModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.to_json_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_model(path: Path) -> PrecursorLogregModel | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PrecursorLogregModel(
        feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
        coef=tuple(payload["coef"]),
        intercept=float(payload["intercept"]),
        scaler_mean=tuple(payload["scaler_mean"]),
        scaler_scale=tuple(payload["scaler_scale"]),
        ema_alpha=float(payload.get("ema_alpha", DEFAULT_EMA_ALPHA)),
        rolling_window_minutes=float(payload.get("rolling_window_minutes", DEFAULT_ROLLING_WINDOW_MIN)),
        train_pos=int(payload.get("train_pos", 0)),
        train_neg=int(payload.get("train_neg", 0)),
        n_wells=int(payload.get("n_wells", 0)),
        oof_auc=payload.get("oof_auc"),
        cv_metrics=dict(payload.get("cv_metrics", {})),
    )
