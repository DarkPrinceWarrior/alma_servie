"""
Universal channel importance analysis for any detector in the generic pipeline.

For each raw channel, measures how much the detector score drops in the
anomaly zone when that channel is disabled.  Generates a self-contained
HTML report designed for customers & field engineers — plain Russian,
no jargon.
"""
from __future__ import annotations

import io
import base64
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

from alma_service.anomaly_specs import DetectionSpec, get_detection_spec
from alma_service.benchmark_metrics import load_intervals as load_intervals_df
from alma_service.detection_artifacts import (
    DEFAULT_DETECTOR,
    benchmark_summary_path,
    load_json,
    normalize_detector_key,
)
from alma_service.engineered_features import PreparedWellData, prepare_engineered_well
from alma_service.generic_detectors import (
    BaseDetector,
    IsolationForestDetector,
    LOFDetector,
    PCASPEDetector,
    PaAnoFeatureDetector,
    FusedDetector,
    set_seed,
)
from alma_service.paano_defaults import (
    MIN_REFERENCE_COVERAGE,
    MIN_TOTAL_COVERAGE,
    REFERENCE_MAX_RATIO,
    REFERENCE_MIN_DAYS,
    REFERENCE_MIN_RATIO,
)
from alma_service.paths import DB_DIR, REPORTS_DIR, ensure_parent
from alma_service.tabular_io import read_table

warnings.filterwarnings("ignore")

ANOMALY_RUNTIME_CONFIG = {
    "negermet": {
        "prepare_patch_size": 64,
        "paano_patch_short": 32,
        "paano_patch_long": 64,
    },
    "pritok": {
        "prepare_patch_size": 96,
        "paano_patch_short": 48,
        "paano_patch_long": 96,
    },
    "salt": {
        "prepare_patch_size": 96,
        "paano_patch_short": 48,
        "paano_patch_long": 96,
    },
}

DETECTOR_LABELS = {
    "pca_spe": "PCA/SPE",
    "fused": "Комбинированный",
    "lof": "LOF",
    "iforest": "Isolation Forest",
    "paano_feat": "PaAno + признаки",
}

DISPLAY_NAMES = {
    "negermet": "Негерметичность НКТ",
    "pritok": "Изменение притока",
    "salt": "Солеотложение",
}

COLOR_THEMES = {
    "negermet": {"accent": "#c0392b", "soft_bg": "#fdf2f2"},
    "pritok": {"accent": "#e67e22", "soft_bg": "#fef5ed"},
    "salt": {"accent": "#0f766e", "soft_bg": "#f0fdfa"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _runtime_config(anomaly_key: str) -> dict[str, Any]:
    return ANOMALY_RUNTIME_CONFIG.get(anomaly_key, ANOMALY_RUNTIME_CONFIG["salt"])


def load_anomaly_data(spec: DetectionSpec, source_path: str | None = None) -> pd.DataFrame:
    if source_path is not None:
        src = Path(source_path)
    else:
        candidates = [DB_DIR / name for name in spec.dataset.source_candidates]
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            raise FileNotFoundError(f"No source dataset found for {spec.anomaly_key}")
    df = read_table(src, dtypes={"well_id": str}, parse_dates=["timestamp"], low_memory=False)
    df["well_id"] = df["well_id"].astype(str).str.strip().str.lower()
    return df.dropna(subset=["timestamp"]).sort_values(["well_id", "timestamp"]).reset_index(drop=True)


def load_intervals(spec: DetectionSpec) -> pd.DataFrame:
    src = spec.dataset.intervals_path
    if not src.exists():
        return pd.DataFrame(columns=["well_id", "start_date", "end_date", "split", "interval_idx"])
    return load_intervals_df(src)


def _build_detector(anomaly_key: str, detector_key: str, device, verbose: bool = False) -> BaseDetector:
    cfg = _runtime_config(anomaly_key)
    if detector_key == "paano_feat":
        return PaAnoFeatureDetector(device=device, patch_short=int(cfg["paano_patch_short"]),
                                    patch_long=int(cfg["paano_patch_long"]), verbose=verbose)
    if detector_key == "pca_spe":
        return PCASPEDetector()
    if detector_key == "lof":
        return LOFDetector()
    if detector_key == "iforest":
        return IsolationForestDetector()
    if detector_key == "fused":
        return FusedDetector(device=device, verbose=verbose,
                             patch_short=int(cfg["paano_patch_short"]),
                             patch_long=int(cfg["paano_patch_long"]))
    raise ValueError(f"Unknown detector: {detector_key}")


def _resolve_detector(spec: DetectionSpec, detector: str | None) -> str:
    if detector:
        return normalize_detector_key(detector)
    benchmark = load_json(benchmark_summary_path(spec))
    selected = benchmark.get("selected_default_detector")
    if selected:
        try:
            return normalize_detector_key(selected)
        except ValueError:
            pass
    return DEFAULT_DETECTOR


# ---------------------------------------------------------------------------
# Channel-level importance
# ---------------------------------------------------------------------------

def _map_features_to_channels(
    feature_columns: list[str],
    raw_columns: list[str],
) -> dict[str, list[int]]:
    channel_to_indices: dict[str, list[int]] = {col: [] for col in raw_columns}
    other_indices: list[int] = []
    for idx, feat in enumerate(feature_columns):
        base = feat.split("::")[0] if "::" in feat else feat
        if base in channel_to_indices:
            channel_to_indices[base].append(idx)
        else:
            other_indices.append(idx)
    if other_indices:
        channel_to_indices["_other"] = other_indices
    return channel_to_indices


def compute_channel_importance(
    anomaly_key: str,
    detector_key: str,
    prepared: PreparedWellData,
    anomaly_mask: np.ndarray,
    device,
    verbose: bool = True,
) -> dict[str, Any] | None:
    set_seed()
    X = prepared.feature_matrix.copy()
    raw_columns = prepared.raw_columns
    feature_columns = prepared.feature_columns

    if not raw_columns or not anomaly_mask.any():
        return None

    channel_map = _map_features_to_channels(feature_columns, raw_columns)

    # Baseline
    baseline_detector = _build_detector(anomaly_key, detector_key, device, verbose=False)
    X_ref = X[prepared.reference_mask]
    baseline_detector.fit_reference(X_ref, mask_ref=prepared.reference_mask)
    baseline_scores = baseline_detector.score_stream(X, mask_all=prepared.stability_mask).primary.astype(np.float32)
    baseline_anom = float(np.mean(baseline_scores[anomaly_mask]))

    if verbose:
        print(f"    Baseline anomaly score: {baseline_anom:.6f}")

    channels = [ch for ch in channel_map if ch != "_other" and channel_map[ch]]

    if verbose:
        print(f"    Вычисление важности {len(channels)} каналов...")

    importance: dict[str, dict[str, float]] = {}
    for rank, ch_name in enumerate(channels, 1):
        indices = channel_map[ch_name]
        X_perturbed = X.copy()
        X_perturbed[:, indices] = 0.0

        det = _build_detector(anomaly_key, detector_key, device, verbose=False)
        det.fit_reference(X_perturbed[prepared.reference_mask], mask_ref=prepared.reference_mask)
        perturbed_scores = det.score_stream(X_perturbed, mask_all=prepared.stability_mask).primary.astype(np.float32)
        perturbed_anom = float(np.mean(perturbed_scores[anomaly_mask]))

        drop = baseline_anom - perturbed_anom
        pct_drop = (drop / baseline_anom * 100) if baseline_anom > 0 else 0.0
        importance[ch_name] = {
            "drop": drop,
            "baseline": baseline_anom,
            "perturbed": perturbed_anom,
            "pct_drop": pct_drop,
        }
        if verbose:
            print(f"      [{rank:2d}/{len(channels)}] {ch_name}: "
                  f"drop={drop:+.6f} ({pct_drop:+.1f}%)")

    return {
        "well_id": prepared.well_id,
        "importance": importance,
        "baseline_scores": baseline_scores,
        "timestamps": prepared.timestamps,
        "anomaly_mask": anomaly_mask,
        "raw_data": prepared.raw_matrix,
        "raw_columns": raw_columns,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _make_bar_b64(result: dict[str, Any], accent: str, display_name: str, top_n: int = 15) -> str:
    imp = result["importance"]
    df = pd.DataFrame([
        {"channel": k, "drop_pct": v["pct_drop"]} for k, v in imp.items()
    ]).sort_values("drop_pct", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df))))
    colors = [accent if v > 0 else "#bdc3c7" for v in df["drop_pct"]]
    ax.barh(range(len(df)), df["drop_pct"].values, color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["channel"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Изменение способности обнаружить аномалию при отключении канала, %")
    ax.set_title(
        f"Скважина {result['well_id']} — Влияние каналов на обнаружение {display_name.lower()}",
        fontsize=11, fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    # Legend for bar colors
    legend_pos = mpatches.Patch(color=accent, label="Канал помогает обнаружить аномалию")
    legend_neg = mpatches.Patch(color="#bdc3c7", label="Канал не помогает / мешает")
    ax.legend(handles=[legend_pos, legend_neg], loc="lower right", fontsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _make_timeseries_b64(
    result: dict[str, Any],
    intervals_df: pd.DataFrame,
    accent: str,
    display_name: str,
    top_n: int = 6,
) -> str:
    well_id = result["well_id"]
    timestamps = result["timestamps"]
    imp = result["importance"]
    raw_columns = result["raw_columns"]
    raw_data = result["raw_data"]

    sorted_channels = sorted(imp.keys(), key=lambda k: abs(imp[k]["pct_drop"]), reverse=True)
    top_channels = sorted_channels[:top_n]

    wi = intervals_df[intervals_df["well_id"] == well_id].sort_values("start_date")

    n_panels = top_n + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.8 * n_panels), sharex=True)
    ts_pd = pd.to_datetime(timestamps)

    # --- Helper to draw anomaly zones on an axis ---
    def _draw_anomaly_zones(ax):
        for zone_idx, (_, row) in enumerate(wi.iterrows()):
            ax.axvspan(row["start_date"], row["end_date"], color="red", alpha=0.12, zorder=0)
            ax.axvline(row["start_date"], color="#16a34a", linewidth=1.2, linestyle="-",
                       label="Начало аномалии" if zone_idx == 0 else None)
            ax.axvline(row["end_date"], color="#dc2626", linewidth=1.2, linestyle="--",
                       label="Конец аномалии" if zone_idx == 0 else None)

    # --- Top panel: detector score ---
    ax0 = axes[0]
    ax0.fill_between(ts_pd, 0, result["baseline_scores"], color=accent, alpha=0.20)
    ax0.plot(ts_pd, result["baseline_scores"], color=accent, linewidth=0.6)
    _draw_anomaly_zones(ax0)
    ax0.set_ylabel("Отклонение от\nнормы (все каналы)", fontsize=8)
    ax0.set_title(
        f"Скважина {well_id} — Показания наиболее значимых каналов",
        fontsize=11, fontweight="bold",
    )
    ax0.grid(True, alpha=0.3)

    # Legend on first axis
    zone_patch = mpatches.Patch(facecolor="red", alpha=0.12, edgecolor="none",
                                label="Зона фактической аномалии")
    start_line = plt.Line2D([0], [0], color="#16a34a", linewidth=1.2, label="Начало аномалии (факт)")
    end_line = plt.Line2D([0], [0], color="#dc2626", linewidth=1.2, linestyle="--",
                          label="Конец аномалии (факт)")
    score_line = plt.Line2D([0], [0], color=accent, linewidth=1, label="Степень отклонения от нормы")
    ax0.legend(handles=[score_line, zone_patch, start_line, end_line],
               loc="upper right", fontsize=7, framealpha=0.9)

    # --- Channel panels ---
    for idx, ch_name in enumerate(top_channels):
        ax = axes[idx + 1]
        pct = imp[ch_name]["pct_drop"]

        if ch_name in raw_columns:
            col_idx = raw_columns.index(ch_name)
            values = raw_data[:, col_idx]
        else:
            values = np.zeros(len(timestamps))

        ax.plot(ts_pd, values, color="tab:blue", linewidth=0.5, alpha=0.8)

        # Y-label: channel name + its influence
        if pct > 0.5:
            influence_text = f"влияние: +{pct:.1f}%"
        elif pct < -0.5:
            influence_text = f"влияние: {pct:.1f}%"
        else:
            influence_text = "влияние: ~0%"
        ax.set_ylabel(f"{ch_name}\n({influence_text})", fontsize=7)

        _draw_anomaly_zones(ax)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Время")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_b64(fig, dpi=110)


# ---------------------------------------------------------------------------
# Per-well summary text
# ---------------------------------------------------------------------------

def _well_summary_html(result: dict[str, Any], accent: str) -> str:
    """Generate a short human-readable summary for a well."""
    imp = result["importance"]
    sorted_items = sorted(imp.items(), key=lambda x: x[1]["pct_drop"], reverse=True)

    positive = [(ch, v) for ch, v in sorted_items if v["pct_drop"] > 0.5]
    negative = [(ch, v) for ch, v in sorted_items if v["pct_drop"] < -0.5]
    neutral = [(ch, v) for ch, v in sorted_items if -0.5 <= v["pct_drop"] <= 0.5]

    lines: list[str] = []

    if positive:
        top3 = ", ".join(f"<b>{ch}</b> ({v['pct_drop']:+.1f}%)" for ch, v in positive[:3])
        lines.append(
            f"<p>✅ <b>Ключевые каналы для обнаружения аномалии</b> (при их отключении "
            f"модель хуже видит аномалию): {top3}"
            + (f" и ещё {len(positive) - 3} канал(а/ов)" if len(positive) > 3 else "")
            + ".</p>"
        )
    else:
        lines.append(
            "<p>ℹ️ У данной скважины <b>ни один канал по отдельности</b> не является "
            "решающим для обнаружения аномалии. Модель использует <b>совокупность изменений "
            "по всем каналам сразу</b> — аномалия проявляется в комбинации показателей, "
            "а не в каком-то одном датчике.</p>"
        )

    if negative:
        top_neg = ", ".join(f"<b>{ch}</b>" for ch, _ in negative[:3])
        lines.append(
            f"<p>⚠️ Каналы, которые <b>вносят шум</b> (при их отключении модель "
            f"даже лучше видит аномалию): {top_neg}"
            + (f" и ещё {len(negative) - 3}" if len(negative) > 3 else "")
            + ".</p>"
        )

    if neutral and not positive:
        lines.append(
            "<p>Все каналы имеют примерно одинаковое влияние — модель опирается "
            "на совместное поведение показателей.</p>"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _build_html(
    spec: DetectionSpec,
    detector_key: str,
    all_results: list[dict[str, Any]],
    intervals_df: pd.DataFrame,
) -> str:
    theme = COLOR_THEMES.get(spec.anomaly_key, COLOR_THEMES["salt"])
    accent = theme["accent"]
    soft_bg = theme["soft_bg"]
    display = DISPLAY_NAMES.get(spec.anomaly_key, spec.display_name)
    title = f"Анализ влияния каналов на обнаружение: {display}"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color: #222; background: #fafafa; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid {accent}; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 35px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; background: #fff; }}
        th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; font-size: 0.92em; }}
        th {{ background-color: {accent}; color: #fff; }}
        tr:nth-child(even) {{ background-color: {soft_bg}; }}
        .positive {{ color: {accent}; font-weight: bold; }}
        .negative {{ color: #95a5a6; }}
        .plot-container {{ margin-bottom: 35px; background: #fff; border: 1px solid #ddd;
                          border-radius: 6px; padding: 15px; }}
        img {{ max-width: 100%; height: auto; }}
        .method-box {{ background: {soft_bg}; border-left: 4px solid {accent}; padding: 15px 20px;
                       margin: 20px 0; border-radius: 4px; line-height: 1.8; }}
        .method-box b {{ color: {accent}; }}
        .summary-box {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
                        padding: 16px 20px; margin: 16px 0; line-height: 1.7; }}
        .summary-box b {{ color: {accent}; }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <div class="method-box">
        <h3>Как читать этот отчёт</h3>
        <p>Модель детекции анализирует показания нескольких каналов (датчиков) скважины
        и на их основе определяет, есть ли аномалия. Этот отчёт показывает,
        <b>какие каналы влияют на обнаружение больше всего</b>.</p>
        <p><b>Методика:</b> мы поочерёдно «отключаем» каждый канал (подставляем вместо
        его показаний среднее нормальное значение) и смотрим, как это повлияло
        на способность модели обнаружить аномалию:</p>
        <ul>
            <li><b>Положительное значение (%)</b> — канал <b>помогает</b> обнаруживать аномалию.
                Чем больше число, тем важнее канал. Без него модель хуже видит аномалию.</li>
            <li><b>Около нуля</b> — канал <b>не влияет</b> на обнаружение в отдельности.</li>
            <li><b>Отрицательное значение (%)</b> — канал <b>мешает</b>: без него модель
                обнаруживает аномалию даже лучше (канал вносит помехи).</li>
        </ul>
        <p><b>На графиках:</b></p>
        <ul>
            <li>🟩 <b>Зелёная вертикальная линия</b> — фактическое начало аномалии</li>
            <li>🟥 <b>Красная пунктирная линия</b> — фактическое окончание аномалии</li>
            <li>🔴 <b>Красная полоса (фон)</b> — весь период аномалии</li>
        </ul>
    </div>
"""

    for result in all_results:
        wid = result["well_id"]
        imp = result["importance"]
        html += f'<h2>Скважина {wid}</h2>\n'

        # Summary
        html += f'<div class="summary-box">\n{_well_summary_html(result, accent)}\n</div>\n'

        # Bar chart
        bar_b64 = _make_bar_b64(result, accent, display, top_n=15)
        html += f"""
    <div class="plot-container">
        <img src="data:image/png;base64,{bar_b64}" alt="Влияние каналов {wid}">
    </div>
"""

        # Timeseries
        ts_b64 = _make_timeseries_b64(result, intervals_df, accent, display, top_n=6)
        html += f"""
    <div class="plot-container">
        <img src="data:image/png;base64,{ts_b64}" alt="Показания каналов {wid}">
    </div>
"""

        # Simplified table
        sorted_imp = sorted(imp.items(), key=lambda x: x[1]["pct_drop"], reverse=True)
        html += """
    <table>
        <tr>
            <th>#</th>
            <th>Канал (датчик)</th>
            <th>Влияние на обнаружение, %</th>
            <th>Роль</th>
        </tr>
"""
        for rank, (ch, v) in enumerate(sorted_imp, 1):
            pct = v["pct_drop"]
            if pct > 0.5:
                cls = "positive"
                role = "Помогает обнаружить"
            elif pct < -0.5:
                cls = "negative"
                role = "Вносит помехи"
            else:
                cls = "negative"
                role = "Не влияет"
            html += f"""
        <tr>
            <td>{rank}</td>
            <td>{ch}</td>
            <td class="{cls}">{pct:+.1f}%</td>
            <td>{role}</td>
        </tr>
"""
        html += "    </table>\n"

    html += """
</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_feature_importance_report(
    anomaly_key: str,
    detector: str | None = None,
    output_path: str | None = None,
    source_path: str | None = None,
    verbose: bool = True,
) -> Path:
    import torch

    spec = get_detection_spec(anomaly_key)
    detector_key = _resolve_detector(spec, detector)
    det_label = DETECTOR_LABELS.get(detector_key, detector_key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    print(f"Аномалия: {spec.display_name}")
    print(f"Детектор: {det_label} ({detector_key})")

    df = load_anomaly_data(spec, source_path=source_path)
    intervals = load_intervals(spec)

    if intervals.empty:
        raise RuntimeError("Нет данных об интервалах аномалий")

    split_map = (
        intervals[["well_id", "split"]]
        .drop_duplicates("well_id")
        .set_index("well_id")["split"]
        .to_dict()
    )

    cfg = _runtime_config(anomaly_key)
    all_results: list[dict[str, Any]] = []

    for wid in sorted(df["well_id"].unique()):
        well_df = df[df["well_id"] == wid].copy()
        print(f"\nСкважина {wid}:")

        set_seed()
        prepared = prepare_engineered_well(
            anomaly_key=anomaly_key,
            well_id=wid,
            split=split_map.get(wid, "train"),
            well_df=well_df,
            patch_size=int(cfg["prepare_patch_size"]),
            reference_min_ratio=REFERENCE_MIN_RATIO,
            reference_max_ratio=REFERENCE_MAX_RATIO,
            reference_min_days=REFERENCE_MIN_DAYS,
            min_reference_coverage=MIN_REFERENCE_COVERAGE,
            min_total_coverage=MIN_TOTAL_COVERAGE,
        )
        if prepared is None:
            print("  Пропуск: недостаточно данных после препроцессинга")
            continue

        well_intervals = intervals[intervals["well_id"] == wid].sort_values("start_date")
        anomaly_mask = np.zeros(len(prepared.timestamps), dtype=bool)
        for _, row in well_intervals.iterrows():
            idx_mask = (
                (prepared.timestamps >= np.datetime64(row["start_date"])) &
                (prepared.timestamps <= np.datetime64(row["end_date"]))
            )
            anomaly_mask |= idx_mask

        if not anomaly_mask.any():
            print("  Пропуск: нет аномальных точек в данных")
            continue

        print(f"  Подготовлено: {prepared.detail['points']} точек, "
              f"{len(prepared.raw_columns)} каналов, "
              f"{int(anomaly_mask.sum())} аномальных точек")

        result = compute_channel_importance(
            anomaly_key, detector_key, prepared, anomaly_mask, device, verbose=verbose
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("Ни одна скважина не прошла анализ")

    html = _build_html(spec, detector_key, all_results, intervals)

    if output_path is None:
        output_path = str(
            REPORTS_DIR / f"{spec.dataset.output_prefix}_{detector_key}_feature_importance.html"
        )
    out = ensure_parent(Path(output_path))
    out.write_text(html, encoding="utf-8")
    print(f"\nОтчёт сохранён: {out}")
    return out
