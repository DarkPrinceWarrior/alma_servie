# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tooling

For any file search or grep in the current git indexed directory use fff tools.

## Project Purpose

**alma_servie** — oil-well anomaly detection system for three anomaly types:
- `negermet` (`Негерметичность`) — sharp pressure drops
- `pritok` (`Приток`) — gradual pressure trends
- `salt` (`Солеотложение`) — salt deposition, with additional soft-sensor derived features

End-to-end pipeline: Excel exports → Parquet datasets → blind detection → interactive HTML reports.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Key Commands

### Build datasets
```bash
python scripts/datasets/build_negermet_dataset.py --freq 15s
python scripts/datasets/build_pritok_dataset.py --freq 2min
python scripts/datasets/build_salt_dataset.py --freq 2min
# Or all at once:
bash scripts/run_full_dataset_build.sh
```

### Run detection
```bash
# Unified stack (preferred): detectors = pca_spe | paano_feat
python scripts/detection/detect_negermet.py --detector pca_spe
python scripts/detection/detect_pritok.py --detector pca_spe
python scripts/detection/detect_salt.py --detector pca_spe

# Legacy PaAno baseline:
python scripts/detection/detect_negermet_paano.py
python scripts/detection/detect_pritok_paano.py
python scripts/detection/detect_salt_paano.py

# Full benchmark:
bash scripts/run_full_detection_benchmark.sh
```

### Generate reports
```bash
python scripts/reports/generate_negermet_paano_report.py [--detector pca_spe]
python scripts/reports/generate_pritok_paano_report.py [--detector pca_spe]
python scripts/reports/generate_salt_paano_report.py [--detector pca_spe]
```
By default, report reads detector from `artifacts/results/*_benchmark_summary.json`.

### Evaluate onset quality
```bash
python scripts/evaluation/evaluate_onset_metrics.py \
  --anomaly salt --detector pca_spe --name salt_pca_spe
```

No linting, formatting, or test runner is configured. After changes, run the affected script directly to check for import/runtime errors.

## Architecture

### Data flow
```
data/raw/*.xlsx
  → scripts/datasets/build_*_dataset.py
  → db/*_anomaly_database_*.parquet + db/*_intervals.parquet
  → scripts/detection/detect_*.py
  → db/*_<detector>_scores.parquet + artifacts/results/*
  → scripts/reports/generate_*_report.py
  → artifacts/reports/*_<detector>_report.html
```

### Library: `alma_service/`
The shared library that all scripts import from. Key modules:
- `paths.py` — all canonical paths; `PROJECT_ROOT` resolved from file location so scripts work from any cwd
- `dataset_config.py` — well lists, train/test split, parameter renames (single source of truth for data config)
- `generic_detection.py` — `run_detection()` unified entry point; onset config, tune grids, runtime config per anomaly type
- `paano_pipeline.py` — `run_detection()` legacy PaAno entry point
- `onset_detection.py` — `detect_causal_onsets()`, numba-JIT compiled CUSUM + EMA; calibrated via `CausalThresholds`

### Detector interface (`BaseDetector`)
All detectors implement:
- `fit_reference(X_ref, mask_ref)` — train on reference (train-split) data
- `score_stream(X_all, mask_all)` → anomaly scores

Concrete implementations: `PaAnoFeatureDetector`, `PCASPEDetector`, `SharedPaAnoDetector`.

### Staged blind pipeline (all three anomaly types)
1. Causal preprocessing
2. Instability mask
3. Engineered features
4. Unified onset layer (Optuna/TPE-tuned on train wells only, then applied to both splits)

`salt` uses the same pipeline but adds soft-sensor derived features.

### `paano/` submodule
The PaAno neural library (ICLR 2026 paper). Used via `PaAnoFeatureDetector` and the legacy scripts. Entry points: `paano/main.py`, `paano/train.py`, `paano/model.py`.

## Conventions

- `from __future__ import annotations` at the top of every module
- Type hints throughout; union types via `X | Y` (Python 3.10+)
- `@dataclass` for data-holding structs (e.g. `DetectorScoreOutput`, `CausalThresholds`)
- No docstrings — names are self-documenting
- All heavy I/O in Parquet; JSON only for summary/config/tuning outputs
- `torch.compile` enabled only for local `PatchEncoder` with a safe fallback

## Output locations
- Raw datasets: `db/*_anomaly_database_*.parquet`
- Intervals + splits: `db/*_intervals.parquet`
- Per-point scores: `db/*_<detector>_scores.parquet`
- Predicted starts: `db/*_<detector>_predicted_starts.parquet`
- Detector config/tuning: `db/*_<detector>_config.json`, `db/*_<detector>_tuning.json`
- Result tables: `artifacts/results/*_<detector>_results.parquet` + `.summary.json`
- Benchmark summary: `artifacts/results/*_benchmark_summary.json`
- HTML reports: `artifacts/reports/*_<detector>_report.html`
- Model weights: `models/`

`db/`, `artifacts/`, and `models/` are gitignored — do not commit their contents.
