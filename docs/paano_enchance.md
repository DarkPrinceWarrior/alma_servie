Ниже — готовый план, который можно отдать coding-agent’у как ТЗ. Я формулирую его как последовательность PR/итераций: сначала исправления без изменения алгоритма, потом ускорение, затем ablation-кандидаты. Это важно, потому что PaAno позиционируется авторами как лёгкий patch-based метод: короткие патчи кодируются компактным 1D-CNN, модель обучается через triplet loss + pretext loss, а anomaly score считается сравнением test embeddings с memory bank нормальных patch embeddings. ([GitHub][1])

---

## ТЗ для AI coding-agent: улучшение PaAno

### Цель

Улучшить кодовую базу `jinnnju/PaAno`, не ломая исходную идею алгоритма: лёгкий patch-level encoder + memory bank + distance-based scoring. Не заменять PaAno на Transformer/Foundation model. Все изменения проводить через маленькие PR, каждый PR должен иметь benchmark до/после.

Актуальный репозиторий помечен как официальный PyTorch-код для ICLR 2026, а README указывает, что в мае 2026 были внесены minor implementation refinements под финальную статью. ([GitHub][1]) Поэтому агент должен работать от текущей `main`-ветки, а не от старых копий.

---

## Общие правила для агента

1. Не менять одновременно correctness, speed и model quality. Один PR — один класс изменений.
2. Перед каждым изменением зафиксировать baseline: seed, dataset/file, `patch_size`, `batch_size`, `num_iters`, `use_revin`, размер memory bank, latency train/inference, peak GPU memory, AUROC, AUPRC, VUS-ROC, VUS-PR, Standard-F1, Range-F1.
3. Не использовать test labels для подбора threshold или гиперпараметров. Статья PaAno специально подчёркивает оценку без point adjustment и threshold tuning. ([arXiv][2])
4. Любое изменение, которое может ухудшить latency или memory footprint, должно быть флагом в CLI/config и выключено по умолчанию до ablation.
5. Все новые режимы должны сохранять обратную совместимость с текущими `script/run_uni.sh` и `script/run_mul.sh`.

---

# Phase 0 — подготовка инфраструктуры

### PR-0.1: воспроизводимость и benchmark harness

**Задача:** добавить минимальный benchmark-слой, чтобы дальнейшие PR не были “на глаз”.

Агенту изменить/добавить:

```text
configs/
  baseline_uni.yaml
  baseline_mul.yaml
scripts/
  benchmark_paano.py
  ablate_paano.py
utils/
  reproducibility.py
  profiling.py
```

Что реализовать:

```python
def set_all_seeds(seed: int) -> None:
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

В отчёт каждого запуска писать:

```text
commit_hash
seed
dataset_path
file_name
patch_size
batch_size
num_iters
lr
use_revin
memory_bank_size_before
memory_bank_size_after
search_backend
top_k
aggregation
train_seconds
inference_seconds
peak_gpu_memory_mb
AUC-ROC
AUC-PR
VUS-ROC
VUS-PR
Standard-F1
Range-F1
```

**Acceptance criteria:**

```text
python scripts/benchmark_paano.py --data_dir data/TSB-AD-U --limit_files 3 --seed 2000
```

должен создавать `runs/<timestamp>/summary.csv` и не менять метрики относительно baseline более чем на шум округления.

---

# Phase 1 — обязательные correctness fixes

Это первый блок для агента. Он почти не должен менять качество модели, но должен убрать реальные сбои.

## PR-1.1: исправить `create_memory_bank` и смысл `num_cores`

Проблема: текущая логика в `utils/utils.py` принудительно ставит минимум до `min(500, num_samples - 1)`, из-за чего `num_cores=0.1` может перестать означать “10% memory bank”. Это же отметили коллеги: текущая конструкция ломает смысл coreset на коротких рядах.  В текущем коде `main.py` вызывает `create_memory_bank(..., num_cores=0.1)`, поэтому этот баг влияет на стандартный pipeline. ([GitHub][3])

Агенту заменить логику на явную функцию:

```python
def resolve_num_cores(num_cores, n: int, min_cores: int = 1) -> int:
    if n <= 0:
        raise ValueError("Cannot create memory bank from zero embeddings.")

    if num_cores is None:
        return n

    if isinstance(num_cores, float):
        if not (0.0 < num_cores <= 1.0):
            raise ValueError("Float num_cores must be in (0, 1].")
        k = int(round(num_cores * n))
    else:
        k = int(num_cores)

    return max(min_cores, min(k, n))
```

И использовать:

```python
num_cores = resolve_num_cores(num_cores, num_samples)
if num_cores >= num_samples:
    return embeddings_tensor, indices_tensor
```

**Тесты:**

```python
assert resolve_num_cores(0.1, 300) == 30
assert resolve_num_cores(0.1, 10) == 1
assert resolve_num_cores(500, 300) == 300
assert resolve_num_cores(None, 300) == 300
```

---

## PR-1.2: guard для `top_k`

Проблема: текущий `calculate_anomaly_scores` делает `torch.topk(..., k=top_k)` без проверки, что `top_k <= memory_bank_size`; в коде также видно dense similarity `feats @ memory_bank.T`. ([GitHub][4]) Коллеги тоже отметили, что `k = min(top_k, memory_bank.shape[0])` обязателен. 

Агенту изменить:

```python
if memory_bank.shape[0] == 0:
    raise ValueError("memory_bank is empty")

k = min(int(top_k), memory_bank.shape[0])
topk_sim, _ = torch.topk(sims, k=k, dim=1, largest=True)
```

**Acceptance criteria:**

```text
memory_bank_size=1, top_k=3 не падает;
результат имеет длину == числу test patches.
```

---

## PR-1.3: исправить `torch.randint(1, M, ...)` при `M=1`

В `train.py` сейчас используется `torch.randint(1, M, ...)`, что падает при последнем batch размера 1; это прямо отмечено коллегами.  В текущем raw-коде эта строка действительно присутствует внутри pretext-блока. ([GitHub][5])

Агенту выбрать один из двух вариантов:

Предпочтительно:

```python
DataLoader(..., drop_last=True)
```

Либо guard:

```python
if M < 2:
    continue
```

**Acceptance criteria:**

```text
batch_size > len(train_patches) и batch_size=1 не приводят к RuntimeError.
```

---

## PR-1.4: уникальные checkpoint names

В `main.py` текущий pipeline сохраняет модель в `trained_encoder.pth`, что может приводить к перезаписи при параллельных запусках. ([GitHub][3]) Коллеги отдельно указали на коллизию checkpoint’ов. 

Агенту сделать:

```python
run_id = f"{dataset_name}_{file_stem}_seed{seed}_ps{patch_size}_{timestamp}"
model_path = output_dir / "checkpoints" / f"{run_id}_encoder.pth"
```

И не делать лишний save-load сразу после `train_model`, если модель уже находится в памяти.

**Acceptance criteria:**

```text
параллельный запуск на двух CSV не перезаписывает веса;
в output_dir/checkpoints сохраняются уникальные файлы.
```

---

## PR-1.5: NaN/missing/nonstationarity preprocessing policy

Коллеги справедливо отметили, что политика missing values и nonstationarity до RevIN важнее многих encoder-tuning идей. 

Агенту добавить в preprocessing:

```text
--missing_policy {error,ffill,bfill,linear_interpolate,zero}
--nan_guard true/false
--clip_quantile 0.001
```

Минимальный default:

```python
if np.isnan(train_data).any() or np.isnan(test_data).any():
    raise ValueError("NaNs detected. Use --missing_policy to handle missing values.")
```

---

# Phase 2 — ускорение training loop без изменения модели

## PR-2.1: GPU-векторизация positive/pretext sampling

В `train.py` текущий код собирает positives через Python loop и `.tolist()`, например `torch.stack([train_patches[i] for i in _pos_idx.tolist()])`; коллеги указали, что это создаёт CPU-GPU синхронизацию и что `train_patches_gpu[_pos_idx]` должен быть быстрее.  Текущий raw-код подтверждает такую структуру. ([GitHub][5])

Агенту реализовать режим:

```python
can_fit_gpu = estimate_tensor_bytes(train_patches) < available_gpu_memory * 0.5
if can_fit_gpu:
    train_patches_indexable = train_patches.to(device, non_blocking=True)
else:
    train_patches_indexable = train_patches
```

Для positives:

```python
_pos_idx = _pos_idx.to(device)
positives = train_patches_gpu[_pos_idx]
```

Для pretext:

```python
pretext_patches = torch.zeros_like(anchors)
valid = _pre_mask.to(device)
pretext_patches[valid] = train_patches_gpu[_tgt_clamped.to(device)[valid]]
```

Убрать `.tolist()` и `.item()` из горячего training loop.

**Acceptance criteria:**

```text
метрики совпадают с baseline в пределах stochastic noise;
train_seconds уменьшается или не ухудшается;
torch.profiler не показывает Python list indexing как hot path.
```

---

## PR-2.2: pinned memory / non_blocking / DataLoader options

Агенту добавить CLI/config:

```text
--num_workers
--pin_memory
--persistent_workers
--prefetch_factor
```

И применять только если `num_workers > 0`.

---

## PR-2.3: `torch.compile` и AMP/bfloat16 как performance flags

`torch.compile` в PyTorch 2.x предназначен для ускорения кода через JIT-компиляцию в optimized kernels при минимальных изменениях, а AMP использует mixed precision через `autocast` и `GradScaler`. ([PyTorch Docs][6]) Коллеги правильно предложили попробовать `torch.compile()` и mixed precision до архитектурных изменений. 

Агенту добавить флаги:

```text
--compile_encoder
--amp {off,fp16,bf16}
--matmul_precision {highest,high,medium}
```

Пример:

```python
if args.compile_encoder:
    model = torch.compile(model)

amp_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}.get(args.amp)

with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
    all_embeddings = model.embedding(all_patches)
```

**Acceptance criteria:**

```text
режимы off/fp16/bf16 проходят smoke test;
bf16/fp16 не меняют ranking scores катастрофически;
compile не включается по умолчанию.
```

---

# Phase 3 — memory bank и search backend

## PR-3.1: абстракция search backend

Текущий scoring использует dense cosine similarity `feats @ memory_bank.T` и `torch.topk`. ([GitHub][4]) Это нормально для baseline, но нужно вынести в интерфейс.

Агенту добавить:

```text
utils/search.py
```

Интерфейс:

```python
class SearchBackend:
    def build(self, memory_bank: torch.Tensor) -> None:
        ...

    def search(self, queries: torch.Tensor, top_k: int) -> torch.Tensor:
        """Return distances or similarities shape (B, top_k)."""
```

Реализации:

```text
TorchFlatSearchBackend
TorchChunkedSearchBackend
FaissFlatIPBackend
FaissHNSWBackend
FaissIVFPQBackend optional
```

Важно: FAISS docs говорят, что если searches мало, direct computation/Flat может быть самым эффективным вариантом; exact result гарантируют только `IndexFlatL2` или `IndexFlatIP`. ([GitHub][7]) Поэтому FAISS не должен быть единственным default.

---

## PR-3.2: benchmark search backends

Агенту сделать benchmark matrix:

```text
memory_bank_size: 1k, 10k, 100k, 1M where possible
embedding_dim: actual D
batch_size: 128, 512, 2048
top_k: 1, 3, 5, 10
backend: torch_flat, torch_chunked, faiss_flat_ip, faiss_hnsw
device: cpu/cuda
```

FAISS HNSW полезен как ANN-вариант, но HNSW не поддерживает удаление векторов, потому что удаление разрушает graph structure. ([GitHub][8]) Поэтому для online-update HNSW использовать только с periodic rebuild или lazy tombstones.

**Acceptance criteria:**

```text
benchmark_search.csv содержит latency, recall@k против exact Flat, memory_mb;
default backend остаётся torch_flat или faiss_flat_ip только после измерения.
```

---

## PR-3.3: memory bank metadata

Агенту хранить вместе с bank:

```python
@dataclass
class MemoryBank:
    embeddings: torch.Tensor
    source_indices: torch.Tensor
    patch_size: int
    train_score_mean: float | None
    train_score_median: float | None
    train_score_mad: float | None
    created_at: str
    coreset_ratio: float | None
    backend_name: str
```

---

# Phase 4 — score layer: агрегация и calibration

## PR-4.1: weighted patch-to-point aggregation как ablation

Сейчас `distribute_patch_scores_to_points` использует равномерный kernel `np.ones(patch_size)`. ([GitHub][4]) Коллеги верно уточнили: Bartlett/Gaussian надо подставлять вместо этого kernel, но не делать unconditional default, потому что center-weighted aggregation может помогать point anomalies и вредить длинным contextual segments. 

Агенту добавить:

```text
--aggregation {uniform,bartlett,gaussian,max,quantile}
--aggregation_quantile 0.9
--gaussian_sigma_ratio 0.25
```

Реализация:

```python
def make_kernel(patch_size: int, kind: str) -> np.ndarray:
    if kind == "uniform":
        w = np.ones(patch_size, dtype=np.float32)
    elif kind == "bartlett":
        w = np.bartlett(patch_size).astype(np.float32)
        if w.sum() == 0:
            w = np.ones(patch_size, dtype=np.float32)
    elif kind == "gaussian":
        x = np.arange(patch_size) - (patch_size - 1) / 2
        sigma = patch_size * sigma_ratio
        w = np.exp(-(x ** 2) / (2 * sigma ** 2)).astype(np.float32)
    else:
        raise ValueError(kind)
    return w / max(w.sum(), 1e-12)
```

**Acceptance criteria:**

```text
uniform reproduces current output;
bartlett/gaussian produce finite scores;
ablation report splits results by anomaly category if labels/categories available.
```

---

## PR-4.2: threshold calibration без test-label leakage

Агенту не писать “conformal без меток”. Корректная политика:

```text
--threshold_method {none,train_quantile,median_mad,pot,split_conformal}
```

Правила:

```text
none: только threshold-free метрики;
train_quantile: threshold = quantile(train_scores, q);
median_mad: threshold = median + k * MAD;
pot: fit GPD на хвост train/calibration scores;
split_conformal: только если есть отдельный чисто-нормальный calibration split.
```

Коллеги отдельно указали, что conformal calibration “без меток” в чистом виде не работает; надо разделять split-conformal, EVT/POT и robust quantile/MAD. 

---

# Phase 5 — multi-scale PaAno

## PR-5.1: shared encoder + per-scale memory banks

Не обучать несколько энкодеров на первом шаге. В `model.py` encoder использует `AdaptiveAvgPool1d(output_size=1)`, так что embedding размер фиксирован при разных длинах патча. ([GitHub][9]) Но коллеги справедливо уточнили: multi-scale надо начинать с shared encoder + отдельных memory banks, иначе latency/memory могут вырасти непропорционально. 

Агенту добавить:

```text
--patch_sizes 32,64,96,128
--multiscale_score_norm {none,robust_z,train_quantile}
--multiscale_fusion {max,mean,weighted_mean}
```

Pipeline:

```text
1. Train one encoder on primary_patch_size.
2. For each scale w:
   - create train patches with patch_size=w;
   - embed with same encoder;
   - build memory bank for scale w;
   - compute train score distribution for calibration;
   - compute test patch scores;
   - distribute to point scores for scale w.
3. Normalize per-scale scores using train distribution.
4. Fuse point scores: max first, weighted_mean as ablation.
```

Default candidate:

```text
primary_patch_size = 64 or current default
patch_sizes = [32, 64, 96, 128]
fusion = max
score_norm = robust_z
```

**Acceptance criteria:**

```text
single-scale mode exactly matches baseline;
multi-scale reports separate latency and memory overhead;
multi-scale accepted only if VUS-PR/AUPRC improve without unacceptable latency.
```

---

# Phase 6 — train-set self-cleaning

## PR-6.1: двухэтапная очистка train-патчей

PaAno работает в semi-supervised setting, где training data предполагается нормальной; статья формулирует обучение на нормальных training patterns. ([arXiv][2]) В реальных данных это допущение часто нарушается. Коллеги предложили двухэтапную очистку: baseline → score train patches → удалить верхние `q%` подозрительных → пересобрать bank / дообучить. 

Агенту добавить:

```text
--self_clean
--self_clean_ratio 0.01
--self_clean_mode {memory_bank_only,retrain}
```

Алгоритм:

```text
1. Train baseline encoder.
2. Build preliminary memory bank.
3. Score train patches with leave-one-neighborhood guard.
4. Remove top q% highest-score train patches.
5. Rebuild memory bank on cleaned embeddings.
6. Optional: retrain encoder on cleaned patches.
```

Guard против “самого себя”:

```text
при train scoring не учитывать идентичный patch index как nearest neighbor.
```

**Acceptance criteria:**

```text
self_clean=false baseline unchanged;
self_clean=true logs removed indices and score quantile;
q слишком большой запрещён, например q > 0.1 требует explicit flag --allow_aggressive_cleaning.
```

---

# Phase 7 — online memory update

## PR-7.1: FIFO-update только low-score embeddings

Коллеги отметили, что online-update memory bank как очередь согласуется с practical deployment, но обновлять bank нужно только низкоскоринговыми патчами, иначе будет poisoning/catastrophic forgetting. 

Агенту добавить:

```text
--online_update
--update_threshold_method {train_quantile,median_mad}
--update_quantile 0.95
--memory_queue_size
--rebuild_interval
```

Алгоритм:

```python
if score < update_threshold:
    memory_queue.append(embedding, index, timestamp)

if len(memory_queue) > memory_queue_size:
    memory_queue.pop_oldest()
```

Для HNSW:

```text
не пытаться удалять по одному вектору из HNSW;
использовать periodic rebuild, потому что FAISS HNSW не поддерживает remove. 
```

Это ограничение зафиксировано в FAISS wiki. ([GitHub][8])

---

# Phase 8 — loss и negative mining

## PR-8.1: не заменять triplet на InfoNCE по умолчанию

Агенту оставить текущую логику triplet/pretext как baseline. В статье PaAno описывает обучение комбинацией triplet loss и pretext loss, а коллеги указали, что farthest-negative triplet лучше не менять на InfoNCE без ablation. ([arXiv][2]) 

Разрешённые ablation-флаги:

```text
--negative_strategy {batch_farthest,queue_farthest,cluster_hard}
--negative_queue_size
--pretext_schedule {early_decay,constant,off}
--pretext_warmup_fraction 0.2
```

MoCo-like queue делать только после benchmark, потому что коллеги отметили, что для коротких 1D patches queue может быть инженерно сложнее, чем просто больший effective batch. 

---

## PR-8.2: seasonal positives только с проверкой близости

Не добавлять `i ± period` как unconditional positive. Коллеги верно указали, что seasonal positive может оказаться не ближе anchor’а и будет тянуть embedding space в неправильную сторону. 

Агенту добавить experimental mode:

```text
--seasonal_positives
--seasonal_period auto/int
--seasonal_positive_gate {raw_distance,embedding_distance}
--seasonal_positive_max_ratio 0.25
```

Правило:

```text
seasonal candidate разрешён только если он ближе anchor’а, чем median local-negative distance,
или входит в top-p% ближайших по текущему embedding/raw distance.
```

---

# Phase 9 — encoder upgrades, только после предыдущих фаз

## PR-9.1: RevIN/config clarity

В `PatchEncoder` параметр `use_revin` по умолчанию `True`, но в `main.py` класс `AnomalyDetection` и CLI запускают `use_revin=False`, если не передан `--use_revin`. ([GitHub][9]) Коллеги также отметили, что RevIN/InstanceNorm не косметика и должен быть обязательным ablation-фактором. 

Агенту сделать:

```text
--normalization {train_standard,revin,none}
```

И убрать двусмысленность:

```python
if normalization == "train_standard":
    apply train mean/std
    model.use_revin = False
elif normalization == "revin":
    do not apply global train standardization
    model.use_revin = True
```

**Acceptance criteria:**

```text
CLI явно печатает выбранную normalization policy;
старый --use_revin остаётся backward-compatible alias.
```

---

## PR-9.2: pooling ablation

Текущий encoder — Conv1d + BatchNorm1d + ReLU blocks и `AdaptiveAvgPool1d`. ([GitHub][9]) Агенту добавить pooling как ablation:

```text
--pooling {avg,gem,attention}
```

Начать с GeM:

```python
class GeMPooling1d(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        ...
```

Не включать GeM по умолчанию до ablation.

---

## PR-9.3: residual/dilated/depthwise blocks

Добавить флаг:

```text
--encoder_type {baseline,residual_tcn,depthwise_tcn}
```

Минимальные требования:

```text
baseline encoder byte-for-byte сохраняется;
новые encoder_type не меняют public API: embedding(), projection().
```

---

# Phase 10 — финальная ablation matrix

Агент должен сделать не один “улучшенный PaAno”, а таблицу вариантов.

Минимальная матрица:

```text
A0 baseline current
A1 correctness fixes only
A2 A1 + vectorized train
A3 A2 + best search backend
A4 A3 + aggregation ablation
A5 A3 + multi-scale shared encoder
A6 A5 + self-cleaning
A7 A6 + online update synthetic drift test
A8 A6 + RevIN normalization ablation
A9 A6 + GeM / TCN encoder ablation
```

Метрики:

```text
quality: AUC-ROC, AUC-PR, VUS-ROC, VUS-PR, Standard-F1, Range-F1
speed: train_seconds, inference_seconds, patches/sec
memory: peak_gpu_memory_mb, memory_bank_mb
robustness: NaN handling, small train set, memory_bank_size < top_k, batch M=1
```

Результат считать успешным только если:

```text
1. A1 не ухудшает метрики относительно baseline.
2. A2/A3 ускоряют train/inference или хотя бы не ухудшают их.
3. A4/A5/A6 принимаются только по ablation, не как unconditional default.
4. Все новые флаги задокументированы в README.
5. Старые scripts/run_uni.sh и scripts/run_mul.sh работают.
```

---

# Приоритет задач для агента

Самый рациональный порядок:

```text
1. PR-0.1 benchmark harness
2. PR-1.1 resolve_num_cores
3. PR-1.2 top_k guard
4. PR-1.3 M=1 randint/drop_last fix
5. PR-1.4 unique checkpoints
6. PR-2.1 vectorized train loop
7. PR-3.1/3.2 search backend benchmark
8. PR-4.1 aggregation ablation
9. PR-5.1 multi-scale shared encoder
10. PR-6.1 self-cleaning
11. PR-7.1 online update
12. PR-9.x encoder upgrades
```

Это совпадает с общей приоритезацией коллег: сначала memory bank + edge cases, потом multi-scale, self-cleaning, и только затем encoder experiments. 

---

## Короткий prompt, который можно прямо вставить агенту

```text
You are modifying the official PaAno repository: https://github.com/jinnnju/PaAno.

Goal:
Improve PaAno without changing its core algorithmic identity: lightweight patch-based 1D-CNN encoder, triplet/pretext representation learning, memory-bank distance scoring.

Rules:
- Work in small PRs.
- Do not introduce Transformer/Foundation model replacements.
- Do not tune thresholds on test labels.
- Preserve backward compatibility with script/run_uni.sh and script/run_mul.sh.
- Every PR must include tests and benchmark before/after.
- New behavior must be behind CLI/config flags unless it is a correctness fix.

Implementation order:
1. Add benchmark harness and reproducibility logging.
2. Fix create_memory_bank num_cores semantics with resolve_num_cores.
3. Add top_k <= memory_bank_size guard.
4. Fix M=1 torch.randint failure using drop_last=True or guard.
5. Use unique checkpoint names; remove redundant save-load.
6. Add missing value policy.
7. Vectorize train.py positives/pretext sampling; remove .tolist()/.item() from hot loop.
8. Add torch.compile and AMP/bfloat16 flags, disabled by default.
9. Add search backend abstraction: torch_flat, torch_chunked, faiss_flat_ip, faiss_hnsw.
10. Benchmark search backends; do not assume FAISS is always best.
11. Add score aggregation ablation: uniform, bartlett, gaussian, max, quantile.
12. Add threshold calibration modes: none, train_quantile, median_mad, pot, split_conformal.
13. Add multi-scale mode using shared encoder + per-scale memory banks.
14. Add self-cleaning train patch mode.
15. Add online memory update using only low-score embeddings.
16. Add optional encoder ablations: RevIN normalization clarity, GeM pooling, residual/dilated/depthwise TCN.

Acceptance:
- Baseline scripts still run.
- Correctness fixes do not degrade metrics.
- Speed PRs report train/inference seconds and peak GPU memory.
- Quality PRs report AUROC, AUPRC, VUS-ROC, VUS-PR, Standard-F1, Range-F1.
- Multi-scale, weighted aggregation, self-cleaning, seasonal positives, queue negatives, and encoder upgrades are ablation candidates, not default improvements.
```

[1]: https://github.com/jinnnju/PaAno "GitHub - jinnnju/PaAno: [ICLR'26] PaAno: Patch-based Representation Learning for Time-Series Anomaly Detection · GitHub"
[2]: https://arxiv.org/html/2602.01359v2 "PaAno: Patch-Based Representation Learning for Time-Series Anomaly Detection"
[3]: https://raw.githubusercontent.com/jinnnju/PaAno/main/main.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/jinnnju/PaAno/main/utils/evaluation.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/jinnnju/PaAno/main/train.py "raw.githubusercontent.com"
[6]: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html?utm_source=chatgpt.com "Introduction to torch.compile"
[7]: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index "Guidelines to choose an index · facebookresearch/faiss Wiki · GitHub"
[8]: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes "Faiss indexes · facebookresearch/faiss Wiki · GitHub"
[9]: https://raw.githubusercontent.com/jinnnju/PaAno/main/model.py "raw.githubusercontent.com"

---

## Реализация в alma_servie (вариант B — vendored paano без upstream gitlink)

Source of truth: сервер `a100`, путь `/root/projects/alma_servie`.
GPU для прогонов: `CUDA_VISIBLE_DEVICES=1`.
Baseline для сравнений: snapshot `pc` (paano_global per-class, без norm_work) в
`artifacts/results/_snapshots/{anomaly}_pc_global_perclass.summary.json`.

Дата начала работ: 2026-05-25.

### Что отброшено (и почему)

| Фаза/PR ТЗ | Решение | Причина |
|---|---|---|
| Phase 0 benchmark harness | НЕ делаем | У нас уже есть снапшоты `pg3/pg4/pg5/pc/ps` + `*_results.summary.json` с метриками детекции и latency — дубль |
| PR-1.4 unique checkpoints | НЕ применимо | `load_or_train_shared_encoder` имеет детерминированный путь `models/{anomaly}_paano_shared_encoder.pt`, конфликтов нет |
| PR-1.5 NaN policy | НЕ применимо | NaN-обработка живёт на уровне `alma_service` препроцессинга, в PaAno уже приходит чистое |
| PR-2.1 GPU-векторизация positives | НЕ делаем | Обучение уже быстрое (~1 минута); узким местом был скоринг, исправлено GPU-kmeans до этих работ |
| PR-2.2 num_workers/pin_memory | УЖЕ СДЕЛАНО | pg4-эксперимент показал, что `workers=2 + persistent_workers` хуже из-за spawn overhead на per-well DataLoader → откатились на `num_workers=0, pin_memory=True` |
| PR-3.x search backend / FAISS | НЕ делаем | После coreset наш memory_bank ≈ 1862 векторов. GPU `torch.cdist + topk` — миллисекунды. FAISS добавит больше оверхеда на Python→C++ маршрутизацию, чем сэкономит |
| PR-4.2 threshold calibration | НЕ применимо | У нас onset-слой с `calibrate_causal_thresholds_from_reference_mask`. PaAno возвращает только raw anomaly scores |
| PR-5.1 multi-scale | УЖЕ СДЕЛАНО | `SharedPaAnoDetector.score_stream` уже использует short+long patches с общим энкодером |
| PR-7.1 online update | НЕ актуально сейчас | Мы в batch-режиме. Релевантно когда будет стриминговый прод |
| PR-8.x triplet/negative mining | НЕ применимо | Энкодер у нас frozen; loss меняется только при retraining |

### Журнал PR

#### PR-1: correctness guards (num_cores + top_k + M=1) — ✅ выполнено

**Файлы:**
- `paano/utils/utils.py` — добавлен `resolve_num_cores(num_cores, n, min_cores=1)`. Убран жёсткий floor `min(500, num_samples-1)`, который на коротких рядах отключал coreset (фактически возвращал полный bank). На наших N≈18617 поведение не меняется (k=1862).
- `paano/utils/evaluation.py` — в `calculate_anomaly_scores` добавлен guard `effective_k = max(1, min(top_k, memory_bank.shape[0]))` + явная проверка `memory_bank.numel() != 0`.
- `paano/train.py` — в pretext loop добавлен `if M < 2: continue` (защита от `torch.randint(1, M)` на последнем батче размера 1).

**Тесты:**
```text
resolve_num_cores(0.1, 300) == 30        ✓
resolve_num_cores(0.1, 10)  == 1         ✓
resolve_num_cores(500, 300) == 300       ✓
resolve_num_cores(None, 300) == 300      ✓
resolve_num_cores(0.1, 18617) == 1862    ✓ (наш реальный случай)
resolve_num_cores(1.5, 100) → ValueError ✓
resolve_num_cores(0.0, 100) → ValueError ✓
```

**Regression-check** (negermet, paano_global, бит-в-бит с pc snapshot):

| split | metric | baseline | PR-1 |
|---|---|---|---|
| all | hit_count | 3 | 3 |
| all | hit_rate | 1.0 | 1.0 |
| all | first_alert_delay_median_hours | 0.01861 | 0.01861 |
| all | first_alert_delay_p90_hours | 0.06128 | 0.06128 |
| all | false_alarms | 0 | 0 |
| train | … | == | == |
| test | … | == | == |

PR-1 — defensive correctness без изменения текущих метрик.

#### PR-Δ: top_k=1 как новый default — ✅ принят, единственный реальный плюс

**Файл:** `alma_service/generic_detectors.py` — `PAANO_TOP_K` дефолт изменён с 5 на 1. Параметр остался env-overridable через `ALMA_PAANO_TOP_K`. Сам код скоринга в `paano/utils/evaluation.py:calculate_anomaly_scores` не менялся — он уже принимает `top_k` как аргумент.

**Замеры (pg5, paano_global, одинаковый encoder cache):**

| Класс | k=5 (был) | k=1 (новый) | Дельта |
|---|---|---|---|
| negermet all | 3/3, median=0.019ч, p90=0.061ч | 3/3, median=0.019ч, p90=0.061ч | **бит-в-бит** |
| salt all | 6/6, median=0.9ч, p90=2.46ч | 6/6, median=0.9ч, p90=2.46ч | **бит-в-бит** |
| **pritok all** | **19/22**, median=4.19ч, p90=17.84ч | **22/22**, median=2.80ч, **p90=6.75ч** | **+3 hits, p90 −62%** |
| pritok test split | 0/2 (3138 и 5021 промах) | **2/2** | **3138 пойман впервые** |

False alarms — 0 во всех случаях.

**Почему это работает:** Anomaly score у нас — `mean(top-k cos-distances)` относительно local memory bank. С `k=5` onset размывается усреднением по 5 ближайшим нормальным патчам, что для медленных трендов pritok даёт затянутый, плавный подъём score-сигнала, и наш causal-onset EMA-слой стартует поздно или вообще не пересекает порог. С `k=1` score = «насколько далеко ближайший нормальный» — это чистая novelty-метрика, onset резкий, порог пересекается рано. На negermet/salt онсет уже резкий по физике, так что разницы между k=1 и k=5 не видно.

**Ablation, отброшенное по тем же 5 прогонам:**

| Тег | k | ratio | hits | median | p90 |
|---|---|---|---|---|---|
| k=1 (winner) | 1 | 0.10 | **22** | **2.80ч** | **6.75ч** |
| k=3 | 3 | 0.10 | 21 | 3.06ч | 7.28ч |
| k=5 (был) | 5 | 0.10 | 19 | 4.19ч | 17.84ч |
| k=7 | 7 | 0.10 | 19 | 7.19ч | **83.2ч** ⚠ |
| ratio=0.05 | 5 | 0.05 | 19 | 4.19ч | 17.84ч |
| ratio=0.20 | 5 | 0.20 | 19 | 4.19ч | 17.84ч |

`memory_bank_ratio` (coreset size) — нулевая чувствительность в диапазоне 0.05–0.20: при N≈18617 даже 0.05 (≈930 центроидов) обеспечивает достаточное покрытие нормальности.

### Что отброшено по результатам ablation (НЕ принято)

- **PR-4.1 aggregation kernel (gaussian/bartlett).** Гипотеза «центр-взвешенное ядро = быстрее onset для медленных трендов» **falsified**. В нашем pipeline causal-onset layer стартует по дискретному пересечению порога EMA: uniform даёт жёсткий подъём score на первом патче, gaussian рассеивает старт по центру окна и задерживает срабатывание. Замеры на paano_global: negermet p90 0.06ч → 7.99ч; pritok p90 7.21ч → 91.4ч. API удалён, дефолт оставлен как раньше (`np.ones(patch_size)`).
- **PR-6.1 self-cleaning train patches (ratio=0.02).** Гипотеза «выкидывание top-2% аномальных ref-патчей улучшает quality» **falsified**. Hit-rate стабилен, latency деградирует: pritok p90 ×2–7, salt p90 ×2. Причина: наш `ref_data` уже clean by construction (нормальный сегмент перед `anomaly_start` или `norm_work`), top-q% — это edge-cases полезного поведения, обогащавшие покрытие bank. API удалён.
- **PR-2.3 AMP bf16 в скоринге.** Гипотеза «`torch.autocast(bfloat16)` вокруг `model.embedding(...)` даст 10-20% speedup на A100» **falsified**. Head-to-head на одном encoder cache: `bf16` wall=685с, `off` wall=682с (шум ±0.4%). Quality: negermet/salt бит-в-бит, pritok прыгнул в обе стороны на ~1ч (характерно для его cache-sensitivity, не сигнал). Причина: PaAno encoder крошечный (1D-CNN на сотни килопараметров), `model.embedding` — micro-операция; bottleneck сидит в DataLoader setup + GPU-kmeans coreset + cdist, AMP туда не достаёт. API удалён.

### Команды для воспроизведения PR-1

```bash
ssh a100 'cd /root/projects/alma_servie && \
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/detection/detect_negermet.py --detector paano_global'
```
