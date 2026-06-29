# 🖥️ A100-сервер: диагностика межкарточной связи и план ускорения

> **Статус:** диагностика завершена (2026-06-11, хост `zeta`).
> **Главный вывод:** NVLink на этом сервере **физически отсутствует** — это особенность железа, не настройки. Ускорение берём **софтом**.
>
> ⛔ **Процедура с `nvidia-fabricmanager` (прежняя версия этого файла) — отменена.** Причина — ниже.

---

## 1. Что это за сервер (проверено)

| Параметр | Значение |
|---|---|
| GPU | **6× NVIDIA A100-SXM4-40GB** (40 ГБ каждая, 240 ГБ суммарно), CC 8.0, драйвер **580.95.05** |
| Хост | Proxmox VE, хост `zeta` (kernel `6.8.12-pve`, Debian 12) |
| Контейнер | **LXC CT 135** (`ruslan`, Debian 13) — не KVM-VM |
| CPU | 2× Intel Xeon Platinum **8338C** @ 2.6 ГГц, 128 потоков (контейнер видит **64**) |
| RAM | ~1.0 ТиБ, swap 0 |
| Межкарточная связь | **только PCIe Gen4 ×16** (NVLink нет), P2P между картами выключен |

### Карта GPU → NUMA → CPU

| GPU | Bus-Id | NUMA | CPU (хост) | Внутри острова |
|---|---|---|---|---|
| 0,1,2 | `4F/53/57` | node 0 | `0-31,64-95` | PXB |
| 3,4,5 | `CE/D5/D6` | node 1 | `32-63,96-127` | PXB; **4↔5 = PIX** (ближайшие) |
| 0↔3 и т.п. | — | cross | — | **SYS** (PCIe + UPI между сокетами) |

---

## 2. Вывод: NVLink отсутствует (и почему fabricmanager не поможет)

Проверка на хосте `zeta` дала три независимых подтверждения:

1. **`nvidia-smi nvlink -c`** — у карт **ноль линков** (печатаются только заголовки `GPU N: …`, без строк `Link 0/1/…`). Драйвер не видит ни одного NVLink.
2. **`dmesg | grep -iE 'nvlink|nvswitch|fabric'`** — **пусто**. Драйвер при инициализации ничего про NVLink/NVSwitch не сообщил → инициализировать нечего.
3. **`nvidia-smi topo -m`** — чистый PCIe (`PXB/PIX/SYS`), ни одного `NV#`; NVSwitch в `lspci` отсутствует.

**Интерпретация:** это SXM4-модули **без NVLink-разводки** в данном шасси (форм-фактор SXM, но связь только по PCIe — типично для SXM4 на PCIe-карьерах). NVLink включается **только железом** (HGX-плата с NVSwitch / NVLink-мосты), софтом — никак.

> `nvidia-fabricmanager` управляет **NVSwitch**, которого здесь нет → ставить его бессмысленно.
> `NCCL_P2P_DISABLE=1` для этого бокса — **правильная постоянная настройка**, а не временный костыль.

### Потолок шины
$$\text{PCIe 4.0 ×16} \approx 31.5\ \tfrac{\text{ГБ}}{\text{с}} \quad\ll\quad \underbrace{600\ \tfrac{\text{ГБ}}{\text{с}}}_{\text{NVLink 3 — недоступен}}$$

Практически — ещё ниже: при выключенном P2P коллективы NCCL идут **через host-RAM** (sysmem-staging) + латентность.

---

## 3. Что делать — план ускорения (софт)

### 🥇 Tier 1 — независимые задачи, 1 на GPU (главный выигрыш)
Для PaAno / YOLO / time-series модель влезает в одну 40 ГБ → межкарточный обмен не нужен:
$$T_{\text{indep}} = T/N,\qquad \text{обмен}=0 \;\Rightarrow\; \text{идеальный }\times 6$$

```bash
# 5 задач на GPU 1..5 (GPU0 оставляем под whisperx), с NUMA-пиннингом
declare -A NODE=( [1]=0 [2]=0 [3]=1 [4]=1 [5]=1 )
mkdir -p runs
for g in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$g \
  numactl --cpunodebind=${NODE[$g]} --membind=${NODE[$g]} \
    uv run python train.py --fold $g 2>&1 | tee runs/fold_$g.log &
done
wait
```

### 🥈 Tier 2 — per-GPU ускорения (всегда, ×1.5–3 независимо от числа карт)
```python
import torch
torch.set_float32_matmul_precision("high")            # TF32 на matmul
torch.backends.cudnn.benchmark = True                  # CV, фиксированные размеры
model = model.to(memory_format=torch.channels_last)    # CV
model = torch.compile(model)                            # PyTorch 2.11
opt = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
with torch.autocast("cuda", dtype=torch.bfloat16):     # A100, без GradScaler
    ...
```
- **DataLoader не должен голодать GPU:** `num_workers=8–12`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`; `.to(device, non_blocking=True)`.
- Убрать `.item()/.cpu()` в горячем цикле (CPU-синхронизация убивает throughput).
- Память впритык → `gradient_checkpointing` + больше batch.

### 🥉 Tier 3 — одна модель на нескольких картах → DDP **внутри острова**
Только если модель реально нужна на нескольких GPU (а не влезает в одну):
```bash
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  numactl --cpunodebind=0 --membind=0 \
  uv run torchrun --nproc_per_node=3 train_ddp.py --grad-accum 4
```
- Держать DDP **в одном острове** (`{0,1,2}` или `{3,4,5}`) — без перехода через UPI.
- `gradient_as_bucket_view=True`, bf16-градиенты, **grad-accumulation** (реже all-reduce).
- **TP=4+ / cross-остров — НЕ делать**: упрётся в PCIe+UPI.

---

## 4. Хостовые/инфраструктурные действия (на `zeta`)

1. **Освободить GPU0** — на нём висят 2 процесса `whisperx` (≈8.4 ГБ). Остановить сервис **или** просто работать на `CUDA_VISIBLE_DEVICES=1..5`.
2. **Дать контейнеру больше ядер** (CT 135 видит 64 из 128):
   ```bash
   pct set 135 -cores 96      # если соседние контейнеры позволяют
   ```
3. **Persistence mode** — на хосте `nvidia-smi -pm 1` (на `zeta` уже `On`).
4. **ECC** сейчас `Enabled`. Для research можно выключить и вернуть ~1–2 ГБ/карту: на хосте `nvidia-smi -e 0` + reboot (компромисс надёжность↔память).

---

## 5. ⛔ Чего НЕ делать
- **Не ставить** `nvidia-fabricmanager` — NVSwitch отсутствует.
- **Не убирать** `NCCL_P2P_DISABLE=1` — P2P по PCIe тут недоступен.
- **Не делать** tensor-parallel на 4+ карт / между сокетами — бутылочное горло на UPI.

---

## Приложение A. Доказательства (вывод с хоста `zeta`)

```text
$ nvidia-smi nvlink -c
GPU 0..5: NVIDIA A100-SXM4-40GB (UUID …)        # ← ни одной строки "Link N" = линков нет

$ nvidia-smi nvlink -s
GPU 0..5: NVML: Unable to retrieve NVLink information as all links are inActive

$ nvidia-smi topo -m
GPU0..2  -> PXB между собой, SYS до GPU3..5, NUMA0
GPU3..5  -> PXB/PIX между собой, SYS до GPU0..2, NUMA1
(ни одного NV#)

$ dmesg | grep -iE 'nvlink|nvswitch|fabric'      # ← пусто
$ lspci | grep -i nvswitch                        # ← пусто
```

## Приложение B. Как перепроверить (если поменяется железо)
```bash
nvidia-smi nvlink -c                  # появятся строки Link N → NVLink есть
nvidia-smi topo -m                    # NV# в матрице → NVLink активен
lspci -nn | grep -i 10de              # есть Bridge/Switch помимо 6 GPU → NVSwitch есть
dmesg | grep -iE 'nvlink|nvswitch|fabric'
```
Если хоть одно из этого станет положительным после смены платы/мостов — тогда (и только тогда) актуальна процедура с `nvidia-fabricmanager`.
