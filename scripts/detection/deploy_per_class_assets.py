from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTRACT_PATH = PROJECT_ROOT / "configs" / "alma_global_deploy_contract.json"
ENCODER_STEM = "global_normality_paano_shared_encoder"
BANK_STEM = "population_memory_bank_paano_global_global_normality"
MODELS_DIR = PROJECT_ROOT / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bank_shape(path: Path) -> list[int]:
    import numpy as np

    bank = np.load(path, allow_pickle=True)
    return [int(dim) for dim in np.asarray(bank["features"]).shape]


def _pair_paths(name: str) -> tuple[Path, Path]:
    if name == "default":
        return MODELS_DIR / f"{ENCODER_STEM}.pt", MODELS_DIR / f"{BANK_STEM}.npz"
    return MODELS_DIR / f"{ENCODER_STEM}.{name}.pt", MODELS_DIR / f"{BANK_STEM}.{name}.npz"


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def cmd_status(_: argparse.Namespace) -> int:
    contract = _load_contract()
    print(f"Контракт: {CONTRACT_PATH} (снимок {contract.get('snapshot_at')})")
    for name in contract.get("pairs", {}):
        encoder, bank = _pair_paths(name)
        for label, path in (("энкодер", encoder), ("банк", bank)):
            mark = "OK" if path.exists() else "ОТСУТСТВУЕТ"
            extra = f" sha256={_sha256(path)[:12]}…" if path.exists() else ""
            print(f"  [{name}] {label}: {path.name} — {mark}{extra}")
    return 0


def cmd_verify(_: argparse.Namespace) -> int:
    contract = _load_contract()
    failures: list[str] = []
    for name, pair in contract.get("pairs", {}).items():
        for key, sha_key in (("encoder", "encoder_sha256"), ("bank", "bank_sha256")):
            path = PROJECT_ROOT / pair[key]
            if not path.exists():
                failures.append(f"[{name}] {path} отсутствует")
                continue
            actual = _sha256(path)
            if actual != pair[sha_key]:
                failures.append(f"[{name}] {path.name}: sha256 {actual[:12]}… != контракт {pair[sha_key][:12]}…")
    default_source = str(contract.get("default_pair_source", ""))
    if default_source and default_source in contract.get("pairs", {}):
        default_pair = contract["pairs"]["default"]
        source_pair = contract["pairs"][default_source]
        if default_pair["encoder_sha256"] != source_pair["encoder_sha256"]:
            failures.append(f"default-энкодер не совпадает с парой {default_source}")
        if default_pair["bank_sha256"] != source_pair["bank_sha256"]:
            failures.append(f"default-банк не совпадает с парой {default_source}")
    if failures:
        print("ПРОВЕРКА НЕ ПРОЙДЕНА:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("Контракт деплоя подтверждён: все пары на месте, sha256 совпадают.")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    contract = _load_contract()
    for name in list(contract.get("pairs", {})):
        encoder, bank = _pair_paths(name)
        if not encoder.exists() or not bank.exists():
            print(f"  [{name}] пропуск snapshot: файлы не найдены")
            continue
        contract["pairs"][name] = {
            "encoder": str(encoder.relative_to(PROJECT_ROOT)),
            "encoder_sha256": _sha256(encoder),
            "bank": str(bank.relative_to(PROJECT_ROOT)),
            "bank_sha256": _sha256(bank),
            "bank_shape": _bank_shape(bank),
        }
    contract["snapshot_at"] = datetime.now().strftime("%Y-%m-%d")
    CONTRACT_PATH.write_text(
        json.dumps(contract, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
    )
    print(f"Контракт обновлён по текущему состоянию models/: {CONTRACT_PATH}")
    return 0


def _install_one(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Источник не найден: {src}")
    if dst.exists():
        backup = dst.with_name(dst.name + datetime.now().strftime(".bak-%Y%m%d-%H%M%S"))
        shutil.copy2(dst, backup)
        print(f"  бэкап: {backup.name}")
    shutil.copy2(src, dst)
    print(f"  установлено: {src} -> {dst.name}")


def cmd_install(args: argparse.Namespace) -> int:
    encoder_dst, bank_dst = _pair_paths(str(args.anomaly))
    _install_one(Path(args.encoder), encoder_dst)
    _install_one(Path(args.bank), bank_dst)
    if args.set_default:
        default_encoder, default_bank = _pair_paths("default")
        _install_one(Path(args.encoder), default_encoder)
        _install_one(Path(args.bank), default_bank)
    print("Готово. Зафиксируйте новое состояние: deploy_per_class_assets.py snapshot, затем verify.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Боевые per-class пары paano_global: проверка и установка по контракту configs/alma_global_deploy_contract.json.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Показать состояние пар в models/.").set_defaults(func=cmd_status)
    sub.add_parser("verify", help="Сверить models/ с контрактом (sha256).").set_defaults(func=cmd_verify)
    sub.add_parser("snapshot", help="Перезаписать контракт по текущему состоянию models/.").set_defaults(func=cmd_snapshot)
    install = sub.add_parser("install", help="Установить пару (с бэкапом существующей).")
    install.add_argument("--anomaly", choices=["negermet", "pritok", "salt"], required=True)
    install.add_argument("--encoder", required=True, help="Путь к исходному .pt")
    install.add_argument("--bank", required=True, help="Путь к исходному .npz")
    install.add_argument("--set-default", action="store_true", help="Также сделать пару дефолтной.")
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
