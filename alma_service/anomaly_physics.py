from __future__ import annotations

from dataclasses import dataclass

from alma_service.pressure_trend import PRESSURE_COL


@dataclass(frozen=True)
class PhysicalBranchSpec:
    groups: dict[str, tuple[str, ...]]
    description: str


PRITOK_PHYSICAL_SPEC = PhysicalBranchSpec(
    description="intake pressure trend / drift",
    groups={
        "pressure_trend": (PRESSURE_COL,),
    },
)

NEGERMET_PHYSICAL_SPEC = PhysicalBranchSpec(
    description="pressure step with electrical/load response",
    groups={
        "pressure_step": (PRESSURE_COL,),
        "load_response": (
            "Выходной ток ПЧ",
            "Ток на фазе А",
            "Ток на фазе В",
            "Ток на фазе С",
            "Активная выходная мощность",
            "Полная выходная мощность",
            "Коэффициент загрузки ПЭД",
        ),
        "regime_context": ("Выходная частота",),
        "thermal_vibration_support": (
            "Температура масла двигателя",
            "Температура на приёме насоса",
            "Вибрация Х",
            "Вибрация Y",
            "Вибрация Z",
            "Вибрация ХY",
            "Вибрация ХYZ",
            "Вибрация ХXYZ",
        ),
    },
)

SALT_PHYSICAL_SPEC = PhysicalBranchSpec(
    description="multichannel salt deposition drift / PCA-SPE residual",
    groups={
        "pressure_efficiency": (
            PRESSURE_COL,
            "soft::pressure_freq_ratio",
            "soft::pressure_freq_gap",
        ),
        "power_efficiency": (
            "Активная выходная мощность",
            "Полная выходная мощность",
            "soft::power_freq_ratio",
            "soft::power_freq_gap",
        ),
        "load_current": (
            "Выходной ток ПЧ",
            "Коэффициент загрузки ПЭД",
            "Ток на фазе А",
            "Ток на фазе В",
            "Ток на фазе С",
            "Дисбаланс токов",
            "soft::current_unbalance",
        ),
        "electrical_imbalance": (
            "Дисбаланс напряжений",
            "Дисбаланс токов",
            "soft::current_unbalance",
            "soft::voltage_unbalance",
        ),
        "thermal": (
            "Температура масла двигателя",
            "Температура на приёме насоса",
        ),
        "mechanical": (
            "Вибрация Х",
            "Вибрация Y",
            "Вибрация Z",
            "Вибрация ХY",
            "Вибрация ХYZ",
            "Вибрация ХXYZ",
            "soft::vibration_vector",
        ),
    },
)

PHYSICAL_BRANCH_SPECS = {
    "pritok": PRITOK_PHYSICAL_SPEC,
    "negermet": NEGERMET_PHYSICAL_SPEC,
    "salt": SALT_PHYSICAL_SPEC,
}


def physical_spec_for(anomaly_key: str) -> PhysicalBranchSpec | None:
    return PHYSICAL_BRANCH_SPECS.get(str(anomaly_key))
