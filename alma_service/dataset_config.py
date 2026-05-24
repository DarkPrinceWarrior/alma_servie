from __future__ import annotations

from functools import lru_cache

from alma_service.paths import DATA_DIR, MODEL_PARAMS_PATH
from alma_service.tabular_io import read_excel_sheet


PARAM_RENAME = {
    "Давление на выкиде ЭЦН": "Давление на приеме насоса кгс/см²",
}

WELLS_DIR_67D = DATA_DIR / "67_days"

# Скважины, у которых сырьё длиннее окна сводки — обрезать по data_start/data_end.
CLIP_TO_SUMMARY_BOUNDS_WELLS: frozenset[str] = frozenset({"408", "4039"})

NEGREMET_WELL_FILES = {
    "3509г": WELLS_DIR_67D / "69-3509г.xlsx",
    "524": WELLS_DIR_67D / "29-524.xlsx",
    "5271г": WELLS_DIR_67D / "335-5271г.xlsx",
}

PRITOK_WELL_FILES = {
    "1449": WELLS_DIR_67D / "132-1449.xlsx",
    "4203у": WELLS_DIR_67D / "255-4203у.xlsx",
    "1508": WELLS_DIR_67D / "133-1508.xlsx",
    "1442л": WELLS_DIR_67D / "133-1442л.xlsx",
    "5021": WELLS_DIR_67D / "127-5021.xlsx",
    "3138": WELLS_DIR_67D / "74-3138.xlsx",
    "1714": WELLS_DIR_67D / "42-1714.xlsx",
    "691": WELLS_DIR_67D / "39-691.xlsx",
    "1062": WELLS_DIR_67D / "200-1062.xlsx",
    "602": WELLS_DIR_67D / "35-602.xlsx",
    "792": WELLS_DIR_67D / "46-792.xlsx",
    "610": WELLS_DIR_67D / "34-610.xlsx",
    "1809": WELLS_DIR_67D / "168-1809.xlsx",
    "816": WELLS_DIR_67D / "45-816.xlsx",
    "5144г": WELLS_DIR_67D / "167б-5144г.xlsx",
    "790": WELLS_DIR_67D / "45-790.xlsx",
    "713": WELLS_DIR_67D / "42-713.xlsx",
    "1756": WELLS_DIR_67D / "50-1756.xlsx",
    "3027": WELLS_DIR_67D / "47-3027.xlsx",
    "129л": WELLS_DIR_67D / "2-129л.xlsx",
    "3261": WELLS_DIR_67D / "47-3261.xlsx",
    "1395": WELLS_DIR_67D / "126-1395.xlsx",
}

SALT_WELL_FILES = {
    "408": WELLS_DIR_67D / "20-408.xlsx",
    "4039": WELLS_DIR_67D / "56-4039.xlsx",
    "3269": WELLS_DIR_67D / "47-3269.xlsx",
    "149г": WELLS_DIR_67D / "1-149г.xlsx",
    "3244г": WELLS_DIR_67D / "72-3244г.xlsx",
    "3245": WELLS_DIR_67D / "47-3245.xlsx",
}

TEST_WELLS = {
    "negermet": {"524"},
    "pritok": {"3138", "5021"},
    "salt": {"3244г", "4039"},
}


@lru_cache(maxsize=1)
def load_model_parameters() -> frozenset[str]:
    df = read_excel_sheet(
        MODEL_PARAMS_PATH,
        sheet_id=1,
        has_header=False,
        infer_schema_length=50,
        raise_if_empty=False,
    )
    params = []
    if not df.empty:
        first_col = df.iloc[:, 0]
        for value in first_col.tolist():
            if value is None:
                continue
            value = str(value).strip()
            if value:
                params.append(value)
    return frozenset(params)


def normalize_param_name(name: str) -> str:
    return PARAM_RENAME.get(name, name)


def split_for_well(anomaly_name: str, well_id: str) -> str:
    normalized_well_id = str(well_id).strip().lower()
    return "test" if normalized_well_id in TEST_WELLS.get(anomaly_name, set()) else "train"
