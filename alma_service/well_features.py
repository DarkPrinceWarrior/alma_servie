from __future__ import annotations

import pandas as pd


def get_well_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in ("timestamp", "well_id") and not df[column].isna().all()
    ]
