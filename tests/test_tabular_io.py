from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from alma_service.tabular_io import read_table, write_dataset_tables


class TabularIoTests(unittest.TestCase):
    def test_roundtrip_parquet_and_csv(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:02:00"]),
                "well_id": ["a1", "a1"],
                "value": [1.25, 2.5],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            parquet_path = base / "sample.parquet"
            csv_path = base / "sample.csv"
            write_dataset_tables(df, parquet_path=parquet_path, csv_path=csv_path)

            parquet_df = read_table(parquet_path, dtypes={"well_id": str}, parse_dates=["timestamp"])
            csv_df = read_table(csv_path, dtypes={"well_id": str}, parse_dates=["timestamp"])

            self.assertEqual(list(parquet_df["well_id"]), ["a1", "a1"])
            self.assertEqual(list(csv_df["well_id"]), ["a1", "a1"])
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(parquet_df["timestamp"]))
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(csv_df["timestamp"]))
            self.assertEqual(list(parquet_df["value"]), [1.25, 2.5])
            self.assertEqual(list(csv_df["value"]), [1.25, 2.5])


if __name__ == "__main__":
    unittest.main()
