import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Display basic info about a parquet file.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--head", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_parquet(args.path)
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))
    print(df.head(args.head))


if __name__ == "__main__":
    main()
