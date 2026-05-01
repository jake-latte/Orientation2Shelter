#!/usr/bin/env python3
"""
Split every `data.csv` under a database tree into `train.csv` and `test.csv`.

Expected structure:
training_data/
├── 0/
│   ├── 0/
│   │   ├── data.csv
│   │   └── meta.json
│   ├── 1/
│   │   ├── data.csv
│   │   └── meta.json
│   └── ...
├── 1/
│   └── ...
└── ...

For each session folder containing `data.csv`, this script:
- randomly assigns a proportion of rows to the test split
- assigns the remainder to the train split
- writes `train.csv` and `test.csv` into the same folder

Usage:
    python split_dataset.py /path/to/training_data --test-proportion 0.2
    python split_dataset.py /path/to/training_data --test-proportion 0.2 --seed 123
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl


def is_int_named_dir(path: Path) -> bool:
    return path.is_dir() and path.name.isdigit()


def split_one_csv(
    csv_path: Path,
    test_proportion: float,
    seed: int | None = None,
    overwrite: bool = False,
) -> None:
    session_dir = csv_path.parent
    train_path = session_dir / "train.csv"
    test_path = session_dir / "test.csv"

    if not overwrite and (train_path.exists() or test_path.exists()):
        print(f"Skipping {csv_path} (train.csv or test.csv already exists)")
        return

    df = pl.read_csv(csv_path)

    n_rows = df.height
    if n_rows == 0:
        print(f"Skipping {csv_path} (empty CSV)")
        return

    # Add a random column, sort by it, then split.
    # This gives an unbiased random partition without requiring pandas.
    shuffled = (
        df.with_row_index("row_idx")
          .with_columns(
              pl.int_range(0, n_rows).shuffle(seed=seed).alias("_perm")
          )
          .sort("_perm")
          .drop("_perm")
    )

    n_test = int(round(test_proportion * n_rows))
    n_test = max(0, min(n_rows, n_test))

    test_df = shuffled.head(n_test).drop("row_idx")
    train_df = shuffled.slice(n_test).drop("row_idx")

    train_df.write_csv(train_path)
    test_df.write_csv(test_path)

    print(
        f"Wrote {train_path} ({train_df.height} rows) and "
        f"{test_path} ({test_df.height} rows)"
    )


def scan_database(
    root: Path,
    test_proportion: float,
    seed: int | None = None,
    overwrite: bool = False,
) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    animal_dirs = sorted(p for p in root.iterdir() if is_int_named_dir(p))

    for animal_dir in animal_dirs:
        session_dirs = sorted(p for p in animal_dir.iterdir() if is_int_named_dir(p))

        for session_dir in session_dirs:
            csv_path = session_dir / "data.csv"
            if csv_path.exists():
                # Optional: vary the seed by folder so all files do not get
                # identical shuffle patterns when row counts match.
                file_seed = None if seed is None else (seed + hash(str(csv_path)) % (10**9))
                split_one_csv(
                    csv_path=csv_path,
                    test_proportion=test_proportion,
                    seed=file_seed,
                    overwrite=overwrite,
                )
            else:
                print(f"Skipping {session_dir} (no data.csv)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the training_data root directory",
    )
    parser.add_argument(
        "--test-proportion",
        type=float,
        required=True,
        help="Fraction of rows to place into test.csv, e.g. 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing train.csv and test.csv if present",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.test_proportion <= 1.0):
        raise ValueError("--test-proportion must lie in [0, 1]")

    scan_database(
        root=args.root,
        test_proportion=args.test_proportion,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()