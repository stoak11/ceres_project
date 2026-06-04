#!/usr/bin/env python3
"""Train a catalog model and save a versioned run under LOCAL_REGISTRY_PATH."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceres_package.ml_logic.model_specs import MODEL_CATALOG
from ceres_package.ml_logic.registry import upload_run_to_gcs
from ceres_package.ml_logic.train import TemporalSplit, train_and_register


def _read_mart(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mart", type=Path, required=True, help="Mart (csv/parquet): DEPT_ID, ANNEE, RENDEMENT, features")
    p.add_argument("--model-id", choices=list(MODEL_CATALOG), required=True)
    p.add_argument("--features", nargs="*", help="Feature columns (required for sklearn models)")
    p.add_argument("--train-max-year", type=int, required=True, help="Train on years <= this")
    p.add_argument("--test-min-year", type=int, required=True, help="Evaluate on years >= this")
    p.add_argument("--upload-gcs", action="store_true", help="After save, upload run folder to GCS")
    args = p.parse_args()

    df = _read_mart(args.mart)
    skip = {"DEPT_ID", "ANNEE", "RENDEMENT", "dept_nom", "REGION"}
    if args.features:
        feature_cols = list(args.features)
    else:
        feature_cols = [
            c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])
        ]

    if args.model_id != "lag1" and not feature_cols:
        p.error("No feature columns; pass --features")

    split = TemporalSplit(args.train_max_year, args.test_min_year)
    run_id = train_and_register(df, args.model_id, feature_cols, split)

    if args.upload_gcs:
        upload_run_to_gcs(run_id)

    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
