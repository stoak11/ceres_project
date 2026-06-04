"""
Train on a mart, evaluate temporal holdout, persist via registry.save_run.

Promotion and GCS upload are separate steps (see registry / scripts/ml).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from ceres_package.ml_logic.model_specs import get_model_spec, predict_from_spec
from ceres_package.ml_logic.registry import save_run


@dataclass(frozen=True)
class TemporalSplit:
    """Train on years <= train_max_year; evaluate on years >= test_min_year."""

    train_max_year: int
    test_min_year: int

    def __post_init__(self) -> None:
        if self.test_min_year <= self.train_max_year:
            raise ValueError(
                f"test_min_year ({self.test_min_year}) must be > train_max_year ({self.train_max_year})"
            )

    def apply(
        self,
        df: pd.DataFrame,
        *,
        year_col: str = "ANNEE",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = df[df[year_col] <= self.train_max_year].copy()
        test = df[df[year_col] >= self.test_min_year].copy()
        if train.empty:
            raise ValueError(f"No training rows with {year_col} <= {self.train_max_year}")
        if test.empty:
            raise ValueError(f"No test rows with {year_col} >= {self.test_min_year}")
        return train, test


def holdout_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    valid = y_pred.notna() & y_true.notna()
    if not valid.any():
        return {"mae": float("nan"), "n_test": 0.0}
    yt = y_true.loc[valid]
    yp = y_pred.loc[valid]
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "n_test": float(int(valid.sum())),
    }


def train_and_register(
    df: pd.DataFrame,
    model_id: str,
    feature_cols: list[str],
    split: TemporalSplit,
    *,
    target_col: str = "RENDEMENT",
    dept_col: str = "DEPT_ID",
    year_col: str = "ANNEE",
    extra_params: dict[str, Any] | None = None,
) -> str:
    """
    Fit (or apply rule), score temporal holdout, save run. Returns run_id.

    Does not promote to production and does not upload to GCS.
    """
    spec = get_model_spec(model_id)
    train_df, test_df = split.apply(df, year_col=year_col)

    pipeline: Pipeline | None = None
    if spec.kind == "sklearn":
        if spec.build_pipeline is None:
            raise ValueError(f"{model_id}: no build_pipeline")
        if not feature_cols:
            raise ValueError(f"{model_id}: feature_cols required")
        pipeline = spec.build_pipeline()
        pipeline.fit(train_df[feature_cols], train_df[target_col])
        y_pred = predict_from_spec(
            spec,
            test_df,
            feature_cols,
            pipeline=pipeline,
            dept_col=dept_col,
            year_col=year_col,
            target_col=target_col,
        )
    elif spec.kind == "rule":
        if model_id == "lag1":
            # Lag needs full history per dept (train years inform test lags).
            y_pred = predict_from_spec(
                spec,
                df,
                feature_cols,
                dept_col=dept_col,
                year_col=year_col,
                target_col=target_col,
            ).reindex(test_df.index)
        else:
            raise ValueError(f"Unknown rule: {model_id}")
    else:
        raise ValueError(f"Unknown kind: {spec.kind}")

    metrics = holdout_metrics(test_df[target_col], y_pred)

    manifest = save_run(
        model_id=model_id,
        pipeline=pipeline,
        feature_cols=feature_cols,
        metrics=metrics,
        params={
            "train_max_year": split.train_max_year,
            "test_min_year": split.test_min_year,
            **(extra_params or {}),
        },
        target_col=target_col,
        dept_col=dept_col,
        year_col=year_col,
        train_years=sorted(train_df[year_col].unique().tolist()),
        test_years=sorted(test_df[year_col].unique().tolist()),
    )
    return manifest.run_id
