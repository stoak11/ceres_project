"""
Model catalog: how to build, fit, and score each registered model type.

Storage (manifest + joblib) lives in registry.py — not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class ModelSpec:
    """One catalog entry: one way to train and predict."""

    model_id: str
    display_name: str
    kind: str  # "sklearn" | "rule"
    build_pipeline: Callable[[], Pipeline] | None = None


def _histgb_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=5,
                    learning_rate=0.06,
                    max_iter=250,
                    random_state=42,
                ),
            ),
        ]
    )


def _ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )


def _random_forest_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def predict_lag1(
    df: pd.DataFrame,
    *,
    dept_col: str,
    year_col: str,
    target_col: str,
) -> pd.Series:
    """
    Dept×year yield lag (previous harvest). Needs target_col in df (for scoring / eval).
    """
    d = df.copy()
    d[dept_col] = d[dept_col].astype(str)
    d = d.sort_values([dept_col, year_col])
    preds = d.groupby(dept_col, group_keys=False)[target_col].shift(1)
    return preds.reindex(df.index)


def predict_sklearn(
    pipeline: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise KeyError(f"Missing feature columns: {sorted(missing)}")
    return pd.Series(pipeline.predict(df[feature_cols]), index=df.index)


def predict_from_spec(
    spec: ModelSpec,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    pipeline: Pipeline | None = None,
    dept_col: str = "DEPT_ID",
    year_col: str = "ANNEE",
    target_col: str = "RENDEMENT",
) -> pd.Series:
    """Score rows using catalog logic for this model_id."""
    if spec.kind == "sklearn":
        if pipeline is None:
            raise ValueError(f"{spec.model_id}: fitted pipeline required")
        return predict_sklearn(pipeline, df, feature_cols)
    if spec.kind == "rule":
        if spec.model_id == "lag1":
            return predict_lag1(df, dept_col=dept_col, year_col=year_col, target_col=target_col)
        raise ValueError(f"Unknown rule model: {spec.model_id}")
    raise ValueError(f"Unknown kind: {spec.kind}")


MODEL_CATALOG: dict[str, ModelSpec] = {
    "lag1": ModelSpec(
        model_id="lag1",
        display_name="Lag-1 persistence (benchmark)",
        kind="rule",
    ),
    "histgb_default": ModelSpec(
        model_id="histgb_default",
        display_name="HistGB + median imputer",
        kind="sklearn",
        build_pipeline=_histgb_pipeline,
    ),
    "ridge_default": ModelSpec(
        model_id="ridge_default",
        display_name="Ridge + median imputer",
        kind="sklearn",
        build_pipeline=_ridge_pipeline,
    ),
    "random_forest_default": ModelSpec(
        model_id="random_forest_default",
        display_name="Random Forest + median imputer",
        kind="sklearn",
        build_pipeline=_random_forest_pipeline,
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    if model_id not in MODEL_CATALOG:
        raise KeyError(f"Unknown model_id '{model_id}'. Choose from: {list(MODEL_CATALOG)}")
    return MODEL_CATALOG[model_id]
