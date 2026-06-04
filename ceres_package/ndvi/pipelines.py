"""Named seasonal NDVI feature pipelines for yield benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ceres_package.ndvi.constants import NDVI_CONTRAST_COLS, NDVI_COUNT_COLS, NDVI_MED_COLS

FeatureFn = Callable[[pd.DataFrame, pd.Index], pd.DataFrame]


@dataclass(frozen=True)
class FeaturePipeline:
    name: str
    description: str
    build: FeatureFn


def _select_columns(cols: list[str]) -> FeatureFn:
    def _build(df: pd.DataFrame, train_idx: pd.Index) -> pd.DataFrame:
        del train_idx
        return df[cols].copy()

    return _build


def _spring_anomaly(df: pd.DataFrame, train_idx: pd.Index) -> pd.DataFrame:
    train = df.loc[train_idx]
    anomaly = df["modis_spring_med"] - df["dept_code"].map(
        train.groupby("dept_code")["modis_spring_med"].mean()
    )
    out = pd.DataFrame({"modis_spring_anomaly": anomaly}, index=df.index)
    return out


def _phenology_plus_anomaly(df: pd.DataFrame, train_idx: pd.Index) -> pd.DataFrame:
    base = df[
        [
            "modis_planting_med",
            "modis_winter_med",
            "modis_spring_med",
            "modis_summer_med",
            *NDVI_CONTRAST_COLS,
        ]
    ].copy()
    return pd.concat([base, _spring_anomaly(df, train_idx)], axis=1)


PIPELINES: dict[str, FeaturePipeline] = {
    "spring_only": FeaturePipeline(
        "spring_only",
        "Spring NDVI (Mar–Jun) only.",
        _select_columns(["modis_spring_med"]),
    ),
    "spring_summer": FeaturePipeline(
        "spring_summer",
        "Spring + summer + senescence contrast.",
        _select_columns(["modis_spring_med", "modis_summer_med", "modis_summer_minus_spring"]),
    ),
    "phenology_core": FeaturePipeline(
        "phenology_core",
        "Planting through summer with phenology contrasts.",
        _select_columns(
            [
                "modis_planting_med",
                "modis_winter_med",
                "modis_spring_med",
                "modis_summer_med",
                *NDVI_CONTRAST_COLS,
            ]
        ),
    ),
    "all_seasons_med": FeaturePipeline(
        "all_seasons_med",
        "All season medians including annual.",
        _select_columns(NDVI_MED_COLS),
    ),
    "phenology_anomaly": FeaturePipeline(
        "phenology_anomaly",
        "Phenology + dept-relative spring anomaly (train-only).",
        _phenology_plus_anomaly,
    ),
    "full_with_quality": FeaturePipeline(
        "full_with_quality",
        "Season medians, contrasts, and observation counts.",
        _select_columns(NDVI_MED_COLS + NDVI_CONTRAST_COLS + NDVI_COUNT_COLS),
    ),
}


def list_pipelines() -> list[str]:
    return list(PIPELINES.keys())


def build_features(
    pipeline_name: str,
    df: pd.DataFrame,
    train_idx: pd.Index,
) -> pd.DataFrame:
    """Build feature matrix ``X`` for a named pipeline."""
    features = PIPELINES[pipeline_name].build(df, train_idx)
    return features.replace([np.inf, -np.inf], np.nan)
