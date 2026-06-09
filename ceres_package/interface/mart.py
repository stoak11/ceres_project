"""
Build the feature matrix used at training time.

Mirrors ceres_package.ml_logic.ml_model.model_feature_eng + ml_pipeline.prepare_data.
Uses random_state=42 / test_size=0.2 so MinMaxScaler matches the fitted XGBoost.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ceres_package.interface.config import RANDOM_STATE, TEST_SIZE
from ceres_package.ml_logic.data import load_from_gcp
from ceres_package.ml_logic.ml_feature_eng import build_engineered_features_mart
from ceres_package.ml_logic.ml_preprocess import (
    create_clean_target,
    merge_dataframes,
    merge_dataframes_meteo,
    merge_sol_y,
    preprocess_ndvi,
)
from ceres_package.ml_logic.meteo_preproc import fast_impute_ml, get_crop_season

ID_COLS = [
    "RENDEMENT",
    "REGION",
    "TYPE BLE",
    "SURFACE",
    "PRODUCTION",
    "dept_nom",
    "DEPT_x",
    "DEPT_y",
    "ANNEE",
    "DEPT_ID",
    "harvest_year",
]

METEO_VALUE_COLS = [
    "temp_air_c",
    "temp_min_c",
    "temp_max_c",
    "temp_rosee_c",
    "humidite_relative_pct",
    "humidite_min_pct",
    "humidite_max_pct",
    "vent_moyen_10m_ms",
    "rafale_max_ms",
    "tension_vapeur_hpa",
    "temp_sol_10cm_c",
    "temp_min_sol_10cm_c",
    "heure_humidite_min",
    "heure_humidite_max",
    "direction_vent_deg",
    "hauteur_neige_sol_cm",
    "precipitations_1h_mm",
    "duree_precipitations_min",
    "duree_gel_min",
    "insolation_min",
    "rayonnement_global_jcm2",
    "duree_humidite_inf40_min",
    "duree_humidite_sup80_min",
]

METEO_DROP_COLS = [
    "duree_humectation_foliaire_min",
    "temp_sol_100cm_c",
    "temp_sol_50cm_c",
    "temp_sol_20cm_c",
    "etat_sol",
    "pression_mer_hpa",
    "temp_surface_sol_c",
]


def _zfill_dept(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DEPT_ID"] = out["DEPT_ID"].astype(str).str.zfill(2)
    return out


def _preprocess_meteo(df_meteo: pd.DataFrame) -> pd.DataFrame:
    """Daily meteo → ML features input (add_datetime_features_ml is broken upstream)."""
    df = df_meteo.copy()
    if "dept_id" in df.columns:
        df = df.rename(columns={"dept_id": "DEPT_ID"})
    df = _zfill_dept(df)
    df["day"] = pd.to_datetime(df["day"])
    df["harvest_year"] = df["day"].dt.year + (df["day"].dt.month >= 9).astype(int)
    df["saison"] = df["day"].map(get_crop_season)
    df = df[~df["DEPT_ID"].isin(["75", "92", "93", "94"])]
    df = df.drop(columns=[c for c in METEO_DROP_COLS if c in df.columns])
    return fast_impute_ml(df, METEO_VALUE_COLS)


def build_merged_dataframe() -> pd.DataFrame:
    df_meteo = _preprocess_meteo(load_from_gcp("meteo_daily"))
    df_eng = _zfill_dept(build_engineered_features_mart(df_meteo))

    merged = merge_sol_y(create_clean_target(load_from_gcp("production")), load_from_gcp("soil"))
    merged = _zfill_dept(merge_dataframes(_zfill_dept(merged), _zfill_dept(preprocess_ndvi(load_from_gcp("ndvi_month")))))
    merged = merged.rename(columns={"ANNEE": "harvest_year"})
    return merge_dataframes_meteo(merged, df_eng)


def build_features_and_scaler():
    """
    Returns (meta, scaler) where meta has DEPT_ID, harvest_year, RENDEMENT
    and scaler is fit on X_train (random_state=42).
    """
    df = build_merged_dataframe().dropna(subset=["RENDEMENT"]).copy()
    meta = df[["DEPT_ID", "harvest_year", "RENDEMENT"]].reset_index(drop=True)
    meta["DEPT_ID"] = meta["DEPT_ID"].astype(str).str.zfill(2)

    drop_cols = [c for c in ID_COLS if c in df.columns]
    X = df.drop(columns=drop_cols).reset_index(drop=True)

    X_train, _, _, _, _, _ = train_test_split(
        X,
        df["RENDEMENT"].reset_index(drop=True),
        meta,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return meta, X, scaler
