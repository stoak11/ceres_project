import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import optuna

from ceres_package.ml_logic.data import load_from_gcp
from ceres_package.ml_logic.ml_preprocess import (
    create_clean_target, preprocess_ndvi, preprocess_meteo_ml,
    merge_sol_y, merge_dataframes_meteo, merge_dataframes
)
from ceres_package.ml_logic.ml_feature_eng import build_engineered_features_mart
from ceres_package.ml_logic.ml_pipeline import prepare_data


# ============================================================
# CONSTANTES
# ============================================================

LOW_IMPORTANCE_FEATURES = [
    'STATUT_QUALITE', 'phh2o_100-200cm_mean', 'vernalisation_nb_jours_0_10C',
    'nb_risques_rayonnement_faible', 'floraison_nb_jours_tx_gt_25',
    'floraison_nb_jours_tx_gt_30', 'floraison_max_consecutive_tx_gt_25',
    'floraison_max_consecutive_tx_gt_30', 'floraison_heat_degree_days_25',
    'floraison_heat_degree_days_30', 'floraison_episodes_tx_gt_30_ge3',
    'floraison_days_heat_drought', 'remplissage_storm_event_count'
]


# ============================================================
# DATA
# ============================================================

def load_and_merge_ml():
    """Charge et fusionne toutes les sources de données."""
    df_soil   = load_from_gcp("soil")
    df_target = load_from_gcp("production")
    df_ndvi   = load_from_gcp("ndvi_month")
    df_meteo  = load_from_gcp('meteo_daily')

    df_target_clean = create_clean_target(df_target)
    df_ndvi_preprocessed = preprocess_ndvi(df_ndvi)
    df_meteo_preprocessed = preprocess_meteo_ml(df_meteo)
    df_meteo_eng = build_engineered_features_mart(df_meteo_preprocessed)

    merged_df = merge_sol_y(df_target_clean, df_soil)
    merged_df = merge_dataframes(merged_df, df_ndvi_preprocessed)
    merged_df = merged_df.rename(columns={'ANNEE': 'harvest_year'})
    merged_df = merge_dataframes_meteo(merged_df, df_meteo_eng)

    return merged_df


def train_test_split_ml():
    """Charge, merge, isole 2025, split train/test."""
    merged_df = load_and_merge_ml()

    # Isoler 2025 pour la démo
    df_2025   = merged_df[merged_df['harvest_year'] == 2025]
    df_train  = merged_df[merged_df['harvest_year'] != 2025]

    print(f"Données 2025 isolées : {len(df_2025)} lignes")
    print(f"Données d'entraînement : {len(df_train)} lignes")

    X_train, X_test, y_train, y_test = prepare_data(
        df_train, target_column="RENDEMENT", test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, df_2025


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def drop_low_importance_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in LOW_IMPORTANCE_FEATURES if c in df.columns]
    return df.drop(columns=cols)


def create_soil_meteo_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['wv0033_deep_x_heat_filling']  = df['wv0033_100-200cm_mean'] * df['remplissage_nb_jours_tx_gt_25']
    df['wv0033_mid_x_dry_filling']    = df['wv0033_30-60cm_mean']   * df['remplissage_max_consecutive_dry_days']
    df['silt_deep_x_rain_flowering']  = df['silt_100-200cm_mean']   * df['floraison_cumul_pluie_10j_max']
    df['wv0033_deep_x_warm_winter']   = df['wv0033_100-200cm_mean'] * df['hiver_anormalement_chaud']
    return df


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    soil_vars = ['silt', 'sand', 'clay', 'wv0033', 'wv0010', 'wv1500',
                 'cfvo', 'ocd', 'soc', 'nitrogen', 'cec', 'phh2o', 'bdod']

    for var in soil_vars:
        surface_cols = [c for c in [f'{var}_0-5cm_mean', f'{var}_5-15cm_mean'] if c in df.columns]
        medium_cols  = [c for c in [f'{var}_15-30cm_mean', f'{var}_30-60cm_mean'] if c in df.columns]
        deep_cols    = [c for c in [f'{var}_60-100cm_mean', f'{var}_100-200cm_mean'] if c in df.columns]

        if surface_cols: df[f'{var}_surface_mean'] = df[surface_cols].mean(axis=1)
        if medium_cols:  df[f'{var}_medium_mean']  = df[medium_cols].mean(axis=1)
        if deep_cols:    df[f'{var}_deep_mean']    = df[deep_cols].mean(axis=1)
        if surface_cols and deep_cols:
            df[f'{var}_gradient'] = df[surface_cols].mean(axis=1) - df[deep_cols].mean(axis=1)

    if 'silt_surface_mean' in df.columns and 'sand_surface_mean' in df.columns:
        df['ratio_silt_sand_surface'] = df['silt_surface_mean'] / (df['sand_surface_mean'] + 1e-6)
        df['ratio_silt_sand_deep']    = df['silt_deep_mean']    / (df['sand_deep_mean']    + 1e-6)

    for zone in ['surface', 'medium', 'deep']:
        if f'wv0033_{zone}_mean' in df.columns and f'wv1500_{zone}_mean' in df.columns:
            df[f'AWC_{zone}'] = df[f'wv0033_{zone}_mean'] - df[f'wv1500_{zone}_mean']
        if f'wv0033_{zone}_mean' in df.columns and f'wv0010_{zone}_mean' in df.columns:
            df[f'retention_ratio_{zone}'] = df[f'wv0033_{zone}_mean'] / (df[f'wv0010_{zone}_mean'] + 1e-6)

    if 'floraison_cumul_pluie_10j_max' in df.columns and 'remplissage_nb_jours_tx_gt_25' in df.columns:
        df['stress_floraison_chaleur_secheresse'] = (
            df['remplissage_nb_jours_tx_gt_25'] * (1 / (df['floraison_cumul_pluie_10j_max'] + 1e-6))
        )
    if 'remplissage_heatwave_length_max' in df.columns and 'remplissage_max_consecutive_dry_days' in df.columns:
        df['stress_remplissage_combine'] = (
            df['remplissage_heatwave_length_max'] * df['remplissage_max_consecutive_dry_days']
        )
    if 'hiver_anormalement_chaud' in df.columns and 'AWC_deep' in df.columns:
        df['stress_vernal_x_AWC'] = df['hiver_anormalement_chaud'] * df['AWC_deep']

    return df


def create_final_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    soil_vars = ['silt', 'sand', 'clay', 'wv0033', 'wv0010', 'wv1500',
                 'cfvo', 'ocd', 'soc', 'nitrogen', 'cec', 'phh2o', 'bdod']

    for var in soil_vars:
        surface_cols = [c for c in [f'{var}_0-5cm_mean', f'{var}_5-15cm_mean'] if c in df.columns]
        medium_cols  = [c for c in [f'{var}_15-30cm_mean', f'{var}_30-60cm_mean'] if c in df.columns]
        deep_cols    = [c for c in [f'{var}_60-100cm_mean', f'{var}_100-200cm_mean'] if c in df.columns]

        if f'{var}_gradient' not in df.columns and surface_cols and deep_cols:
            df[f'{var}_gradient'] = df[surface_cols].mean(axis=1) - df[deep_cols].mean(axis=1)
        if medium_cols and deep_cols:
            df[f'{var}_gradient_mid_deep'] = df[medium_cols].mean(axis=1) - df[deep_cols].mean(axis=1)

    depth_suffixes = ['0-5cm_mean', '5-15cm_mean', '15-30cm_mean',
                      '30-60cm_mean', '60-100cm_mean', '100-200cm_mean']
    cols_to_drop = [c for c in df.columns if any(c.endswith(s) for s in depth_suffixes)]
    df = df.drop(columns=cols_to_drop)

    return df


def full_feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering."""
    X = drop_low_importance_features(X)
    X = create_soil_meteo_interactions(X)
    X = create_advanced_features(X)
    X = create_final_features(X)
    print(f"✅ Feature engineering terminé — {X.shape[1]} features")
    return X


# ============================================================
# MODÈLE
# ============================================================

def objective(trial, X_train, y_train):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 500, 1500),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'colsample_bynode':  trial.suggest_float('colsample_bynode', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 15),
        'gamma':             trial.suggest_float('gamma', 0.0, 1.0),
    }
    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(random_state=42, n_jobs=-1, **params))
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5,
                             scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()


def train_model(X_train, y_train, n_trials=200):
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )
    print(f"\n✅ Meilleur MAE CV : {study.best_value:.4f}")
    print(f"✅ Meilleurs params :\n{study.best_params}")

    best_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(random_state=42, n_jobs=-1, **study.best_params))
    ])
    best_pipe.fit(X_train, y_train)
    return best_pipe, study


def evaluate_and_predict(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae  = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2   = r2_score(y_test, predictions)

    print(f"\n✅ Résultats test set :")
    print(f"   MAE  : {mae:.4f} q/ha")
    print(f"   RMSE : {rmse:.4f} q/ha")
    print(f"   R²   : {r2:.4f}")

    return predictions, mae, rmse, r2


# ============================================================
# MAIN
# ============================================================

def _main():

    # --- Données ---
    print("⏳ Chargement et préparation des données...")
    X_train, X_test, y_train, y_test, df_2025 = train_test_split_ml()
    print(f"✅ Données prêtes\n")

    # --- Feature engineering ---
    print("⏳ Feature engineering...")
    X_train_fe = full_feature_engineering(X_train)
    X_test_fe  = full_feature_engineering(X_test)
    print(f"✅ Features prêtes — {X_train_fe.shape[1]} features\n")

    # --- Entraînement ---
    print("⏳ Entraînement Optuna XGBoost...")
    best_pipe, study = train_model(X_train_fe, y_train, n_trials=200)
    print(f"✅ Modèle entraîné\n")

    # --- Évaluation ---
    print("⏳ Évaluation sur le test set...")
    predictions, mae, rmse, r2 = evaluate_and_predict(best_pipe, X_test_fe, y_test)

    # --- Prédictions 2025 ---
    print("\n⏳ Prédictions 2025...")
    X_2025 = df_2025.drop(columns=['RENDEMENT', 'DEPT_ID', 'harvest_year',
                                    'DEPT_x', 'DEPT_y', 'REGION'], errors='ignore')
    X_2025_fe  = full_feature_engineering(X_2025)
    preds_2025 = best_pipe.predict(X_2025_fe)

    df_2025_out = df_2025[['DEPT_ID', 'harvest_year', 'RENDEMENT']].copy()
    df_2025_out['rendement_predit'] = preds_2025

    # Métriques sur 2025
    mask_notna = df_2025['RENDEMENT'].notna().values
    y_2025_real = df_2025['RENDEMENT'].dropna().values
    y_2025_pred = preds_2025[mask_notna]

    mae_2025  = mean_absolute_error(y_2025_real, y_2025_pred)
    rmse_2025 = np.sqrt(mean_squared_error(y_2025_real, y_2025_pred))
    r2_2025   = r2_score(y_2025_real, y_2025_pred)

    print(f"✅ Prédictions 2025 :")
    print(f"   MAE  : {mae_2025:.4f} q/ha")
    print(f"   RMSE : {rmse_2025:.4f} q/ha")
    print(f"   R²   : {r2_2025:.4f}")
    print(df_2025_out.head(10))

    return best_pipe, study, df_2025_out


if __name__ == "__main__":
    _main()
