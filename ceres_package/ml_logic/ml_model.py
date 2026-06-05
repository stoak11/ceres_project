from __future__ import annotations
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from ceres_package.ml_logic.data import load_from_gcp
from ceres_package.ml_logic.ml_pipeline import evaluate_and_predict, prepare_data, train_model
from ceres_package.ml_logic.ml_preprocess import create_clean_target, merge_dataframes, merge_sol_y, preprocess_ndvi

"""
Legacy exploratory XGBoost script (pre-registry).

Run explicitly:  PYTHONPATH=. python -m ceres_package.ml_logic.ml_model

Do not import this module — use train_and_register + model_specs instead.
"""


def _main() -> None:

    # --- Chargement des données brutes depuis GCP ---
    df_soil = load_from_gcp("soil")
    df_target = load_from_gcp("production")
    df_ndvi = load_from_gcp("ndvi_month")

    # --- Préprocessing ---
    df_target_clean = create_clean_target(df_target)
    df_ndvi_preprocessed = preprocess_ndvi(df_ndvi)

    # --- Fusion des sources de données ---
    merged_df = merge_sol_y(df_target_clean, df_soil)
    merged_df = merge_dataframes(merged_df, df_ndvi_preprocessed)

    # --- Entraînement ---
    X_train, X_test, y_train, y_test = prepare_data(merged_df)
    model = train_model(X_train, y_train)

    # --- Évaluation ---
    predictions = evaluate_and_predict(model, X_test, y_test)
    print(predictions.head())
    print(
        "mae",  mean_absolute_error(y_test, predictions),
        "rmse", mean_squared_error(y_test, predictions),
        "r2",   r2_score(y_test, predictions)
        )


if __name__ == "__main__":
    _main()
