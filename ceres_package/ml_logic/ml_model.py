import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .data import load_from_gcp
from .ml_preprocess import create_clean_target, merge_dataframes, merge_sol_y, preprocess_meteo, preprocess_ndvi
from .ml_pipeline import prepare_data, train_model, evaluate_and_predict

# Récupération des données
df_soil = load_from_gcp("soil")
# df_meteo = load_from_gcp("meteo")
df_target = load_from_gcp("production")
df_ndvi = load_from_gcp("ndvi")

# Prétraitement des données
df_target_clean = create_clean_target(df_target)
# df_meteo_preprocessed = preprocess_meteo(df_meteo)
df_ndvi_preprocessed = preprocess_ndvi(df_ndvi)
merged_df = merge_sol_y(df_target_clean, df_soil)
# merged_df = merge_dataframes(merged_df, df_meteo_preprocessed)
merged_df = merge_dataframes(merged_df, df_ndvi_preprocessed)

# Préparation des données pour le modèle
X_train, X_test, y_train, y_test = prepare_data(merged_df)

# Entraînement du modèle
model = train_model(X_train, y_train)

# Évaluation du modèle et prédictions
predictions = evaluate_and_predict(model, X_test, y_test)
