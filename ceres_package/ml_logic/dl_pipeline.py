import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from ceres_package.ml_logic.registry import save_results, save_model
from ceres_package.ml_logic.data import load_from_gcp
from ceres_package.ml_logic.dl_preprocess import build_engineered_features_mart_dl
from ceres_package.ml_logic.ml_preprocess import create_clean_target, merge_dataframes, merge_sol_y, preprocess_ndvi
from ceres_package.ml_logic.meteo_preproc import add_datetime_features_dl, add_harvest_year_dl,  time_cycle_dl
from ceres_package.params import LOCAL_REGISTRY_PATH

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def general_dl_df_agro():
    """
    Pipeline DL avec features agronomiques calculées depuis meteo_hourly.
    Branche météo : GRU sur 8760h × n_meteo_features
    Branche statique : sol + NDVI + features agronomiques
    """
    df_meteo = load_from_gcp("meteo_hourly")
    df_feat_meteo = add_datetime_features_dl(df_meteo)
    df_feat_meteo = add_harvest_year_dl(df_feat_meteo)
    df_soil = load_from_gcp("soil")
    df_target = load_from_gcp("production")
    df_ndvi = load_from_gcp("ndvi_month")

    # --- Préprocessing ---
    df_target_clean = create_clean_target(df_target)
    df_ndvi_preprocessed = preprocess_ndvi(df_ndvi)

    # --- Exclusions ---
    DEPTS_EXCLUS = {'75', '92', '93', '94'}

    # Filtrer avant de calculer les features agro
    df_feat_meteo_filtered = df_feat_meteo[
        (~df_feat_meteo['DEPT_ID'].isin(DEPTS_EXCLUS)) &
        (df_feat_meteo['harvest_year'] > 2010) &
        (df_feat_meteo['harvest_year'] <= 2024)
    ]

    # --- Features agronomiques depuis hourly ---
    print("⏳ Calcul des features agronomiques...")
    df_agro = build_engineered_features_mart_dl(df_feat_meteo_filtered, combo_id="all")
    print(f"✅ Features agro : {df_agro.shape}")

    # --- Fusion des sources statiques ---
    df_merged = merge_sol_y(df_target_clean, df_soil)
    df_merged = merge_dataframes(df_merged, df_ndvi_preprocessed)

    df_meteo_clean = df_feat_meteo[
        (~df_feat_meteo['DEPT_ID'].isin(DEPTS_EXCLUS)) &
        (df_feat_meteo['harvest_year'] <= 2024)
    ]
    df_meteo_clean = df_meteo_clean[df_meteo_clean['harvest_year'] > 2010]
    df_meteo_clean = df_meteo_clean[
        ~((df_meteo_clean['DEPT_ID'] == '13') & (df_meteo_clean['harvest_year'] == 2012))
    ]

    df_merged_clean = df_merged[~df_merged['DEPT_ID'].isin(DEPTS_EXCLUS)]
    df_merged_clean = df_merged_clean[df_merged_clean['ANNEE'] > 2010]

    df_agro_clean = df_agro[
        (~df_agro['DEPT_ID'].isin(DEPTS_EXCLUS)) &
        (df_agro['harvest_year'] > 2010) &
        (df_agro['harvest_year'] <= 2024)
    ]
    df_agro_clean = df_agro_clean[
        ~((df_agro_clean['DEPT_ID'] == '13') & (df_agro_clean['harvest_year'] == 2012))
    ]

    # --- Vérifications ---
    print(f"Paires dept x harvest_year meteo  : {df_meteo_clean.groupby(['DEPT_ID', 'harvest_year']).ngroups}")
    print(f"Paires dept x ANNEE merged        : {len(df_merged_clean)}")
    print(f"Paires dept x harvest_year agro   : {len(df_agro_clean)}")

    # --- Imputation météo ---
    df_meteo_clean = df_meteo_clean.drop(columns=['duree_humectation_foliaire_min', 'etat_sol'], errors='ignore')

    cols_to_impute = [col for col in df_meteo_clean.columns
                      if df_meteo_clean[col].isnull().any()
                      and col not in ['datetime', 'DEPT_ID', 'saison', 'annee', 'harvest_year', 'heure', 'jour', 'mois']]

    print(f"Colonnes à imputer : {len(cols_to_impute)}")
    for col in cols_to_impute:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df_meteo_clean[col] = df_meteo_clean.groupby(['DEPT_ID', 'heure'])[col].transform(lambda x: x.fillna(x.median()))
            df_meteo_clean[col] = df_meteo_clean.groupby('DEPT_ID')[col].transform(lambda x: x.fillna(x.median()))
            df_meteo_clean[col] = df_meteo_clean[col].fillna(df_meteo_clean[col].median())
        print(f"  ✓ {col}")

    # --- Cyclic encoding ---
    df_meteo_clean = time_cycle_dl(df_meteo_clean)

    # --- Merge static + agro ---
    df_static_full = df_merged_clean.merge(
        df_agro_clean,
        left_on=['DEPT_ID', 'ANNEE'],
        right_on=['DEPT_ID', 'harvest_year'],
        how='inner'
    ).drop(columns='harvest_year')

    # --- Features ---
    METEO_FEATURES = [col for col in df_meteo_clean.columns if col not in
                      ['datetime', 'DEPT_ID', 'annee', 'harvest_year']]
    STATIC_FEATURES = [col for col in df_merged_clean.columns if col not in
                       ['DEPT_ID', 'DEPT_x', 'DEPT_y', 'ANNEE', 'RENDEMENT', 'STATUT_QUALITE']]
    AGRO_FEATURES = [col for col in df_agro_clean.columns if col not in ['DEPT_ID', 'harvest_year']]
    ALL_STATIC = STATIC_FEATURES + AGRO_FEATURES

    print(f"Features météo : {len(METEO_FEATURES)} | statiques : {len(STATIC_FEATURES)} | agro : {len(AGRO_FEATURES)}")

    # --- Tri ---
    df_meteo_clean  = df_meteo_clean.sort_values(['DEPT_ID', 'harvest_year', 'datetime'])
    df_static_full  = df_static_full.sort_values(['DEPT_ID', 'ANNEE'])

    pairs       = list(zip(df_static_full['DEPT_ID'], df_static_full['ANNEE']))
    pair_to_idx = {p: i for i, p in enumerate(pairs)}
    SEQ_LEN     = 8760

    X_meteo  = np.zeros((len(pairs), SEQ_LEN, len(METEO_FEATURES)), dtype=np.float32)
    X_static = df_static_full[ALL_STATIC].values.astype(np.float32)
    y        = df_static_full['RENDEMENT'].values.astype(np.float32).reshape(-1, 1)

    for (dept, harvest_year), group in df_meteo_clean.groupby(['DEPT_ID', 'harvest_year'], sort=False):
        idx = pair_to_idx.get((dept, harvest_year))
        if idx is None:
            continue
        seq = group[METEO_FEATURES].values
        n = min(len(seq), SEQ_LEN)
        X_meteo[idx, :n, :] = seq[:n]

    print(f"\nX_meteo  : {X_meteo.shape}")
    print(f"X_static : {X_static.shape}")
    print(f"y        : {y.shape}")

    return X_meteo, X_static, y, pairs


def general_dl_df():

    df_meteo = load_from_gcp("meteo_hourly")
    df_feat_meteo = add_datetime_features_dl(df_meteo)
    df_feat_meteo = add_harvest_year_dl(df_feat_meteo)
    df_soil = load_from_gcp("soil")
    df_target = load_from_gcp("production")
    df_ndvi = load_from_gcp("ndvi_month")

    # --- Préprocessing ---
    df_target_clean = create_clean_target(df_target)
    df_ndvi_preprocessed = preprocess_ndvi(df_ndvi)

    # --- Fusion des sources de données ---
    df_merged = merge_sol_y(df_target_clean, df_soil)
    df_merged = merge_dataframes(df_merged, df_ndvi_preprocessed)

    # Départements urbains sans agriculture à exclure
    DEPTS_EXCLUS = {'75', '92', '93', '94'}

    # Filtrer les deux DataFrames
    df_meteo_clean = df_feat_meteo[
        (~df_feat_meteo['DEPT_ID'].isin(DEPTS_EXCLUS)) &
        (df_feat_meteo['harvest_year'] <= 2024)
    ]

    df_merged_clean = df_merged[
        ~df_merged['DEPT_ID'].isin(DEPTS_EXCLUS)
    ]

    # Exclure harvest_year 2010 (données incomplètes — météo démarre en jan 2010)
    df_meteo_clean = df_meteo_clean[df_meteo_clean['harvest_year'] > 2010]
    df_merged_clean = df_merged_clean[df_merged_clean['ANNEE'] > 2010]

    # Vérifier ce qu'il reste
    print(f"Paires dept x harvest_year meteo : {df_meteo_clean.groupby(['DEPT_ID', 'harvest_year']).ngroups}")
    print(f"Paires dept x ANNEE merged : {len(df_merged_clean)}")

    # Retirer la paire orpheline
    df_meteo_clean = df_meteo_clean[
        ~((df_meteo_clean['DEPT_ID'] == '13') & (df_meteo_clean['harvest_year'] == 2012))
    ]

    # Vérification finale
    print(f"Paires meteo : {df_meteo_clean.groupby(['DEPT_ID', 'harvest_year']).ngroups}")
    print(f"Paires merged : {len(df_merged_clean)}")

    # Colonnes à supprimer
    COLS_TO_DROP = ['duree_humectation_foliaire_min', 'etat_sol']
    df_meteo_clean = df_meteo_clean.drop(columns=COLS_TO_DROP, errors='ignore')

    # Colonnes à imputer
    cols_to_impute = [col for col in df_meteo_clean.columns
                    if df_meteo_clean[col].isnull().any()
                    and col not in ['datetime', 'DEPT_ID', 'saison', 'annee', 'harvest_year', 'heure', 'jour', 'mois']]

    print(f"Colonnes à imputer : {len(cols_to_impute)}")

    # Imputation colonne par colonne pour éviter de charger toutes les colonnes en RAM
    for col in cols_to_impute:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # 1. Médiane par DEPT_ID × heure
            df_meteo_clean[col] = df_meteo_clean.groupby(['DEPT_ID', 'heure'])[col].transform(
                lambda x: x.fillna(x.median())
            )
            # 2. Fallback : médiane par DEPT_ID uniquement
            df_meteo_clean[col] = df_meteo_clean.groupby('DEPT_ID')[col].transform(
                lambda x: x.fillna(x.median())
            )
            # 3. Fallback final : médiane globale (cas extrême)
            df_meteo_clean[col] = df_meteo_clean[col].fillna(df_meteo_clean[col].median())
        print(f"  ✓ {col}")

    # Vérification
    remaining = df_meteo_clean[cols_to_impute].isnull().sum()
    print(remaining[remaining > 0])
    print("Done" if remaining.sum() == 0 else "Valeurs nulles restantes !")

    # Encoding cyclique des variables temporelles
    df_meteo_clean = time_cycle_dl(df_meteo_clean)

    # Colonnes features météo (on exclut les colonnes non-numériques et identifiants)
    METEO_FEATURES = [col for col in df_meteo_clean.columns if col not in
                    ['datetime', 'DEPT_ID', 'annee', 'harvest_year']]

    # Colonnes features statiques (sol + NDVI, on exclut target et identifiants)
    STATIC_FEATURES = [col for col in df_merged_clean.columns if col not in
                    ['DEPT_ID', 'DEPT_x', 'DEPT_y', 'ANNEE', 'RENDEMENT', 'STATUT_QUALITE']]

    print(f"Features météo : {len(METEO_FEATURES)}")
    print(f"Features statiques : {len(STATIC_FEATURES)}")

    # Tri pour garantir l'ordre cohérent
    df_meteo_clean = df_meteo_clean.sort_values(['DEPT_ID', 'harvest_year', 'datetime'])
    df_merged_clean = df_merged_clean.sort_values(['DEPT_ID', 'ANNEE'])

    pairs = list(zip(df_merged_clean['DEPT_ID'], df_merged_clean['ANNEE']))
    pair_to_idx = {p: i for i, p in enumerate(pairs)}

    SEQ_LEN = 8760

    X_meteo  = np.zeros((len(pairs), SEQ_LEN, len(METEO_FEATURES)), dtype=np.float32)
    X_static = df_merged_clean[STATIC_FEATURES].values.astype(np.float32)
    y        = df_merged_clean['RENDEMENT'].values.astype(np.float32).reshape(-1, 1)

    # Un seul groupby au lieu de 1259 filtres
    for (dept, harvest_year), group in df_meteo_clean.groupby(['DEPT_ID', 'harvest_year'], sort=False):
        idx = pair_to_idx.get((dept, harvest_year))
        if idx is None:
            continue
        seq = group[METEO_FEATURES].values
        n = min(len(seq), SEQ_LEN)
        X_meteo[idx, :n, :] = seq[:n]

    print(f"X_meteo  : {X_meteo.shape}")
    print(f"X_static : {X_static.shape}")
    print(f"y        : {y.shape}")

    return X_meteo, X_static, y, pairs

def train_val_test_dl(X_meteo, X_static, y, pairs):

    HARVEST_YEARS = np.array([harvest_year for _, harvest_year in pairs])

    # Masks temporels
    train_mask = HARVEST_YEARS <= 2020
    val_mask   = (HARVEST_YEARS == 2021) | (HARVEST_YEARS == 2022)
    test_mask  = HARVEST_YEARS >= 2023

    # --- Météo : calcul mean/std sur le train uniquement ---
    meteo_mean = X_meteo[train_mask].mean(axis=(0, 1))
    meteo_std  = X_meteo[train_mask].std(axis=(0, 1))
    meteo_std[meteo_std == 0] = 1

    X_meteo -= meteo_mean
    X_meteo /= meteo_std

    # --- Statique ---
    static_mean = X_static[train_mask].mean(axis=0)
    static_std  = X_static[train_mask].std(axis=0)
    static_std[static_std == 0] = 1
    X_static = (X_static - static_mean) / static_std

    # # --- Target scaling ---
    # y_mean = y[train_mask].mean()
    # y_std  = y[train_mask].std()
    # y_scaled = (y - y_mean) / y_std

    # --- Split train / val / test ---
    X_meteo_train, X_meteo_val, X_meteo_test   = X_meteo[train_mask],  X_meteo[val_mask],  X_meteo[test_mask]
    X_static_train, X_static_val, X_static_test = X_static[train_mask], X_static[val_mask], X_static[test_mask]
    # y_train, y_val, y_test                      = y_scaled[train_mask], y_scaled[val_mask], y_scaled[test_mask]
    y_train, y_val, y_test                      = y[train_mask], y[val_mask], y[test_mask]

    print(f"Train : {train_mask.sum()} | Val : {val_mask.sum()} | Test : {test_mask.sum()}")
    print(f"\nX_meteo_train  : {X_meteo_train.shape}")
    print(f"X_meteo_val    : {X_meteo_val.shape}")
    print(f"X_meteo_test   : {X_meteo_test.shape}")
    # print(f"\ny_mean={y_mean:.4f}, y_std={y_std:.4f}")

    return X_meteo_train, X_meteo_val, X_meteo_test, X_static_train, X_static_val, X_static_test, y_train, y_val, y_test


def build_gru_two_branch_model(X_meteo_train, X_static_features, hidden_dim=32):

    # --- Branche météo (horaire) ---
    meteo_input = keras.Input(shape=X_meteo_train.shape[1:], name='meteo_input')
    x = layers.GRU(hidden_dim, return_sequences=True, dropout=0.3, recurrent_dropout=0.2, name='gru_1')(meteo_input)
    x = layers.BatchNormalization()(x)
    x = layers.GRU(hidden_dim // 2, return_sequences=False, dropout=0.3, recurrent_dropout=0.2, name='gru_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    meteo_embedding = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='meteo_embedding')(x)

    # --- Branche statique (sol + NDVI) ---
    static_input = keras.Input(shape=X_static_features.shape[1:], name='static_input')
    s = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(static_input)
    s = layers.BatchNormalization()(s)
    s = layers.Dropout(0.4)(s)
    static_embedding = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name='static_embedding')(s)

    # --- Fusion ---
    merged = layers.Concatenate()([meteo_embedding, static_embedding])
    out = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(merged)
    out = layers.BatchNormalization()(out)
    out = layers.Dropout(0.4)(out)
    output = layers.Dense(1, name='rendement')(out)

    model = keras.Model(
        inputs=[meteo_input, static_input],
        outputs=output
    )
    return model


def build_lstm_two_branch_model(X_meteo_train, X_static_features, hidden_dim=64):

    # --- Branche météo (horaire) ---
    meteo_input = keras.Input(shape=X_meteo_train.shape[1:], name='meteo_input')
    x = layers.LSTM(hidden_dim, return_sequences=True, name='lstm_1')(meteo_input)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(hidden_dim // 2, return_sequences=False, name='lstm_2')(x)
    x = layers.Dropout(0.2)(x)
    meteo_embedding = layers.Dense(32, activation='relu', name='meteo_embedding')(x)

    # --- Branche statique (sol + NDVI) ---
    static_input = keras.Input(shape=X_static_features.shape[1:], name='static_input')
    s = layers.Dense(64, activation='relu')(static_input)
    s = layers.Dropout(0.2)(s)
    static_embedding = layers.Dense(32, activation='relu', name='static_embedding')(s)

    # --- Fusion ---
    merged = layers.Concatenate()([meteo_embedding, static_embedding])
    out = layers.Dense(32, activation='relu')(merged)
    out = layers.Dropout(0.2)(out)
    output = layers.Dense(1, name='rendement')(out)

    model = keras.Model(
        inputs=[meteo_input, static_input],
        outputs=output
    )
    return model

def gru_compile(X_meteo_train, X_static_features):

    model = build_gru_two_branch_model(
        X_meteo_train, X_static_features
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    model.summary()
    return model

def lstm_compile(X_meteo_train, X_static_features):

    model = build_lstm_two_branch_model(
        X_meteo_train, X_static_features
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    model.summary()
    return model

def fit_gru(model, X_meteo_train, X_meteo_val, X_static_train, X_static_val, y_train, y_val):

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        [X_meteo_train, X_static_train],
        y_train,
        validation_data=([X_meteo_val, X_static_val], y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return history

def fit_lstm(model, X_meteo_train, X_meteo_val, X_static_train, X_static_val, y_train, y_val):

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        [X_meteo_train, X_static_train],
        y_train,
        validation_data=([X_meteo_val, X_static_val], y_val),
        epochs=1500,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return history


def _main(model_type: str = 'gru'):
    """
    model_type : 'gru' | 'gru_agro' | 'lstm'
    """
    assert model_type in ('gru', 'gru_agro', 'lstm'), "model_type doit être 'gru', 'gru_agro' ou 'lstm'"

    # --- Vérification des dossiers de sauvegarde AVANT de lancer ---
    for subdir in ['models', 'params', 'metrics']:
        path = os.path.join(LOCAL_REGISTRY_PATH, subdir)
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except PermissionError:
            raise PermissionError(f"❌ Pas de droits d'écriture sur {path} — vérifie LOCAL_REGISTRY_PATH dans ton .env avant de lancer l'entraînement")

    print("✅ Dossiers de sauvegarde vérifiés\n")

    # --- Données ---
    print("⏳ Chargement et préparation des données...")
    if model_type == 'gru_agro':
        X_meteo, X_static, y, pairs = general_dl_df_agro()
    else:
        X_meteo, X_static, y, pairs = general_dl_df()
    print(f"✅ Données prêtes — {len(pairs)} paires dept x année\n")

    # --- Split & normalisation ---
    print("⏳ Normalisation et split train/val/test...")
    X_meteo_train, X_meteo_val, X_meteo_test, \
    X_static_train, X_static_val, X_static_test, \
    y_train, y_val, y_test = train_val_test_dl(X_meteo, X_static, y, pairs)
    print(f"✅ Split effectué\n")

    # --- Compilation ---
    print(f"⏳ Construction et compilation du modèle ({model_type.upper()})...")
    if model_type in ('gru', 'gru_agro'):
        model = gru_compile(X_meteo_train, X_static_train)
    else:
        model = lstm_compile(X_meteo_train, X_static_train)
    print(f"✅ Modèle compilé\n")

    # --- Entraînement ---
    print("⏳ Entraînement du modèle...")
    if model_type in ('gru', 'gru_agro'):
        history = fit_gru(
            model,
            X_meteo_train, X_meteo_val,
            X_static_train, X_static_val,
            y_train, y_val
        )
    else:
        history = fit_lstm(
            model,
            X_meteo_train, X_meteo_val,
            X_static_train, X_static_val,
            y_train, y_val
        )
    print(f"✅ Entraînement terminé — {len(history.history['loss'])} epochs\n")

    # --- Évaluation finale ---
    print("⏳ Évaluation sur le test set (2023-2024)...")
    test_loss, test_mae = model.evaluate([X_meteo_test, X_static_test], y_test)
    print(f"\n✅ Résultats finaux :")
    print(f"   Test loss (MSE) : {test_loss:.4f}")
    print(f"   Test MAE        : {test_mae:.2f} q/ha")

    # --- Sauvegarde ---
    print("\n⏳ Sauvegarde des résultats et du modèle...")
    save_results(
        params={
            'model_type': model_type,
            'seq_len': 8760,
            'hidden_dim': 32,
            'batch_size': 32,
            'epochs_run': len(history.history['loss']),
        },
        metrics={
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'val_loss_min': float(min(history.history['val_loss'])),
            'val_mae_min': float(min(history.history['val_mae'])),
        }
    )
    save_model(model)


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'gru'
    _main(model_type=model_type)
