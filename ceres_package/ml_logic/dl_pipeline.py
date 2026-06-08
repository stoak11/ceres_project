import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ceres_package.ml_logic.registry import save_results, save_model
from ceres_package.ml_logic.data import load_from_gcp
from ceres_package.ml_logic.ml_preprocess import create_clean_target, merge_dataframes, merge_sol_y, preprocess_ndvi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_crop_season(date: pd.Timestamp) -> str:
    month, day = date.month, date.day

    if month in (9, 10, 11):
        return 'semis'
    elif month == 12 or month in (1, 2):
        return 'vernalisation'
    elif month == 3 or (month == 4 and day <= 15):
        return 'tallage'
    elif (month == 4 and day > 15) or (month == 5 and day <= 15):
        return 'montaison'
    elif (month == 5 and day > 15) or (month == 6 and day <= 15):
        return 'floraison'
    elif (month == 6 and day > 15) or month in (7, 8):
        return 'remplissage'
    raise ValueError(f"Date inattendue : {date}")

def add_datetime_features_dl(df: pd.DataFrame) -> pd.DataFrame:
    # Conversion to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Vectorial Extraction
    dt = df['datetime'].dt
    df['heure'] = dt.hour.astype('int8')
    df['jour']  = dt.day.astype('int8')
    df['mois']  = dt.month.astype('int8')
    df['annee'] = dt.year.astype('int16')

    # Saison agronomique — map sur les valeurs uniques pour éviter 13M appels
    unique_dates = df['datetime'].drop_duplicates()
    season_map = {ts: get_crop_season(ts) for ts in unique_dates}
    df['saison'] = df['datetime'].map(season_map).astype('category')

    # Encoding dept_id en 2 digits (01, 02, ..., 95)
    df['DEPT_ID'] = df['dept_id'].astype(int).astype(str).str.zfill(2)
    df = df.drop(columns='dept_id')

    return df


def general_dl_df():

    df_meteo = load_from_gcp("meteo_hourly")
    df_feat_meteo = add_datetime_features_dl(df_meteo)
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
        (df_feat_meteo['annee'] <= 2024)
    ]

    df_merged_clean = df_merged[
        ~df_merged['DEPT_ID'].isin(DEPTS_EXCLUS)
    ]

    # Vérifier ce qu'il reste
    print(f"Paires deptxannée meteo : {df_meteo_clean.groupby(['DEPT_ID', 'annee']).ngroups}")
    print(f"Paires deptxannée merged : {len(df_merged_clean)}")

    df_meteo_clean = df_meteo_clean[
        ~((df_meteo_clean['DEPT_ID'] == '13') & (df_meteo_clean['annee'] == 2012))
    ]

    # Vérification finale
    print(f"Paires meteo : {df_meteo_clean.groupby(['DEPT_ID', 'annee']).ngroups}")
    print(f"Paires merged : {len(df_merged_clean)}")

    # Colonnes à supprimer
    COLS_TO_DROP = ['duree_humectation_foliaire_min', 'etat_sol']
    df_meteo_clean = df_meteo_clean.drop(columns=COLS_TO_DROP)

    # Colonnes à imputer
    cols_to_impute = [col for col in df_meteo_clean.columns
                    if df_meteo_clean[col].isnull().any()
                    and col not in ['datetime', 'DEPT_ID', 'saison', 'annee', 'heure', 'jour', 'mois']]

    print(f"Colonnes à imputer : {len(cols_to_impute)}")

    # Imputation colonne par colonne pour éviter de charger toutes les colonnes en RAM
    for col in cols_to_impute:
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

    # Colonnes features météo (on exclut les colonnes non-numériques et identifiants)
    METEO_FEATURES = [col for col in df_meteo_clean.columns if col not in
                    ['datetime', 'DEPT_ID', 'annee', 'saison', 'heure', 'jour', 'mois']]

    # Colonnes features statiques (sol + NDVI, on exclut target et identifiants)
    STATIC_FEATURES = [col for col in df_merged_clean.columns if col not in
                    ['DEPT_ID', 'DEPT_x', 'DEPT_y', 'ANNEE', 'RENDEMENT', 'STATUT_QUALITE']]

    print(f"Features météo : {len(METEO_FEATURES)}")
    print(f"Features statiques : {len(STATIC_FEATURES)}")

    # Tri pour garantir l'ordre cohérent
    df_meteo_clean = df_meteo_clean.sort_values(['DEPT_ID', 'annee', 'datetime'])
    df_merged_clean = df_merged_clean.sort_values(['DEPT_ID', 'ANNEE'])

    pairs = list(zip(df_merged_clean['DEPT_ID'], df_merged_clean['ANNEE']))
    pair_to_idx = {p: i for i, p in enumerate(pairs)}

    # Hours per year :
    SEQ_LEN = 8760

    X_meteo  = np.zeros((len(pairs), SEQ_LEN, len(METEO_FEATURES)), dtype=np.float32)
    X_static = df_merged_clean[STATIC_FEATURES].values.astype(np.float32)
    y        = df_merged_clean['RENDEMENT'].values.astype(np.float32).reshape(-1, 1)

    # Un seul groupby au lieu de 1349 filtres
    for (dept, annee), group in df_meteo_clean.groupby(['DEPT_ID', 'annee'], sort=False):
        idx = pair_to_idx.get((dept, annee))
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

    ANNEES = np.array([annee for _, annee in pairs])

    # Masks temporels
    train_mask = ANNEES <= 2020
    val_mask   = (ANNEES == 2021) | (ANNEES == 2022)
    test_mask  = ANNEES >= 2023

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


def build_gru_two_branch_model(X_meteo_train, X_static_features, hidden_dim=64):

    # --- Branche météo (horaire) ---
    meteo_input = keras.Input(shape = X_meteo_train.shape[1:], name='meteo_input')
    x = layers.GRU(hidden_dim, return_sequences=True, name='gru_1')(meteo_input)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(hidden_dim // 2, return_sequences=False, name='gru_2')(x)
    x = layers.Dropout(0.2)(x)
    meteo_embedding = layers.Dense(32, activation='relu', name='meteo_embedding')(x)

    # --- Branche statique (sol + NDVI) ---
    static_input = keras.Input(shape = X_static_features.shape[1:], name='static_input')
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


def _main():

    # --- Données ---
    print("⏳ Chargement et préparation des données...")
    X_meteo, X_static, y, pairs = general_dl_df()
    print(f"✅ Données prêtes — {len(pairs)} paires dept×année\n")

    # --- Split & normalisation ---
    print("⏳ Normalisation et split train/val/test...")
    X_meteo_train, X_meteo_val, X_meteo_test, \
    X_static_train, X_static_val, X_static_test, \
    y_train, y_val, y_test = train_val_test_dl(X_meteo, X_static, y, pairs)
    print(f"✅ Split effectué\n")

    # --- Compilation ---
    print("⏳ Construction et compilation du modèle...")
    model = gru_compile(X_meteo_train, X_static_train)
    print(f"✅ Modèle compilé\n")

    # --- Entraînement ---
    print("⏳ Entraînement du modèle...")
    history = fit_gru(
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
            'seq_len': 8760,
            'hidden_dim': 64,
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
    _main()
