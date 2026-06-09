import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from ceres_package.ml_logic.registry import save_model


def train_test_split_data(merged_df):
    X = merged_df.drop(columns=["RENDEMENT", "DEPT_ID", "ANNEE"])
    y = merged_df["RENDEMENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def prepare_data(df, target_column="RENDEMENT", test_size=0.2, random_state=42):
    """
    Nettoie le DataFrame des valeurs nulles sur la cible, sépare les features (X)
    de la target (y) et sépare le jeu en train/test.
    """
    print("⏳ Préparation et nettoyage des données...")

    # 1. Suppression des lignes où la cible est vide
    df_clean = df.dropna(subset=[target_column]).copy()

    # 2. Séparation X et y (on retire la cible et les colonnes d'identifiants/temporelles)
    # Ajustez cette liste selon les noms exacts de vos colonnes à exclure du training
    columns_to_drop = [target_column, "REGION", "TYPE BLE", "SURFACE", "PRODUCTION", "dept_nom", "DEPT_x",
                       "DEPT_y", "ANNEE", "DEPT_ID", "harvest_year"]
    columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]

    X = df_clean.drop(columns=columns_to_drop)
    y = df_clean[target_column]

    # 3. Split Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"✅ Données prêtes ! Train: {X_train.shape[0]} lignes | Test: {X_test.shape[0]} lignes")
    return X_train, X_test, y_train, y_test


def random_search_pipe(X_train, y_train, param_grid=None, n_iter=50, cv=5, random_state=42):
    """
    Initialise le Pipeline (Scaler + XGBoost) et lance la recherche d'hyperparamètres.
    """
    print("🚀 Initialisation du Pipeline et du RandomizedSearchCV...")

    # 1. Définition du pipeline (uniquement scaling et modèle)
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", XGBRegressor(random_state=random_state))
    ])

    # 2. Grille de paramètres par défaut si aucune n'est fournie
    if param_grid is None:
        param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__subsample": [0.8, 1.0]
        }

    # 3. Configuration du Search (Optimisation sur le R²)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    print(f"🏋️‍♂️ Entraînement en cours ({n_iter} combinaisons, CV={cv})...")
    search.fit(X_train, y_train)

    print(f"🏆 Meilleur score R² trouvé en CV : {search.best_score_:.4f}")
    return search.best_estimator_


def evaluate_and_predict(model, X_test, y_test):
    """
    Évalue le modèle entraîné sur le jeu de test et affiche les métriques de performance.
    """
    print("📊 Évaluation du modèle sur le jeu de test...")

    # Prédictions
    predictions = model.predict(X_test)

    # Calcul des métriques
    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"🎯 --- RÉSULTATS ---")
    print(f"   Score R² : {r2:.4f}")
    print(f"---------------------")

    return predictions

def scale_features(X_train, X_test):
    MinMax = MinMaxScaler()
    X_train_scaled = MinMax.fit_transform(X_train)
    X_test_scaled = MinMax.transform(X_test)

    return X_train_scaled, X_test_scaled

def train_save(X_train_scaled, y_train):
    XGB = XGBRegressor(learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.8)

    model = XGB.fit(X_train_scaled, y_train)

    model_final = save_model(model)

    return model
