"""
LEGACY — notebook / exploration only (grid search, MinMaxScaler pipelines).

Production path: model_specs.MODEL_CATALOG + train_and_register + registry.
Do not import this module from packaged ingest or API code.
"""
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.linear_model import Lasso, ElasticNet, BayesianRidge


def train_test_split_data(merged_df):
    X = merged_df.drop(columns=["RENDEMENT", "DEPT_ID", "ANNEE"])
    y = merged_df["RENDEMENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

# ── shared preprocessing + model slot ───────────────────────────────────────
def make_pipe(estimator) -> Pipeline:
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", estimator),
    ])

# ── 5 linear models + their grids (hyperparams on "model__...") ─────────────
MODELS = {
    "ols": (
        make_pipe(LinearRegression()),
        {},  # no tunable params (or fit_intercept, positive, etc.)
    ),
    "ridge": (
        make_pipe(Ridge(random_state=42)),
        {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
    "lasso": (
        make_pipe(Lasso(random_state=42, max_iter=10_000)),
        {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
    ),
    "elasticnet": (
        make_pipe(ElasticNet(random_state=42, max_iter=10_000)),
        {
            "model__alpha": [0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
    ),
    "bayesian_ridge": (
        make_pipe(BayesianRidge()),
        {},  # optional: {"model__alpha_1": [...], "model__alpha_2": [...]}
    ),

    "xgboost": (
        make_pipe(XGBRegressor(random_state=42, n_jobs=-1)),
        {"model__n_estimators": [400, 450, 500], "model__learning_rate": [0.01, 0.1, 0.2]},
    ),
}

def run_grid_search_all_models(
    X_train,
    y_train,
    *,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
) -> dict:
    """
    Legacy exploratory grid search (not persisted).

    Prefer train_and_register() + registry for production paths.
    """
    from sklearn.model_selection import GridSearchCV

    results = {}
    for name, (pipe, param_grid) in MODELS.items():
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        results[name] = search
    return results


def pick_best_from_grid(results: dict):
    """Return (best_name, best_search, best_pipe) from run_grid_search_all_models."""
    best_name = min(results, key=lambda k: -results[k].best_score_)
    best_search = results[best_name]
    return best_name, best_search, best_search.best_estimator_


def randomized_grid_search(model, param_grid, X_train, y_train, n_iter=100, cv=5, scoring="neg_root_mean_squared_error"):
    search = RandomizedSearchCV(
        model=model,
        param_grid=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search, search.best_estimator_, search.best_params_, search.cv_results_
