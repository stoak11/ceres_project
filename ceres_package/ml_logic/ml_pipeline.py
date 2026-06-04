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
