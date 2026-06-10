import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ceres_package.ml_logic.registry import load_model

app = FastAPI()
app.state.model = load_model()
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

X_TEST_GCS_PATH = "gs://ceres-ai-bucket/demo/X_test_for_predict.csv"

@app.on_event("startup")
def load_reference_data():
    app.state.X_test = pd.read_csv(X_TEST_GCS_PATH)


# http://127.0.0.1:8000/predict?DEPT_ID=51&harvest_year=2020
@app.get("/predict")
def predict(DEPT_ID: str, harvest_year: int):
    """
    Filtre X_test sur (DEPT_ID, harvest_year),
    retire les colonnes métadonnées + RENDEMENT avant le predict,
    et retourne {'prediction': ..., 'reel': ...}
    """
    X_test: pd.DataFrame = app.state.X_test

    # --- Filtrage ---
    mask = (
        (X_test["DEPT_ID"].astype(str) == str(DEPT_ID)) &
        (X_test["harvest_year"] == harvest_year)
    )
    row = X_test[mask]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune donnée pour DEPT_ID={DEPT_ID} et harvest_year={harvest_year}"
        )

    # --- Récupération du réel avant de le retirer ---
    rendement_reel = float(row["RENDEMENT"].values[0])

    # --- Colonnes à exclure (métadonnées + cible) ---
    cols_to_drop = [c for c in ["DEPT_ID", "harvest_year", "RENDEMENT"] if c in row.columns]
    X_new = row.drop(columns=cols_to_drop)

    # --- Prédiction ---
    model = app.state.model
    assert model is not None

    y_pred = model.predict(X_new)

    return {"prediction": float(y_pred[0]), "reel": rendement_reel}


@app.get("/")
def root():
    return {"greeting": "Hello"}
