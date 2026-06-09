"""
Ceres predict API.

  uvicorn ceres_package.interface.api.fast:app --reload
  GET /predict?dept=51&year=2023
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ceres_package.interface.predictor import get_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()
    yield


app = FastAPI(title="Ceres Predict API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "ceres-predict"}


@app.get("/ping")
def ping():
    return {"response": "pong"}


@app.get("/health")
def health():
    get_predictor()
    return {"status": "ready", "model_loaded": True}


@app.get("/predict")
def predict_yield(
    dept: str = Query(..., description="Code département, ex. 51"),
    year: int = Query(..., ge=2011, le=2030, description="Année de récolte"),
):
    try:
        return get_predictor().predict(dept, year)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
