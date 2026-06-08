import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1. On initialise l'application FastAPI
app = FastAPI()

# 2. On crée une route "Root" pour vérifier que l'API est en ligne
@app.get("/")
def root():
    return {"status": "API Ceres opérationnelle"}

# 3. La route Ping-Pong demandée
@app.get("/ping")
def ping():
    return {"response": "pong"}
