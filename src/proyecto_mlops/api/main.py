"""API REST de predicción de churn (FastAPI)."""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from proyecto_mlops.api.model_loader import load_model
from proyecto_mlops.api.schemas import CustomerData, PredictionResponse
from proyecto_mlops.config import DECISION_THRESHOLD
from proyecto_mlops.data.prepare import clean_features

logger = logging.getLogger("churn-api")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al arrancar. Si falla, la API no arranca."""
    STATE["model"], STATE["version"] = load_model()
    logger.warning("Modelo cargado: churn-model versión %s", STATE["version"])
    yield
    STATE.clear()


app = FastAPI(
    title="Predicción de churn - MedTelecomunicaciones",
    description="Recibe los datos de un cliente y devuelve su riesgo de abandono.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Comprueba que la API responde y qué versión del modelo usa."""
    return {"status": "ok", "model_version": STATE["version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Predice el churn de un cliente."""
    # Misma limpieza que en el entrenamiento; el preprocesamiento lo hace el Pipeline.
    frame = clean_features(pd.DataFrame([customer.model_dump()]))
    probability = float(STATE["model"].predict_proba(frame)[0, 1])
    return PredictionResponse(
        prediction=int(probability >= DECISION_THRESHOLD),
        churn_probability=round(probability, 4),
        model_version=STATE["version"],
    )
