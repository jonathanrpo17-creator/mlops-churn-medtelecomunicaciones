"""Datos y objetos de prueba compartidos.

Ningún test descarga datos, usa el CSV real ni consulta el Model Registry:
todo se construye aquí con datos sintéticos, así que corren en cualquier
máquina y en pocos segundos.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from proyecto_mlops.api import main as api_main
from proyecto_mlops.api.schemas import EXAMPLE_CUSTOMER
from proyecto_mlops.config import EXPECTED_RAW_COLUMNS
from proyecto_mlops.data.prepare import build_dataset
from proyecto_mlops.models.pipeline import build_pipeline

CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def make_raw_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Réplica sintética del CSV original (mismas columnas y formatos)."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    monthly = np.round(rng.uniform(18, 118, n), 2)
    df = pd.DataFrame({"customerID": [f"ID-{i:05d}" for i in range(n)]})
    for column, values in CATEGORIES.items():
        df[column] = rng.choice(values, n)
    df["tenure"] = tenure
    df["MonthlyCharges"] = monthly
    # En el CSV real TotalCharges llega como texto y vacío cuando tenure es 0.
    df["TotalCharges"] = [
        " " if t == 0 else str(round(m * t, 2)) for m, t in zip(monthly, tenure)
    ]
    df["Churn"] = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    return df[EXPECTED_RAW_COLUMNS]


class FixedProbabilityModel:
    """Modelo falso que siempre devuelve la misma probabilidad de churn."""

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, frame):
        row = [1 - self.probability, self.probability]
        return np.array([row] * len(frame))


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture
def fitted_pipeline(raw_df):
    X, y = build_dataset(raw_df)
    return build_pipeline().fit(X, y)


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    """Redirige el registro de predicciones a una carpeta temporal."""
    path = tmp_path / "logs" / "predictions.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(path))
    return path


@pytest.fixture
def make_client(monkeypatch, log_file):
    """Fábrica de clientes de la API con el modelo que se indique."""

    def _make(model):
        monkeypatch.setattr(api_main, "load_model", lambda: (model, "test"))
        return TestClient(api_main.app)

    return _make


@pytest.fixture
def client(make_client, fitted_pipeline):
    """API con un Pipeline real entrenado con datos sintéticos."""
    with make_client(fitted_pipeline) as test_client:
        yield test_client


@pytest.fixture
def customer() -> dict:
    return dict(EXAMPLE_CUSTOMER)
