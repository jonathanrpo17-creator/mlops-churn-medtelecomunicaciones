"""Carga del modelo que sirve la API."""

import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from proyecto_mlops.config import ALIAS_PRODUCTION, MLFLOW_TRACKING_URI, MODEL_NAME


def load_model():
    """Devuelve (modelo, versión).

    Por defecto carga el alias `production` del Model Registry. Si existe la
    variable MODEL_URI (por ejemplo dentro de Docker), carga esa ruta local.
    """
    model_uri = os.getenv("MODEL_URI")
    if model_uri:
        model = mlflow.sklearn.load_model(model_uri)
        version_file = Path(model_uri) / "model_version.txt"
        if version_file.exists():
            return model, version_file.read_text().strip()
        return model, os.getenv("MODEL_VERSION", "local")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS_PRODUCTION)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS_PRODUCTION}")
    return model, str(version.version)
