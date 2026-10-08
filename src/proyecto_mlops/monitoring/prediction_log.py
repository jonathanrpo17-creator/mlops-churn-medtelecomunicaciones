"""Registro simple de predicciones: una línea JSON por petición."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from proyecto_mlops.config import PREDICTION_LOG_FILE

logger = logging.getLogger("churn-api")


def log_path() -> Path:
    """Ruta del registro (se puede cambiar con la variable PREDICTION_LOG_PATH)."""
    return Path(os.getenv("PREDICTION_LOG_PATH", PREDICTION_LOG_FILE))


def log_prediction(
    features: dict,
    probability: float,
    prediction: int,
    model_version: str,
    latency_ms: float,
) -> None:
    """Añade una línea al registro. Un fallo aquí nunca debe romper la API."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_version,
        "prediction": prediction,
        "churn_probability": probability,
        "latency_ms": round(latency_ms, 2),
        "features": features,
    }
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
    except OSError:
        logger.exception("No se pudo escribir el registro de predicciones")
