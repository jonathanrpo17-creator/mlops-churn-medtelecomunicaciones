"""Exporta el modelo en producción a una carpeta local (la usa Docker)."""

import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from proyecto_mlops.config import ALIAS_PRODUCTION, MLFLOW_TRACKING_URI, MODEL_NAME

EXPORT_DIR = Path("models/production")


def export_production_model(destination: Path = EXPORT_DIR) -> str:
    """Copia el modelo del alias `production` a `destination`. Devuelve la versión."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS_PRODUCTION)

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@{ALIAS_PRODUCTION}",
        dst_path=str(destination),
    )
    (destination / "model_version.txt").write_text(str(version.version))
    return str(version.version)


if __name__ == "__main__":
    exported = export_production_model()
    print(f"Modelo versión {exported} exportado a {EXPORT_DIR}")
