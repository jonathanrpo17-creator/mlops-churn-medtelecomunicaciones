"""Fase 6: pipeline de entrenamiento orquestado con Prefect.

Cada etapa es una tarea (task) y el pipeline completo es un flujo (flow):
cargar -> validar -> preparar -> entrenar candidatos -> elegir champion
-> registrar y promover el modelo.
"""

import sys

import mlflow
from prefect import flow, get_run_logger, task

from proyecto_mlops.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    RAW_DATA_PATH,
)
from proyecto_mlops.data.load import load_raw_data
from proyecto_mlops.data.prepare import build_dataset, split_data
from proyecto_mlops.models.candidates import get_candidates
from proyecto_mlops.models.experiments import (
    file_sha256,
    run_candidate,
    select_champion,
)
from proyecto_mlops.models.registry import register_and_promote


@task(name="1-cargar-y-validar-datos", retries=2, retry_delay_seconds=5)
def load_data():
    """Descarga si hace falta, lee y valida el CSV. Reintenta si falla."""
    raw_df = load_raw_data()
    get_run_logger().info("Datos cargados y validados: %d filas", len(raw_df))
    return raw_df


@task(name="2-preparar-y-dividir")
def prepare(raw_df):
    """Limpieza, X/y y división train/test."""
    X, y = build_dataset(raw_df)
    splits = split_data(X, y)
    get_run_logger().info(
        "Train: %d filas, test: %d filas", len(splits[0]), len(splits[1])
    )
    return splits


@task(name="3-entrenar-candidato")
def train_candidate(name, classifier, splits, data_sha256):
    """Un run de MLflow por candidato."""
    result = run_candidate(name, classifier, splits, data_sha256)
    get_run_logger().info(
        "%s: Recall CV %.3f, F1 CV %.3f",
        name,
        result["cv_recall_mean"],
        result["cv_f1_mean"],
    )
    return result


@task(name="4-elegir-champion")
def choose_champion(results):
    """Aplica el criterio de negocio y etiqueta el run ganador."""
    champion = select_champion(results)
    mlflow.MlflowClient().set_tag(champion["run_id"], "champion", "true")
    get_run_logger().info("Champion: %s", champion["name"])
    return champion


@task(name="5-registrar-y-promover")
def register_model():
    """Registra la versión, la valida y mueve los alias."""
    register_and_promote()


@flow(name="entrenamiento-churn", log_prints=True)
def training_pipeline():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    raw_df = load_data()
    splits = prepare(raw_df)
    data_sha256 = file_sha256(RAW_DATA_PATH)

    results = [
        train_candidate(name, classifier, splits, data_sha256)
        for name, classifier in get_candidates().items()
    ]
    champion = choose_champion(results)
    register_model()
    return champion


if __name__ == "__main__":
    if "--serve" in sys.argv:
        # Programa el pipeline: todos los lunes a las 6:00 a. m.
        training_pipeline.serve(name="churn-training-semanal", cron="0 6 * * 1")
    else:
        training_pipeline()
