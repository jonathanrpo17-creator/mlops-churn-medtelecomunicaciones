"""Fase 5: Model Registry. Registrar, validar, promover y volver atrás."""

import sys

import mlflow
from mlflow import MlflowClient

from proyecto_mlops.config import (
    ALIAS_PRODUCTION,
    ALIAS_STAGING,
    MIN_CV_RECALL,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
)
from proyecto_mlops.data.load import load_raw_data
from proyecto_mlops.data.prepare import build_dataset


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def get_champion_run(client: MlflowClient):
    """Run más reciente marcado con champion=true en el experimento."""
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError("No existe el experimento. Ejecuta primero experiments.")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.champion = 'true'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No hay un run champion. Ejecuta primero experiments.")
    return runs[0]


def register_champion(client: MlflowClient):
    """Crea una nueva versión del modelo a partir del run champion."""
    run = get_champion_run(client)
    metrics = run.data.metrics
    version = mlflow.register_model(
        f"runs:/{run.info.run_id}/model",
        MODEL_NAME,
        tags={
            "source_run_id": run.info.run_id,
            "model_name": run.data.params.get("model_name", ""),
            "data_sha256": run.data.params.get("data_sha256", ""),
            "cv_recall_mean": f"{metrics['cv_recall_mean']:.4f}",
            "cv_f1_mean": f"{metrics['cv_f1_mean']:.4f}",
        },
    )
    client.update_registered_model(
        MODEL_NAME,
        description=(
            "Predice el riesgo de abandono (churn) de clientes de "
            "MedTelecomunicaciones. Entrada: 19 variables del cliente. "
            "Salida: clase (0/1) y probabilidad de churn."
        ),
    )
    client.update_model_version(
        MODEL_NAME,
        version.version,
        description=(
            f"Champion '{run.data.params.get('model_name')}'. "
            f"Recall CV {metrics['cv_recall_mean']:.3f}, "
            f"F1 CV {metrics['cv_f1_mean']:.3f}."
        ),
    )
    return version, run


def validate_version(client: MlflowClient, version, run) -> list[str]:
    """Puerta de calidad antes de producción. Devuelve la lista de problemas."""
    problems = []

    recall = run.data.metrics["cv_recall_mean"]
    if recall < MIN_CV_RECALL:
        problems.append(f"Recall CV {recall:.3f} < mínimo {MIN_CV_RECALL}")

    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version.version}")
        X, _ = build_dataset(load_raw_data())
        proba = model.predict_proba(X.head(5))[:, 1]
        if len(proba) != 5 or not ((proba >= 0) & (proba <= 1)).all():
            problems.append("Las probabilidades no están en el rango [0, 1]")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"El modelo no carga o no predice: {exc}")

    return problems


def set_alias(client: MlflowClient, alias: str, version) -> None:
    client.set_registered_model_alias(MODEL_NAME, alias, str(version))


def show_registry(client: MlflowClient) -> None:
    model = client.get_registered_model(MODEL_NAME)
    print(f"Modelo: {model.name}")
    for alias, version in sorted(model.aliases.items()):
        print(f"  alias {alias:<11} -> versión {version}")
    for mv in client.search_model_versions(f"name = '{MODEL_NAME}'"):
        print(f"  versión {mv.version}: {mv.description}")


def register_and_promote() -> None:
    client = get_client()
    version, run = register_champion(client)
    set_alias(client, ALIAS_STAGING, version.version)
    problems = validate_version(client, version, run)
    if problems:
        print("NO se promueve a producción:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        client.set_model_version_tag(MODEL_NAME, version.version, "validated", "true")
        set_alias(client, ALIAS_PRODUCTION, version.version)
        print(f"Versión {version.version} validada y promovida a producción.")
    show_registry(client)


def rollback(version: str) -> None:
    """Vuelve a una versión anterior moviendo el alias de producción."""
    client = get_client()
    set_alias(client, ALIAS_PRODUCTION, version)
    print(f"Producción ahora apunta a la versión {version}.")
    show_registry(client)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "rollback":
        rollback(sys.argv[2])
    else:
        register_and_promote()
