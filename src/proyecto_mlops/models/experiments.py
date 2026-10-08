"""Fase 4: comparación de modelos con MLflow Tracking y elección del champion."""

import hashlib
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate

from proyecto_mlops.config import (
    BASELINE_NAME,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TEST_SIZE,
)
from proyecto_mlops.data.load import load_raw_data
from proyecto_mlops.data.prepare import build_dataset, split_data
from proyecto_mlops.models.candidates import get_candidates
from proyecto_mlops.models.evaluate import compute_metrics, estimated_cost
from proyecto_mlops.models.pipeline import build_pipeline


def file_sha256(path: Path) -> str:
    """Huella del CSV: permite saber con qué datos se entrenó cada run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_candidate(name, classifier, splits, data_sha256) -> dict:
    """Entrena un candidato y registra todo en un run de MLflow."""
    X_train, X_test, y_train, y_test = splits
    pipeline = build_pipeline(classifier=classifier)

    with mlflow.start_run(run_name=name) as run:
        # 1) Parámetros: todo lo necesario para reproducir el run.
        mlflow.log_param("model_name", name)
        mlflow.log_param("model_class", type(classifier).__name__)
        mlflow.log_params(classifier.get_params())
        mlflow.log_params(
            {
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "cv_folds": CV_FOLDS,
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
                "n_numeric_features": len(NUMERIC_FEATURES),
                "n_categorical_features": len(CATEGORICAL_FEATURES),
                "data_sha256": data_sha256,
            }
        )

        # 2) Validación cruzada SOLO sobre entrenamiento (el test no se toca).
        cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=["accuracy", "recall", "f1"]
        )
        cv_metrics = {
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            "cv_recall_mean": scores["test_recall"].mean(),
            "cv_recall_std": scores["test_recall"].std(),
            "cv_f1_mean": scores["test_f1"].mean(),
        }

        # 3) Entrenamiento final y evaluación en test.
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_metrics = {
            f"test_{k}": v for k, v in compute_metrics(y_test, y_pred).items()
        }
        cost = estimated_cost(y_test, y_pred)
        test_metrics.update({f"test_{k}": v for k, v in cost.items()})

        mlflow.log_metrics({**cv_metrics, **test_metrics})

        # 4) Artefactos: informe de clasificación y matriz de confusión.
        mlflow.log_text(
            classification_report(y_test, y_pred), "classification_report.txt"
        )
        cm = pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            index=["real_no_churn", "real_churn"],
            columns=["pred_no_churn", "pred_churn"],
        )
        mlflow.log_text(cm.to_csv(), "confusion_matrix.csv")

        # 5) Modelo: el Pipeline completo (preprocesamiento + clasificador).
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=X_train.head(3),
        )

    return {"name": name, "run_id": run.info.run_id, **cv_metrics, **test_metrics}


def select_champion(results: list[dict]) -> dict:
    """Criterio de negocio escrito de antemano.

    Entre los modelos cuyo F1 de validación cruzada no es peor que el del
    baseline, gana el de mayor Recall de validación cruzada.
    """
    baseline = next(r for r in results if r["name"] == BASELINE_NAME)
    elegibles = [r for r in results if r["cv_f1_mean"] >= baseline["cv_f1_mean"]]
    return max(elegibles, key=lambda r: r["cv_recall_mean"])


def run_experiments() -> tuple[pd.DataFrame, dict]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X, y = build_dataset(load_raw_data())
    splits = split_data(X, y)
    data_sha256 = file_sha256(RAW_DATA_PATH)

    results = [
        run_candidate(name, clf, splits, data_sha256)
        for name, clf in get_candidates().items()
    ]
    champion = select_champion(results)
    mlflow.MlflowClient().set_tag(champion["run_id"], "champion", "true")

    table = pd.DataFrame(results).set_index("name")
    return table, champion


if __name__ == "__main__":
    table, champion = run_experiments()
    columnas = [
        "cv_recall_mean",
        "cv_f1_mean",
        "test_recall",
        "test_f1",
        "test_accuracy",
        "test_business_cost_usd",
    ]
    print(table[columnas].round(4).to_string())
    print(f"\nChampion: {champion['name']} (run_id {champion['run_id']})")
