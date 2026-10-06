"""Entrenamiento del modelo base (se extiende con MLflow en la Fase 4)."""

from proyecto_mlops.config import CATEGORICAL_FEATURES
from proyecto_mlops.data.load import load_raw_data
from proyecto_mlops.data.prepare import build_dataset, split_data
from proyecto_mlops.models.evaluate import compute_metrics
from proyecto_mlops.models.pipeline import build_pipeline


def train_baseline(categorical_features=CATEGORICAL_FEATURES):
    """Carga -> validación -> limpieza -> split -> entrenamiento -> métricas."""
    raw_df = load_raw_data()
    X, y = build_dataset(raw_df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_pipeline(categorical_features=categorical_features)
    model.fit(X_train, y_train)

    metrics = compute_metrics(y_test, model.predict(X_test))
    return model, metrics


if __name__ == "__main__":
    _, metrics = train_baseline()
    for name, value in metrics.items():
        print(f"{name:<10} {value:.4f}")
