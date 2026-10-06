"""Etapa de evaluación: métricas de negocio para churn."""

from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
