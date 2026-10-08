"""Etapa de evaluación: métricas de negocio para churn."""

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from proyecto_mlops.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def estimated_cost(y_true, y_pred) -> dict[str, float]:
    """Costo simplificado de los errores, con los costos del caso de negocio.

    Supuesto: solo cuentan los errores. Un falso negativo (cliente que se va y
    no detectamos) cuesta 300 USD; un falso positivo (campaña innecesaria),
    40 USD. No es una métrica de selección: sirve para interpretar el Recall.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "business_cost_usd": float(fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE),
    }
