"""Informe de monitoreo: resume el registro de predicciones y busca deriva."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from proyecto_mlops.config import (
    CATEGORICAL_FEATURES,
    MIN_LOGGED_ROWS,
    NUMERIC_FEATURES,
    PSI_ALERT,
    PSI_WARNING,
)
from proyecto_mlops.data.load import load_raw_data
from proyecto_mlops.data.prepare import build_dataset
from proyecto_mlops.monitoring.prediction_log import log_path

EPSILON = 1e-4  # evita dividir entre cero cuando un grupo no tiene datos


def read_log(path: Path) -> pd.DataFrame:
    """Lee el registro y devuelve una fila por predicción."""
    if not path.exists():
        raise SystemExit(f"No hay registro en {path}. Haz predicciones primero.")
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        raise SystemExit(f"El registro {path} está vacío.")
    frame = pd.json_normalize(rows)
    frame.columns = [name.removeprefix("features.") for name in frame.columns]
    return frame


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Índice de estabilidad de la población entre dos distribuciones."""
    expected = np.clip(expected, EPSILON, None)
    actual = np.clip(actual, EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI de una variable numérica con cortes por deciles del entrenamiento."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    return psi(expected, actual)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI de una variable categórica comparando la proporción de cada valor."""
    categories = sorted(set(reference) | set(current), key=str)
    expected = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    actual = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    return psi(expected.to_numpy(), actual.to_numpy())


def drift_status(value: float) -> str:
    """Traduce el PSI a una etiqueta (umbrales de config.py)."""
    if value >= PSI_ALERT:
        return "ALERTA"
    if value >= PSI_WARNING:
        return "atención"
    return "estable"


def print_summary(log: pd.DataFrame, reference_churn_rate: float) -> None:
    """Resumen de uso y del comportamiento del modelo (no necesita etiquetas)."""
    print("=== Resumen de predicciones ===")
    print(f"Predicciones registradas : {len(log)}")
    first, last = log["timestamp"].min(), log["timestamp"].max()
    print(f"Periodo (UTC)            : {first} -> {last}")
    versions = log["model_version"].value_counts().to_dict()
    print(f"Versiones del modelo     : {versions}")
    print(f"Probabilidad media       : {log['churn_probability'].mean():.3f}")
    print(f"Clientes marcados churn  : {log['prediction'].mean():.1%}")
    print(f"Churn real en entrenam.  : {reference_churn_rate:.1%} (solo referencia)")
    print(
        f"Latencia mediana / p95   : {log['latency_ms'].median():.1f} ms / "
        f"{log['latency_ms'].quantile(0.95):.1f} ms"
    )


def print_drift(log: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Compara las entradas recibidas con los datos de entrenamiento."""
    results = []
    for column in NUMERIC_FEATURES:
        current = pd.to_numeric(log[column])
        results.append((column, psi_numeric(reference[column], current)))
    for column in CATEGORICAL_FEATURES:
        results.append((column, psi_categorical(reference[column], log[column])))
    results.sort(key=lambda item: item[1], reverse=True)

    print("\n=== Deriva de las entradas (PSI frente al entrenamiento) ===")
    for column, value in results[:8]:
        print(f"{column:<18} PSI={value:6.3f}  {drift_status(value)}")
    alerts = sum(value >= PSI_ALERT for _, value in results)
    warnings = sum(PSI_WARNING <= value < PSI_ALERT for _, value in results)
    print(
        f"\nVariables en alerta: {alerts} | en atención: {warnings} "
        f"| total revisadas: {len(results)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Informe de monitoreo del modelo.")
    parser.add_argument("--path", help="Ruta del registro (por defecto, el de la API).")
    args = parser.parse_args()

    log = read_log(Path(args.path) if args.path else log_path())
    X_ref, y_ref = build_dataset(load_raw_data())
    print_summary(log, float(y_ref.mean()))
    if len(log) < MIN_LOGGED_ROWS:
        print(
            f"\nHay menos de {MIN_LOGGED_ROWS} predicciones: "
            "el análisis de deriva aún no es fiable."
        )
        return
    print_drift(log, X_ref)


if __name__ == "__main__":
    main()
