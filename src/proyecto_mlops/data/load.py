"""Etapa 1 y 2: carga de datos y validación del esquema."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from proyecto_mlops.config import (
    DATA_URL,
    EXPECTED_RAW_COLUMNS,
    RAW_DATA_PATH,
    TARGET,
)


def download_data(path: Path = RAW_DATA_PATH, url: str = DATA_URL) -> Path:
    """Descarga el CSV solo si todavía no existe en disco."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)
    return path


def validate_raw_data(df: pd.DataFrame) -> None:
    """Falla con un mensaje claro si los datos no tienen el formato esperado."""
    if df.empty:
        raise ValueError("El conjunto de datos está vacío.")

    missing = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en los datos: {missing}")

    if df[TARGET].isna().any():
        raise ValueError(f"La columna {TARGET} tiene valores nulos.")

    invalid = set(df[TARGET].unique()) - {"Yes", "No"}
    if invalid:
        raise ValueError(f"Valores inesperados en {TARGET}: {sorted(invalid)}")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Descarga (si hace falta), lee y valida el CSV original."""
    csv_path = download_data(path)
    df = pd.read_csv(csv_path)
    validate_raw_data(df)
    return df
