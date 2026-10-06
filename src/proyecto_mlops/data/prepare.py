"""Etapa 3 y 4: limpieza, construcción de X e y, y división train/test."""

import pandas as pd
from sklearn.model_selection import train_test_split

from proyecto_mlops.config import (
    FEATURES,
    ID_COLUMN,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza determinista, compartida por entrenamiento y API.

    - Elimina customerID (identificador, no aporta patrón).
    - Convierte TotalCharges a número; vacíos -> 0.0 (clientes con tenure=0).
    No aprende nada de los datos, por eso no genera fuga de datos.
    """
    out = df.copy()
    out = out.drop(columns=[ID_COLUMN], errors="ignore")
    total = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = total.fillna(0.0)
    return out


def build_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y): limpia, quita duplicados y mapea Churn a 1/0."""
    df = clean_features(raw_df).drop_duplicates()
    X = df[FEATURES]
    y = df[TARGET].map({"Yes": 1, "No": 0})
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    """División 80/20 estratificada y reproducible."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
