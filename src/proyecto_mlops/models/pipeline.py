"""Pipeline de scikit-learn: preprocesamiento + clasificador en un solo objeto."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from proyecto_mlops.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def build_pipeline(
    classifier=None,
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
) -> Pipeline:
    """Construye el Pipeline sin entrenar.

    Si no se indica clasificador se usa la Regresión Logística del notebook.
    """
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(numeric_features)),
            (
                "cat",
                OneHotEncoder(
                    sparse_output=False, drop="first", handle_unknown="ignore"
                ),
                list(categorical_features),
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[("preprocesamiento", preprocesador), ("clasificador", classifier)]
    )
