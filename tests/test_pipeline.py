"""Pipeline de scikit-learn: columnas, probabilidades y categorías nuevas."""

import pandas as pd
import pytest

from proyecto_mlops.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from proyecto_mlops.data.prepare import build_dataset
from proyecto_mlops.models.pipeline import build_pipeline


def test_el_preprocesamiento_cubre_todas_las_variables():
    """Ninguna variable del modelo puede quedar fuera por un descuido."""
    preprocesador = build_pipeline().named_steps["preprocesamiento"]
    cubiertas = [c for _, _, columnas in preprocesador.transformers for c in columnas]
    assert sorted(cubiertas) == sorted(FEATURES)
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_las_probabilidades_estan_entre_cero_y_uno(fitted_pipeline, raw_df):
    X, _ = build_dataset(raw_df)
    proba = fitted_pipeline.predict_proba(X)[:, 1]
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_una_categoria_nueva_no_rompe_la_prediccion(fitted_pipeline, raw_df):
    """Decisión 5: handle_unknown='ignore' evita errores con valores no vistos."""
    X, _ = build_dataset(raw_df)
    nuevo = X.head(1).copy()
    nuevo["PaymentMethod"] = "Crypto"
    proba = fitted_pipeline.predict_proba(nuevo)[:, 1]
    assert 0 <= proba[0] <= 1


def test_el_pipeline_es_reproducible(raw_df):
    X, y = build_dataset(raw_df)
    a = build_pipeline().fit(X, y).predict_proba(X)
    b = build_pipeline().fit(X, y).predict_proba(X)
    assert a == pytest.approx(b)


def test_el_pipeline_rechaza_datos_sin_una_columna(fitted_pipeline, raw_df):
    X, _ = build_dataset(raw_df)
    with pytest.raises(Exception):
        fitted_pipeline.predict_proba(X.drop(columns=["Contract"]))


def test_el_pipeline_acepta_un_cliente_de_la_api(fitted_pipeline, customer):
    from proyecto_mlops.data.prepare import clean_features

    frame = clean_features(pd.DataFrame([customer]))
    assert fitted_pipeline.predict_proba(frame).shape == (1, 2)
