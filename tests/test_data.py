"""Carga, validación y limpieza de datos."""

import pandas as pd
import pytest

from proyecto_mlops.config import FEATURES, ID_COLUMN, TARGET
from proyecto_mlops.data import load
from proyecto_mlops.data.load import validate_raw_data
from proyecto_mlops.data.prepare import build_dataset, clean_features, split_data


def test_clean_features_convierte_total_charges_vacio_en_cero(raw_df):
    raw_df.loc[0, "TotalCharges"] = " "
    cleaned = clean_features(raw_df)
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])


def test_clean_features_elimina_el_identificador(raw_df):
    assert ID_COLUMN not in clean_features(raw_df).columns


def test_clean_features_no_modifica_el_original(raw_df):
    antes = raw_df.copy()
    clean_features(raw_df)
    pd.testing.assert_frame_equal(raw_df, antes)


def test_clean_features_acepta_los_datos_de_la_api(customer):
    """La API envía TotalCharges numérico y sin customerID: debe funcionar."""
    cleaned = clean_features(pd.DataFrame([customer]))
    assert cleaned.loc[0, "TotalCharges"] == customer["TotalCharges"]


def test_build_dataset_devuelve_features_y_objetivo_binario(raw_df):
    X, y = build_dataset(raw_df)
    assert list(X.columns) == FEATURES
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


def test_build_dataset_elimina_duplicados(raw_df):
    con_duplicados = pd.concat([raw_df, raw_df.iloc[[0]].copy()], ignore_index=True)
    con_duplicados.loc[len(con_duplicados) - 1, ID_COLUMN] = "OTRO-ID"
    # Mismo cliente con distinto ID: tras quitar el ID es una fila repetida.
    X, _ = build_dataset(con_duplicados)
    assert len(X) == len(raw_df)


def test_split_data_es_reproducible_y_estratificado(raw_df):
    X, y = build_dataset(raw_df)
    primera = split_data(X, y)
    segunda = split_data(X, y)
    pd.testing.assert_frame_equal(primera[0], segunda[0])
    _, X_test, _, y_test = primera
    assert len(X_test) == pytest.approx(0.20 * len(X), abs=2)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.03)


def test_validate_raw_data_acepta_datos_correctos(raw_df):
    validate_raw_data(raw_df)  # no debe lanzar excepción


def test_validate_raw_data_rechaza_datos_vacios(raw_df):
    with pytest.raises(ValueError, match="vacío"):
        validate_raw_data(raw_df.iloc[0:0])


def test_validate_raw_data_rechaza_columnas_faltantes(raw_df):
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_raw_data(raw_df.drop(columns=["Contract"]))


def test_validate_raw_data_rechaza_objetivo_con_nulos(raw_df):
    raw_df.loc[0, TARGET] = None
    with pytest.raises(ValueError, match="nulos"):
        validate_raw_data(raw_df)


def test_validate_raw_data_rechaza_objetivo_con_valores_raros(raw_df):
    raw_df.loc[0, TARGET] = "Maybe"
    with pytest.raises(ValueError, match="Valores inesperados"):
        validate_raw_data(raw_df)


def test_download_data_no_descarga_si_el_archivo_existe(tmp_path, monkeypatch):
    archivo = tmp_path / "datos.csv"
    archivo.write_text("a,b\n1,2\n")

    def no_deberia_llamarse(*args, **kwargs):
        raise AssertionError("Se intentó descargar un archivo que ya existe")

    monkeypatch.setattr(load, "urlretrieve", no_deberia_llamarse)
    assert load.download_data(archivo) == archivo
