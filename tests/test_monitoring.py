"""Registro de predicciones e indicadores de deriva (PSI)."""

import json

import numpy as np
import pandas as pd
import pytest

from proyecto_mlops.monitoring import report
from proyecto_mlops.monitoring.prediction_log import log_prediction


def leer_registro(ruta):
    return [json.loads(linea) for linea in ruta.read_text().splitlines()]


def test_la_api_registra_cada_prediccion_valida(client, customer, log_file):
    client.post("/predict", json=customer)
    client.post("/predict", json=customer)
    filas = leer_registro(log_file)
    assert len(filas) == 2
    assert filas[0]["model_version"] == "test"
    assert filas[0]["features"] == customer
    assert filas[0]["latency_ms"] >= 0


def test_una_peticion_invalida_no_se_registra(client, customer, log_file):
    customer["InternetService"] = "Crypto"
    assert client.post("/predict", json=customer).status_code == 422
    assert not log_file.exists()


def test_log_prediction_crea_la_carpeta_y_agrega_lineas(log_file):
    log_prediction({"tenure": 1}, 0.7, 1, "4", 12.3456)
    log_prediction({"tenure": 2}, 0.2, 0, "4", 8.0)
    filas = leer_registro(log_file)
    assert [f["prediction"] for f in filas] == [1, 0]
    assert filas[0]["latency_ms"] == 12.35


def test_un_fallo_al_escribir_el_registro_no_rompe_la_api(
    make_client, monkeypatch, tmp_path, customer, fitted_pipeline
):
    bloqueo = tmp_path / "archivo.txt"
    bloqueo.write_text("no soy una carpeta")
    # La ruta cuelga de un archivo, así que no se puede crear la carpeta.
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(bloqueo / "predictions.jsonl"))
    with make_client(fitted_pipeline) as test_client:
        assert test_client.post("/predict", json=customer).status_code == 200


def test_psi_coincide_con_un_calculo_hecho_a_mano():
    # (0.25 - 0.5) * ln(0.25 / 0.5) + (0.75 - 0.5) * ln(0.75 / 0.5) = 0.2747
    esperado = np.array([0.5, 0.5])
    actual = np.array([0.25, 0.75])
    assert report.psi(esperado, actual) == pytest.approx(0.2747, abs=1e-4)


def test_psi_es_cero_con_distribuciones_iguales():
    datos = pd.Series(np.random.default_rng(0).normal(size=2000))
    assert report.psi_numeric(datos, datos) == pytest.approx(0, abs=1e-9)


def test_psi_detecta_un_desplazamiento_numerico():
    rng = np.random.default_rng(0)
    referencia = pd.Series(rng.normal(0, 1, 5000))
    desplazada = pd.Series(rng.normal(1.5, 1, 500))
    assert report.psi_numeric(referencia, desplazada) > 0.25


def test_psi_cuenta_los_valores_fuera_del_rango_de_entrenamiento():
    """Un cliente fuera del rango visto en entrenamiento no puede ignorarse."""
    referencia = pd.Series(range(100))  # 10 intervalos con 10 % cada uno
    fuera_de_rango = pd.Series([1000] * 50)  # todo cae en el último intervalo
    assert report.psi_numeric(referencia, fuera_de_rango) == pytest.approx(
        8.283, abs=0.01
    )


def test_psi_categorico_detecta_un_cambio_de_proporciones():
    referencia = pd.Series(["A"] * 50 + ["B"] * 50)
    estable = pd.Series(["A"] * 52 + ["B"] * 48)
    cambiada = pd.Series(["A"] * 95 + ["B"] * 5)
    assert report.psi_categorical(referencia, estable) < 0.10
    assert report.psi_categorical(referencia, cambiada) > 0.25


def test_psi_categorico_maneja_categorias_que_no_estaban_en_la_referencia():
    referencia = pd.Series(["A"] * 100)
    nueva = pd.Series(["A"] * 50 + ["Z"] * 50)
    assert np.isfinite(report.psi_categorical(referencia, nueva))


@pytest.mark.parametrize(
    "valor, etiqueta",
    [(0.05, "estable"), (0.10, "atención"), (0.20, "atención"), (0.25, "ALERTA")],
)
def test_drift_status_usa_los_umbrales_de_config(valor, etiqueta):
    assert report.drift_status(valor) == etiqueta


def test_read_log_falla_con_un_mensaje_claro_si_no_hay_registro(tmp_path):
    with pytest.raises(SystemExit, match="No hay registro"):
        report.read_log(tmp_path / "no_existe.jsonl")


def test_read_log_aplana_las_variables_de_entrada(log_file):
    log_prediction({"tenure": 5, "Contract": "One year"}, 0.3, 0, "4", 1.0)
    tabla = report.read_log(log_file)
    assert {"tenure", "Contract", "churn_probability"} <= set(tabla.columns)
    assert tabla.loc[0, "tenure"] == 5
