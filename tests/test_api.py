"""API: contrato de entrada y salida, validación y umbral de decisión."""

import pytest

from tests.conftest import FixedProbabilityModel


def test_health_informa_la_version_del_modelo(client):
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"status": "ok", "model_version": "test"}


def test_predict_devuelve_prediccion_probabilidad_y_version(client, customer):
    respuesta = client.post("/predict", json=customer)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == {"prediction", "churn_probability", "model_version"}
    assert cuerpo["prediction"] in (0, 1)
    assert 0 <= cuerpo["churn_probability"] <= 1
    assert cuerpo["model_version"] == "test"


def test_valor_fuera_del_catalogo_devuelve_422_y_nombra_el_campo(client, customer):
    customer["InternetService"] = "Crypto"
    respuesta = client.post("/predict", json=customer)
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == ["body", "InternetService"]


def test_un_campo_obligatorio_faltante_devuelve_422(client, customer):
    del customer["Contract"]
    assert client.post("/predict", json=customer).status_code == 422


def test_un_campo_extra_devuelve_422(client, customer):
    customer["customerID"] = "7590-VHVEG"
    assert client.post("/predict", json=customer).status_code == 422


def test_un_numero_fuera_de_rango_devuelve_422(client, customer):
    customer["tenure"] = -1
    assert client.post("/predict", json=customer).status_code == 422


def test_total_charges_puede_omitirse_con_tenure_cero(client, customer):
    customer["tenure"] = 0
    del customer["TotalCharges"]
    assert client.post("/predict", json=customer).status_code == 200


@pytest.mark.parametrize(
    "probabilidad, clase",
    [(0.49, 0), (0.50, 1), (0.51, 1)],
)
def test_el_umbral_de_decision_es_0_5(make_client, customer, probabilidad, clase):
    with make_client(FixedProbabilityModel(probabilidad)) as test_client:
        cuerpo = test_client.post("/predict", json=customer).json()
    assert cuerpo["prediction"] == clase
    assert cuerpo["churn_probability"] == pytest.approx(probabilidad)
