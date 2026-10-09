"""Métricas, costo de negocio y regla de selección del champion."""

import pytest

from proyecto_mlops.config import BASELINE_NAME
from proyecto_mlops.models.evaluate import compute_metrics, estimated_cost
from proyecto_mlops.models.experiments import select_champion


def test_compute_metrics_con_un_caso_conocido():
    # TP=2, FN=1, FP=1, TN=2  ->  recall 2/3, accuracy 4/6, f1 2/3
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    metricas = compute_metrics(y_true, y_pred)
    assert metricas["recall"] == pytest.approx(2 / 3)
    assert metricas["accuracy"] == pytest.approx(4 / 6)
    assert metricas["f1"] == pytest.approx(2 / 3)


def test_estimated_cost_usa_los_costos_del_caso_de_negocio():
    # 2 falsos negativos (2 x 300 USD) + 1 falso positivo (40 USD) = 640 USD
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 0]
    costo = estimated_cost(y_true, y_pred)
    assert costo["false_negatives"] == 2
    assert costo["false_positives"] == 1
    assert costo["business_cost_usd"] == 640.0


def resultado(nombre, f1, recall):
    return {"name": nombre, "cv_f1_mean": f1, "cv_recall_mean": recall}


def test_el_champion_es_el_de_mayor_recall_entre_los_elegibles():
    resultados = [
        resultado(BASELINE_NAME, 0.60, 0.55),
        resultado("balanceado", 0.62, 0.80),
        resultado("otro", 0.61, 0.70),
    ]
    assert select_champion(resultados)["name"] == "balanceado"


def test_un_modelo_con_f1_menor_que_el_baseline_no_puede_ganar():
    resultados = [
        resultado(BASELINE_NAME, 0.60, 0.55),
        resultado("recall_alto_pero_f1_bajo", 0.40, 0.95),
        resultado("razonable", 0.61, 0.65),
    ]
    assert select_champion(resultados)["name"] == "razonable"


def test_si_nadie_supera_al_baseline_gana_el_baseline():
    resultados = [
        resultado(BASELINE_NAME, 0.60, 0.55),
        resultado("peor", 0.50, 0.90),
    ]
    assert select_champion(resultados)["name"] == BASELINE_NAME
