# MLOps · Predicción de churn en MedTelecomunicaciones

[![CI](https://github.com/jonathanrpo17-creator/mlops-churn-medtelecomunicaciones/actions/workflows/ci.yml/badge.svg)](https://github.com/jonathanrpo17-creator/mlops-churn-medtelecomunicaciones/actions/workflows/ci.yml)

Proyecto final del curso de MLOps (Universidad de Medellín). Convierte un notebook de
regresión logística en un proyecto reproducible: pipeline de scikit-learn, experimentos
y registro de modelos con MLflow, orquestación con Prefect, API con FastAPI, imagen
Docker, monitoreo de deriva, pruebas y CI. Todo corre en local, sin nube.

**Equipo:** `LEIDY TATIANA ARBOLEDA VELEZ`, `CINNDY VANESSA VELASQUEZ CASTAÑO`, `JONATHAN RESTREPO JARAMILLO` · **Entrega:** 13 de octubre de 2026

---

## 1. Problema de negocio (hipotético)

MedTelecomunicaciones pierde clientes que cancelan su servicio (*churn*). Retener a uno
cuesta mucho menos que conseguir uno nuevo, pero la campaña de retención no se puede
lanzar a todos. El modelo estima la probabilidad de cancelación de cada cliente para
que el área de fidelización priorice a quién contactar.

Costos **ilustrativos** (supuestos del equipo, no datos reales de una empresa):

| Error | Qué significa | Costo (USD) |
|---|---|---|
| Falso negativo | El cliente se iba y no lo detectamos | 300 |
| Falso positivo | Hicimos una campaña a quien no se iba | 40 |

Como perder un cliente cuesta ~7.5 veces más que una campaña innecesaria, **se prioriza
el recall** (detectar a quienes se van) sin dejar caer demasiado el F1.

## 2. Métricas de éxito y alcance

- **Métrica principal:** recall de la clase "churn" en validación cruzada (5 folds).
- **Métrica de control:** F1 en validación cruzada (no puede ser peor que el baseline).
- **Métrica de negocio:** costo total en el conjunto de prueba con los costos de la sección 1.
- **Regla de campeón (escrita antes de entrenar):** entre los modelos con F1 CV ≥ baseline,
  gana el de mayor recall CV.
- **Compuerta de producción:** un modelo solo pasa a `production` si su recall CV ≥ 0.60
  (supuesto del equipo, pendiente de validar con el área de fidelización).

**MVP incluido:** entrenamiento comparado, registro con alias, API de predicción,
contenedor, monitoreo sin etiquetas, pruebas y CI.
**Fuera de alcance:** nube, reentrenamiento automático por deriva, búsqueda de
hiperparámetros, XGBoost, dashboards (Grafana).

## 3. Resultados

Dataset: Telco Customer Churn (IBM), 7 043 clientes, 26.4 % de churn.
Partición 80/20 estratificada, `random_state=212`.

| Modelo | Recall CV | F1 CV | Recall test | F1 test | Accuracy test | Costo test (USD) |
|---|---|---|---|---|---|---|
| logreg_baseline | 0.5455 | 0.5942 | 0.5484 | 0.5982 | 0.8050 | 54 640 |
| **logreg_balanced (campeón)** | **0.7993** | 0.6266 | 0.8011 | 0.6228 | 0.7431 | **33 680** |
| random_forest | 0.5077 | 0.5791 | 0.5188 | 0.5831 | 0.8036 | 57 580 |
| random_forest_balanced | 0.7569 | 0.6315 | 0.7742 | 0.6365 | 0.7658 | 35 000 |
| hist_gradient_boosting | 0.5212 | 0.5681 | 0.5457 | 0.5901 | 0.7993 | 55 220 |

El campeón baja el costo de prueba ~38 % frente al baseline (54 640 → 33 680 USD) a cambio
de menor accuracy: marca más clientes como posible churn (≈41 % del tráfico normal, frente
a 26.4 % de churn real). Es una consecuencia buscada de priorizar recall.

## 4. Arquitectura

```
CSV (data/raw) ─► validación y limpieza ─► split 80/20 ─► 5 candidatos (CV + test)
                                                              │  cada uno = 1 run en MLflow
                                                              ▼
                              regla del campeón ─► Model Registry (alias staging → production)
                                                              │
                                   export ─► models/production ─► imagen Docker
                                                              │
                          FastAPI /predict ─► logs/predictions.jsonl ─► informe PSI (deriva)
```

Prefect orquesta el tramo de entrenamiento y registro. La función `clean_features` es la
misma en entrenamiento y en la API, para evitar *training-serving skew*.

```
.
├── src/proyecto_mlops/
│   ├── config.py          # constantes en un solo lugar
│   ├── data/              # descarga, validación y preparación
│   ├── models/            # pipeline sklearn, candidatos, experimentos, registro, export
│   ├── pipelines/         # flujo de Prefect
│   ├── api/               # FastAPI, esquemas Pydantic, carga del modelo
│   └── monitoring/        # log de predicciones, informe PSI, simulador
├── tests/                 # 50 pruebas (datos sintéticos, sin red ni MLflow)
├── notebooks/FundLogR.ipynb   # EDA y notebook original
├── docs/ejemplo_cliente.json  # cuerpo de ejemplo para /predict
├── Dockerfile, .dockerignore
├── .github/workflows/ci.yml
└── pyproject.toml, uv.lock
```

## 5. Cómo reproducirlo (paso a paso)

Probado en Windows 11 (PowerShell) y desde un clon limpio en Linux. Todos los comandos
se ejecutan **desde la raíz del proyecto**.

**Requisitos:** [uv](https://docs.astral.sh/uv/), Git, conexión a internet (para bajar el
CSV la primera vez) y, solo para el paso 8, Docker Desktop. Python 3.12 lo gestiona uv.

> Si el proyecto está dentro de OneDrive y `uv sync` falla con el error 396, ejecuta
> antes `$env:UV_LINK_MODE = "copy"` (o clona en una carpeta fuera de OneDrive).

### Paso 1 · Clonar e instalar
```powershell
git clone https://github.com/jonathanrpo17-creator/mlops-churn-medtelecomunicaciones.git
cd mlops-churn-medtelecomunicaciones
uv sync
```
Resultado: se crea `.venv` con las dependencias exactas de `uv.lock`.

### Paso 2 · Pruebas
```powershell
uv run pytest -q
```
Resultado: `50 passed`. No necesita datos ni red.

### Paso 3 · Datos
No se versionan. El primer comando que los necesite descarga el CSV a
`data/raw/telco_churn.csv` (fuente: repositorio público de IBM; la URL está en
`config.py`). Si prefieres hacerlo a mano, guarda ese archivo con ese nombre.

### Paso 4 · Entrenar y comparar modelos (MLflow)
```powershell
uv run python -m proyecto_mlops.models.experiments
```
Resultado: tabla con los 5 modelos (la de la sección 3) y el campeón marcado. Los runs
quedan en `mlflow.db`.

### Paso 5 · Registrar y promover
```powershell
uv run python -m proyecto_mlops.models.registry
```
Resultado: versión registrada como `churn-model`, alias `staging`, validación (recall CV ≥ 0.60)
y, si pasa, alias `production`. Para volver a una versión anterior (rollback):
```powershell
uv run python -m proyecto_mlops.models.registry rollback 1
```
Para explorar los experimentos en el navegador:
```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
(abre http://127.0.0.1:5000). El número de versión depende de cuántas veces hayas
registrado; en una instalación limpia es la 1.

### Paso 6 · Pipeline con Prefect (alternativa a los pasos 4 y 5)
```powershell
uv run python -m proyecto_mlops.pipelines.training_flow
```
Ejecuta el mismo proceso como flujo de 5 tareas con reintentos. Con `--serve` lo programa
(lunes 6:00 a. m., despliegue `churn-training-semanal`), pero deja la terminal ocupada
y requiere un servidor Prefect en ejecución; se documenta, no es necesario para la demo.

### Paso 7 · API local
```powershell
uv run uvicorn proyecto_mlops.api.main:app --port 8000
```
- Documentación interactiva: http://127.0.0.1:8000/docs
- Salud: `Invoke-RestMethod http://127.0.0.1:8000/health`
- Predicción:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -InFile docs/ejemplo_cliente.json
```
Resultado esperado (clon limpio, modelo campeón):
```json
{"prediction": 1, "churn_probability": 0.8757, "model_version": "1"}
```
Un valor fuera del catálogo (por ejemplo `"InternetService": "Satellite"`) devuelve **422**.
Cada petición exitosa agrega una línea a `logs/predictions.jsonl`.

### Paso 8 · Docker
Primero exporta el modelo de producción (la imagen es autocontenida, no necesita MLflow):
```powershell
uv run python -m proyecto_mlops.models.export
docker build -t churn-api .
docker run --rm -p 8000:8000 -v "${PWD}/logs:/app/logs" churn-api
```
Resultado: la misma API en http://127.0.0.1:8000. La primera construcción tarda entre 6 y
10 minutos. La imagen ocupa ~487 MB de contenido (~2.2 GB en disco).

### Paso 9 · Monitoreo
Con la API en marcha, en otra terminal:
```powershell
uv run python -m proyecto_mlops.monitoring.simulate --n 1000          # tráfico normal
uv run python -m proyecto_mlops.monitoring.report                     # informe
uv run python -m proyecto_mlops.monitoring.simulate --n 300 --drift   # tráfico con deriva
uv run python -m proyecto_mlops.monitoring.report
```
Resultado: con tráfico normal todas las variables quedan en PSI < 0.01 (estable); con
deriva, `tenure`, `TotalCharges`, `Contract` y `MonthlyCharges` pasan a **ALERTA** y el
churn predicho sube a ~62 %. Para repetir la demo desde cero, borra `logs/predictions.jsonl`.

### Calidad de código
```powershell
uv run black --check src tests
uv run flake8 src tests
```
El mismo conjunto (más pytest) corre en GitHub Actions en cada push.

## 6. Referencia de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Devuelve `{"status": "ok", "model_version": ...}` |
| POST | `/predict` | Recibe los 19 campos del cliente; devuelve `prediction` (0/1), `churn_probability` y `model_version` |
| GET | `/docs` | Swagger UI con esquema y ejemplo |

Validación estricta: campos categóricos como `Literal` (valores del dataset), `tenure`
entre 0 y 120, y campos desconocidos rechazados (`extra="forbid"`). El umbral de decisión
es 0.5 (`prediction = probability >= 0.5`).

## 7. Decisiones técnicas

| Decisión | Alternativa | Por qué |
|---|---|---|
| uv + `uv.lock` | pip / Poetry | Entornos exactos y rápidos; un solo archivo de verdad |
| Pipeline sklearn único (`ColumnTransformer`) | Preprocesar a mano | Mismo código en entrenamiento y API; sin fuga de datos |
| `handle_unknown="ignore"` | Fallar ante categorías nuevas | La API no se cae por un valor inesperado |
| Mantener regresión logística (balanceada) | RF / boosting | Gana en recall CV; es explicable y barata |
| Umbral 0.5 | Optimizarlo por costo | Se prefirió `class_weight="balanced"`; el umbral se deja como mejora |
| MLflow con SQLite y aliases | Stages clásicos | Los stages están obsoletos; los aliases permiten rollback inmediato |
| Prefect | Scripts sueltos | Reintentos, trazabilidad y programación |
| Imagen con modelo exportado | Imagen que consulta MLflow | Arranca sin depender de la base de MLflow |
| PSI sin etiquetas | Métricas de desempeño | Las etiquetas reales llegan tarde; la deriva se ve antes |
| Tests con datos sintéticos y modelo falso | Usar el CSV real | Rápidos, sin red, corren en CI |

## 8. Cumplimiento de la guía del curso

| Requisito | Dónde está |
|---|---|
| Métricas de éxito, alcance, problema de negocio | Secciones 1 y 2 |
| Timeline con tiempos y responsables | Sección 10 |
| Repositorio Git, uv, dependencias | `pyproject.toml`, `uv.lock` |
| EDA y baseline | `notebooks/FundLogR.ipynb`, modelo `logreg_baseline` |
| MLflow: tracking, parámetros, métricas, CV | `models/experiments.py` |
| Algoritmos comparados | Regresión logística, Random Forest, HistGradientBoosting |
| Model Registry: versiones, tags, staging/production | `models/registry.py` |
| Prefect: flujo y scheduling | `pipelines/training_flow.py` (`--serve`) |
| Validación y logging de datos | `data/load.py`, `data/prepare.py` |
| Dockerfile | `Dockerfile` |
| API FastAPI con validación | `api/` |
| Diseño de monitoreo | `monitoring/` y sección 9 |
| Unit tests | `tests/` (50) |
| flake8 y black | `.flake8`, CI |
| CI con GitHub Actions | `.github/workflows/ci.yml` |
| Documentación y guías | Este README |

**No incluido:** XGBoost, búsqueda de hiperparámetros, pre-commit, despliegue en nube
(`deploy.yml`), reentrenamiento automático.

## 9. Monitoreo: qué mide y qué no

- **Aplicación:** latencia de cada predicción (`latency_ms`) y versión del modelo en el log; la API devuelve 422 ante entradas inválidas.
- **Datos (deriva):** PSI por variable contra el dataset de entrenamiento. Numéricas: deciles
  de la referencia. Categóricas: proporción por categoría. Umbrales: < 0.10 estable,
  0.10–0.25 atención, > 0.25 alerta.
- **Predicciones:** proporción de clientes marcados como churn.
- **Mínimo de datos:** el informe de deriva exige 200 predicciones registradas; con menos, el
  PSI es ruido (≈ (bins − 1) / n).
- **No mide** desempeño real del modelo (no hay etiquetas en producción) ni dispara
  reentrenamiento: es informativo.

## 10. Equipo y cronograma

> **Completar antes de entregar** (el curso pide tiempos y responsables).

| Actividad | Responsable | Fecha |
|---|---|---|
| EDA y baseline | `<NOMBRE>` | `<FECHA>` |
| Experimentos y MLflow | `<NOMBRE>` | `<FECHA>` |
| Pipeline Prefect | `<NOMBRE>` | `<FECHA>` |
| API y Docker | `<NOMBRE>` | `<FECHA>` |
| Monitoreo y pruebas | `<NOMBRE>` | `<FECHA>` |
| README y sustentación | `<NOMBRE>` | `<FECHA>` |

## 11. Limitaciones conocidas

- Los costos (300 / 40 USD) y el recall mínimo de 0.60 son supuestos, no datos de negocio.
- El campeón marca ≈ 41 % de los clientes como churn: útil para priorizar, costoso si la
  campaña es cara. Optimizar el umbral por costo sería el siguiente paso.
- Dataset público de ejemplo, sin dimensión temporal: la deriva real solo se demuestra
  con tráfico simulado.
- Docker no se construye en CI porque `models/production` no se versiona.
- El log de predicciones es un archivo local (sin rotación ni concurrencia múltiple).
- Una advertencia de Starlette sobre `httpx` en los tests es inofensiva.

## 12. Licencia y datos

Proyecto académico. Dataset: *Telco Customer Churn*, publicado por IBM.
