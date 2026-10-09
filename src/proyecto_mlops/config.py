"""Configuración central del proyecto: un solo lugar para las constantes."""

from pathlib import Path

# --- Datos -------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
# Ruta relativa a la raíz del proyecto (ejecutar los comandos desde allí).
RAW_DATA_PATH = Path("data/raw/telco_churn.csv")

# --- Reproducibilidad --------------------------------------------------
RANDOM_STATE = 212
TEST_SIZE = 0.20

# --- Columnas (declaradas por nombre, no inferidas por tipo) -----------
TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPECTED_RAW_COLUMNS = [ID_COLUMN] + FEATURES + [TARGET]


# --- Experimentos y MLflow ---------------------------------------------
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "churn-medtelecomunicaciones"
CV_FOLDS = 5
BASELINE_NAME = "logreg_baseline"

# Costos de negocio (USD) del caso MedTelecomunicaciones.
COST_FALSE_NEGATIVE = 300  # cliente que se va y no detectamos
COST_FALSE_POSITIVE = 40  # campaña de retención innecesaria


# --- Model Registry ----------------------------------------------------
MODEL_NAME = "churn-model"
ALIAS_STAGING = "staging"
ALIAS_PRODUCTION = "production"
# Recall mínimo (validación cruzada) para promover un modelo a producción.
# Supuesto del equipo, pendiente de validar con el área de fidelización.
MIN_CV_RECALL = 0.60


# --- API ---------------------------------------------------------------
# Umbral para convertir la probabilidad en clase 0/1 (decisión 3: se mantiene).
DECISION_THRESHOLD = 0.5

# --- Monitoreo (Fase 9) ---------------------------------------------------
# Archivo donde la API registra cada predicción (una línea JSON por petición).
PREDICTION_LOG_FILE = "logs/predictions.jsonl"
# Umbrales habituales del PSI: < 0.10 estable, 0.10-0.25 atención, > 0.25 alerta.
PSI_WARNING = 0.10
PSI_ALERT = 0.25
# Con menos predicciones registradas el informe de deriva no es fiable.
MIN_LOGGED_ROWS = 200
