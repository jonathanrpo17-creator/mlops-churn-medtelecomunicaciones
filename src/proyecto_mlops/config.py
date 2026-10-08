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
