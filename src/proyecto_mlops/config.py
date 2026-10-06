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
