"""Genera tráfico de prueba hacia la API (normal o con deriva simulada)."""

import argparse
import json
import urllib.error
import urllib.request
from collections import Counter

from proyecto_mlops.config import RANDOM_STATE
from proyecto_mlops.data.load import load_raw_data
from proyecto_mlops.data.prepare import build_dataset


def apply_drift(rows: list[dict]) -> list[dict]:
    """Simula un cambio de población: contratos mensuales, recientes y más caros."""
    for row in rows:
        row["Contract"] = "Month-to-month"
        row["tenure"] = min(row["tenure"], 6)
        row["MonthlyCharges"] = round(row["MonthlyCharges"] * 1.5, 2)
        row["TotalCharges"] = round(row["MonthlyCharges"] * row["tenure"], 2)
    return rows


def post(url: str, payload: dict) -> int:
    """Envía un cliente a la API y devuelve el código HTTP."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulador de tráfico para la API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--n", type=int, default=200, help="Número de peticiones.")
    parser.add_argument("--drift", action="store_true", help="Simula deriva.")
    args = parser.parse_args()

    X, _ = build_dataset(load_raw_data())
    sample = X.sample(n=args.n, random_state=RANDOM_STATE, replace=len(X) < args.n)
    rows = json.loads(sample.to_json(orient="records"))
    if args.drift:
        rows = apply_drift(rows)

    codes: Counter = Counter()
    try:
        for row in rows:
            codes[post(args.url, row)] += 1
    except urllib.error.URLError as error:
        raise SystemExit(f"No se pudo conectar con {args.url}: {error.reason}")
    print(f"Peticiones enviadas: {args.n} | códigos HTTP: {dict(codes)}")


if __name__ == "__main__":
    main()
