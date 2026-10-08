FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

COPY models/production ./models/production

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_URI=/app/models/production

EXPOSE 8000

CMD ["uvicorn", "proyecto_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
