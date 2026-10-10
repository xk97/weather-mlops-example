FROM python:3.12-slim

ARG MODEL_VERSION=local
ENV MODEL_PATH=/app/models/weather_model.joblib \
    MODEL_VERSION=${MODEL_VERSION} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt \
    && useradd --system --create-home app

COPY --chown=app:app src/ ./src/
COPY --chown=app:app models/weather_model.joblib ./models/weather_model.joblib

USER app
EXPOSE 8000

CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]