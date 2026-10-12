import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from src.features.build_features import FEATURE_COLUMNS


PREDICTION_REQUESTS = Counter(
    "weather_prediction_requests_total", "Prediction endpoint requests"
)
PREDICTION_ERRORS = Counter(
    "weather_prediction_errors_total", "Prediction endpoint errors"
)
PREDICTIONS = Counter(
    "weather_predictions_total", "Predictions by outcome", ["outcome"]
)
PREDICTION_DURATION = Histogram(
    "weather_prediction_duration_seconds", "Prediction request duration"
)


class WeatherObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    MinTemp: float = Field(allow_inf_nan=False)
    MaxTemp: float = Field(allow_inf_nan=False)
    Rainfall: float = Field(allow_inf_nan=False)
    Humidity3pm: float = Field(allow_inf_nan=False)
    Pressure3pm: float = Field(allow_inf_nan=False)


class PredictionResponse(BaseModel):
    prediction: Literal["No", "Yes"]
    rain_probability: float
    model_version: str


def create_app(
    model_path: Path | None = None,
    model_version: str | None = None,
    model_uri: str | None = None,
) -> FastAPI:
    resolved_model_uri = model_uri or os.environ.get("MODEL_URI")
    resolved_model_path = model_path or Path(
        os.environ.get("MODEL_PATH", "models/weather_model.joblib")
    )
    resolved_model_version = (
        model_version
        or os.environ.get("MODEL_VERSION")
        or resolved_model_uri
        or "local"
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if resolved_model_uri:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            application.state.model = mlflow.sklearn.load_model(resolved_model_uri)
        else:
            application.state.model = joblib.load(resolved_model_path)
        yield

    application = FastAPI(
        title="Weather prediction service",
        version=resolved_model_version,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def observe_prediction_requests(request: Request, call_next):
        if request.url.path != "/predict":
            return await call_next(request)

        PREDICTION_REQUESTS.inc()
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            PREDICTION_ERRORS.inc()
            raise
        finally:
            PREDICTION_DURATION.observe(perf_counter() - started_at)

        if response.status_code >= 400:
            PREDICTION_ERRORS.inc()
        return response

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": resolved_model_version}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(observation: WeatherObservation, request: Request) -> PredictionResponse:
        features = pd.DataFrame(
            [observation.model_dump()], columns=FEATURE_COLUMNS
        ) # type: ignore
        model = request.app.state.model
        positive_class_index = next(
            (index for index, label in enumerate(model.classes_) if label == 1), None
        )
        if positive_class_index is None:
            raise RuntimeError("Loaded model does not contain the positive rain class (1).")

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0, positive_class_index])
        outcome = "Yes" if prediction == 1 else "No"
        PREDICTIONS.labels(outcome.lower()).inc()
        return PredictionResponse(
            prediction=outcome,
            rain_probability=probability,
            model_version=resolved_model_version,
        )

    @application.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()