import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from src.features.build_features import build_features
from src.models.evaluate_model import evaluate_model
from src.models.predict_model import predict_file
from src.models.promote_model import promote_run
from src.models.train_model import train_model
from src.service import app, create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_weather_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed_path = tmp_path / "weather_processed.csv"
    model_path = tmp_path / "weather_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.csv"
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    build_features(PROJECT_ROOT / "data/raw/weather.csv", processed_path)
    train_model(processed_path, model_path)
    metrics = evaluate_model(processed_path, model_path, metrics_path)

    assert model_path.is_file()
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
    assert all(0.0 <= value <= 1.0 for value in metrics.values())

    mlflow_client = MlflowClient(tracking_uri=tracking_uri)
    experiment = mlflow_client.get_experiment_by_name("weather-rain-prediction")
    assert experiment is not None
    runs = mlflow_client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.metrics["f1"] == metrics["f1"]
    assert run.data.params["n_estimators"] == "100"
    assert len(run.data.params["dataset_sha256"]) == 64
    logged_models = mlflow_client.search_logged_models(
        experiment_ids=[experiment.experiment_id]
    )
    assert any(model.source_run_id == run.info.run_id for model in logged_models)
    model_name = "weather-rain-classifier"
    version = promote_run(run.info.run_id, model_name, tracking_uri)
    assert version == "1"
    assert mlflow_client.get_model_version_by_alias(model_name, "champion").run_id == run.info.run_id

    assert promote_run(run.info.run_id, model_name, tracking_uri) == version
    assert len(mlflow_client.search_model_versions(f"name='{model_name}'")) == 1

    registry_uri = f"models:/{model_name}@champion"
    monkeypatch.setenv("MODEL_URI", registry_uri)
    with TestClient(create_app(model_version=f"{model_name}@champion")) as registry_client:
        registry_response = registry_client.post(
            "/predict",
            json={
                "MinTemp": 16.1,
                "MaxTemp": 22.4,
                "Rainfall": 8.2,
                "Humidity3pm": 82,
                "Pressure3pm": 1004.6,
            },
        )
    assert registry_response.status_code == 200
    assert registry_response.json()["model_version"] == f"{model_name}@champion"

    predictions = predict_file(
        PROJECT_ROOT / "data/raw/weather.csv", model_path, predictions_path
    )
    assert predictions_path.is_file()
    assert len(predictions) == 32
    assert set(predictions["RainTomorrowPrediction"]) <= {"No", "Yes"}
    assert predictions["RainTomorrowProbability"].between(0.0, 1.0).all()

    with TestClient(create_app(model_path, model_version="test-version")) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model_version": "test-version",
        }
        response = client.post(
            "/predict",
            json={
                "MinTemp": 16.1,
                "MaxTemp": 22.4,
                "Rainfall": 8.2,
                "Humidity3pm": 82,
                "Pressure3pm": 1004.6,
            },
        )
        assert response.status_code == 200
        assert response.json()["prediction"] in {"No", "Yes"}
        assert 0.0 <= response.json()["rain_probability"] <= 1.0
        assert response.json()["model_version"] == "test-version"

        invalid_response = client.post("/predict", json={"MinTemp": 10})
        assert invalid_response.status_code == 422
        metrics = client.get("/metrics").text
        assert "weather_prediction_requests_total" in metrics
        assert "weather_prediction_errors_total" in metrics
        assert "weather_predictions_total" in metrics


def test_prediction_rejects_missing_features(tmp_path: Path) -> None:
    input_path = tmp_path / "incomplete.csv"
    input_path.write_text("MinTemp,MaxTemp\n10,20\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required feature columns"):
        predict_file(input_path, tmp_path / "unused.joblib", tmp_path / "predictions.csv")


def test_default_service_loads_dvc_model() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_version"] == "local"