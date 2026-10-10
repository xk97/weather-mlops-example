import json
from pathlib import Path

import pandas as pd
import pytest

from src.features.build_features import build_features
from src.models.evaluate_model import evaluate_model
from src.models.predict_model import predict_file
from src.models.train_model import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_weather_pipeline(tmp_path: Path) -> None:
    processed_path = tmp_path / "weather_processed.csv"
    model_path = tmp_path / "weather_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.csv"

    build_features(PROJECT_ROOT / "data/raw/weather.csv", processed_path)
    train_model(processed_path, model_path)
    metrics = evaluate_model(processed_path, model_path, metrics_path)

    assert model_path.is_file()
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
    assert all(0.0 <= value <= 1.0 for value in metrics.values())

    predictions = predict_file(
        PROJECT_ROOT / "data/raw/weather.csv", model_path, predictions_path
    )
    assert predictions_path.is_file()
    assert len(predictions) == 32
    assert set(predictions["RainTomorrowPrediction"]) <= {"No", "Yes"}
    assert predictions["RainTomorrowProbability"].between(0.0, 1.0).all()


def test_prediction_rejects_missing_features(tmp_path: Path) -> None:
    input_path = tmp_path / "incomplete.csv"
    input_path.write_text("MinTemp,MaxTemp\n10,20\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required feature columns"):
        predict_file(input_path, tmp_path / "unused.joblib", tmp_path / "predictions.csv")