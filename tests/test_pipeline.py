import json
from pathlib import Path

from src.features.build_features import build_features
from src.models.evaluate_model import evaluate_model
from src.models.train_model import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_weather_pipeline(tmp_path: Path) -> None:
    processed_path = tmp_path / "weather_processed.csv"
    model_path = tmp_path / "weather_model.joblib"
    metrics_path = tmp_path / "metrics.json"

    build_features(PROJECT_ROOT / "data/raw/weather.csv", processed_path)
    train_model(processed_path, model_path)
    metrics = evaluate_model(processed_path, model_path, metrics_path)

    assert model_path.is_file()
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
    assert all(0.0 <= value <= 1.0 for value in metrics.values())