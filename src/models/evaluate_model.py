import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN


def evaluate_model(input_path: Path, model_path: Path, metrics_path: Path) -> dict[str, float]:
    data = pd.read_csv(input_path)
    _, features_test, _, target_test = train_test_split(
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMN],
        test_size=0.25,
        random_state=42,
        stratify=data[TARGET_COLUMN],
    )
    model = joblib.load(model_path)
    predictions = model.predict(features_test)
    metrics = {
        "accuracy": float(accuracy_score(target_test, predictions)),
        "precision": float(precision_score(target_test, predictions, zero_division=0)),
        "recall": float(recall_score(target_test, predictions, zero_division=0)),
        "f1": float(f1_score(target_test, predictions, zero_division=0)),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment("weather-rain-prediction")
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=type(model).__name__,
    ):
        mlflow.log_params(model.get_params(deep=False))
        mlflow.log_param(
            "dataset_sha256", hashlib.sha256(input_path.read_bytes()).hexdigest()
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("metrics_path", type=Path)
    args = parser.parse_args()
    evaluate_model(args.input_path, args.model_path, args.metrics_path)


if __name__ == "__main__":
    main()